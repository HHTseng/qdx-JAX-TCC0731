"""Train one GNN-QDX policy across multiple (N, K) tasks and validate it.

The default PPO settings follow the STANDARD example in notebooks/demo.ipynb.
The 2,000,000 timestep budget is shared across all training task visits.

Run:
    conda run -n qdx python examples/demo_multitask_nk.py
    conda run -n qdx python examples/demo_multitask_nk.py --dry-run
"""

import argparse
import itertools
import json
import os
from datetime import datetime
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from qdx.envs.graph_code_discovery import GraphCodeDiscovery
from qdx.gnn import GraphPadding
from qdx.make_train import make_actor_critic, make_train
from qdx.simulators.clifford_gates import CliffordGates
from qdx.utils import Utils


TRAIN_N = (6, 7, 8, 9)
VALIDATION_N = (5, 6, 7, 8, 9, 10)
K_VALUES = (1, 2)
TRAIN_TASKS = tuple(itertools.product(TRAIN_N, K_VALUES))
VALIDATION_TASKS = tuple(itertools.product(VALIDATION_N, K_VALUES))

BASE_CONFIG = {
    "MODEL": "GNN",
    "ENV_TYPE": "STANDARD",
    "D": 3,
    "MAX_STEPS": 50,
    "WHICH_GATES": ("cx", "h"),
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "P_I": 0.9,
    "LAMBDA": 10,
    "SEED": 42,
    "LR": 1.0e-3,
    "NUM_ENVS": 16,
    "NUM_STEPS": 50,
    "TOTAL_TIMESTEPS": 2_000_000,
    "UPDATE_EPOCHS": 3,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.02,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.25,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 32,
    "ANNEAL_LR": True,
    "COMPUTE_METRICS": True,
    "GNN_HIDDEN_DIM": 64,
    "GNN_RELATION_DIM": 8,
    "GNN_GATE_DIM": 8,
    "GNN_NUM_LAYERS": 3,
}

# N=10 is the largest validation task. K=1 gives at most nine stabilizers.
PADDING = GraphPadding(
    n_max=10,
    stabilizers_max=9,
    hardware_edges_max=10 * 9,
    actions_max=10 * 9 + 10,  # all directed CX edges + all H actions
)


def all_to_all_graph(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def make_env(n, k, config):
    gates = CliffordGates(n)
    gate_set = [getattr(gates, gate_name) for gate_name in config["WHICH_GATES"]]
    return GraphCodeDiscovery(
        n,
        k,
        config["D"],
        gate_set,
        graph=all_to_all_graph(n),
        max_steps=config["MAX_STEPS"],
        lbda=config["LAMBDA"],
        pI=config["P_I"],
        softness=config["SOFTNESS"],
        graph_padding=PADDING,
    )


def config_for_task(base_config, n, k, timesteps):
    config = dict(base_config)
    config.update({"N": n, "K": k, "TOTAL_TIMESTEPS": int(timesteps)})
    return config


def metric_tail_mean(metrics, name, tail=10):
    if metrics is None:
        return None
    values = np.asarray(metrics[name]).reshape(-1)
    if values.size == 0:
        return None
    return float(np.mean(values[-tail:]))


def default_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(__file__).resolve().parent / "results" / f"demo_multitask_nk_{timestamp}"


def format_metric(value):
    if value is None:
        return "nan"
    return f"{value:.2f}"


def train_multitask(base_config, total_timesteps, train_rounds):
    """Sequential curriculum that carries the same GNN parameters across tasks."""

    visits = len(TRAIN_TASKS) * train_rounds
    rollout_size = base_config["NUM_ENVS"] * base_config["NUM_STEPS"]
    timesteps_per_visit = max(rollout_size, total_timesteps // visits)
    timesteps_per_visit -= timesteps_per_visit % rollout_size

    environments = {
        task: make_env(*task, base_config)
        for task in TRAIN_TASKS
    }
    task_configs = {
        task: config_for_task(base_config, *task, timesteps_per_visit)
        for task in TRAIN_TASKS
    }
    # Each task has different tableau/KL shapes and therefore its own JIT executable.
    trainers = {
        task: jax.jit(make_train(task_configs[task], environments[task]))
        for task in TRAIN_TASKS
    }

    params = None
    rng = jax.random.PRNGKey(base_config["SEED"])
    history = []
    print(
        f"Training {len(TRAIN_TASKS)} tasks for {train_rounds} round(s); "
        f"{timesteps_per_visit:,} timesteps per task visit."
    )

    for round_index in range(train_rounds):
        round_started = time.perf_counter()
        task_order = list(TRAIN_TASKS)
        random.Random(base_config["SEED"] + round_index).shuffle(task_order)
        round_records = []
        for n, k in task_order:
            started = time.perf_counter()
            rng, task_rng = jax.random.split(rng)
            output = jax.block_until_ready(
                trainers[(n, k)](task_rng, params)
            )
            params = output["params"]
            metrics = output["metrics"]
            record = {
                "round": round_index + 1,
                "n": n,
                "k": k,
                "timesteps": timesteps_per_visit,
                "seconds": time.perf_counter() - started,
                "return_tail_mean": metric_tail_mean(
                    metrics, "returned_episode_returns"
                ),
                "length_tail_mean": metric_tail_mean(
                    metrics, "returned_episode_lengths"
                ),
            }
            history.append(record)
            round_records.append(record)
            print(
                f"round={record['round']} N={n} K={k} "
                f"return={format_metric(record['return_tail_mean'])} "
                f"length={format_metric(record['length_tail_mean'])} "
                f"time={record['seconds']:.1f}s"
            )
        round_seconds = time.perf_counter() - round_started
        round_returns = [
            record["return_tail_mean"]
            for record in round_records
            if record["return_tail_mean"] is not None
        ]
        round_lengths = [
            record["length_tail_mean"]
            for record in round_records
            if record["length_tail_mean"] is not None
        ]
        avg_return = float(np.mean(round_returns)) if round_returns else float("nan")
        avg_length = float(np.mean(round_lengths)) if round_lengths else float("nan")
        print(
            f"round={round_index + 1} "
            f"avg_return={avg_return:.2f} "
            f"avg_length={avg_length:.2f} "
            f"total_time={round_seconds:.1f}s"
        )

    return params, history


def distance_up_to_target(n, k, gates, target_distance):
    """Return the first failing Pauli weight, checking through target_distance."""

    utilities = Utils(n, k, gates, softness=n - k)
    for weight in range(1, target_distance + 1):
        failures = int(utilities.check_KL(utilities.error_operators(weight)))
        if failures != 0:
            return weight
    return target_distance + 1


def validate(params, base_config, compute_distance=True):
    results = []
    print(f"Validating on {len(VALIDATION_TASKS)} tasks...")
    for task_index, (n, k) in enumerate(VALIDATION_TASKS):
        env = make_env(n, k, base_config)
        network = make_actor_critic(base_config, env)
        rng = jax.random.PRNGKey(base_config["SEED"] + 10_000 + task_index)
        observation, state = env.reset(rng, None)

        gates = []
        total_reward = 0.0
        final_reward = float("nan")
        final_value = float("nan")
        done = False
        for step in range(base_config["MAX_STEPS"]):
            policy, value = network.apply(params, observation)
            action = int(policy.mode())
            gates.append(env.action_string_stim[action])
            rng, step_rng = jax.random.split(rng)
            observation, state, reward, done, _ = env.step(
                step_rng, state, action, None
            )
            final_reward = float(reward)
            final_value = float(value)
            total_reward += final_reward
            if bool(done):
                break

        distance = (
            distance_up_to_target(n, k, gates, base_config["D"])
            if compute_distance
            else None
        )
        target_met = (
            distance >= base_config["D"]
            if distance is not None
            else bool(jnp.isclose(final_reward, 0.0, atol=1.0e-6))
        )
        result = {
            "n": n,
            "k": k,
            "target_distance": base_config["D"],
            "distance": distance,
            "target_met": target_met,
            "steps": len(gates),
            "total_reward": total_reward,
            "final_reward": final_reward,
            "final_value": final_value,
            "gates": gates,
        }
        results.append(result)
        print(
            f"  N={n} K={k}: distance={distance} "
            f"target_met={target_met} steps={len(gates)}"
        )
    return results


def dry_run(base_config):
    """Check that every requested task shares graph/action tensor shapes."""

    first_env = make_env(*TRAIN_TASKS[0], base_config)
    network = make_actor_critic(base_config, first_env)
    first_observation, _ = first_env.reset(jax.random.PRNGKey(0), None)
    params = network.init(jax.random.PRNGKey(1), first_observation)
    expected_shapes = jax.tree.map(lambda x: x.shape, first_observation)

    for task_index, (n, k) in enumerate(VALIDATION_TASKS):
        env = make_env(n, k, base_config)
        observation, _ = env.reset(jax.random.PRNGKey(task_index + 2), None)
        shapes = jax.tree.map(lambda x: x.shape, observation)
        if shapes != expected_shapes:
            raise ValueError(f"Task {(n, k)} has incompatible graph shapes")
        policy, value = network.apply(params, observation)
        print(
            f"N={n} K={k}: nodes={int(observation.node_mask.sum())}, "
            f"actions={int(observation.action_mask.sum())}, "
            f"logits={policy.logits.shape}, value_shape={value.shape}"
        )


def save_results(output_dir, params, history, validation, run_config):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))
    for name, value in (
        ("train_history.json", history),
        ("validation.json", validation),
        ("run_config.json", run_config),
    ):
        with (output_dir / name).open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
    print(f"Saved checkpoint and results to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=BASE_CONFIG["TOTAL_TIMESTEPS"],
        help="Total PPO timesteps shared by all training task visits.",
    )
    parser.add_argument("--train-rounds", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=BASE_CONFIG["NUM_ENVS"])
    parser.add_argument("--seed", type=int, default=BASE_CONFIG["SEED"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--skip-distance",
        action="store_true",
        help="Skip the more expensive post-rollout distance calculation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build all validation tasks and run shared-parameter forward passes.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir()
    return args


def main():
    args = parse_args()
    if args.train_rounds < 1:
        raise ValueError("--train-rounds must be at least one")
    config = dict(BASE_CONFIG)
    config.update({"NUM_ENVS": args.num_envs, "SEED": args.seed})

    if args.dry_run:
        dry_run(config)
        return

    params, history = train_multitask(
        config,
        total_timesteps=args.total_timesteps,
        train_rounds=args.train_rounds,
    )
    validation = validate(
        params,
        config,
        compute_distance=not args.skip_distance,
    )
    run_config = {
        **config,
        "WHICH_GATES": list(config["WHICH_GATES"]),
        "train_tasks": [list(task) for task in TRAIN_TASKS],
        "validation_tasks": [list(task) for task in VALIDATION_TASKS],
        "requested_total_timesteps": args.total_timesteps,
        "train_rounds": args.train_rounds,
    }
    save_results(args.output_dir, params, history, validation, run_config)


if __name__ == "__main__":
    main()
