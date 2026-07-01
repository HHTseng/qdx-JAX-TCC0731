"""Replay a saved GNN-QDX checkpoint on custom (N, K, D) tasks.

This script is standalone: it loads the checkpoint directly from
`examples/results/demo_multitask_513_nkd_20260701031130`, reconstructs the
model from `run_config.json`, and evaluates custom tasks in two ways:

1. Greedy / maximum-probability action selection.
2. Stochastic sampling, repeated `RANDOM_SAMPLES` times per task.

The greedy output matches the validation print format used by
`examples/demo_multitask_nkd.py`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from qdx.envs.graph_code_discovery import GraphCodeDiscovery
from qdx.gnn import GraphPadding
from qdx.make_train import make_actor_critic
from qdx.simulators.clifford_gates import CliffordGates
from qdx.utils import Utils


SOURCE_RESULT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "demo_multitask_513_nkd_20260701031130"
)

# Default to the tasks that were already in the source checkpoint's validation set.
DEFAULT_CUSTOM_TASKS = (
    (5, 1, 3),
    (6, 1, 3),
    (7, 1, 3),
    (8, 1, 3),
    (8, 2, 3),
    (8, 3, 3),
    (9, 1, 3),
    (9, 2, 3),
    (9, 3, 3),
    (10, 1, 3),
    (10, 2, 3),
    (10, 3, 3),
    (10, 4, 3),
)

RANDOM_SAMPLES = 100
DEFAULT_OUTPUT_PATH = SOURCE_RESULT_DIR / "greedy_sampling_validation.json"


def all_to_all_graph(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def make_env(n, k, d, config, graph_padding):
    gates = CliffordGates(n)
    gate_set = [getattr(gates, gate_name) for gate_name in config["WHICH_GATES"]]
    return GraphCodeDiscovery(
        n,
        k,
        d,
        gate_set,
        graph=all_to_all_graph(n),
        max_steps=config["MAX_STEPS"],
        lbda=config["LAMBDA"],
        pI=config["P_I"],
        softness=config["SOFTNESS"],
        graph_padding=graph_padding,
    )


def normalize_task_specs(task_specs, default_task_specs, default_distance):
    raw_specs = default_task_specs if task_specs is None else task_specs
    normalized = []
    for task in raw_specs:
        if isinstance(task, dict):
            n = task["n"]
            k = task["k"]
            target_distance = task.get(
                "d", task.get("target_distance", default_distance)
            )
        else:
            if len(task) == 2:
                n, k = task
                target_distance = default_distance
            elif len(task) == 3:
                n, k, target_distance = task
            else:
                raise ValueError(
                    "tasks must be (n, k) or (n, k, d) tuples "
                    "or dictionaries with n, k, d"
                )

        n = int(n)
        k = int(k)
        target_distance = int(target_distance)
        if n <= 0 or k <= 0 or target_distance <= 0:
            raise ValueError("task values must be positive integers")
        if k >= n:
            raise ValueError(f"task {(n, k, target_distance)} must satisfy k < n")
        normalized.append(
            {
                "n": n,
                "k": k,
                "target_distance": target_distance,
            }
        )

    if not normalized:
        raise ValueError("at least one task is required")
    return normalized


def build_graph_padding(task_specs):
    if not task_specs:
        raise ValueError("at least one task is required to build graph padding")
    max_n = max(task["n"] for task in task_specs)
    max_stabilizers = max(task["n"] - task["k"] for task in task_specs)
    return GraphPadding(
        n_max=max_n,
        stabilizers_max=max_stabilizers,
        hardware_edges_max=max_n * max(max_n - 1, 0),
    )


def load_saved_run(result_dir):
    run_config_path = result_dir / "run_config.json"
    params_path = result_dir / "params.msgpack"
    if not run_config_path.exists():
        raise FileNotFoundError(f"missing run config: {run_config_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"missing parameter file: {params_path}")

    with run_config_path.open("r", encoding="utf-8") as file:
        run_config = json.load(file)
    return run_config, params_path.read_bytes()


def load_params_for_saved_run(params_bytes, run_config, graph_padding, reference_task):
    reference_config = dict(run_config)
    reference_config["D"] = reference_task["target_distance"]
    env = make_env(
        reference_task["n"],
        reference_task["k"],
        reference_config["D"],
        reference_config,
        graph_padding,
    )
    network = make_actor_critic(reference_config, env)
    template_params = network.init(
        jax.random.PRNGKey(0), env.graph_observation_template()
    )
    return serialization.from_bytes(template_params, params_bytes)


def format_task(task):
    return f"N={task['n']} K={task['k']} D={task['target_distance']}"


def distance_up_to_target(n, k, gates, target_distance):
    """Return the first failing Pauli weight, checking through target_distance."""

    utilities = Utils(n, k, gates, softness=n - k)
    first_failure = target_distance + 1
    for weight in range(1, target_distance + 1):
        error_operators = utilities.error_operators(weight)
        error_count = int(utilities.check_KL(error_operators))
        if error_count != 0:
            first_failure = weight
            break
    return first_failure


def distance_error_stats_up_to_target(n, k, gates, target_distance):
    """Return the first failing Pauli weight and per-d KL error statistics."""

    utilities = Utils(n, k, gates, softness=n - k)
    distance_stats = []
    first_failure = target_distance + 1
    for weight in range(1, target_distance + 1):
        error_operators = utilities.error_operators(weight)
        error_count = int(utilities.check_KL(error_operators))
        total_count = int(error_operators.shape[0])
        error_rate = error_count / total_count if total_count else 0.0
        distance_stats.append(
            {
                "d": weight,
                "error_count": error_count,
                "total_count": total_count,
                "error_count_over_total": f"{error_count}/{total_count}",
                "error_rate": error_rate,
            }
        )
        if error_count != 0 and first_failure == target_distance + 1:
            first_failure = weight
    return first_failure, distance_stats


def format_distance_stats(distance_stats):
    if not distance_stats:
        return "distance_stats=[]"
    formatted = "; ".join(
        (
            f"d={item['d']} "
            f"error_count/total_count={item['error_count_over_total']} "
            f"error_rate={item['error_rate']:.2%}"
        )
        for item in distance_stats
    )
    return f"distance_stats=[{formatted}]"


def aggregate_distance_stats(results):
    stats_by_d = {}
    for result in results:
        for item in result.get("distance_stats", []) or []:
            d = item["d"]
            summary = stats_by_d.setdefault(
                d,
                {
                    "d": d,
                    "error_count": 0,
                    "total_count": 0,
                },
            )
            summary["error_count"] += item["error_count"]
            summary["total_count"] += item["total_count"]

    aggregated = []
    for d in sorted(stats_by_d):
        error_count = stats_by_d[d]["error_count"]
        total_count = stats_by_d[d]["total_count"]
        aggregated.append(
            {
                "d": d,
                "error_count": error_count,
                "total_count": total_count,
                "error_count_over_total": f"{error_count}/{total_count}",
                "error_rate": error_count / total_count if total_count else 0.0,
            }
        )
    return aggregated


def rollout_policy(params, env, network, task_config, rng, use_sampling):
    observation, state = env.reset(rng, None)
    gates = []
    total_reward = 0.0
    final_reward = float("nan")
    final_value = float("nan")
    done = False
    steps = 0
    for _ in range(task_config["MAX_STEPS"]):
        policy, value = network.apply(params, observation)
        final_value = float(value)
        if use_sampling:
            rng, action_rng = jax.random.split(rng)
            action = int(policy.sample(seed=action_rng))
        else:
            action = int(policy.mode())
        gates.append(env.action_string_stim[action])
        rng, step_rng = jax.random.split(rng)
        observation, state, reward, done, _ = env.step(
            step_rng, state, action, None
        )
        final_reward = float(reward)
        total_reward += final_reward
        steps += 1
        if bool(done):
            break
    return gates, total_reward, final_reward, final_value, steps


def build_batched_sampling_rollout(params, env, network, task_config):
    # Batch independent stochastic rollouts with one JAX trace.
    max_steps = int(task_config["MAX_STEPS"])

    def one_rollout(rng):
        observation, state = env.reset(rng, None)
        done = jnp.asarray(False)
        final_reward = jnp.asarray(0.0, dtype=jnp.float32)

        def body(_, carry):
            observation, state, rng, done, final_reward = carry

            def take_step(carry):
                observation, state, rng, done, final_reward = carry
                rng, action_rng = jax.random.split(rng)
                rng, step_rng = jax.random.split(rng)
                policy, _ = network.apply(params, observation)
                action = jnp.asarray(policy.sample(seed=action_rng), dtype=jnp.int32)
                next_observation, next_state, reward, next_done, _ = env.step(
                    step_rng, state, action, None
                )
                next_done = jnp.asarray(next_done, dtype=jnp.bool_)
                return (
                    next_observation,
                    next_state,
                    rng,
                    jnp.logical_or(done, next_done),
                    jnp.asarray(reward, dtype=jnp.float32),
                )

            return jax.lax.cond(
                done,
                lambda carry: carry,
                take_step,
                (observation, state, rng, done, final_reward),
            )

        _, _, _, _, final_reward = jax.lax.fori_loop(
            0,
            max_steps,
            body,
            (observation, state, rng, done, final_reward),
        )
        return final_reward

    return jax.jit(jax.vmap(one_rollout))


def evaluate_greedy(params, run_config, tasks, graph_padding):
    results = []
    print(f"Validating on {len(tasks)} tasks...")
    for task_index, task in enumerate(tasks):
        task_config = dict(run_config)
        task_config["D"] = task["target_distance"]
        env = make_env(
            task["n"], task["k"], task["target_distance"], task_config, graph_padding
        )
        network = make_actor_critic(task_config, env)
        rng = jax.random.PRNGKey(task_config["SEED"] + 10_000 + task_index)
        gates, total_reward, final_reward, final_value, steps = rollout_policy(
            params, env, network, task_config, rng, use_sampling=False
        )

        distance, distance_stats = distance_error_stats_up_to_target(
            task["n"], task["k"], gates, task["target_distance"]
        )
        distance_stats_text = format_distance_stats(distance_stats)
        target_met = distance >= task["target_distance"]
        result = {
            "n": task["n"],
            "k": task["k"],
            "target_distance": task["target_distance"],
            "distance": distance,
            "distance_stats": distance_stats,
            "target_met": target_met,
            "steps": steps,
            "total_reward": total_reward,
            "final_reward": final_reward,
            "final_value": final_value,
            "gates": gates,
        }
        results.append(result)
        print(
            f"  N={task['n']} K={task['k']} D={task['target_distance']}: "
            f"distance={distance} target_met={target_met} steps={steps} "
            f"{distance_stats_text}"
        )

    distance_summary = aggregate_distance_stats(results) if results else None
    if distance_summary:
        print("Distance summary across validation tasks:")
        for item in distance_summary:
            print(
                "  "
                f"d={item['d']} "
                f"error_count/total_count={item['error_count_over_total']} "
                f"error_rate={item['error_rate']:.2%}"
            )

    return {
        "target_distance": (
            tasks[0]["target_distance"]
            if len({task["target_distance"] for task in tasks}) == 1
            else None
        ),
        "target_distances": sorted({task["target_distance"] for task in tasks}),
        "compute_distance": True,
        "tasks": results,
        "distance_summary": distance_summary,
    }


def evaluate_sampling(params, run_config, tasks, graph_padding, num_samples, greedy=None):
    if num_samples <= 0:
        raise ValueError("--random-samples must be a positive integer")

    results = []
    greedy_by_task = {}
    if greedy is not None:
        greedy_by_task = {
            (item["n"], item["k"], item["target_distance"]): item
            for item in greedy.get("tasks", [])
        }
    print(f"Sampling on {len(tasks)} tasks with {num_samples} runs each...")
    for task_index, task in enumerate(tasks):
        task_config = dict(run_config)
        task_config["D"] = task["target_distance"]
        env = make_env(
            task["n"], task["k"], task["target_distance"], task_config, graph_padding
        )
        network = make_actor_critic(task_config, env)
        base_rng = jax.random.PRNGKey(task_config["SEED"] + 20_000 + task_index)
        batched_rollout = build_batched_sampling_rollout(
            params, env, network, task_config
        )
        sample_indices = jnp.arange(num_samples, dtype=jnp.int32)
        sample_rngs = jax.vmap(
            lambda sample_index: jax.random.fold_in(base_rng, sample_index)
        )(sample_indices)
        sampled_final_rewards = np.asarray(batched_rollout(sample_rngs))

        # Success is encoded directly by the terminal reward, so we do not need
        # to reconstruct gates or recompute KL-distance statistics here.
        success_count = int(np.count_nonzero(np.isclose(sampled_final_rewards, 0.0, atol=1.0e-6)))

        success_rate = success_count / num_samples
        greedy_result = greedy_by_task.get(
            (task["n"], task["k"], task["target_distance"])
        )
        result = {
            "n": task["n"],
            "k": task["k"],
            "target_distance": task["target_distance"],
            "sample_count": num_samples,
            "success_count": success_count,
            "success_rate": success_rate,
            "greedy_distance": None if greedy_result is None else greedy_result["distance"],
            "greedy_target_met": None
            if greedy_result is None
            else greedy_result["target_met"],
            "greedy_steps": None if greedy_result is None else greedy_result["steps"],
        }
        results.append(result)
        greedy_text = ""
        if greedy_result is not None:
            greedy_text = (
                f"greedy_distance={greedy_result['distance']} "
                f"greedy_target_met={greedy_result['target_met']} "
            )
        print(
            f"  {format_task(task)}: "
            f"{greedy_text}"
            f"success_rate={success_rate:.2%} "
            f"success_count={success_count}/{num_samples}"
        )

    success_rates = [
        item["success_rate"] for item in results if item["success_rate"] is not None
    ]
    if success_rates:
        print(
            f"Average sampling success rate across tasks: "
            f"{float(np.mean(success_rates)):.2%}"
        )

    return {
        "sample_count": num_samples,
        "tasks": results,
    }


def save_results(output_path, source_result_dir, run_config, tasks, greedy, sampling):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_result_dir": str(source_result_dir),
        "run_config": run_config,
        "tasks": tasks,
        "greedy": greedy,
        "sampling": sampling,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved replay results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        nargs=3,
        type=int,
        metavar=("N", "K", "D"),
        help="Add one evaluation task; repeat the flag for multiple tasks.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=RANDOM_SAMPLES,
        help="Number of stochastic rollouts per task.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_config, params_bytes = load_saved_run(SOURCE_RESULT_DIR)
    if run_config.get("MODEL", "GNN").upper() != "GNN":
        raise ValueError(
            "This replay script expects a GNN checkpoint so it can be reused "
            "across larger padded graph shapes."
        )
    if run_config.get("ENV_TYPE", "STANDARD").upper() != "STANDARD":
        raise ValueError("This replay script expects ENV_TYPE='STANDARD'.")

    tasks = normalize_task_specs(
        args.tasks,
        DEFAULT_CUSTOM_TASKS,
        run_config["D"],
    )
    graph_padding = build_graph_padding(tasks)
    reference_task = max(
        tasks,
        key=lambda item: (item["n"], item["n"] - item["k"], item["target_distance"]),
    )
    params = load_params_for_saved_run(
        params_bytes, run_config, graph_padding, reference_task
    )

    print(f"Loaded saved checkpoint from {SOURCE_RESULT_DIR}")
    greedy = evaluate_greedy(params, run_config, tasks, graph_padding)
    sampling = evaluate_sampling(
        params,
        run_config,
        tasks,
        graph_padding,
        num_samples=args.random_samples,
        greedy=greedy,
    )
    save_results(
        args.output_path,
        SOURCE_RESULT_DIR,
        run_config,
        tasks,
        greedy,
        sampling,
    )


if __name__ == "__main__":
    main()
