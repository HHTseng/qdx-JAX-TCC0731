"""Train and validate one shared GNN-QDX policy on custom (N, K, D) tasks.

Instead of hard-coding all training and validation tasks, you can pass them
one-by-one with repeated ``--train-task`` and ``--validation-task`` flags.
Each task is specified as ``N K D``.

Run:
    conda run -n qdx python examples/demo_multitask_nkd.py
    conda run -n qdx python examples/demo_multitask_nkd.py \
        --train-task 5 1 3 --train-task 6 1 3 \
        --validation-task 7 1 3 --validation-task 8 1 3
    conda run -n qdx python examples/demo_multitask_nkd.py --dry-run
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")

from flax import serialization
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import optax
import jax
import jax.numpy as jnp
import numpy as np

from qdx.envs.graph_code_discovery import GraphCodeDiscovery
from qdx.gnn import GraphPadding
from qdx.make_train import make_actor_critic
from qdx.simulators.clifford_gates import CliffordGates
from qdx.utils import Utils


BASE_CONFIG = {
    "MODEL": "GNN",
    "ENV_TYPE": "STANDARD",
    "D": 3,
    "MAX_STEPS": 50,
    "WHICH_GATES": ("cx", "h", "s", "sqrt_x", "cz", "sqrt_xx"),
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "P_I": 0.9,
    "LAMBDA": 10,
    "SEED": 42,
    "LR": 1.0e-3,
    "NUM_ENVS_PER_TASK": 16,
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

DEFAULT_TRAIN_TASKS = (
    (5, 1, 3),
    (6, 1, 3),
    (7, 1, 3),
)
DEFAULT_VALIDATION_TASKS = (
    (5, 1, 3),
    (6, 1, 3),
    (7, 1, 3),
    (8, 1, 3),
    (8, 2, 3),
    (8, 3, 3),
)


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


def normalize_task_specs(task_specs, default_task_specs):
    raw_specs = default_task_specs if task_specs is None else task_specs
    normalized = []
    for task in raw_specs:
        if isinstance(task, dict):
            n = task["n"]
            k = task["k"]
            d = task.get("d", task.get("target_distance", BASE_CONFIG["D"]))
        else:
            if len(task) != 3:
                raise ValueError(
                    "task specs must be 3-tuples of the form (n, k, d)"
                )
            n, k, d = task

        n = int(n)
        k = int(k)
        d = int(d)
        if n <= 0 or k <= 0 or d <= 0:
            raise ValueError("task values must be positive integers")
        if k >= n:
            raise ValueError(f"task {(n, k, d)} must satisfy k < n")
        normalized.append({"n": n, "k": k, "d": d})

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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: Any


class PPOBatch(NamedTuple):
    obs: Any
    action: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    advantages: jnp.ndarray
    targets: jnp.ndarray


def format_metric(value):
    if value is None:
        return "nan"
    return f"{value:.2f}"


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def completed_episode_mean(info, name):
    if info is None or name not in info:
        return None
    values = np.asarray(info[name]).reshape(-1)
    returned_episode = info.get("returned_episode")
    if returned_episode is not None:
        mask = np.asarray(returned_episode).astype(bool).reshape(-1)
        values = values[mask]
    if values.size == 0:
        return None
    return float(np.mean(values))


def completed_episode_values(info, name):
    if info is None or name not in info:
        return None
    values = np.asarray(info[name]).reshape(-1)
    returned_episode = info.get("returned_episode")
    if returned_episode is not None:
        mask = np.asarray(returned_episode).astype(bool).reshape(-1)
        values = values[mask]
    return values


def rollout_success_stats(traj_batch, max_steps):
    done = np.asarray(traj_batch.done).astype(bool).reshape(-1)
    episode_count = int(np.sum(done))
    episode_lengths = completed_episode_values(
        traj_batch.info, "returned_episode_lengths"
    )
    if episode_lengths is None:
        success_count = 0
    else:
        episode_count = int(episode_lengths.size)
        success_count = int(np.sum(episode_lengths < max_steps))
    timeout_count = episode_count - success_count
    success_rate = success_count / episode_count if episode_count else None
    return {
        "episode_count": episode_count,
        "success_count": success_count,
        "timeout_count": timeout_count,
        "success_rate": success_rate,
    }


def create_train_state(config, network, params, num_updates):
    def linear_schedule(count):
        updates_completed = count // (
            config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        )
        frac = 1.0 - (updates_completed / num_updates)
        return config["LR"] * frac

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def build_rollout_collector(config, env, network):
    num_envs = config["NUM_ENVS_PER_TASK"]

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            delta = (
                transition.reward
                + config["GAMMA"] * next_value * (1 - transition.done)
                - transition.value
            )
            gae = (
                delta
                + config["GAMMA"]
                * config["GAE_LAMBDA"]
                * (1 - transition.done)
                * gae
            )
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def collect_rollout(params, env_state, last_obs, rng):
        def _env_step(runner_state, unused):
            env_state, last_obs, rng = runner_state

            rng, policy_rng = jax.random.split(rng)
            pi, value = network.apply(params, last_obs)
            action = pi.sample(seed=policy_rng)
            log_prob = pi.log_prob(action)

            rng, step_rng = jax.random.split(rng)
            rng_step = jax.random.split(step_rng, num_envs)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, None)
            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs,
                info=info,
            )
            return (env_state, obsv, rng), transition

        (env_state, last_obs, rng), traj_batch = jax.lax.scan(
            _env_step,
            (env_state, last_obs, rng),
            None,
            length=config["NUM_STEPS"],
        )
        _, last_val = network.apply(params, last_obs)
        advantages, targets = _calculate_gae(traj_batch, last_val)
        return (env_state, last_obs, rng), (traj_batch, advantages, targets)

    return jax.jit(collect_rollout)


def build_joint_update_fn(config, network, batch_size):
    def _loss_fn(params, batch):
        pi, value = network.apply(params, batch.obs)
        log_prob = pi.log_prob(batch.action)

        value_pred_clipped = batch.value + (
            value - batch.value
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - batch.targets)
        value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        gae = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1.0e-8
        )
        ratio = jnp.exp(log_prob - batch.log_prob)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(
            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
        ) * gae
        actor_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
        entropy = pi.entropy().mean()

        total_loss = (
            actor_loss
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
        )
        metrics = {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "entropy": entropy,
        }
        return total_loss, metrics

    def _update_minibatch(train_state, batch):
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def update(train_state, batch, rng):
        flat_batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]),
            batch,
        )

        def _update_epoch(update_state, unused):
            train_state, flat_batch, rng = update_state
            rng, permutation_rng = jax.random.split(rng)
            permutation = jax.random.permutation(permutation_rng, batch_size)
            shuffled = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0),
                flat_batch,
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (config["NUM_MINIBATCHES"], -1) + x.shape[1:]
                ),
                shuffled,
            )
            train_state, metrics = jax.lax.scan(
                _update_minibatch, train_state, minibatches
            )
            return (train_state, flat_batch, rng), metrics

        (train_state, _, rng), metrics = jax.lax.scan(
            _update_epoch,
            (train_state, flat_batch, rng),
            None,
            length=config["UPDATE_EPOCHS"],
        )
        return train_state, metrics, rng

    return jax.jit(update)


def initialize_task_state(env, num_envs_per_task, rng):
    rng, reset_rng = jax.random.split(rng)
    reset_rng = jax.random.split(reset_rng, num_envs_per_task)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, None)
    return {
        "env_state": env_state,
        "last_obs": obs,
        "rng": rng,
    }


def merge_task_batches(task_batches):
    return jax.tree_util.tree_map(
        lambda *parts: jnp.concatenate(parts, axis=1), *task_batches
    )


def summarize_loss_metrics(loss_metrics):
    return {
        name: float(np.mean(np.asarray(values)))
        for name, values in loss_metrics.items()
    }


def distance_up_to_target(n, k, gates, target_distance):
    """Return the first failing Pauli weight, checking through target_distance."""

    first_failure, _ = distance_error_stats_up_to_target(
        n, k, gates, target_distance
    )
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


def default_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return (
        Path(__file__).resolve().parent
        / "results"
        / f"demo_multitask_nkd_{timestamp}"
    )


def format_task(task):
    return f"N={task['n']} K={task['k']} D={task['d']}"


def summarize_task_rollout(task, traj_batch, max_steps):
    info = traj_batch.info
    reward_mean = float(np.mean(np.asarray(traj_batch.reward)))
    done_rate = float(np.mean(np.asarray(traj_batch.done)))
    stats = rollout_success_stats(traj_batch, max_steps=max_steps)
    return {
        "n": task["n"],
        "k": task["k"],
        "d": task["d"],
        "target_distance": task["d"],
        "reward_mean": reward_mean,
        "done_rate": done_rate,
        **stats,
        "episode_return_mean": completed_episode_mean(
            info, "returned_episode_returns"
        ),
        "episode_length_mean": completed_episode_mean(
            info, "returned_episode_lengths"
        ),
    }


def compute_training_layout(base_config, total_timesteps, train_tasks):
    if not train_tasks:
        raise ValueError("at least one training task is required")

    rollout_per_task = (
        base_config["NUM_ENVS_PER_TASK"] * base_config["NUM_STEPS"]
    )
    rollout_per_update = rollout_per_task * len(train_tasks)
    if total_timesteps < rollout_per_update:
        raise ValueError(
            "--total-timesteps must cover at least one full joint PPO update "
            f"({rollout_per_update:,} timesteps)."
        )
    num_updates = total_timesteps // rollout_per_update
    actual_total_timesteps = num_updates * rollout_per_update
    if rollout_per_update % base_config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            "NUM_MINIBATCHES must divide the joint rollout size per update "
            f"({rollout_per_update})."
        )
    return {
        "num_updates": int(num_updates),
        "rollout_per_task": int(rollout_per_task),
        "rollout_per_update": int(rollout_per_update),
        "actual_total_timesteps": int(actual_total_timesteps),
        "minibatch_size": int(
            rollout_per_update // base_config["NUM_MINIBATCHES"]
        ),
    }


def train_joint_multitask(base_config, total_timesteps, train_tasks, graph_padding):
    layout = compute_training_layout(base_config, total_timesteps, train_tasks)
    first_task = train_tasks[0]
    first_env = make_env(
        first_task["n"],
        first_task["k"],
        first_task["d"],
        base_config,
        graph_padding,
    )
    network = make_actor_critic(base_config, first_env)

    rng = jax.random.PRNGKey(base_config["SEED"])
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, first_env.graph_observation_template())
    train_state = create_train_state(
        base_config, network, params, layout["num_updates"]
    )
    update_rng = jax.random.fold_in(rng, 10_000)

    task_contexts = []
    for task_index, task in enumerate(train_tasks):
        env = LogWrapper(
            make_env(
                task["n"],
                task["k"],
                task["d"],
                base_config,
                graph_padding,
            )
        )
        task_rng = jax.random.fold_in(rng, task_index)
        task_state = initialize_task_state(
            env, base_config["NUM_ENVS_PER_TASK"], task_rng
        )
        task_contexts.append(
            {
                "task": task,
                "env": env,
                "collector": build_rollout_collector(
                    base_config, env, network
                ),
                **task_state,
            }
        )

    joint_update = build_joint_update_fn(
        base_config,
        network,
        batch_size=layout["rollout_per_update"],
    )

    history = []
    print(
        f"Training {len(train_tasks)} tasks jointly for "
        f"{layout['num_updates']} PPO updates; "
        f"{layout['rollout_per_update']:,} timesteps/update "
        f"({layout['rollout_per_task']:,} per task)."
    )

    for update_index in range(layout["num_updates"]):
        started = time.perf_counter()
        task_batches = []
        task_records = []

        for context in task_contexts:
            runner_state, rollout = context["collector"](
                train_state.params,
                context["env_state"],
                context["last_obs"],
                context["rng"],
            )
            context["env_state"], context["last_obs"], context["rng"] = runner_state
            traj_batch, advantages, targets = rollout
            task_batches.append(
                PPOBatch(
                    obs=traj_batch.obs,
                    action=traj_batch.action,
                    value=traj_batch.value,
                    log_prob=traj_batch.log_prob,
                    advantages=advantages,
                    targets=targets,
                )
            )
            task_records.append(
                summarize_task_rollout(
                    context["task"],
                    traj_batch,
                    max_steps=base_config["MAX_STEPS"],
                )
            )

        combined_batch = merge_task_batches(task_batches)
        update_rng, step_rng = jax.random.split(update_rng)
        train_state, loss_metrics, update_rng = joint_update(
            train_state, combined_batch, step_rng
        )
        loss_summary = summarize_loss_metrics(loss_metrics)

        reward_mean = float(
            np.mean([record["reward_mean"] for record in task_records])
        )
        done_rate = float(np.mean([record["done_rate"] for record in task_records]))
        episode_count = int(sum(record["episode_count"] for record in task_records))
        success_count = int(sum(record["success_count"] for record in task_records))
        timeout_count = int(sum(record["timeout_count"] for record in task_records))
        success_rate = success_count / episode_count if episode_count else None
        episode_return_mean = mean_or_none(
            [record["episode_return_mean"] for record in task_records]
        )
        episode_length_mean = mean_or_none(
            [record["episode_length_mean"] for record in task_records]
        )

        record = {
            "update": update_index + 1,
            "timesteps": (update_index + 1) * layout["rollout_per_update"],
            "seconds": time.perf_counter() - started,
            "reward_mean": reward_mean,
            "done_rate": done_rate,
            "episode_count": episode_count,
            "success_count": success_count,
            "timeout_count": timeout_count,
            "success_rate": success_rate,
            "episode_return_mean": episode_return_mean,
            "episode_length_mean": episode_length_mean,
            "loss": loss_summary,
            "tasks": task_records,
        }
        history.append(record)
        print(
            f"update={record['update']} "
            f"reward={record['reward_mean']:.2f} "
            f"success={format_metric(record['success_rate'])} "
            f"episodes={record['episode_count']} "
            f"return={format_metric(record['episode_return_mean'])} "
            f"length={format_metric(record['episode_length_mean'])} "
            f"loss={record['loss']['total_loss']:.4f} "
            f"time={record['seconds']:.1f}s"
        )

    return train_state.params, history, layout


def validate(params, base_config, validation_tasks, graph_padding, compute_distance=True):
    if not validation_tasks:
        print("Validation skipped: no validation tasks were provided.")
        return {
            "target_distance": None,
            "target_distances": [],
            "compute_distance": compute_distance,
            "tasks": [],
            "distance_summary": [],
        }

    results = []
    target_distances = sorted({task["d"] for task in validation_tasks})
    print(f"Validating on {len(validation_tasks)} tasks...")
    for task_index, task in enumerate(validation_tasks):
        task_config = dict(base_config)
        task_config["D"] = task["d"]
        env = make_env(
            task["n"], task["k"], task["d"], task_config, graph_padding
        )
        network = make_actor_critic(task_config, env)
        rng = jax.random.PRNGKey(task_config["SEED"] + 10_000 + task_index)
        observation, state = env.reset(rng, None)

        gates = []
        total_reward = 0.0
        final_reward = float("nan")
        final_value = float("nan")
        done = False
        for _ in range(task_config["MAX_STEPS"]):
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

        distance_stats = None
        if compute_distance:
            distance, distance_stats = distance_error_stats_up_to_target(
                task["n"], task["k"], gates, task["d"]
            )
            distance_stats_text = format_distance_stats(distance_stats)
        else:
            distance = None
            distance_stats_text = "distance_stats=skipped"
        target_met = (
            distance >= task["d"]
            if distance is not None
            else bool(jnp.isclose(final_reward, 0.0, atol=1.0e-6))
        )
        result = {
            "n": task["n"],
            "k": task["k"],
            "d": task["d"],
            "target_distance": task["d"],
            "distance": distance,
            "distance_stats": distance_stats,
            "target_met": target_met,
            "steps": len(gates),
            "total_reward": total_reward,
            "final_reward": final_reward,
            "final_value": final_value,
            "gates": gates,
        }
        results.append(result)
        print(
            f"  {format_task(task)}: distance={distance} "
            f"target_met={target_met} steps={len(gates)} "
            f"{distance_stats_text}"
        )

    distance_summary = (
        aggregate_distance_stats(results) if compute_distance else None
    )
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
        "target_distance": target_distances[0] if len(target_distances) == 1 else None,
        "target_distances": target_distances,
        "compute_distance": compute_distance,
        "tasks": results,
        "distance_summary": distance_summary,
    }


def dry_run(base_config, train_tasks, validation_tasks, graph_padding):
    """Check that every requested task shares graph/action tensor shapes."""

    check_tasks = train_tasks + validation_tasks
    if not check_tasks:
        raise ValueError("at least one task is required for a dry run")

    first_task = check_tasks[0]
    first_env = make_env(
        first_task["n"],
        first_task["k"],
        first_task["d"],
        base_config,
        graph_padding,
    )
    network = make_actor_critic(base_config, first_env)
    params = network.init(
        jax.random.PRNGKey(1), first_env.graph_observation_template()
    )
    expected_shapes = jax.tree_util.tree_map(
        lambda x: x.shape,
        first_env.graph_observation_template(),
    )

    for task_index, task in enumerate(check_tasks):
        env = make_env(
            task["n"], task["k"], task["d"], base_config, graph_padding
        )
        observation, _ = env.reset(jax.random.PRNGKey(task_index + 2), None)
        shapes = jax.tree_util.tree_map(lambda x: x.shape, observation)
        if shapes != expected_shapes:
            raise ValueError(f"Task {task} has incompatible graph shapes")
        policy, value = network.apply(params, observation)
        print(
            f"{format_task(task)}: nodes={int(observation.node_mask.sum())}, "
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
        help="Total joint-training timesteps across all tasks.",
    )
    parser.add_argument(
        "--num-envs-per-task",
        "--num-envs",
        dest="num_envs_per_task",
        type=int,
        default=BASE_CONFIG["NUM_ENVS_PER_TASK"],
        help="Parallel environments assigned to each training task per PPO update.",
    )
    parser.add_argument(
        "--train-task",
        dest="train_tasks",
        action="append",
        nargs=3,
        type=int,
        metavar=("N", "K", "D"),
        help="Add one training task; repeat the flag for multiple tasks.",
    )
    parser.add_argument(
        "--validation-task",
        dest="validation_tasks",
        action="append",
        nargs=3,
        type=int,
        metavar=("N", "K", "D"),
        help="Add one validation task; repeat the flag for multiple tasks.",
    )
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
        "--skip-validation",
        action="store_true",
        help="Skip the validation pass entirely.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the requested tasks and run shared-parameter forward passes.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir()
    return args


def main():
    args = parse_args()
    config = dict(BASE_CONFIG)
    config.update(
        {
            "NUM_ENVS_PER_TASK": args.num_envs_per_task,
            "SEED": args.seed,
        }
    )

    train_tasks = normalize_task_specs(args.train_tasks, DEFAULT_TRAIN_TASKS)
    validation_tasks = (
        []
        if args.skip_validation
        else normalize_task_specs(args.validation_tasks, DEFAULT_VALIDATION_TASKS)
    )
    graph_padding = build_graph_padding(train_tasks + validation_tasks)

    if args.dry_run:
        dry_run(config, train_tasks, validation_tasks, graph_padding)
        return

    params, history, layout = train_joint_multitask(
        config,
        total_timesteps=args.total_timesteps,
        train_tasks=train_tasks,
        graph_padding=graph_padding,
    )
    validation = validate(
        params,
        config,
        validation_tasks,
        graph_padding=graph_padding,
        compute_distance=not args.skip_distance,
    )
    run_config = {
        **config,
        "WHICH_GATES": list(config["WHICH_GATES"]),
        "train_tasks": train_tasks,
        "validation_tasks": validation_tasks,
        "requested_total_timesteps": args.total_timesteps,
        **layout,
    }
    save_results(args.output_dir, params, history, validation, run_config)


if __name__ == "__main__":
    main()
