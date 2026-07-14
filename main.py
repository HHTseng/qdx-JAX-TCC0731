"""Train and validate one shared GNN-QDX policy on custom (N, K, D) tasks.

Training hyperparameters, task lists, graph choices, and runtime toggles are
loaded from a YAML file in ``configs/``.

Run:
    conda run -n qdx python main.py
    conda run -n qdx python main.py --config configs/main.yaml
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

from qdx.runtime_cache import configure_jax_persistent_cache

RUNTIME_CACHE_DIRS = configure_jax_persistent_cache()

from flax import serialization
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import optax
import jax
import jax.numpy as jnp
import numpy as np

from qdx.make_train import make_actor_critic
from qdx.utils import (
    DEFAULT_CONFIG_PATH,
    build_graph_padding,
    format_task,
    graph_padding_to_dict,
    load_run_settings,
    make_task_env,
)
from validation import run_validation


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


def format_duration(seconds):
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
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


def completed_episode_mean(info, name):
    values = completed_episode_values(info, name)
    if values is None or values.size == 0:
        return None
    return float(np.mean(values))


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

    return jax.jit(update, donate_argnums=(0))


def initialize_task_state(env, num_envs_per_task, rng):
    rng, reset_rng = jax.random.split(rng)
    reset_rng = jax.random.split(reset_rng, num_envs_per_task)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, None)
    return {
        "env_state": env_state,
        "last_obs": obs,
        "rng": rng,
    }


def build_task_context(task, config, graph_padding, network, rng):
    env = LogWrapper(make_task_env(task, config, graph_padding))
    task_state = initialize_task_state(env, config["NUM_ENVS_PER_TASK"], rng)
    return {
        "task": task,
        "env": env,
        "collector": build_rollout_collector(config, env, network),
        **task_state,
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


def summarize_task_rollout(task, traj_batch, max_steps):
    info = traj_batch.info
    reward_mean = float(np.mean(np.asarray(traj_batch.reward)))
    done_rate = float(np.mean(np.asarray(traj_batch.done)))
    stats = rollout_success_stats(traj_batch, max_steps=max_steps)
    return {
        "graph": task["graph"],
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
            "TOTAL_TIMESTEPS/total_timesteps in the YAML config must cover at "
            f"least one full joint PPO update ({rollout_per_update:,} timesteps)."
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


def train_joint_multitask(
    base_config, total_timesteps, train_tasks, train_graph_padding, run_started
):
    layout = compute_training_layout(base_config, total_timesteps, train_tasks)
    first_task = train_tasks[0]
    first_env = make_task_env(first_task, base_config, train_graph_padding)
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
        task_contexts.append(
            build_task_context(
                task,
                base_config,
                train_graph_padding,
                network,
                jax.random.fold_in(rng, task_index),
            )
        )

    joint_update = build_joint_update_fn(
        base_config,
        network,
        batch_size=layout["rollout_per_update"],
    )

    history = []
    training_started = time.perf_counter()
    startup_elapsed = time.perf_counter() - run_started
    print(
        f"Training {len(train_tasks)} tasks jointly for "
        f"{layout['num_updates']} PPO updates; "
        f"{layout['rollout_per_update']:,} timesteps/update "
        f"({layout['rollout_per_task']:,} per task). "
        f"elapsed={format_duration(startup_elapsed)}"
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

        finished = time.perf_counter()
        record = {
            "update": update_index + 1,
            "timesteps": (update_index + 1) * layout["rollout_per_update"],
            "seconds": finished - started,
            "elapsed_seconds": finished - training_started,
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
            f"time={record['seconds']:.1f}s "
            f"elapsed={format_duration(record['elapsed_seconds'])}"
        )

    return train_state.params, history, layout


def dry_run(
    base_config,
    train_tasks,
    validation_tasks,
    train_graph_padding,
    validation_graph_padding,
):
    """Check split-specific shapes and train-parameter reuse across paddings."""

    if not train_tasks:
        raise ValueError("at least one training task is required for a dry run")

    first_task = train_tasks[0]
    first_env = make_task_env(first_task, base_config, train_graph_padding)
    network = make_actor_critic(base_config, first_env)
    params = network.init(
        jax.random.PRNGKey(1), first_env.graph_observation_template()
    )

    def check_tasks(label, tasks, graph_padding):
        if not tasks:
            return

        reference_task = tasks[0]
        reference_env = make_task_env(reference_task, base_config, graph_padding)
        expected_shapes = jax.tree_util.tree_map(
            lambda x: x.shape,
            reference_env.graph_observation_template(),
        )

        for task_index, task in enumerate(tasks):
            env = make_task_env(task, base_config, graph_padding)
            observation, _ = env.reset(jax.random.PRNGKey(task_index + 2), None)
            shapes = jax.tree_util.tree_map(lambda x: x.shape, observation)
            if shapes != expected_shapes:
                raise ValueError(
                    f"{label} task {task} has incompatible graph shapes"
                )
            policy, value = network.apply(params, observation)
            print(
                f"{label} {format_task(task)}: "
                f"nodes={int(observation.node_mask.sum())}, "
                f"actions={int(observation.action_mask.sum())}, "
                f"logits={policy.logits.shape}, value_shape={value.shape}"
            )

    check_tasks("train", train_tasks, train_graph_padding)
    check_tasks("validation", validation_tasks, validation_graph_padding)


def save_training_results(output_dir, params, history, run_config):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))
    for name, value in (
        ("train_history.json", history),
        ("run_config.json", run_config),
    ):
        with (output_dir / name).open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
    print(f"Saved training artifacts to {output_dir}")


def save_validation_results(output_dir, validation):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation.json").open("w", encoding="utf-8") as file:
        json.dump(validation, file, indent=2)
    print(f"Saved validation results to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory from the config file.",
    )
    parser.add_argument(
        "--kl-method",
        choices=("existing", "gf2", "gf2_tableau"),
        default=None,
        help="KL check used by env.step; overrides the YAML config.",
    )
    return parser.parse_args()


def main():
    run_started = time.perf_counter()
    args = parse_args()
    run_settings = load_run_settings(args.config)
    if args.output_dir is not None:
        run_settings["output_dir"] = args.output_dir.expanduser()
    config = run_settings["config"]
    if args.kl_method is not None:
        config = dict(config)
        config["KL_METHOD"] = args.kl_method
    graphs = run_settings["graphs"]
    train_tasks = run_settings["train_tasks"]
    configured_validation_tasks = run_settings["validation_tasks"]
    validation_tasks = (
        [] if run_settings["skip_validation"] else configured_validation_tasks
    )
    train_graph_padding = build_graph_padding(train_tasks)
    validation_graph_padding = (
        build_graph_padding(validation_tasks)
        if validation_tasks
        else train_graph_padding
    )

    print(f"Loaded configuration from {run_settings['config_path']}")

    if run_settings["dry_run"]:
        dry_run(
            config,
            train_tasks,
            validation_tasks,
            train_graph_padding,
            validation_graph_padding,
        )
        return

    params, history, layout = train_joint_multitask(
        config,
        total_timesteps=config["TOTAL_TIMESTEPS"],
        train_tasks=train_tasks,
        train_graph_padding=train_graph_padding,
        run_started=run_started,
    )
    run_config = {
        **config,
        "config_path": run_settings["config_path"],
        "output_dir": str(run_settings["output_dir"]),
        "skip_distance": run_settings["skip_distance"],
        "skip_validation": run_settings["skip_validation"],
        "dry_run": run_settings["dry_run"],
        "graphs": list(graphs),
        "WHICH_GATES": list(config["WHICH_GATES"]),
        "train_tasks": train_tasks,
        "validation_tasks": configured_validation_tasks,
        "train_graph_padding": graph_padding_to_dict(train_graph_padding),
        "validation_graph_padding": (
            None
            if not validation_tasks
            else graph_padding_to_dict(validation_graph_padding)
        ),
        "requested_total_timesteps": config["TOTAL_TIMESTEPS"],
        **layout,
    }
    save_training_results(run_settings["output_dir"], params, history, run_config)
    validation = run_validation(
        params,
        config,
        validation_tasks,
        validation_graph_padding=validation_graph_padding,
        compute_distance=not run_settings["skip_distance"],
    )
    save_validation_results(run_settings["output_dir"], validation)
    total_runtime = time.perf_counter() - run_started
    print(
        f"Total runtime: {format_duration(total_runtime)} "
        f"({total_runtime:.1f}s)"
    )


if __name__ == "__main__":
    main()
