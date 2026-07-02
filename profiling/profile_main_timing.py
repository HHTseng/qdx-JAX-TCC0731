"""Profile startup, one PPO update, and validation for the current main flow.

This script does not change ``main.py`` or ``validation.py``. It reuses the
existing training helpers, runs a single warmed-up PPO update, and records
wall-clock timings for the major phases and their subparts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")

import jax
import jax.numpy as jnp
import numpy as np

import main as train_main
from qdx.make_train import make_actor_critic
from qdx.profiling import TimingRecorder, block_until_ready_tree, timed_section
from qdx.utils import (
    DEFAULT_CONFIG_PATH,
    aggregate_distance_stats,
    build_graph_padding,
    build_task_config,
    distance_error_stats_up_to_target,
    format_distance_stats,
    format_task,
    graph_padding_to_dict,
    load_run_settings,
    make_task_env,
)


def task_timing_label(task):
    return format_task(task).replace(" ", "_")


def build_training_session(
    base_config,
    total_timesteps,
    train_tasks,
    train_graph_padding,
    timing,
):
    with timed_section(timing, "startup/compute_layout"):
        layout = train_main.compute_training_layout(
            base_config, total_timesteps, train_tasks
        )

    first_task = train_tasks[0]
    with timed_section(timing, "startup/build_first_env"):
        first_env = make_task_env(first_task, base_config, train_graph_padding)

    with timed_section(timing, "startup/build_network"):
        network = make_actor_critic(base_config, first_env)

    with timed_section(timing, "startup/init_state"):
        rng = jax.random.PRNGKey(base_config["SEED"])
        rng, init_rng = jax.random.split(rng)
        params = network.init(init_rng, first_env.graph_observation_template())
        train_state = train_main.create_train_state(
            base_config, network, params, layout["num_updates"]
        )
        update_rng = jax.random.fold_in(rng, 10_000)

    task_contexts = []
    for task_index, task in enumerate(train_tasks):
        with timed_section(timing, f"startup/task_contexts/{task_timing_label(task)}"):
            task_contexts.append(
                train_main.build_task_context(
                    task,
                    base_config,
                    train_graph_padding,
                    network,
                    jax.random.fold_in(rng, task_index),
                )
            )

    with timed_section(timing, "startup/build_joint_update_fn"):
        joint_update = train_main.build_joint_update_fn(
            base_config,
            network,
            batch_size=layout["rollout_per_update"],
        )

    return layout, train_state, update_rng, task_contexts, joint_update


def run_profiled_update(
    base_config,
    layout,
    train_state,
    update_rng,
    task_contexts,
    joint_update,
    timing,
    label_prefix,
    update_index=0,
):
    update_started = time.perf_counter()
    task_batches = []
    task_records = []

    for context in task_contexts:
        task_label = task_timing_label(context["task"])
        started = time.perf_counter()
        runner_state, rollout = context["collector"](
            train_state.params,
            context["env_state"],
            context["last_obs"],
            context["rng"],
        )
        block_until_ready_tree((runner_state, rollout))
        if timing is not None:
            timing.record(
                f"{label_prefix}/rollout/{task_label}",
                time.perf_counter() - started,
            )

        context["env_state"], context["last_obs"], context["rng"] = runner_state
        traj_batch, advantages, targets = rollout
        task_batches.append(
            train_main.PPOBatch(
                obs=traj_batch.obs,
                action=traj_batch.action,
                value=traj_batch.value,
                log_prob=traj_batch.log_prob,
                advantages=advantages,
                targets=targets,
            )
        )
        task_records.append(
            train_main.summarize_task_rollout(
                context["task"],
                traj_batch,
                max_steps=base_config["MAX_STEPS"],
            )
        )

    started = time.perf_counter()
    combined_batch = train_main.merge_task_batches(task_batches)
    if timing is not None:
        timing.record(
            f"{label_prefix}/merge_batches",
            time.perf_counter() - started,
        )

    update_rng, step_rng = jax.random.split(update_rng)
    started = time.perf_counter()
    train_state, loss_metrics, update_rng = joint_update(
        train_state, combined_batch, step_rng
    )
    block_until_ready_tree((train_state, loss_metrics))
    if timing is not None:
        timing.record(
            f"{label_prefix}/optimize",
            time.perf_counter() - started,
        )

    started = time.perf_counter()
    loss_summary = train_main.summarize_loss_metrics(loss_metrics)
    reward_mean = float(np.mean([record["reward_mean"] for record in task_records]))
    done_rate = float(np.mean([record["done_rate"] for record in task_records]))
    episode_count = int(sum(record["episode_count"] for record in task_records))
    success_count = int(sum(record["success_count"] for record in task_records))
    timeout_count = int(sum(record["timeout_count"] for record in task_records))
    success_rate = success_count / episode_count if episode_count else None
    episode_return_mean = train_main.mean_or_none(
        [record["episode_return_mean"] for record in task_records]
    )
    episode_length_mean = train_main.mean_or_none(
        [record["episode_length_mean"] for record in task_records]
    )
    if timing is not None:
        timing.record(
            f"{label_prefix}/aggregate_metrics",
            time.perf_counter() - started,
        )

    update_seconds = time.perf_counter() - update_started
    record = {
        "update": update_index + 1,
        "timesteps": (update_index + 1) * layout["rollout_per_update"],
        "seconds": update_seconds,
        "elapsed_seconds": update_seconds,
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
    return train_state, update_rng, record, task_records


def run_profiled_validation(
    params,
    base_config,
    validation_tasks,
    validation_graph_padding,
    compute_distance,
    timing,
):
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
        task_label = task_timing_label(task)

        setup_started = time.perf_counter()
        task_config = build_task_config(base_config, task)
        env = make_task_env(task, task_config, validation_graph_padding)
        network = make_actor_critic(task_config, env)
        rng = jax.random.PRNGKey(task_config["SEED"] + 10_000 + task_index)
        observation, state = env.reset(rng, None)
        if timing is not None:
            timing.record(
                f"validation/setup/{task_label}",
                time.perf_counter() - setup_started,
            )

        rollout_started = time.perf_counter()
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
        if timing is not None:
            timing.record(
                f"validation/rollout/{task_label}",
                time.perf_counter() - rollout_started,
            )

        distance_started = time.perf_counter()
        distance_stats = None
        if compute_distance:
            distance, distance_stats = distance_error_stats_up_to_target(
                task["n"],
                task["k"],
                gates,
                task["d"],
                softness=base_config.get("VALIDATION_SOFTNESS"),
            )
            distance_stats_text = format_distance_stats(distance_stats)
        else:
            distance = None
            distance_stats_text = "distance_stats=skipped"
        if timing is not None:
            timing.record(
                f"validation/distance/{task_label}",
                time.perf_counter() - distance_started,
            )

        target_met = (
            distance >= task["d"]
            if distance is not None
            else bool(jnp.isclose(final_reward, 0.0, atol=1.0e-6))
        )
        result = {
            "graph": task["graph"],
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

    summary_started = time.perf_counter()
    distance_summary = aggregate_distance_stats(results) if compute_distance else None
    if timing is not None:
        timing.record(
            "validation/summary",
            time.perf_counter() - summary_started,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile startup, one PPO update, and validation timing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help=(
            "Skip the warmup PPO update. When omitted, the profiler runs one "
            "extra warmup update so the measured PPO update reflects steady-state "
            "runtime instead of first-call compilation time."
        ),
    )
    return parser.parse_args()


def main():
    run_started = time.perf_counter()
    args = parse_args()
    timing = TimingRecorder()

    with timed_section(timing, "startup/load_run_settings"):
        run_settings = load_run_settings(args.config)

    config = run_settings["config"]
    graphs = run_settings["graphs"]
    train_tasks = run_settings["train_tasks"]
    configured_validation_tasks = run_settings["validation_tasks"]
    validation_tasks = (
        [] if run_settings["skip_validation"] else configured_validation_tasks
    )

    with timed_section(timing, "startup/build_graph_padding/train"):
        train_graph_padding = build_graph_padding(train_tasks)
    with timed_section(timing, "startup/build_graph_padding/validation"):
        validation_graph_padding = (
            build_graph_padding(validation_tasks)
            if validation_tasks
            else train_graph_padding
        )

    print(f"Loaded configuration from {run_settings['config_path']}")

    if run_settings["dry_run"]:
        raise ValueError(
            "profiling mode requires dry_run: false so that startup, update, and "
            "validation can all be measured"
        )
    if not validation_tasks:
        raise ValueError(
            "profiling mode requires at least one validation task and "
            "skip_validation: false"
        )

    layout, train_state, update_rng, task_contexts, joint_update = build_training_session(
        config,
        total_timesteps=config["TOTAL_TIMESTEPS"],
        train_tasks=train_tasks,
        train_graph_padding=train_graph_padding,
        timing=timing,
    )

    if not args.no_warmup:
        warmup_contexts = [context.copy() for context in task_contexts]
        run_profiled_update(
            config,
            layout,
            train_state,
            update_rng,
            warmup_contexts,
            joint_update,
            timing,
            label_prefix="startup/warmup",
            update_index=0,
        )

    train_state, update_rng, record, _ = run_profiled_update(
        config,
        layout,
        train_state,
        update_rng,
        task_contexts,
        joint_update,
        timing,
        label_prefix="ppo_update",
        update_index=0,
    )

    validation = run_profiled_validation(
        train_state.params,
        config,
        validation_tasks,
        validation_graph_padding=validation_graph_padding,
        compute_distance=not run_settings["skip_distance"],
        timing=timing,
    )

    run_config = {
        **config,
        "config_path": run_settings["config_path"],
        "output_dir": str(run_settings["output_dir"]),
        "skip_distance": run_settings["skip_distance"],
        "skip_validation": run_settings["skip_validation"],
        "dry_run": run_settings["dry_run"],
        "profile_timing": True,
        "profile_warmup": not args.no_warmup,
        "profile_updates_executed": 1,
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
    train_main.save_results(
        run_settings["output_dir"],
        train_state.params,
        [record],
        validation,
        run_config,
    )
    timing.save(run_settings["output_dir"] / "timing.json")

    print(timing.format_report())
    total_runtime = time.perf_counter() - run_started
    print(
        f"Total runtime: {train_main.format_duration(total_runtime)} "
        f"({total_runtime:.1f}s)"
    )


if __name__ == "__main__":
    main()
