"""Run validation for a saved GNN-QDX checkpoint.

This script reads validation tasks and model settings from a YAML config,
loads a saved ``params.msgpack`` checkpoint, and runs the same validation flow
used by ``main.py``.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")

import jax
import jax.numpy as jnp

from qdx.make_train import make_actor_critic
from qdx.validation_rollout import (
    build_validation_episode_runner,
    summarize_validation_episode,
)
from qdx.utils import (
    DEFAULT_CONFIG_PATH,
    aggregate_distance_stats,
    build_graph_padding,
    build_task_config,
    distance_error_stats_up_to_target,
    format_distance_stats,
    format_task,
    load_params_from_path,
    load_run_settings,
    make_task_env,
)


def run_validation(
    params,
    base_config,
    validation_tasks,
    validation_graph_padding,
    compute_distance=True,
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
        task_config = build_task_config(base_config, task)
        env = make_task_env(task, task_config, validation_graph_padding)
        network = make_actor_critic(task_config, env)
        validate_episode = build_validation_episode_runner(
            env, network, task_config["MAX_STEPS"]
        )
        rng = jax.random.PRNGKey(task_config["SEED"] + 10_000 + task_index)
        observation, state = env.reset(rng, None)

        rollout = validate_episode(params, observation, state, rng)
        rollout_summary = summarize_validation_episode(
            rollout, env.action_string_stim
        )
        gates = rollout_summary["gates"]
        total_reward = rollout_summary["total_reward"]
        final_reward = rollout_summary["final_reward"]
        final_value = rollout_summary["final_value"]

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
            "steps": rollout_summary["steps"],
            "total_reward": total_reward,
            "final_reward": final_reward,
            "final_value": final_value,
            "gates": gates,
        }
        results.append(result)
        print(
            f"  {format_task(task)}: distance={distance} "
            f"target_met={target_met} steps={rollout_summary['steps']} "
            f"{distance_stats_text}"
        )

    distance_summary = aggregate_distance_stats(results) if compute_distance else None
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


def save_validation(output_path, validation):
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(validation, file, indent=2)
    print(f"Saved validation results to {output_path}")


def run_validation_from_config(
    config_path,
    params_path,
    output_path=None,
    compute_distance=None,
):
    run_settings = load_run_settings(config_path)
    validation_tasks = run_settings["validation_tasks"]
    if not validation_tasks:
        raise ValueError("validation config must provide at least one validation task")

    config = run_settings["config"]
    validation_graph_padding = build_graph_padding(validation_tasks)
    params = load_params_from_path(
        params_path,
        config,
        validation_tasks[0],
        validation_graph_padding,
    )
    should_compute_distance = (
        not run_settings["skip_distance"]
        if compute_distance is None
        else compute_distance
    )

    print(f"Loaded validation config from {run_settings['config_path']}")
    print(f"Loaded model params from {Path(params_path).expanduser()}")
    validation = run_validation(
        params,
        config,
        validation_tasks,
        validation_graph_padding,
        compute_distance=should_compute_distance,
    )
    if output_path is not None:
        save_validation(output_path, validation)
    return validation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file that defines validation tasks and model settings.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        required=True,
        help="Path to the saved params.msgpack checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the validation JSON output.",
    )
    parser.add_argument(
        "--skip-distance",
        action="store_true",
        help="Skip the more expensive post-rollout distance calculation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_validation_from_config(
        config_path=args.config,
        params_path=args.params,
        output_path=args.output,
        compute_distance=False if args.skip_distance else None,
    )


if __name__ == "__main__":
    main()
