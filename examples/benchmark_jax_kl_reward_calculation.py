"""Benchmark the JAX KL/reward kernel on the same validation tasks as GF(2).

The script follows ``examples/benchmark_gf2_distance.py``: it restores the
saved policy, generates one final stabilizer code for every validation task,
and checks physical Pauli weights ``1..d``.  For every task/weight it also
calls :func:`qdx.gf2_distance.benchmark_jax_kl_reward_calculation` to compare
the exact GF(2) kernel with legacy softness ``1``, ``2``, and ``3``.

Run from the repository root::

    conda run -n qdx python examples/benchmark_jax_kl_reward_calculation.py

Use ``--task-limit 1 --num-steps 1 --repetitions 1`` for a small smoke test.
The timed kernel excludes model inference, tableau construction, input error
construction, and JIT warm-up, as defined by the benchmark function.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")

from qdx.runtime_cache import configure_jax_persistent_cache


configure_jax_persistent_cache()

import jax.numpy as jnp

from qdx.gf2_distance import (
    benchmark_jax_kl_reward_calculation,
    precache_pauli_errors,
    stabilizer_check_matrix_from_gates,
    verify_stabilizer_distance_gf2,
)
from qdx.make_train import make_actor_critic
from qdx.profiling import block_until_ready_tree
from qdx.utils import (
    Utils,
    build_graph_padding,
    build_task_config,
    format_task,
    load_params_from_path,
    load_run_settings,
    make_task_env,
)
from qdx.validation_rollout import (
    build_validation_episode_runner,
    summarize_validation_episode,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "513_1373.yaml"
DEFAULT_PARAMS = REPOSITORY_ROOT / "outputs" / "V1-4_513_1373" / "params.msgpack"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "outputs"
    / "V1-4_513_1373"
    / "jax_kl_reward_benchmark.json"
)
SOFTNESS_VALUES = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of repeated scan steps per task/weight benchmark.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of timed executions per task/weight/method.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional smoke-test limit; the default benchmarks every task.",
    )
    return parser.parse_args()


def _build_rollout_context(task, config, graph_padding, task_index):
    task_config = build_task_config(config, task)
    env = make_task_env(task, task_config, graph_padding)
    network = make_actor_critic(task_config, env)
    import jax

    rng = jax.random.PRNGKey(task_config["SEED"] + 10_000 + task_index)
    observation, state = env.reset(rng, None)
    block_until_ready_tree((observation, state))
    return {
        "task": task,
        "env": env,
        "runner": build_validation_episode_runner(
            env,
            network,
            task_config["MAX_STEPS"],
        ),
        "observation": observation,
        "state": state,
        "rng": rng,
    }


def _run_rollout(context, params):
    import time

    started = time.perf_counter()
    rollout = context["runner"](
        params,
        context["observation"],
        context["state"],
        context["rng"],
    )
    block_until_ready_tree(rollout)
    summary = summarize_validation_episode(
        rollout,
        context["env"].action_string_stim,
    )
    summary["seconds"] = time.perf_counter() - started
    return summary


def _legacy_weight_counts(n, k, gates, max_weight):
    counts = {softness: {} for softness in SOFTNESS_VALUES}
    for softness in SOFTNESS_VALUES:
        utilities = Utils(n, k, gates, softness=max(1, min(softness, n - k)))
        for weight in range(1, max_weight + 1):
            errors = utilities.error_operators(weight)
            counts[softness][weight] = int(utilities.check_KL(errors))
    return counts


def _weight_probability(n, weight, p_identity):
    p_single = (1.0 - float(p_identity)) / 3.0
    return float((p_single**int(weight)) * (float(p_identity) ** (n - weight)))


def _save_report(path, report):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    os.replace(temporary_path, path)


def _distance_summary(gf2_result, softness_counts, weight_totals):
    gf2_by_weight = {
        int(item["weight"]): item["violation_count"]
        for item in gf2_result["weight_stats"]
    }
    result = {
        "target_distance": gf2_result["target_distance"],
        "gf2_target_met": gf2_result["target_met"],
        "gf2_estimated_distance": gf2_result["estimated_distance_label"],
        "softness": {},
        "mismatches": [],
    }
    for softness, counts in softness_counts.items():
        first_violation = next(
            (weight for weight in sorted(counts) if counts[weight]),
            None,
        )
        estimated = first_violation or gf2_result["target_distance"] + 1
        result["softness"][str(softness)] = {
            "target_met": first_violation is None
            or first_violation >= gf2_result["target_distance"],
            "estimated_distance": estimated,
            "estimated_distance_label": (
                str(estimated) if first_violation else f">={estimated}"
            ),
            "weight_stats": [
                {
                    "weight": weight,
                    "violation_count": counts[weight],
                    "total_count": weight_totals[weight],
                }
                for weight in sorted(counts)
            ],
        }
        for weight in sorted(gf2_by_weight.keys() & counts.keys()):
            if gf2_by_weight[weight] != counts[weight]:
                result["mismatches"].append(
                    {
                        "softness": softness,
                        "weight": weight,
                        "gf2_error_count": gf2_by_weight[weight],
                        "softness_error_count": counts[weight],
                    }
                )
    return result


def main():
    args = parse_args()
    if args.num_steps < 1 or args.repetitions < 1:
        raise ValueError("--num-steps and --repetitions must be positive")
    if args.task_limit is not None and args.task_limit < 1:
        raise ValueError("--task-limit must be positive")
    if not args.params.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.params}")

    settings = load_run_settings(args.config)
    config = settings["config"]
    validation_tasks = settings["validation_tasks"]
    if args.task_limit is not None:
        validation_tasks = validation_tasks[: args.task_limit]
    graph_padding = build_graph_padding(validation_tasks)

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.params}")
    print(
        f"Validation tasks: {len(validation_tasks)} "
        f"(graphs={','.join(settings['graphs'])})"
    )
    print("Building task-specific model rollouts...")
    contexts = [
        _build_rollout_context(task, config, graph_padding, task_index)
        for task_index, task in enumerate(validation_tasks)
    ]
    params = load_params_from_path(
        args.params,
        config,
        validation_tasks[0],
        graph_padding,
    )

    print("JAX warm-up: compiling and synchronizing every validation rollout...")
    warmup_started = time.perf_counter()
    for context in contexts:
        rollout = context["runner"](
            params,
            context["observation"],
            context["state"],
            context["rng"],
        )
        block_until_ready_tree(rollout)
    rollout_warmup_seconds = time.perf_counter() - warmup_started

    print("Generating final stabilizers with formal rollout timing...")
    task_rollouts = []
    for context in contexts:
        rollout = _run_rollout(context, params)
        task = context["task"]
        task_rollouts.append(
            {
                "task": task,
                "rollout": rollout,
                "check_matrix": stabilizer_check_matrix_from_gates(
                    task["n"], task["k"], rollout["gates"]
                ),
            }
        )

    print("Pre-caching exact-weight Pauli errors...")
    cache_started = time.perf_counter()
    precache_pauli_errors(
        (item["task"]["n"], item["task"]["d"]) for item in task_rollouts
    )
    error_cache_seconds = time.perf_counter() - cache_started

    task_results = []
    total_timings = {name: 0.0 for name in ("gf2", "softness_1", "softness_2", "softness_3")}
    mismatch_count = 0
    for task_index, item in enumerate(task_rollouts):
        task = item["task"]
        n = int(task["n"])
        max_weight = min(int(task["d"]), n)
        print(f"[{task_index + 1:3d}/{len(task_rollouts)}] {format_task(task)}")

        gf2_result = verify_stabilizer_distance_gf2(
            item["check_matrix"],
            task["d"],
            max_weight=max_weight,
            stop_at_first_logical_weight=False,
        ).to_dict()
        softness_counts = _legacy_weight_counts(
            n, int(task["k"]), item["rollout"]["gates"], max_weight
        )
        weight_totals = {
            weight: int(Utils(n, int(task["k"]), item["rollout"]["gates"], 1)
                        .error_operators(weight).shape[0])
            for weight in range(1, max_weight + 1)
        }
        distance = _distance_summary(gf2_result, softness_counts, weight_totals)
        mismatch_count += len(distance["mismatches"])

        weight_results = []
        for weight in range(1, max_weight + 1):
            from qdx.gf2_distance import cached_exact_weight_pauli_errors

            errors = jnp.asarray(cached_exact_weight_pauli_errors(n, weight))
            probability = _weight_probability(n, weight, config["P_I"])
            probabilities = jnp.full((errors.shape[0],), probability, dtype=jnp.float32)
            benchmark = benchmark_jax_kl_reward_calculation(
                item["check_matrix"],
                errors,
                probabilities,
                config["LAMBDA"],
                num_steps=args.num_steps,
                repetitions=args.repetitions,
                print_summary=False,
            )
            for method, values in benchmark["timings"].items():
                total_timings[method] += float(values["total_seconds"])
            comparisons = benchmark["gf2_comparisons"]
            weight_results.append(
                {
                    "weight": weight,
                    "error_count": gf2_result["weight_stats"][weight - 1][
                        "violation_count"
                    ],
                    "total_count": int(errors.shape[0]),
                    "probability_per_error": probability,
                    "benchmark": benchmark,
                    "legacy_error_counts": {
                        str(softness): softness_counts[softness][weight]
                        for softness in SOFTNESS_VALUES
                    },
                    "jax_matches_legacy": {
                        f"softness_{softness}": comparisons[f"softness_{softness}"][
                            "error_count_equal"
                        ]
                        for softness in SOFTNESS_VALUES
                    },
                }
            )
            print(
                f"  weight={weight} errors={errors.shape[0]} "
                f"gf2_mean={benchmark['timings']['gf2']['mean_seconds_per_step']:.9f}s"
            )

        task_results.append(
            {
                "task_index": task_index,
                "task_number": task_index + 1,
                **task,
                "rollout": item["rollout"],
                "stabilizer_check_matrix": item["check_matrix"].tolist(),
                "distance": distance,
                "weights": weight_results,
            }
        )

    report = {
        "config_path": str(args.config.resolve()),
        "params_path": str(args.params.resolve()),
        "task_count": len(task_results),
        "graphs": list(settings["graphs"]),
        "settings": {
            "num_steps": args.num_steps,
            "repetitions": args.repetitions,
            "task_limit": args.task_limit,
            "probability_model": "p_single^weight * p_identity^(n-weight)",
            "p_identity": config["P_I"],
            "lambda": config["LAMBDA"],
        },
        "warmup": {
            "rollout_seconds": rollout_warmup_seconds,
            "error_cache_seconds": error_cache_seconds,
        },
        "timing_totals": total_timings,
        "mismatch_count": mismatch_count,
        "tasks": task_results,
    }
    _save_report(args.output, report)

    print("\nBenchmark summary")
    print(f"  tasks: {len(task_results)}")
    for method, seconds in total_timings.items():
        print(f"  {method:10s} total: {seconds:.6f}s")
    print(f"  distance mismatches: {mismatch_count}")
    print(f"  JSON: {args.output}")


if __name__ == "__main__":
    main()
