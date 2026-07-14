"""Benchmark exact GF(2) and legacy softness distance verification.

By default this script reads every validation task from
``configs/513_1373.yaml`` (including its graph expansion), restores
``outputs/V1-4_513_1373/params.msgpack``, performs JAX warm-up, and writes a
per-task JSON report next to the checkpoint.

Run from the repository root::

    conda run -n qdx python examples/benchmark_gf2_distance.py
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

import jax
import jax.numpy as jnp
import numpy as np

from qdx.gf2_distance import (
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
    / "gf2_softness_benchmark.json"
)
SOFTNESS_VALUES = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gf2-chunk-size",
        type=int,
        default=262_144,
        help="Maximum number of Pauli errors in one NumPy GF(2) batch.",
    )
    parser.add_argument(
        "--softness-chunk-size",
        type=int,
        default=None,
        help=(
            "Optional fixed JAX batch size for legacy softness checks. "
            "The default processes each weight's complete error array, "
            "matching main.py exactly."
        ),
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
    started = time.perf_counter()
    rollout = context["runner"](
        params,
        context["observation"],
        context["state"],
        context["rng"],
    )
    block_until_ready_tree(rollout)
    seconds = time.perf_counter() - started
    summary = summarize_validation_episode(
        rollout,
        context["env"].action_string_stim,
    )
    summary["seconds"] = seconds
    return summary


def _fixed_softness_batch(errors, start, chunk_size):
    batch = errors[start : start + chunk_size]
    padding = chunk_size - int(batch.shape[0])
    if padding:
        filler = jnp.broadcast_to(errors[0], (padding, errors.shape[1]))
        batch = jnp.concatenate((batch, filler), axis=0)
    return batch, padding


def _softness_result(n, k, gates, target_distance, softness, chunk_size=None):
    """Run the same legacy softness calculation used by ``main.py``.

    With ``chunk_size=None`` (the default), every weight is passed to
    ``Utils.check_KL`` as one complete array, exactly as in
    ``qdx.utils.distance_error_stats_up_to_target``. Chunking remains
    available as an explicit memory-saving option.
    """

    utilities = Utils(n, k, gates, softness=max(1, min(int(softness), n - k)))
    stats = []
    first_violation = None
    for weight in range(1, target_distance + 1):
        errors = utilities.error_operators(weight)
        if chunk_size is None:
            # This is intentionally the same call shape as main.py.
            violation_count = int(utilities.check_KL(errors))
        else:
            filler_violation = int(utilities.check_KL(errors[:1]))
            violation_count = 0
            for start in range(0, int(errors.shape[0]), chunk_size):
                batch, padding = _fixed_softness_batch(errors, start, chunk_size)
                violation_count += int(utilities.check_KL(batch))
                violation_count -= padding * filler_violation
        total_count = int(errors.shape[0])
        stats.append(
            {
                "weight": weight,
                "d": weight,
                "violation_count": violation_count,
                "error_count": violation_count,
                "total_count": total_count,
                "error_count_over_total": f"{violation_count}/{total_count}",
                "violation_rate": (
                    violation_count / total_count if total_count else 0.0
                ),
                "error_rate": (
                    violation_count / total_count if total_count else 0.0
                ),
            }
        )
        if violation_count and first_violation is None:
            first_violation = weight

    distance_is_exact = first_violation is not None
    estimated_distance = (
        first_violation if distance_is_exact else target_distance + 1
    )
    return {
        "target_distance": target_distance,
        "max_weight_checked": target_distance,
        "target_met": first_violation is None or first_violation >= target_distance,
        "estimated_distance": estimated_distance,
        "estimated_distance_label": (
            str(estimated_distance)
            if distance_is_exact
            else f">={estimated_distance}"
        ),
        "distance_is_exact": distance_is_exact,
        "first_logical_weight": first_violation,
        "weight_stats": stats,
        "softness": int(softness),
    }


def _warm_up_softness(task_rollouts, chunk_size):
    """Compile every unique legacy verifier shape before formal timing."""

    representative = {}
    for item in task_rollouts:
        task = item["task"]
        representative.setdefault((task["n"], task["k"]), item)

    for (n, k), item in representative.items():
        gates = item["rollout"]["gates"]
        for softness in SOFTNESS_VALUES:
            utilities = Utils(n, k, gates, softness=max(1, min(softness, n - k)))
            for weight in range(1, item["task"]["d"] + 1):
                errors = utilities.error_operators(weight)
                if chunk_size is None:
                    block_until_ready_tree(utilities.check_KL(errors))
                else:
                    batch, _ = _fixed_softness_batch(errors, 0, chunk_size)
                    block_until_ready_tree(utilities.check_KL(batch))
                    block_until_ready_tree(utilities.check_KL(errors[:1]))


def _time_verifier(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    seconds = time.perf_counter() - started
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    result["seconds"] = seconds
    return result


def _stats_by_weight(result):
    return {int(item["weight"]): item for item in result["weight_stats"]}


def _comparison_reasons(gf2_result, softness_result):
    reasons = []
    if gf2_result["target_met"] != softness_result["target_met"]:
        reasons.append("target_met")
    if gf2_result["estimated_distance"] != softness_result["estimated_distance"]:
        reasons.append("estimated_distance")

    gf2_weights = _stats_by_weight(gf2_result)
    softness_weights = _stats_by_weight(softness_result)
    differing_weights = [
        weight
        for weight in sorted(gf2_weights.keys() & softness_weights.keys())
        if gf2_weights[weight]["violation_count"]
        != softness_weights[weight]["violation_count"]
    ]
    if differing_weights:
        reasons.append(
            "violation_count@" + ",".join(str(weight) for weight in differing_weights)
        )
    return reasons


def _format_weight_counts(result):
    return ",".join(
        f"w{item['weight']}={item['violation_count']}/{item['total_count']}"
        for item in result["weight_stats"]
    )


def _save_report(path, report):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    if args.gf2_chunk_size < 1 or (
        args.softness_chunk_size is not None
        and args.softness_chunk_size < 1
    ):
        raise ValueError("chunk sizes must be positive")
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
        check_matrix = stabilizer_check_matrix_from_gates(
            task["n"],
            task["k"],
            rollout["gates"],
        )
        task_rollouts.append(
            {
                "task": task,
                "rollout": rollout,
                "check_matrix": check_matrix,
            }
        )

    print("Pre-caching exact-weight Pauli errors outside verifier timings...")
    cache_started = time.perf_counter()
    precache_pauli_errors(
        (item["task"]["n"], item["task"]["d"])
        for item in task_rollouts
    )
    error_cache_seconds = time.perf_counter() - cache_started

    print("JAX warm-up: compiling legacy softness verifier shapes...")
    warmup_started = time.perf_counter()
    _warm_up_softness(task_rollouts, args.softness_chunk_size)
    softness_warmup_seconds = time.perf_counter() - warmup_started

    print("Running formal verifier benchmark...")
    task_results = []
    mismatches = []
    method_totals = {"gf2": 0.0, **{f"softness_{s}": 0.0 for s in SOFTNESS_VALUES}}
    for task_index, item in enumerate(task_rollouts):
        task = item["task"]
        gf2_result = _time_verifier(
            verify_stabilizer_distance_gf2,
            item["check_matrix"],
            task["d"],
            max_weight=task["d"],
            chunk_size=args.gf2_chunk_size,
        )
        verifier_results = {"gf2": gf2_result}
        method_totals["gf2"] += gf2_result["seconds"]

        for softness in SOFTNESS_VALUES:
            method = f"softness_{softness}"
            result = _time_verifier(
                _softness_result,
                task["n"],
                task["k"],
                item["rollout"]["gates"],
                task["d"],
                softness,
                args.softness_chunk_size,
            )
            verifier_results[method] = result
            method_totals[method] += result["seconds"]
            reasons = _comparison_reasons(gf2_result, result)
            if reasons:
                mismatches.append(
                    {
                        # Keep task_index zero-based for the JSON schema and
                        # add a one-based task_number for human-readable logs.
                        "task_index": task_index,
                        "task_number": task_index + 1,
                        **task,
                        "method": method,
                        "reasons": reasons,
                        "gf2_target_met": gf2_result["target_met"],
                        "softness_target_met": result["target_met"],
                        "gf2_estimated_distance": gf2_result[
                            "estimated_distance_label"
                        ],
                        "softness_estimated_distance": result[
                            "estimated_distance_label"
                        ],
                    }
                )

        task_record = {
            "task_index": task_index,
            "task_number": task_index + 1,
            **task,
            "rollout": item["rollout"],
            "stabilizer_check_matrix": item["check_matrix"].tolist(),
            "verifiers": verifier_results,
        }
        task_results.append(task_record)
        print(f"[{task_index + 1:3d}/{len(task_rollouts)}] {format_task(task)}")
        for method, result in verifier_results.items():
            print(
                f"  {method:10s} target_met={str(result['target_met']):5s} "
                f"distance={result['estimated_distance_label']:>3s} "
                f"time={result['seconds']:.6f}s "
                f"violations=[{_format_weight_counts(result)}]"
            )

    report = {
        "config_path": str(args.config.resolve()),
        "params_path": str(args.params.resolve()),
        "task_count": len(task_results),
        "graphs": list(settings["graphs"]),
        "settings": {
            "gf2_chunk_size": args.gf2_chunk_size,
            "softness_chunk_size": args.softness_chunk_size,
            "task_limit": args.task_limit,
            "gf2_max_weight": "target_distance",
        },
        "warmup": {
            "rollout_seconds": rollout_warmup_seconds,
            "softness_seconds": softness_warmup_seconds,
            "error_cache_seconds": error_cache_seconds,
        },
        "timing_totals": {
            "rollout_seconds": sum(
                item["rollout"]["seconds"] for item in task_results
            ),
            "verifiers": method_totals,
        },
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "tasks": task_results,
    }
    _save_report(args.output, report)

    print("\nBenchmark summary")
    print(f"  tasks: {len(task_results)}")
    print(f"  rollout total: {report['timing_totals']['rollout_seconds']:.6f}s")
    for method, seconds in method_totals.items():
        print(f"  {method:10s} total: {seconds:.6f}s")
    print(f"  GF(2) mismatches: {len(mismatches)}")
    if mismatches:
        for mismatch in mismatches:
            print(
                "  "
                f"#{mismatch['task_number']} GRAPH={mismatch['graph']} "
                f"N={mismatch['n']} K={mismatch['k']} D={mismatch['d']} "
                f"vs {mismatch['method']}: {','.join(mismatch['reasons'])}"
            )
    else:
        print("  none")
    print(f"  JSON: {args.output}")


if __name__ == "__main__":
    main()
