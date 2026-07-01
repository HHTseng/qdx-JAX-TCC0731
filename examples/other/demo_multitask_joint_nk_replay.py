"""Replay a saved joint-multitask GNN-QDX checkpoint on larger (N, K, D) tasks.

Edit the constants near the top of the file to choose the saved result directory
and the list of validation tasks. The script reads `params.msgpack` and
`validation.json` from that directory, expands graph padding automatically to
cover the requested tasks, and recomputes `distance` and `distance_stats`.
"""

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


# Edit these in place.
SOURCE_RESULT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "demo_multitask_joint_nk_20260629121606"
)
CUSTOM_VALIDATION_TASKS = (
    {"n": 14, "k": 1, "d": 3},
    {"n": 14, "k": 2, "d": 3},
    {"n": 15, "k": 1, "d": 4},
    {"n": 16, "k": 2, "d": 4},
)
CUSTOM_VALIDATION_OUTPUT = SOURCE_RESULT_DIR / "validation_larger_tasks.json"


def all_to_all_graph(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def make_env(n, k, config, graph_padding):
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
        graph_padding=graph_padding,
    )


def normalize_validation_tasks(tasks, default_distance):
    normalized = []
    for task in tasks:
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
                    "validation tasks must be (n, k) or (n, k, d) tuples "
                    "or dictionaries with n, k, d"
                )
        normalized.append(
            {
                "n": int(n),
                "k": int(k),
                "target_distance": int(target_distance),
            }
        )
    return normalized


def build_graph_padding_for_tasks(task_specs):
    if not task_specs:
        raise ValueError("at least one validation task is required")
    max_n = max(item["n"] for item in task_specs)
    max_stabilizers = max(item["n"] - item["k"] for item in task_specs)
    return GraphPadding(
        n_max=max_n,
        stabilizers_max=max_stabilizers,
        hardware_edges_max=max_n * max(max_n - 1, 0),
    )


def load_saved_run(result_dir):
    run_config_path = result_dir / "run_config.json"
    validation_path = result_dir / "validation.json"
    params_path = result_dir / "params.msgpack"
    if not run_config_path.exists():
        raise FileNotFoundError(f"missing run config: {run_config_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"missing validation file: {validation_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"missing parameter file: {params_path}")

    with run_config_path.open("r", encoding="utf-8") as file:
        run_config = json.load(file)
    with validation_path.open("r", encoding="utf-8") as file:
        validation = json.load(file)
    return run_config, validation, params_path.read_bytes()


def summarize_saved_validation(validation):
    return {
        "target_distance": validation.get("target_distance"),
        "target_distances": validation.get("target_distances"),
        "compute_distance": validation.get("compute_distance"),
        "task_count": len(validation.get("tasks") or []),
        "distance_summary": validation.get("distance_summary") or [],
    }


def print_saved_validation_summary(validation, prefix=""):
    summary = summarize_saved_validation(validation)
    print(
        f"{prefix}tasks={summary['task_count']} "
        f"compute_distance={summary['compute_distance']}"
    )
    if summary["target_distances"] is None:
        print(f"{prefix}target_distance={summary['target_distance']}")
    else:
        print(f"{prefix}target_distances={summary['target_distances']}")
    if summary["distance_summary"]:
        print(f"{prefix}distance summary:")
        for item in summary["distance_summary"]:
            print(
                f"{prefix}  d={item['d']} "
                f"錯誤次數/總次數={item['error_count_over_total']} "
                f"錯誤率={item['error_rate']:.2%}"
            )


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
            f"錯誤次數/總次數={item['error_count_over_total']} "
            f"錯誤率={item['error_rate']:.2%}"
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


def load_params_for_saved_run(params_bytes, run_config, graph_padding, reference_task):
    reference_config = dict(run_config)
    reference_config["D"] = reference_task["target_distance"]
    env = make_env(
        reference_task["n"],
        reference_task["k"],
        reference_config,
        graph_padding=graph_padding,
    )
    network = make_actor_critic(reference_config, env)
    template_params = network.init(
        jax.random.PRNGKey(0), env.graph_observation_template()
    )
    return serialization.from_bytes(template_params, params_bytes)


def validate_saved_run(
    params,
    run_config,
    task_specs,
    graph_padding,
    compute_distance=True,
):
    results = []
    target_distances = sorted({item["target_distance"] for item in task_specs})
    print(f"Validating on {len(task_specs)} tasks...")

    for task_index, task_spec in enumerate(task_specs):
        n = task_spec["n"]
        k = task_spec["k"]
        target_distance = task_spec["target_distance"]
        task_config = dict(run_config)
        task_config["D"] = target_distance
        env = make_env(n, k, task_config, graph_padding=graph_padding)
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
                n, k, gates, target_distance
            )
            distance_stats_text = format_distance_stats(distance_stats)
        else:
            distance = None
            distance_stats_text = "distance_stats=skipped"

        target_met = (
            distance >= target_distance
            if distance is not None
            else bool(jnp.isclose(final_reward, 0.0, atol=1.0e-6))
        )
        result = {
            "n": n,
            "k": k,
            "target_distance": target_distance,
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
            f"  N={n} K={k} D={target_distance}: distance={distance} "
            f"target_met={target_met} steps={len(gates)} {distance_stats_text}"
        )

    distance_summary = aggregate_distance_stats(results) if compute_distance else None
    if distance_summary:
        print("Distance summary across validation tasks:")
        for item in distance_summary:
            print(
                "  "
                f"d={item['d']} "
                f"錯誤次數/總次數={item['error_count_over_total']} "
                f"錯誤率={item['error_rate']:.2%}"
            )

    return {
        "target_distance": target_distances[0] if len(target_distances) == 1 else None,
        "target_distances": target_distances,
        "compute_distance": compute_distance,
        "tasks": results,
        "distance_summary": distance_summary,
    }


def run_saved_validation(
    result_dir,
    task_specs,
    output_path=CUSTOM_VALIDATION_OUTPUT,
    compute_distance=True,
):
    run_config, source_validation, params_bytes = load_saved_run(result_dir)
    if run_config.get("MODEL", "GNN").upper() != "GNN":
        raise ValueError(
            "This replay script expects a GNN checkpoint so it can be reused "
            "across larger padded graph shapes."
        )
    if run_config.get("ENV_TYPE", "STANDARD").upper() != "STANDARD":
        raise ValueError("This replay script expects ENV_TYPE='STANDARD'.")

    normalized_tasks = normalize_validation_tasks(task_specs, run_config["D"])
    graph_padding = build_graph_padding_for_tasks(normalized_tasks)
    reference_task = max(
        normalized_tasks,
        key=lambda item: (item["n"], item["n"] - item["k"], item["target_distance"]),
    )
    params = load_params_for_saved_run(
        params_bytes, run_config, graph_padding, reference_task
    )

    print(f"Loaded saved checkpoint from {result_dir}")
    print_saved_validation_summary(source_validation, prefix="  source ")

    validation = validate_saved_run(
        params,
        run_config,
        normalized_tasks,
        graph_padding=graph_padding,
        compute_distance=compute_distance,
    )
    validation["source_result_dir"] = str(result_dir)
    validation["source_validation_summary"] = summarize_saved_validation(
        source_validation
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(validation, file, indent=2)
        print(f"Saved custom validation to {output_path}")

    return validation


def main():
    run_saved_validation(
        SOURCE_RESULT_DIR,
        CUSTOM_VALIDATION_TASKS,
        output_path=CUSTOM_VALIDATION_OUTPUT,
        compute_distance=True,
    )


if __name__ == "__main__":
    main()
