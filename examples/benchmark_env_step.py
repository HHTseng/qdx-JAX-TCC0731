"""Benchmark batched validation ``env.step`` with legacy KL and GF(2) checks.

The benchmark follows the saved policy's validation trajectories from
``configs/513_1373.yaml`` but times only the batched environment transitions.
Model inference, validation-trajectory generation, JIT compilation, and the
initial environment reset are excluded from the reported step timings.

Run from the repository root::

    conda run -n qdx python examples/benchmark_env_step.py

By default this benchmarks the All-to-All validation tasks with 8, 16, 32, 64,
and 128 environments per task, and writes the report to
``outputs/V1-4_513_1373/env_step_benchmark.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import numpy as np
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")

from qdx.runtime_cache import configure_jax_persistent_cache


configure_jax_persistent_cache()

import jax
import jax.numpy as jnp

from qdx.profiling import block_until_ready_tree
from qdx.gf2_distance import (
    error_weight_indices_upto,
    jax_exact_gf2_kl,
    jax_tableau_kl,
)
from qdx.make_train import make_actor_critic
from qdx.utils import (
    build_graph_padding,
    build_task_config,
    format_task,
    load_params_from_path,
    load_run_settings,
    make_task_env,
)
from qdx.validation_rollout import summarize_validation_episode


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "513_1373.yaml"
DEFAULT_PARAMS = REPOSITORY_ROOT / "outputs" / "V1-4_513_1373" / "params.msgpack"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "outputs"
    / "V1-4_513_1373"
    / "env_step_benchmark.json"
)
DEFAULT_NUM_ENVS = (8, 16, 32, 64, 128)
METHODS = ("softness_1", "softness_2", "softness_3", "gf2", "gf2_tableau")


StepFunction = Callable[[Any, Any, Any, Any], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--num-envs-per-task",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_ENVS,
        help="Batch sizes to benchmark; defaults to 8 16 32 64 128.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of timed executions for each task/method/batch-size.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional smoke-test limit after filtering to All-to-All tasks.",
    )
    return parser.parse_args()


def _build_validation_runner(
    env, network, step_fn: StepFunction, max_steps: int, diagnostic_size: int = 0,
):
    """Build a greedy validation runner using an explicitly selected step."""

    max_steps = int(max_steps)

    def validate_episode(params, observation, state, rng):
        zero = jnp.zeros((), dtype=jnp.float32)
        initial_carry = (
            observation,
            state,
            rng,
            jnp.asarray(False),
            zero,
            zero,
            zero,
            jnp.zeros((), dtype=jnp.int32),
        )

        def scan_step(carry, _):
            def active_step(active_carry):
                (
                    observation,
                    state,
                    rng,
                    _done,
                    total_reward,
                    _final_reward,
                    _final_value,
                    step_count,
                ) = active_carry
                policy, value = network.apply(params, observation)
                action = jnp.asarray(policy.mode(), dtype=jnp.int32)
                rng, step_rng = jax.random.split(rng)
                next_observation, next_state, reward, next_done, info = step_fn(
                    step_rng, state, action, None
                )
                next_carry = (
                    next_observation,
                    next_state,
                    rng,
                    next_done,
                    total_reward + reward,
                    reward,
                    value,
                    step_count + 1,
                )
                empty_counts = jnp.zeros((diagnostic_size,), dtype=jnp.int32)
                empty_rates = jnp.zeros((diagnostic_size,), dtype=jnp.float32)
                error_count_by_weight = info.get(
                    "error_count_by_weight", empty_counts
                )
                total_count_by_weight = info.get(
                    "total_count_by_weight", empty_counts
                )
                error_rate_by_weight = info.get("error_rate_by_weight", empty_rates)
                return next_carry, {
                    "action_ids": action,
                    "rewards": reward,
                    "dones": next_done,
                    "error_count_by_weight": error_count_by_weight,
                    "total_count_by_weight": total_count_by_weight,
                    "error_rate_by_weight": error_rate_by_weight,
                }

            def skip_step(skip_carry):
                (
                    observation,
                    state,
                    rng,
                    done,
                    total_reward,
                    final_reward,
                    final_value,
                    step_count,
                ) = skip_carry
                return skip_carry, {
                    "action_ids": jnp.asarray(-1, dtype=jnp.int32),
                    "rewards": jnp.zeros_like(total_reward),
                    "dones": done,
                    "error_count_by_weight": jnp.zeros((diagnostic_size,), dtype=jnp.int32),
                    "total_count_by_weight": jnp.zeros((diagnostic_size,), dtype=jnp.int32),
                    "error_rate_by_weight": jnp.zeros((diagnostic_size,), dtype=jnp.float32),
                }

            return jax.lax.cond(
                carry[3], skip_step, active_step, carry
            )

        final_carry, outputs = jax.lax.scan(
            scan_step,
            initial_carry,
            None,
            length=max_steps,
        )
        (
            _observation,
            _state,
            _rng,
            done,
            total_reward,
            final_reward,
            final_value,
            steps,
        ) = final_carry
        return {
            "action_ids": outputs["action_ids"],
            "rewards": outputs["rewards"],
            "dones": outputs["dones"],
            "error_count_by_weight": outputs["error_count_by_weight"],
            "total_count_by_weight": outputs["total_count_by_weight"],
            "error_rate_by_weight": outputs["error_rate_by_weight"],
            "done": done,
            "total_reward": total_reward,
            "final_reward": final_reward,
            "final_value": final_value,
            "steps": steps,
        }

    return jax.jit(validate_episode)


def _build_gf2_step(env, implementation: str = "rref") -> StepFunction:
    """Build an exact GF(2) step using RREF or direct tableau coordinates."""

    error_weights = jnp.asarray(
        error_weight_indices_upto(env.n_qubits_physical, env.d),
        dtype=jnp.int32,
    )
    max_weight = min(int(env.d) - 1, int(env.n_qubits_physical))
    weight_values = jnp.arange(1, max_weight + 1, dtype=jnp.int32)
    if int(error_weights.shape[0]) != int(env.E_mu.shape[0]):
        raise ValueError("error-weight metadata does not match env.E_mu")

    def gf2_step(key, state, action, params=None):
        if params is None:
            params = env.default_params
        _key_step, key_reset = jax.random.split(key)
        new_pending_action_mask = env.update_pending_action_mask(
            state.pending_action_mask,
            action,
        )
        state_step = type(state)(
            tableau=(state.tableau @ env.actions[action]) % 2,
            time=state.time + 1,
            pending_action_mask=new_pending_action_mask,
        )
        if implementation == "tableau":
            result = jax_tableau_kl(
                state_step.tableau,
                env.n_qubits_logical,
                env.E_mu,
                env.p_mu,
                env.lbda,
                error_weights=error_weights,
                weight_values=weight_values,
            )
        elif implementation == "rref":
            check_matrix = state_step.tableau[
                env.n_qubits_physical + env.n_qubits_logical :
            ]
            result = jax_exact_gf2_kl(
                check_matrix,
                env.E_mu,
                env.p_mu,
                env.lbda,
                error_weights=error_weights,
                weight_values=weight_values,
            )
        else:
            raise ValueError(f"unsupported GF(2) implementation: {implementation}")

        reward = result.reward
        done = jnp.logical_or(
            result.error_count == 0,
            state_step.time >= env.max_steps,
        )
        obs_step = jax.lax.stop_gradient(env.get_obs(state_step))
        state_step = jax.lax.stop_gradient(state_step)
        info = {
            "discount": env.discount(state_step, params),
            "error_count": result.error_count,
            "logical_error_probability": result.logical_error_probability,
            "error_count_by_weight": result.error_count_by_weight,
            "total_count_by_weight": result.total_count_by_weight,
            "error_rate_by_weight": result.error_rate_by_weight,
        }
        obs, next_state = jax.lax.cond(
            done,
            lambda _: env.reset_env(key_reset, params),
            lambda _: (obs_step, state_step),
            operand=None,
        )
        return obs, next_state, reward, done, info

    return jax.jit(gf2_step)


def _method_softness(method: str) -> int | None:
    if method in {"gf2", "gf2_tableau"}:
        return None
    prefix, _, value = method.partition("_")
    if prefix != "softness" or value not in {"1", "2", "3"}:
        raise ValueError(f"unsupported benchmark method: {method}")
    return int(value)


def _distance_stats_from_rollout(rollout, steps):
    rates = np.asarray(rollout["error_rate_by_weight"], dtype=np.float32)
    if rates.shape[-1] == 0:
        return []
    index = max(0, int(steps) - 1)
    counts = np.asarray(rollout["error_count_by_weight"])[index]
    totals = np.asarray(rollout["total_count_by_weight"])[index]
    rates = rates[index]
    return [
        {
            "d": weight + 1,
            "error_count": int(counts[weight]),
            "total_count": int(totals[weight]),
            "error_count_over_total": (
                f"{int(counts[weight])}/{int(totals[weight])}"
            ),
            "error_rate": float(rates[weight]),
        }
        for weight in range(rates.shape[0])
    ]

def _build_method_context(task, base_config, graph_padding, method, params, index):

    softness = _method_softness(method)
    task_config = build_task_config(base_config, task)
    if softness is not None:
        task_config["SOFTNESS"] = softness
    else:
        # Softness only affects the legacy verifier. Keep the environment's
        # static graph/model shape identical for the GF(2) variant.
        task_config["SOFTNESS"] = 1

    env = make_task_env(task, task_config, graph_padding)
    network = make_actor_critic(task_config, env)
    if softness is not None:
        step_fn = env.step
        diagnostic_size = 0
    else:
        implementation = "tableau" if method == "gf2_tableau" else "rref"
        step_fn = _build_gf2_step(env, implementation)
        diagnostic_size = min(int(env.d) - 1, int(env.n_qubits_physical))
    runner = _build_validation_runner(
        env,
        network,
        step_fn,
        task_config["MAX_STEPS"],
        diagnostic_size,
    )
    rng = jax.random.PRNGKey(task_config["SEED"] + 10_000 + index)
    observation, state = env.reset(rng, None)
    started = time.perf_counter()
    rollout = runner(params, observation, state, rng)
    block_until_ready_tree(rollout)
    validation_seconds = time.perf_counter() - started
    summary = summarize_validation_episode(
        rollout,
        env.action_string_stim,
    )
    actions = rollout["action_ids"][: summary["steps"]]
    summary["distance_stats"] = _distance_stats_from_rollout(rollout, summary["steps"])
    return {
        "env": env,
        "step_fn": step_fn,
        "observation": observation,
        "state": state,
        "actions": actions,
        "summary": summary,
        "validation_seconds": validation_seconds,
    }


def _broadcast_tree(value, batch_size: int):
    return jax.tree_util.tree_map(
        lambda item: jnp.broadcast_to(item, (batch_size,) + item.shape),
        value,
    )


def _build_batched_step_runner(step_fn: StepFunction, num_envs: int):
    """Build a JIT runner for a fixed action sequence and batch size."""

    num_envs = int(num_envs)

    def run(observation, state, keys, actions):
        def scan_step(carry, action):
            observation, state, keys = carry
            key_pairs = jax.vmap(jax.random.split)(keys)
            step_keys = key_pairs[:, 0]
            next_keys = key_pairs[:, 1]
            batched_actions = jnp.broadcast_to(action, (num_envs,))
            next_observation, next_state, rewards, dones, _ = jax.vmap(
                step_fn,
                in_axes=(0, 0, 0, None),
            )(step_keys, state, batched_actions, None)
            return (next_observation, next_state, next_keys), (
                rewards,
                dones,
            )

        (observation, state, keys), (rewards, dones) = jax.lax.scan(
            scan_step,
            (observation, state, keys),
            actions,
        )
        return {
            "observation": observation,
            "state": state,
            "keys": keys,
            "rewards": rewards,
            "dones": dones,
        }

    return jax.jit(run)


def _time_batched_steps(context, num_envs: int, repetitions: int):
    actions = context["actions"]
    steps = int(actions.shape[0])
    if steps == 0:
        return {
            "total_seconds": 0.0,
            "mean_seconds_per_batched_step": 0.0,
            "mean_seconds_per_env_step": 0.0,
            "env_steps_per_run": 0,
            "batched_steps_per_run": 0,
            "repetitions": int(repetitions),
            "warmup_seconds": 0.0,
        }

    initial_observation = _broadcast_tree(context["observation"], num_envs)
    initial_state = _broadcast_tree(context["state"], num_envs)
    initial_keys = jax.random.split(
        jax.random.PRNGKey(70_000 + num_envs),
        num_envs,
    )
    runner = _build_batched_step_runner(context["step_fn"], num_envs)
    warmup_started = time.perf_counter()
    warmup_output = runner(
        initial_observation,
        initial_state,
        initial_keys,
        actions,
    )
    block_until_ready_tree(warmup_output)
    warmup_seconds = time.perf_counter() - warmup_started

    started = time.perf_counter()
    output = None
    for _ in range(int(repetitions)):
        output = runner(
            initial_observation,
            initial_state,
            initial_keys,
            actions,
        )
        block_until_ready_tree(output)
    total_seconds = time.perf_counter() - started
    batched_steps = steps * int(repetitions)
    env_steps = batched_steps * int(num_envs)
    return {
        "total_seconds": total_seconds,
        "mean_seconds_per_batched_step": total_seconds / batched_steps,
        "mean_seconds_per_env_step": total_seconds / env_steps,
        "env_steps_per_run": steps * int(num_envs),
        "batched_steps_per_run": steps,
        "repetitions": int(repetitions),
        "warmup_seconds": warmup_seconds,
        "final_done": bool(jnp.all(output["dones"][-1])),
    }


def _save_report(path, report):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    if any(value < 1 for value in args.num_envs_per_task):
        raise ValueError("--num-envs-per-task values must be positive")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.task_limit is not None and args.task_limit < 1:
        raise ValueError("--task-limit must be positive")
    if not args.params.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.params}")

    settings = load_run_settings(args.config)
    validation_tasks = [
        task for task in settings["validation_tasks"] if task["graph"] == "All-to-All"
    ]
    if args.task_limit is not None:
        validation_tasks = validation_tasks[: args.task_limit]
    if not validation_tasks:
        raise ValueError("no All-to-All validation tasks found in the config")

    config = settings["config"]
    graph_padding = build_graph_padding(validation_tasks)
    params = load_params_from_path(
        args.params,
        config,
        validation_tasks[0],
        graph_padding,
    )

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.params}")
    print(
        f"All-to-All validation tasks: {len(validation_tasks)}; "
        f"num_envs_per_task={args.num_envs_per_task}"
    )
    print("Generating validation action trajectories and running env.step benchmark...")
    task_records = []
    totals = {
        str(num_envs): {
            method: {
                "total_seconds": 0.0,
                "total_env_steps": 0,
                "total_batched_steps": 0,
            }
            for method in METHODS
        }
        for num_envs in args.num_envs_per_task
    }

    for task_index, task in enumerate(validation_tasks):
        print(f"[{task_index + 1:3d}/{len(validation_tasks)}] {format_task(task)}")
        task_record = {
            "task_index": task_index,
            "task_number": task_index + 1,
            **task,
            "validation": {},
            "timings": {},
        }
        for method in METHODS:
            context = _build_method_context(
                task,
                config,
                graph_padding,
                method,
                params,
                task_index,
            )
            task_record["validation"][method] = context["summary"]
            print(
                f"  {method:10s} validation_steps={context['summary']['steps']:2d} "
                f"compile+rollout={context['validation_seconds']:.6f}s"
            )
            for num_envs in args.num_envs_per_task:
                size_record = task_record["timings"].setdefault(str(num_envs), {})
                timing = _time_batched_steps(
                    context,
                    num_envs,
                    args.repetitions,
                )
                size_record[method] = timing
                totals[str(num_envs)][method]["total_seconds"] += timing[
                    "total_seconds"
                ]
                totals[str(num_envs)][method]["total_env_steps"] += (
                    timing["env_steps_per_run"] * args.repetitions
                )
                totals[str(num_envs)][method]["total_batched_steps"] += (
                    timing["batched_steps_per_run"] * args.repetitions
                )
            # A full validation run can create hundreds of task/method/shape
            # executables. Release each method before compiling the next one.
            del context
            jax.clear_caches()
            gc.collect()
        task_records.append(task_record)

    for num_envs, method_totals in totals.items():
        gf2_seconds = method_totals["gf2"]["total_seconds"]
        for method, values in method_totals.items():
            env_steps = values["total_env_steps"]
            batched_steps = values["total_batched_steps"]
            values["mean_seconds_per_env_step"] = (
                values["total_seconds"] / env_steps if env_steps else 0.0
            )
            values["mean_seconds_per_batched_step"] = (
                values["total_seconds"] / batched_steps if batched_steps else 0.0
            )
            if method == "gf2":
                values["speedup_vs_gf2"] = 1.0
            else:
                values["speedup_vs_gf2"] = (
                    values["total_seconds"] / gf2_seconds
                    if gf2_seconds
                    else None
                )

    report = {
        "benchmark": "batched_validation_env_step",
        "config_path": str(args.config.expanduser().resolve()),
        "params_path": str(args.params.expanduser().resolve()),
        "graph": "All-to-All",
        "methods": list(METHODS),
        "settings": {
            "num_envs_per_task": [int(value) for value in args.num_envs_per_task],
            "repetitions": int(args.repetitions),
            "task_limit": args.task_limit,
            "timed_scope": (
                "batched env.step only; validation action generation, "
                "JIT compilation, and reset excluded"
            ),
            "speedup_vs_gf2": "method total seconds / GF(2) total seconds",
        },
        "task_count": len(task_records),
        "tasks": task_records,
        "totals": totals,
    }
    _save_report(args.output, report)

    print("\nBenchmark summary")
    for num_envs, method_totals in totals.items():
        print(f"  num_envs_per_task={num_envs}")
        for method, values in method_totals.items():
            print(
                f"    {method:10s} "
                f"mean_env_step={values['mean_seconds_per_env_step']:.9f}s "
                f"total={values['total_seconds']:.6f}s "
                f"speedup_vs_gf2={values['speedup_vs_gf2']:.3f}"
            )
    print(f"  JSON: {args.output}")


if __name__ == "__main__":
    main()
