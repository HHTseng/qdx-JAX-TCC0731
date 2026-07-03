"""Shared helpers for JIT-compiled greedy validation rollouts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def build_validation_episode_runner(env, network, max_steps):
    """Return a JIT-compiled greedy validation episode runner."""

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

        def _scan_step(carry, _):
            def _active_step(active_carry):
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
                next_observation, next_state, reward, next_done, _ = env.step(
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
                outputs = {
                    "action_ids": action,
                    "rewards": reward,
                    "dones": next_done,
                }
                return next_carry, outputs

            def _skip_step(skip_carry):
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
                outputs = {
                    "action_ids": jnp.asarray(-1, dtype=jnp.int32),
                    "rewards": jnp.zeros_like(total_reward),
                    "dones": done,
                }
                return (
                    observation,
                    state,
                    rng,
                    done,
                    total_reward,
                    final_reward,
                    final_value,
                    step_count,
                ), outputs

            done = carry[3]
            return jax.lax.cond(done, _skip_step, _active_step, carry)

        final_carry, outputs = jax.lax.scan(
            _scan_step,
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
            "done": done,
            "total_reward": total_reward,
            "final_reward": final_reward,
            "final_value": final_value,
            "steps": steps,
        }

    return jax.jit(validate_episode)


def summarize_validation_episode(rollout, action_strings):
    """Decode a JIT rollout result into host-side validation metadata."""

    step_count = int(np.asarray(rollout["steps"]))
    action_ids = np.asarray(rollout["action_ids"], dtype=np.int32)[:step_count]
    gates = [action_strings[int(action)] for action in action_ids]
    if step_count:
        final_reward = float(np.asarray(rollout["final_reward"]))
        final_value = float(np.asarray(rollout["final_value"]))
    else:
        final_reward = float("nan")
        final_value = float("nan")
    return {
        "done": bool(np.asarray(rollout["done"])),
        "steps": step_count,
        "total_reward": float(np.asarray(rollout["total_reward"])),
        "final_reward": final_reward,
        "final_value": final_value,
        "gates": gates,
    }
