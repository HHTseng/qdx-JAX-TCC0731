"""Exact stabilizer-distance verification over GF(2).

This module is intentionally independent of :class:`qdx.utils.Utils`, whose
``softness``-based KL check remains available unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import comb
import re
import time
from typing import Iterable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from qdx.runtime_cache import (
    build_exact_weight_error_operators,
    build_s_structure,
    load_or_build_array_bundle,
)
from qdx.simulators import TableauSimulator


_GATE_PATTERN = re.compile(r"^\.([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)$")


class JaxKLResult(NamedTuple):
    """Exact KL/reward values produced entirely on the JAX device.

    error_cost has the same positive-cost meaning as CodeDiscovery.check_KL.
    reward is the value used by CodeDiscovery.step_env: -error_cost.
    """

    logical_error_mask: jax.Array
    commutes_mask: jax.Array
    in_stabilizer_mask: jax.Array
    error_count: jax.Array
    logical_error_probability: jax.Array
    error_cost: jax.Array
    reward: jax.Array
    terminal: jax.Array
    error_count_by_weight: jax.Array
    total_count_by_weight: jax.Array
    error_rate_by_weight: jax.Array


class JaxKLStepScalars(NamedTuple):
    """Compact scalar output used by the repeated-kernel benchmark."""

    error_count: jax.Array
    logical_error_probability: jax.Array
    error_cost: jax.Array
    reward: jax.Array
    terminal: jax.Array


def jax_gf2_rref(matrix: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute GF(2) RREF with fixed-shape JAX control flow.

    The pivot array has one entry per input row; unused entries are -1. There
    are no host conversions or data-dependent Python operations, so this is
    safe under jit, vmap, and lax.scan.
    """

    reduced = jnp.asarray(matrix, dtype=jnp.uint8) & jnp.uint8(1)
    row_count, column_count = reduced.shape
    pivots = jnp.full((row_count,), -1, dtype=jnp.int32)
    pivot_row = jnp.asarray(0, dtype=jnp.int32)
    if row_count == 0:
        return reduced, pivots, pivot_row

    row_indices = jnp.arange(row_count, dtype=jnp.int32)

    def eliminate_column(column, carry):
        current, current_pivots, next_pivot_row = carry
        candidates = (row_indices >= next_pivot_row) & current[:, column].astype(
            jnp.bool_
        )
        has_pivot = jnp.any(candidates)
        selected_row = jnp.argmax(candidates.astype(jnp.int32))
        safe_pivot_row = jnp.minimum(next_pivot_row, row_count - 1)

        def apply_pivot(operand):
            matrix_value, pivot_values = operand
            selected = matrix_value[selected_row]
            displaced = matrix_value[safe_pivot_row]
            matrix_value = matrix_value.at[safe_pivot_row].set(selected)
            matrix_value = matrix_value.at[selected_row].set(displaced)

            pivot_vector = matrix_value[safe_pivot_row]
            eliminate = matrix_value[:, column] & (
                row_indices != safe_pivot_row
            ).astype(jnp.uint8)
            matrix_value ^= eliminate[:, None] * pivot_vector[None, :]
            pivot_values = pivot_values.at[safe_pivot_row].set(column)
            return matrix_value, pivot_values

        current, current_pivots = jax.lax.cond(
            has_pivot,
            apply_pivot,
            lambda operand: operand,
            (current, current_pivots),
        )
        next_pivot_row += has_pivot.astype(jnp.int32)
        return current, current_pivots, next_pivot_row

    return jax.lax.fori_loop(
        0,
        column_count,
        eliminate_column,
        (reduced, pivots, pivot_row),
    )


def jax_gf2_row_space_mask(
    vectors: jax.Array,
    reduced_basis: jax.Array,
    pivot_columns: jax.Array,
) -> jax.Array:
    """Test a fixed batch of vectors against a JAX GF(2) RREF basis."""

    remainders = jnp.asarray(vectors, dtype=jnp.uint8) & jnp.uint8(1)
    basis = jnp.asarray(reduced_basis, dtype=jnp.uint8) & jnp.uint8(1)
    pivots = jnp.asarray(pivot_columns, dtype=jnp.int32)
    row_count = basis.shape[0]
    if row_count == 0:
        return ~jnp.any(remainders.astype(jnp.bool_), axis=1)

    def reduce_by_row(row, current):
        pivot = pivots[row]
        active = pivot >= 0
        safe_pivot = jnp.maximum(pivot, 0)
        eliminate = current[:, safe_pivot] & active.astype(jnp.uint8)
        return current ^ eliminate[:, None] * basis[row][None, :]

    remainders = jax.lax.fori_loop(0, row_count, reduce_by_row, remainders)
    return ~jnp.any(remainders.astype(jnp.bool_), axis=1)


def _jax_commutes_with_stabilizers(
    error_operators: jax.Array,
    check_matrix: jax.Array,
) -> jax.Array:
    errors = jnp.asarray(error_operators, dtype=jnp.uint8) & jnp.uint8(1)
    stabilizers = jnp.asarray(check_matrix, dtype=jnp.uint8) & jnp.uint8(1)
    n_qubits = stabilizers.shape[1] // 2
    symplectic_products = errors[:, :n_qubits] @ stabilizers[:, n_qubits:].T
    symplectic_products ^= errors[:, n_qubits:] @ stabilizers[:, :n_qubits].T
    return ~jnp.any((symplectic_products & jnp.uint8(1)).astype(jnp.bool_), axis=1)


def _jax_kl_result_from_masks(
    logical_error_mask: jax.Array,
    commutes_mask: jax.Array,
    in_stabilizer_mask: jax.Array,
    error_probabilities: jax.Array,
    lbda: float | jax.Array,
    error_weights: jax.Array | None = None,
    weight_values: jax.Array | None = None,
) -> JaxKLResult:
    probabilities = jnp.asarray(error_probabilities)
    logical = jnp.asarray(logical_error_mask, dtype=jnp.bool_)
    error_count = jnp.sum(logical, dtype=jnp.int32)
    logical_probability = jnp.sum(
        jnp.where(logical, probabilities, jnp.zeros_like(probabilities))
    )
    error_cost = jnp.asarray(lbda, dtype=probabilities.dtype) * logical_probability
    if error_weights is None:
        error_count_by_weight = jnp.zeros((0,), dtype=jnp.int32)
        total_count_by_weight = jnp.zeros((0,), dtype=jnp.int32)
        error_rate_by_weight = jnp.zeros((0,), dtype=probabilities.dtype)
    else:
        if weight_values is None:
            raise ValueError(
                "weight_values is required when error_weights is supplied"
            )
        weights = jnp.asarray(error_weights, dtype=jnp.int32)
        values = jnp.asarray(weight_values, dtype=jnp.int32)
        weight_mask = weights[:, None] == values[None, :]
        error_count_by_weight = jnp.sum(
            logical[:, None] & weight_mask,
            axis=0,
            dtype=jnp.int32,
        )
        total_count_by_weight = jnp.sum(weight_mask, axis=0, dtype=jnp.int32)
        error_rate_by_weight = jnp.where(
            total_count_by_weight > 0,
            error_count_by_weight.astype(probabilities.dtype)
            / total_count_by_weight.astype(probabilities.dtype),
            jnp.zeros_like(total_count_by_weight, dtype=probabilities.dtype),
        )
    return JaxKLResult(
        commutes_mask=jnp.asarray(commutes_mask, dtype=jnp.bool_),
        logical_error_mask=logical,
        in_stabilizer_mask=jnp.asarray(in_stabilizer_mask, dtype=jnp.bool_),
        error_count=error_count,
        logical_error_probability=logical_probability,
        error_cost=error_cost,
        reward=-error_cost,
        terminal=error_count == 0,
        error_count_by_weight=error_count_by_weight,
        total_count_by_weight=total_count_by_weight,
        error_rate_by_weight=error_rate_by_weight,
    )


def jax_exact_gf2_kl(
    check_matrix: jax.Array,
    error_operators: jax.Array,
    error_probabilities: jax.Array,
    lbda: float | jax.Array,
    *,
    error_weights: jax.Array | None = None,
    weight_values: jax.Array | None = None,
) -> JaxKLResult:
    """Return exact KL error count, weighted cost, reward, and terminal.

    This is the device-only replacement kernel for the calculation currently
    performed inside CodeDiscovery.check_KL. Inputs must already have fixed
    shapes. In an environment, pass state.tableau[n + k:], env.E_mu, env.p_mu,
    and env.lbda.

    An operator is a logical error exactly when it commutes with every row of
    the check matrix but is not in that matrix's complete GF(2) row space.
    error_count is the number of such operators. logical_error_probability is
    their supplied probability mass, error_cost is lambda times that mass, and
    reward is its negative.

    terminal means the encoding condition error_count == 0. The environment's
    full done condition must still OR this with state.time >= max_steps.

    During training, env.E_mu is built once with physical weights 1..d-1 and
    env.p_mu contains the corresponding per-Pauli channel probabilities.
    Existing post-rollout distance validation instead checks individual exact
    weights and primarily consumes error_count; if weighted validation reward
    is wanted, its probability vector must match that supplied static error
    set. The kernel never changes or infers the requested weight range.
    """

    stabilizers = jnp.asarray(check_matrix, dtype=jnp.uint8) & jnp.uint8(1)
    errors = jnp.asarray(error_operators, dtype=jnp.uint8) & jnp.uint8(1)
    reduced, pivots, _rank = jax_gf2_rref(stabilizers)
    in_stabilizer = jax_gf2_row_space_mask(errors, reduced, pivots)
    commutes = _jax_commutes_with_stabilizers(errors, stabilizers)
    logical = commutes & ~in_stabilizer
    return _jax_kl_result_from_masks(
        logical,
        commutes,
        in_stabilizer,
        error_probabilities,
        lbda,
        error_weights,
        weight_values,
    )


def jax_tableau_kl(
    tableau: jax.Array,
    n_logical: int,
    error_operators: jax.Array,
    error_probabilities: jax.Array,
    lbda: float | jax.Array,
    *,
    error_weights: jax.Array | None = None,
    weight_values: jax.Array | None = None,
) -> JaxKLResult:
    """Check KL conditions directly from a complete Clifford tableau.

    The QDX row convention stores the n-k stabilizer generators in the last
    rows. For a symplectic tableau T, T-inverse = Omega @ T.T @ Omega.
    Transforming each error into reference coordinates avoids the per-step RREF
    and row-space elimination used by jax_exact_gf2_kl. The returned per-weight
    arrays use the weights supplied through ``weight_values``.
    """
    full_tableau = jnp.asarray(tableau, dtype=jnp.uint8) & jnp.uint8(1)
    errors = jnp.asarray(error_operators, dtype=jnp.uint8) & jnp.uint8(1)
    if full_tableau.ndim != 2 or full_tableau.shape[0] != full_tableau.shape[1]:
        raise ValueError("tableau must have shape [2n, 2n]")
    if full_tableau.shape[1] % 2:
        raise ValueError("tableau must have an even width")
    n_qubits = full_tableau.shape[1] // 2
    n_logical = int(n_logical)
    if not 0 <= n_logical <= n_qubits:
        raise ValueError("n_logical must satisfy 0 <= n_logical <= n_qubits")
    if errors.ndim != 2 or errors.shape[1] != 2 * n_qubits:
        raise ValueError("error_operators must have shape [errors, 2n]")

    # E @ Omega swaps the X and Z halves. Multiplication by T.T then gives
    # E @ Omega @ T.T; the final Omega only swaps the coordinate halves, so
    # the two coordinate slices needed below can be selected directly.
    transformed = (
        jnp.concatenate((errors[:, n_qubits:], errors[:, :n_qubits]), axis=1)
        @ full_tableau.T
    ) & jnp.uint8(1)
    commutes = ~jnp.any(transformed[:, n_qubits + n_logical :], axis=1)
    outside_stabilizer = jnp.concatenate(
        (transformed[:, n_qubits:], transformed[:, :n_logical]),
        axis=1,
    )
    in_stabilizer = ~jnp.any(outside_stabilizer, axis=1)
    logical = commutes & ~in_stabilizer
    return _jax_kl_result_from_masks(
        logical,
        commutes,
        in_stabilizer,
        error_probabilities,
        lbda,
        error_weights,
        weight_values,
    )


def jax_softness_kl(
    check_matrix: jax.Array,
    error_operators: jax.Array,
    error_probabilities: jax.Array,
    s_structure: jax.Array,
    lbda: float | jax.Array,
) -> JaxKLResult:
    """Device-only form of current softness KL, for benchmarking only."""

    stabilizers = jnp.asarray(check_matrix, dtype=jnp.uint8) & jnp.uint8(1)
    errors = jnp.asarray(error_operators, dtype=jnp.uint8) & jnp.uint8(1)
    enumerated_stabilizers = (
        jnp.asarray(s_structure, dtype=jnp.uint8) @ stabilizers
    ) & jnp.uint8(1)
    in_stabilizer = jnp.any(
        jnp.all(
            errors[:, None, :] == enumerated_stabilizers[None, :, :],
            axis=-1,
        ),
        axis=1,
    )
    commutes = _jax_commutes_with_stabilizers(errors, stabilizers)
    logical = commutes & ~in_stabilizer
    return _jax_kl_result_from_masks(
        logical,
        commutes,
        in_stabilizer,
        error_probabilities,
        lbda,
    )


def _compact_jax_kl_result(result: JaxKLResult) -> JaxKLStepScalars:
    return JaxKLStepScalars(
        error_count=result.error_count,
        logical_error_probability=result.logical_error_probability,
        error_cost=result.error_cost,
        reward=result.reward,
        terminal=result.terminal,
    )


def _block_jax_tree(value):
    return jax.tree_util.tree_map(lambda item: item.block_until_ready(), value)


def benchmark_jax_kl_reward_calculation(
    check_matrices: jax.Array,
    error_operators: jax.Array,
    error_probabilities: jax.Array,
    lbda: float,
    *,
    num_steps: int = 100,
    repetitions: int = 5,
    print_summary: bool = True,
) -> dict[str, object]:
    """Benchmark only repeated per-step KL/reward calculation.

    Model inference, tableau construction, error/cache construction, and JIT
    compilation are excluded. Every method is warmed and synchronized first.
    """

    if int(num_steps) < 1 or int(repetitions) < 1:
        raise ValueError("num_steps and repetitions must be positive")
    matrices = jnp.asarray(check_matrices, dtype=jnp.uint8)
    if matrices.ndim == 2:
        matrices = jnp.broadcast_to(matrices, (int(num_steps),) + matrices.shape)
    elif matrices.ndim == 3:
        num_steps = int(matrices.shape[0])
    else:
        raise ValueError("check_matrices must have shape [m, 2n] or [steps, m, 2n]")

    errors = jnp.asarray(error_operators, dtype=jnp.uint8)
    probabilities = jnp.asarray(error_probabilities)
    num_stabilizers = int(matrices.shape[1])
    method_kernels = {"gf2": jax_exact_gf2_kl}
    for softness in (1, 2, 3):
        structure = jnp.asarray(
            build_s_structure(num_stabilizers, softness), dtype=jnp.uint8
        )
        method_kernels[f"softness_{softness}"] = (
            lambda matrix, operators, probs, scale, structure=structure: (
                jax_softness_kl(matrix, operators, probs, structure, scale)
            )
        )

    runners = {}
    warmup_seconds = {}
    for name, kernel in method_kernels.items():
        def run_steps(matrix_sequence, step_kernel=kernel):
            def step(carry, matrix):
                del carry
                result = _compact_jax_kl_result(
                    step_kernel(matrix, errors, probabilities, lbda)
                )
                return None, result

            return jax.lax.scan(step, None, matrix_sequence)[1]

        runner = jax.jit(run_steps)
        started = time.perf_counter()
        _block_jax_tree(runner(matrices))
        warmup_seconds[name] = time.perf_counter() - started
        runners[name] = runner

    outputs = {}
    timings = {}
    calculations = int(num_steps) * int(repetitions)
    for name, runner in runners.items():
        started = time.perf_counter()
        output = None
        for _ in range(int(repetitions)):
            output = _block_jax_tree(runner(matrices))
        total_seconds = time.perf_counter() - started
        outputs[name] = output
        timings[name] = {
            "total_seconds": total_seconds,
            "mean_seconds_per_step": total_seconds / calculations,
            "calculations": calculations,
            "warmup_seconds_excluded": warmup_seconds[name],
        }

    gf2_output = outputs["gf2"]
    comparisons = {}
    gf2_seconds = timings["gf2"]["total_seconds"]
    for softness in (1, 2, 3):
        name = f"softness_{softness}"
        output = outputs[name]
        comparisons[name] = {
            "speedup": timings[name]["total_seconds"] / gf2_seconds,
            "reward_allclose": bool(
                np.asarray(jnp.allclose(output.reward, gf2_output.reward))
            ),
            "error_count_equal": bool(
                np.asarray(jnp.all(output.error_count == gf2_output.error_count))
            ),
            "terminal_equal": bool(
                np.asarray(jnp.all(output.terminal == gf2_output.terminal))
            ),
        }

    result = {
        "num_steps": int(num_steps),
        "repetitions": int(repetitions),
        "calculations_per_method": calculations,
        "timings": timings,
        "speedup_definition": "softness total seconds / GF(2) total seconds",
        "gf2_comparisons": comparisons,
    }
    if print_summary:
        print("JAX KL/reward kernel benchmark")
        print(f"  steps={num_steps} repetitions={repetitions}")
        for name, values in timings.items():
            print(
                f"  {name:10s} mean={values['mean_seconds_per_step']:.9f}s "
                f"total={values['total_seconds']:.6f}s"
            )
        for name, values in comparisons.items():
            print(
                f"  gf2 vs {name}: speedup={values['speedup']:.3f}x "
                f"reward_equal={values['reward_allclose']} "
                f"error_equal={values['error_count_equal']} "
                f"terminal_equal={values['terminal_equal']}"
            )
    return result


@dataclass(frozen=True)
class WeightVerificationStats:
    """Exact counts for one physical Pauli weight."""

    weight: int
    violation_count: int
    total_count: int
    commuting_count: int
    stabilizer_count: int

    @property
    def violation_rate(self) -> float:
        return self.violation_count / self.total_count if self.total_count else 0.0

    def to_dict(self) -> dict[str, int | float | str]:
        result = asdict(self)
        result.update(
            {
                "d": self.weight,
                "error_count": self.violation_count,
                "error_count_over_total": (
                    f"{self.violation_count}/{self.total_count}"
                ),
                "error_rate": self.violation_rate,
            }
        )
        return result


@dataclass(frozen=True)
class GF2DistanceResult:
    """Result of an exact search over the requested weight range."""

    target_distance: int
    max_weight_checked: int
    target_met: bool
    estimated_distance: int
    distance_is_exact: bool
    first_logical_weight: int | None
    weight_stats: tuple[WeightVerificationStats, ...]
    logical_error: np.ndarray | None = None

    @property
    def estimated_distance_label(self) -> str:
        prefix = "" if self.distance_is_exact else ">="
        return f"{prefix}{self.estimated_distance}"

    def to_dict(self) -> dict[str, object]:
        return {
            "target_distance": self.target_distance,
            "max_weight_checked": self.max_weight_checked,
            "target_met": self.target_met,
            "estimated_distance": self.estimated_distance,
            "estimated_distance_label": self.estimated_distance_label,
            "distance_is_exact": self.distance_is_exact,
            "first_logical_weight": self.first_logical_weight,
            "weight_stats": [item.to_dict() for item in self.weight_stats],
            "logical_error": (
                None if self.logical_error is None else self.logical_error.tolist()
            ),
        }


def _as_binary_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only binary values")
    return np.ascontiguousarray(array, dtype=np.uint8)


def gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the reduced row echelon form and pivot columns over GF(2)."""

    reduced = _as_binary_matrix(matrix, name="matrix").copy()
    row = 0
    pivots: list[int] = []
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[row:, column])
        if not candidates.size:
            continue
        pivot_row = row + int(candidates[0])
        if pivot_row != row:
            reduced[[row, pivot_row]] = reduced[[pivot_row, row]]

        eliminate = reduced[:, column].astype(bool)
        eliminate[row] = False
        reduced[eliminate] ^= reduced[row]
        pivots.append(column)
        row += 1
        if row == reduced.shape[0]:
            break

    return reduced[:row], tuple(pivots)


def gf2_row_space_mask(
    vectors: np.ndarray,
    reduced_basis: np.ndarray,
    pivot_columns: Sequence[int],
) -> np.ndarray:
    """Vectorized membership test against a GF(2) RREF row basis."""

    candidates = _as_binary_matrix(vectors, name="vectors").copy()
    basis = _as_binary_matrix(reduced_basis, name="reduced_basis")
    pivots = tuple(int(column) for column in pivot_columns)
    if candidates.shape[1] != basis.shape[1]:
        raise ValueError("vectors and reduced_basis must have the same width")
    if len(pivots) != basis.shape[0]:
        raise ValueError("one pivot column is required for every basis row")

    for row, column in enumerate(pivots):
        eliminate = candidates[:, column].astype(bool)
        candidates[eliminate] ^= basis[row]
    return ~np.any(candidates, axis=1)


def symplectic_commutation_mask(
    errors: np.ndarray,
    check_matrix: np.ndarray,
) -> np.ndarray:
    """Return which Pauli errors commute with every stabilizer generator."""

    errors = _as_binary_matrix(errors, name="errors")
    check_matrix = _validate_check_matrix(check_matrix, validate_commutation=False)
    if errors.shape[1] != check_matrix.shape[1]:
        raise ValueError("errors and check_matrix must have the same width")

    n = check_matrix.shape[1] // 2
    products = errors[:, :n] @ check_matrix[:, n:].T
    products ^= errors[:, n:] @ check_matrix[:, :n].T
    return ~np.any(products & np.uint8(1), axis=1)


def _validate_check_matrix(
    check_matrix: np.ndarray,
    *,
    validate_commutation: bool,
) -> np.ndarray:
    check_matrix = _as_binary_matrix(check_matrix, name="check_matrix")
    if not check_matrix.shape[1] or check_matrix.shape[1] % 2:
        raise ValueError("check_matrix must have nonzero even width [H_X | H_Z]")
    if validate_commutation and check_matrix.shape[0]:
        n = check_matrix.shape[1] // 2
        products = check_matrix[:, :n] @ check_matrix[:, n:].T
        products ^= check_matrix[:, n:] @ check_matrix[:, :n].T
        if np.any(products & np.uint8(1)):
            raise ValueError("check_matrix contains anticommuting stabilizer rows")
    return check_matrix


@lru_cache(maxsize=None)
def cached_exact_weight_pauli_errors(n_qubits: int, weight: int) -> np.ndarray:
    """Load or build an immutable, shared array of exact-weight Pauli errors."""

    n_qubits = int(n_qubits)
    weight = int(weight)
    arrays = load_or_build_array_bundle(
        "utils_exact_weight_error_operators",
        {
            "n_qubits_physical": n_qubits,
            "eval_weight": weight,
        },
        lambda: {
            "error_ops": build_exact_weight_error_operators(n_qubits, weight),
        },
    )
    errors = np.ascontiguousarray(arrays["error_ops"], dtype=np.uint8)
    errors.setflags(write=False)
    return errors


@lru_cache(maxsize=None)
def error_weight_indices_upto(
    n_qubits: int,
    code_distance: int,
) -> np.ndarray:
    """Return the physical weight for each environment error operator.

    The ordering matches runtime_cache.build_error_operators_upto: all
    weight-1 operators, then weight-2 operators, and so on through d-1.
    """
    n_qubits = int(n_qubits)
    max_weight = min(int(code_distance) - 1, n_qubits)
    if max_weight < 1:
        result = np.zeros((0,), dtype=np.int32)
    else:
        result = np.concatenate(
            [
                np.full(
                    comb(n_qubits, weight) * (3 ** weight),
                    weight,
                    dtype=np.int32,
                )
                for weight in range(1, max_weight + 1)
            ]
        )
    result.setflags(write=False)
    return result


def precache_pauli_errors(task_ranges: Iterable[tuple[int, int]]) -> None:

    """Preload ``(n_qubits, max_weight)`` error ranges before timed work."""

    unique_ranges = {(int(n), int(max_weight)) for n, max_weight in task_ranges}
    for n_qubits, max_weight in sorted(unique_ranges):
        for weight in range(1, min(n_qubits, max_weight) + 1):
            cached_exact_weight_pauli_errors(n_qubits, weight)


def verify_stabilizer_distance_gf2(
    check_matrix: np.ndarray,
    target_distance: int,
    *,
    max_weight: int | None = None,
    chunk_size: int | None = None,
    stop_at_first_logical_weight: bool = True,
    validate_stabilizers: bool = True,
) -> GF2DistanceResult:
    """Exactly verify distance using symplectic and GF(2) row-space tests.

    By default only weights ``1..target_distance-1`` are searched, which is
    sufficient to decide whether the target is met. Set ``max_weight`` to the
    target itself when an exact count at that weight is also desired.
    """

    target_distance = int(target_distance)
    if target_distance < 1:
        raise ValueError("target_distance must be positive")
    required_max_weight = target_distance - 1
    resolved_max_weight = (
        required_max_weight if max_weight is None else int(max_weight)
    )
    if resolved_max_weight < required_max_weight:
        raise ValueError("max_weight must be at least target_distance - 1")
    if chunk_size is not None and int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")

    check_matrix = _validate_check_matrix(
        check_matrix,
        validate_commutation=validate_stabilizers,
    )
    n_qubits = check_matrix.shape[1] // 2
    resolved_max_weight = min(resolved_max_weight, n_qubits)
    reduced_basis, pivots = gf2_rref(check_matrix)

    stats: list[WeightVerificationStats] = []
    first_logical_weight: int | None = None
    first_logical_error: np.ndarray | None = None
    for weight in range(1, resolved_max_weight + 1):
        errors = cached_exact_weight_pauli_errors(n_qubits, weight)
        batch_size = len(errors) if chunk_size is None else int(chunk_size)
        violation_count = 0
        commuting_count = 0
        stabilizer_count = 0

        for start in range(0, len(errors), batch_size):
            batch = errors[start : start + batch_size]
            commutes = symplectic_commutation_mask(batch, check_matrix)
            normalizer_errors = batch[commutes]
            commuting_count += int(commutes.sum())
            if not len(normalizer_errors):
                continue
            in_row_space = gf2_row_space_mask(
                normalizer_errors,
                reduced_basis,
                pivots,
            )
            stabilizer_count += int(in_row_space.sum())
            logical_count = int((~in_row_space).sum())
            violation_count += logical_count
            if logical_count and first_logical_error is None:
                first_logical_error = normalizer_errors[~in_row_space][0].copy()

        stats.append(
            WeightVerificationStats(
                weight=weight,
                violation_count=violation_count,
                total_count=len(errors),
                commuting_count=commuting_count,
                stabilizer_count=stabilizer_count,
            )
        )
        if violation_count and first_logical_weight is None:
            first_logical_weight = weight
            if stop_at_first_logical_weight:
                break

    checked_through = stats[-1].weight if stats else 0
    distance_is_exact = first_logical_weight is not None
    estimated_distance = (
        first_logical_weight if distance_is_exact else checked_through + 1
    )
    target_met = (
        first_logical_weight is None or first_logical_weight >= target_distance
    )
    return GF2DistanceResult(
        target_distance=target_distance,
        max_weight_checked=checked_through,
        target_met=target_met,
        estimated_distance=estimated_distance,
        distance_is_exact=distance_is_exact,
        first_logical_weight=first_logical_weight,
        weight_stats=tuple(stats),
        logical_error=first_logical_error,
    )


def stabilizer_check_matrix_from_tableau(
    tableau: np.ndarray,
    n_logical: int,
) -> np.ndarray:
    """Extract stabilizer generators from a QDX Clifford tableau."""

    tableau = _as_binary_matrix(tableau, name="tableau")
    if tableau.shape[0] != tableau.shape[1] or tableau.shape[1] % 2:
        raise ValueError("tableau must be a square 2n by 2n binary matrix")
    n_qubits = tableau.shape[0] // 2
    n_logical = int(n_logical)
    if not 0 <= n_logical < n_qubits:
        raise ValueError("n_logical must satisfy 0 <= n_logical < n_qubits")
    return np.ascontiguousarray(tableau[n_qubits + n_logical :])


def stabilizer_check_matrix_from_gates(
    n_qubits: int,
    n_logical: int,
    gates: Sequence[str],
) -> np.ndarray:
    """Apply QDX ``.gate(args)`` strings and return final stabilizers."""

    simulator = TableauSimulator(int(n_qubits))
    for gate in gates:
        match = _GATE_PATTERN.fullmatch(str(gate).strip())
        if match is None:
            raise ValueError(f"invalid QDX gate string: {gate!r}")
        gate_name, raw_arguments = match.groups()
        gate_method = getattr(simulator, gate_name, None)
        if gate_method is None or gate_name.startswith("_"):
            raise ValueError(f"unsupported tableau gate: {gate_name!r}")
        arguments = (
            tuple(int(item.strip()) for item in raw_arguments.split(","))
            if raw_arguments.strip()
            else ()
        )
        gate_method(*arguments)
    tableau = np.asarray(simulator.current_tableau[0], dtype=np.uint8)
    return stabilizer_check_matrix_from_tableau(tableau, n_logical)
