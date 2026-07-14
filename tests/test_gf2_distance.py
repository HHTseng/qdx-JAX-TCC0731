import unittest

import jax
import jax.numpy as jnp
import numpy as np

from qdx.gf2_distance import (
    gf2_rref,
    gf2_row_space_mask,
    jax_exact_gf2_kl,
    jax_tableau_kl,
    jax_softness_kl,
    stabilizer_check_matrix_from_gates,
    symplectic_commutation_mask,
    verify_stabilizer_distance_gf2,
)
from qdx.runtime_cache import build_s_structure
from qdx.simulators import TableauSimulator


def check_matrix(*paulis):
    n_qubits = len(paulis[0])
    matrix = np.zeros((len(paulis), 2 * n_qubits), dtype=np.uint8)
    for row, pauli in enumerate(paulis):
        for qubit, operator in enumerate(pauli):
            if operator in "XY":
                matrix[row, qubit] = 1
            if operator in "YZ":
                matrix[row, n_qubits + qubit] = 1
    return matrix





class GF2DistanceTest(unittest.TestCase):
    def test_jax_tableau_kl_matches_rref_and_reports_weight_rates(self):
        simulator = TableauSimulator(3)
        simulator.h(0)
        simulator.cx(0, 1)
        simulator.s(2)
        tableau = simulator.current_tableau[0]
        stabilizers = tableau[4:]
        errors = jnp.asarray(
            check_matrix("XII", "ZII", "IXI", "IZI", "IIZ", "IXX")
        )
        probabilities = jnp.asarray(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            dtype=jnp.float32,
        )
        weights = jnp.asarray([1, 1, 1, 1, 1, 2], dtype=jnp.int32)
        weight_values = jnp.asarray([1, 2], dtype=jnp.int32)

        exact = jax.jit(jax_exact_gf2_kl)(
            stabilizers,
            errors,
            probabilities,
            10.0,
            error_weights=weights,
            weight_values=weight_values,
        )
        direct = jax.jit(jax_tableau_kl, static_argnums=1)(
            tableau,
            1,
            errors,
            probabilities,
            10.0,
            error_weights=weights,
            weight_values=weight_values,
        )

        np.testing.assert_array_equal(
            np.asarray(direct.logical_error_mask),
            np.asarray(exact.logical_error_mask),
        )
        np.testing.assert_array_equal(
            np.asarray(direct.commutes_mask),
            np.asarray(exact.commutes_mask),
        )
        np.testing.assert_array_equal(
            np.asarray(direct.in_stabilizer_mask),
            np.asarray(exact.in_stabilizer_mask),
        )
        np.testing.assert_array_equal(
            np.asarray(direct.error_count_by_weight),
            np.asarray(exact.error_count_by_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(direct.total_count_by_weight),
            [5, 1],
        )
        np.testing.assert_allclose(
            np.asarray(direct.error_rate_by_weight),
            np.asarray(exact.error_rate_by_weight),
        )
        self.assertAlmostEqual(float(direct.reward), float(exact.reward), places=6)


    def test_jax_exact_kl_logical_mask_count_and_weighted_reward(self):
        stabilizers = jnp.asarray(check_matrix("ZZI", "IZZ"))
        errors = jnp.asarray(check_matrix("ZIZ", "ZII", "XII"))
        probabilities = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)

        result = jax.jit(jax_exact_gf2_kl)(
            stabilizers,
            errors,
            probabilities,
            10.0,
        )

        np.testing.assert_array_equal(
            np.asarray(result.in_stabilizer_mask), [True, False, False]
        )
        np.testing.assert_array_equal(
            np.asarray(result.commutes_mask), [True, True, False]
        )
        np.testing.assert_array_equal(
            np.asarray(result.logical_error_mask), [False, True, False]
        )
        self.assertEqual(int(result.error_count), 1)
        self.assertAlmostEqual(
            float(result.logical_error_probability), 0.3, places=6
        )
        self.assertAlmostEqual(float(result.error_cost), 3.0, places=6)
        self.assertAlmostEqual(float(result.reward), -3.0, places=6)
        self.assertFalse(bool(result.terminal))

    def test_jax_exact_kl_uses_full_row_space_not_softness(self):
        stabilizers = jnp.asarray(check_matrix("ZZI", "IZZ"))
        errors = jnp.asarray(check_matrix("ZIZ", "ZII", "XII"))
        probabilities = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)
        exact = jax_exact_gf2_kl(stabilizers, errors, probabilities, 10.0)
        softness_one = jax_softness_kl(
            stabilizers,
            errors,
            probabilities,
            jnp.asarray(build_s_structure(2, 1)),
            10.0,
        )
        softness_two = jax_softness_kl(
            stabilizers,
            errors,
            probabilities,
            jnp.asarray(build_s_structure(2, 2)),
            10.0,
        )

        self.assertEqual(int(exact.error_count), 1)
        self.assertEqual(int(softness_one.error_count), 2)
        self.assertAlmostEqual(float(softness_one.reward), -5.0, places=6)
        self.assertEqual(int(softness_two.error_count), int(exact.error_count))
        self.assertAlmostEqual(
            float(softness_two.reward), float(exact.reward), places=6
        )

    def test_jax_exact_kl_supports_vmap_and_scan(self):
        stabilizers = jnp.asarray(check_matrix("ZZI", "IZZ"))
        errors = jnp.asarray(check_matrix("ZIZ", "ZII", "XII"))
        probabilities = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)
        batch = jnp.stack((stabilizers, stabilizers))

        vmapped = jax.jit(
            jax.vmap(jax_exact_gf2_kl, in_axes=(0, None, None, None))
        )(batch, errors, probabilities, 10.0)

        def scan_step(carry, matrix):
            del carry
            result = jax_exact_gf2_kl(matrix, errors, probabilities, 10.0)
            return None, result.error_count

        _, scanned = jax.jit(
            lambda matrices: jax.lax.scan(scan_step, None, matrices)
        )(batch)
        np.testing.assert_array_equal(np.asarray(vmapped.error_count), [1, 1])
        np.testing.assert_array_equal(np.asarray(scanned), [1, 1])

    def test_jax_exact_kl_terminal_when_no_logical_error_remains(self):
        result = jax_exact_gf2_kl(
            jnp.asarray(check_matrix("ZZI", "IZZ")),
            jnp.asarray(check_matrix("ZIZ", "XII")),
            jnp.asarray([0.4, 0.6], dtype=jnp.float32),
            10.0,
        )
        self.assertEqual(int(result.error_count), 0)
        self.assertAlmostEqual(float(result.reward), 0.0, places=6)
        self.assertTrue(bool(result.terminal))

    def test_rref_row_space_membership_with_redundant_generators(self):
        generators = np.asarray(
            [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]],
            dtype=np.uint8,
        )
        reduced, pivots = gf2_rref(generators)
        vectors = np.asarray(
            [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(
            gf2_row_space_mask(vectors, reduced, pivots),
            [True, True, False],
        )

    def test_symplectic_commutation(self):
        stabilizers = check_matrix("ZZI", "IZZ")
        errors = check_matrix("XII", "ZII", "IIZ")
        np.testing.assert_array_equal(
            symplectic_commutation_mask(errors, stabilizers),
            [False, True, True],
        )

    def test_three_qubit_repetition_code_has_weight_one_logical(self):
        result = verify_stabilizer_distance_gf2(
            check_matrix("ZZI", "IZZ"),
            target_distance=2,
        )

        self.assertFalse(result.target_met)
        self.assertTrue(result.distance_is_exact)
        self.assertEqual(result.estimated_distance, 1)
        self.assertEqual(result.weight_stats[0].violation_count, 3)

    def test_five_qubit_code_has_exact_distance_three(self):
        five_qubit_code = check_matrix("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")

        target_result = verify_stabilizer_distance_gf2(
            five_qubit_code,
            target_distance=3,
        )
        exact_result = verify_stabilizer_distance_gf2(
            five_qubit_code,
            target_distance=3,
            max_weight=3,
        )

        self.assertTrue(target_result.target_met)
        self.assertFalse(target_result.distance_is_exact)
        self.assertEqual(target_result.estimated_distance_label, ">=3")
        self.assertTrue(exact_result.target_met)
        self.assertTrue(exact_result.distance_is_exact)
        self.assertEqual(exact_result.estimated_distance, 3)
        self.assertGreater(exact_result.weight_stats[-1].violation_count, 0)

    def test_gate_helper_matches_qdx_tableau_layout(self):
        no_gate_check = stabilizer_check_matrix_from_gates(2, 1, [])
        after_h = stabilizer_check_matrix_from_gates(2, 1, [".h(1)"])

        np.testing.assert_array_equal(no_gate_check, check_matrix("IZ"))
        np.testing.assert_array_equal(after_h, check_matrix("IX"))

    def test_rejects_anticommuting_stabilizers(self):
        with self.assertRaisesRegex(ValueError, "anticommuting"):
            verify_stabilizer_distance_gf2(
                check_matrix("X", "Z"),
                target_distance=2,
            )


if __name__ == "__main__":
    unittest.main()
