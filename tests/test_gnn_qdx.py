import unittest

import jax
import jax.numpy as jnp
import numpy as np

from qdx.envs.graph_code_discovery import GraphCodeDiscovery
from qdx.gnn import GNNQDXActorCritic, GraphPadding
from qdx.make_train import make_train
from qdx.simulators.clifford_gates import CliffordGates


def make_env(n, padding):
    gates = CliffordGates(n)
    hardware_edges = []
    for i in range(n - 1):
        hardware_edges.extend([(i, i + 1), (i + 1, i)])
    return GraphCodeDiscovery(
        n,
        1,
        2,
        [gates.h, gates.s, gates.cx],
        graph=hardware_edges,
        max_steps=3,
        softness=1,
        graph_padding=padding,
    )


class GNNQDXSmokeTest(unittest.TestCase):
    def test_variable_size_forward_action_mapping_and_step(self):
        padding = GraphPadding(
            n_max=4,
            stabilizers_max=3,
            hardware_edges_max=6,
        )
        small_env = make_env(3, padding)
        large_env = make_env(4, padding)
        small_obs, small_state = small_env.reset(jax.random.PRNGKey(0), None)
        large_obs, _ = large_env.reset(jax.random.PRNGKey(1), None)

        self.assertEqual(small_obs.node_features.shape, large_obs.node_features.shape)
        self.assertEqual(small_obs.edge_features.shape, large_obs.edge_features.shape)
        self.assertEqual(small_obs.action_mask.shape, large_obs.action_mask.shape)
        self.assertEqual(int(small_obs.node_mask.sum()), 3 + (3 - 1))
        self.assertEqual(int(small_obs.action_mask.sum()), small_env.num_actions)

        network = GNNQDXActorCritic(
            num_gate_types=3,
            hidden_dim=16,
            num_gnn_layers=1,
        )
        params = network.init(jax.random.PRNGKey(2), small_obs)
        small_policy, small_value = network.apply(params, small_obs)
        large_policy, large_value = network.apply(params, large_obs)

        self.assertEqual(small_policy.logits.shape, small_obs.action_mask.shape)
        self.assertEqual(large_policy.logits.shape, large_obs.action_mask.shape)
        self.assertEqual(small_value.shape, ())
        self.assertEqual(large_value.shape, ())
        self.assertTrue(bool(jnp.all(jnp.isfinite(small_policy.logits))))
        self.assertTrue(
            bool(jnp.all(small_policy.logits[~small_obs.action_mask] < -1.0e8))
        )

        first_two_qubit_action = 2 * small_env.n_qubits_physical
        self.assertEqual(
            small_env.action_descriptor(first_two_qubit_action),
            {
                "type": "two",
                "gate": "CNOT",
                "control": 0,
                "target": 1,
            },
        )
        np.testing.assert_array_equal(
            np.asarray(small_env.gate_matrix_for_action(first_two_qubit_action)),
            np.asarray(small_env.actions[first_two_qubit_action]),
        )

        action = jnp.asarray(1)
        expected_tableau = (small_state.tableau @ small_env.actions[action]) % 2
        next_obs, next_state, reward, done, _ = small_env.step(
            jax.random.PRNGKey(3), small_state, action, None
        )
        self.assertFalse(bool(done))
        np.testing.assert_array_equal(
            np.asarray(next_state.tableau), np.asarray(expected_tableau)
        )
        self.assertTrue(bool(jnp.isfinite(reward)))
        self.assertAlmostEqual(float(next_obs.global_features[0]), 1.0 / 3.0)


    def test_one_ppo_update(self):
        padding = GraphPadding(
            n_max=3,
            stabilizers_max=2,
            hardware_edges_max=4,
        )
        env = make_env(3, padding)
        config = {
            "MODEL": "GNN",
            "TOTAL_TIMESTEPS": 1,
            "NUM_STEPS": 1,
            "NUM_ENVS": 1,
            "NUM_MINIBATCHES": 1,
            "UPDATE_EPOCHS": 1,
            "LR": 3.0e-4,
            "ACTIVATION": "tanh",
            "HIDDEN_DIM": 8,
            "GNN_HIDDEN_DIM": 8,
            "GNN_NUM_LAYERS": 1,
            "ANNEAL_LR": False,
            "MAX_GRAD_NORM": 0.5,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "VF_COEF": 0.5,
            "ENT_COEF": 0.01,
            "COMPUTE_METRICS": False,
            "MAX_STEPS": 3,
        }
        output = jax.jit(make_train(config, env))(jax.random.PRNGKey(4))
        parameter_leaves = jax.tree_util.tree_leaves(output["params"])
        self.assertTrue(
            all(bool(jnp.all(jnp.isfinite(x))) for x in parameter_leaves)
        )

if __name__ == "__main__":
    unittest.main()
