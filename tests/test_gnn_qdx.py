import unittest

import jax
import jax.numpy as jnp
import numpy as np

from qdx.envs.graph_code_discovery import GraphCodeDiscovery
from qdx.gnn import GNNQDXActorCritic, GraphObservationBuilder, GraphPadding
from qdx.gnn.model import _aggregate_messages, _edge_group_masks, _masked_mean
from qdx.gnn.observation import (
    CHECK_Q_TO_S,
    CHECK_S_TO_Q,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    HW_Q_TO_Q,
    NODE_FEATURE_DIM,
)
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
        self.assertEqual(small_obs.node_features.shape[-1], NODE_FEATURE_DIM)
        self.assertEqual(small_obs.edge_features.shape, large_obs.edge_features.shape)
        self.assertEqual(small_obs.edge_features.shape[-1], EDGE_FEATURE_DIM)
        self.assertEqual(small_obs.global_features.shape, (GLOBAL_FEATURE_DIM,))
        self.assertEqual(small_obs.action_mask.shape, large_obs.action_mask.shape)
        self.assertEqual(
            small_obs.action_edge_indices.shape, large_obs.action_edge_indices.shape
        )
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
        self.assertAlmostEqual(float(next_obs.global_features[1]), 2.0 / 3.0)

    def test_v12_feature_builder_emits_expected_stats(self):
        gates = CliffordGates(3)
        builder = GraphObservationBuilder(
            n=3,
            k=1,
            d=2,
            max_steps=4,
            gates=[gates.h, gates.s, gates.cx],
            hardware_edges=[(0, 1), (1, 0), (1, 2), (2, 1)],
            padding=GraphPadding(
                n_max=3,
                stabilizers_max=2,
                hardware_edges_max=4,
            ),
        )
        check = jnp.asarray(
            [
                [1, 0, 1, 0, 1, 1],
                [0, 1, 0, 0, 0, 1],
            ],
            dtype=jnp.uint8,
        )

        obs = builder.build(check, jnp.asarray(1, dtype=jnp.int32))

        self.assertEqual(obs.node_features.shape, (builder.max_nodes, NODE_FEATURE_DIM))
        self.assertEqual(obs.edge_features.shape, (builder.max_edges, EDGE_FEATURE_DIM))
        self.assertEqual(obs.global_features.shape, (GLOBAL_FEATURE_DIM,))
        self.assertEqual(obs.action_edge_indices.shape, (builder.max_actions,))

        np.testing.assert_allclose(
            np.asarray(obs.edge_features[:3]),
            np.pad(
                np.asarray(
                    [
                        [1.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                ((0, 0), (0, EDGE_FEATURE_DIM - 5)),
            ),
        )

        hw_start = 2 * builder.n * builder.num_stabilizers
        np.testing.assert_allclose(
            np.asarray(
                obs.edge_features[hw_start : hw_start + len(builder.hardware_edges)]
            ),
            np.pad(
                np.asarray(
                    [
                        [
                            np.log1p(1.0),
                            np.log1p(2.0),
                            0.5,
                            0.0,
                            0.0,
                            0.0,
                            -0.5,
                            1.0,
                        ],
                        [
                            np.log1p(2.0),
                            np.log1p(1.0),
                            0.5,
                            0.0,
                            0.0,
                            0.0,
                            0.5,
                            -1.0,
                        ],
                        [
                            np.log1p(2.0),
                            np.log1p(1.0),
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.5,
                        ],
                        [
                            np.log1p(1.0),
                            np.log1p(2.0),
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            -0.5,
                        ],
                    ],
                    dtype=np.float32,
                ),
                ((0, 0), (5, 0)),
            ),
            atol=1.0e-6,
        )

        np.testing.assert_array_equal(
            np.asarray(obs.action_edge_indices[: 2 * builder.n]),
            np.zeros(2 * builder.n, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(
                obs.action_edge_indices[
                    2 * builder.n : 2 * builder.n + len(builder.hardware_edges)
                ]
            ),
            np.arange(hw_start, hw_start + len(builder.hardware_edges), dtype=np.int32),
        )

        q0 = np.asarray(obs.node_features[0])
        q2 = np.asarray(obs.node_features[2])
        s0 = np.asarray(obs.node_features[builder.n_max])

        np.testing.assert_allclose(
            q0[[0, 2, 3, 5, 9, 10, 13, 15, 18]],
            np.asarray([1.0, 0.5, 0.5, 0.5, 0.6, 1.5, 0.5, 1.0, 1.0]),
            atol=5.0e-6,
        )
        np.testing.assert_allclose(
            q2[[2, 4, 6, 7, 15, 16, 17, 18]],
            np.asarray([1.0, 1.0, 0.5, 0.5, 0.0, 0.5, 0.5, -0.5]),
            atol=5.0e-6,
        )
        np.testing.assert_allclose(
            s0[[0, 1, 2, 3, 4, 10, 12, 15, 16, 17, 19, 20]],
            np.asarray(
                [
                    0.0,
                    1.0,
                    1.0,
                    2.0 / 3.0,
                    2.0 / 3.0,
                    1.0,
                    2.0,
                    1.0 / 3.0,
                    1.0 / 3.0,
                    1.0 / 3.0,
                    np.log1p(3.0),
                    1.2,
                ]
            ),
            atol=5.0e-6,
        )
        np.testing.assert_allclose(
            np.asarray(obs.global_features),
            np.asarray(
                [
                    0.25,
                    0.75,
                    1.0 / 3.0,
                    2.0 / 3.0,
                    np.log1p(2.5),
                    0.2,
                    np.log1p(5.0 / 3.0),
                    np.sqrt(2.0) / 5.0,
                    np.log1p(4.0 / 3.0),
                ]
            ),
            atol=1.0e-6,
        )

    def test_v13_separates_check_and_hardware_edge_groups(self):
        messages = jnp.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [100.0, 1000.0],
                [200.0, 2000.0],
                [999.0, 9999.0],
            ],
            dtype=jnp.float32,
        )
        receivers = jnp.asarray([0, 0, 0, 1, 1], dtype=jnp.int32)
        relation_ids = jnp.asarray(
            [CHECK_S_TO_Q, CHECK_Q_TO_S, HW_Q_TO_Q, HW_Q_TO_Q, CHECK_S_TO_Q],
            dtype=jnp.int32,
        )
        edge_mask = jnp.asarray([True, True, True, True, False])

        check_mask, hw_mask = _edge_group_masks(relation_ids, edge_mask)
        agg_check = _aggregate_messages(messages, receivers, check_mask, 2)
        agg_hw = _aggregate_messages(messages, receivers, hw_mask, 2)
        check_pool = _masked_mean(messages, check_mask)
        hw_pool = _masked_mean(messages, hw_mask)

        np.testing.assert_allclose(
            np.asarray(agg_check),
            np.asarray([[1.5, 15.0], [0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(agg_hw),
            np.asarray([[100.0, 1000.0], [200.0, 2000.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(check_pool), np.asarray([1.5, 15.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            np.asarray(hw_pool), np.asarray([150.0, 1500.0], dtype=np.float32)
        )

    def test_v13_model_uses_edge_grouped_node_and_global_inputs(self):
        padding = GraphPadding(
            n_max=4,
            stabilizers_max=3,
            hardware_edges_max=6,
        )
        env = make_env(3, padding)
        obs, _ = env.reset(jax.random.PRNGKey(5), None)

        network = GNNQDXActorCritic(
            num_gate_types=3,
            hidden_dim=16,
            num_gnn_layers=1,
        )
        params = network.init(jax.random.PRNGKey(6), obs)

        node_kernel = params['params']['node_update_mlp_0']['dense_0']['kernel']
        global_kernel = params['params']['global_update_mlp_0']['dense_0']['kernel']
        two_action_kernel = params['params']['two_action_mlp']['dense_0']['kernel']

        self.assertEqual(node_kernel.shape, (4 * network.hidden_dim, network.hidden_dim))
        self.assertEqual(global_kernel.shape, (5 * network.hidden_dim, network.hidden_dim))
        self.assertEqual(
            two_action_kernel.shape,
            (3 * network.hidden_dim + network.gate_dim, network.hidden_dim),
        )

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
