"""Graph-observation interface over unchanged QDX environment dynamics."""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from qdx.envs.code_discovery import CodeDiscovery, EnvParams, EnvState
from qdx.gnn.observation import GraphObservation, GraphObservationBuilder, GraphPadding


class GraphCodeDiscovery(CodeDiscovery):
    """CodeDiscovery with padded GraphObservation outputs.

    Reward calculation, terminal logic, action matrices, and tableau transitions are
    inherited without modification from CodeDiscovery.
    """

    def __init__(self, *args, graph_padding: Optional[GraphPadding] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_builder = GraphObservationBuilder(
            n=self.n_qubits_physical,
            k=self.n_qubits_logical,
            d=self.d,
            max_steps=self.max_steps,
            gates=self.gates,
            hardware_edges=self.graph,
            padding=graph_padding,
        )
        if len(self.graph_builder.action_descriptors) != self.num_actions:
            raise ValueError("graph action ordering does not match environment actions")

    def get_obs(
        self, state: EnvState, params: Optional[EnvParams] = None
    ) -> GraphObservation:
        check_matrix = self.get_observation(state.tableau)
        return self.graph_builder.build(check_matrix, state.time)

    @partial(jax.jit, static_argnames=("self",))
    def step(self, key, state, action, params=None):
        """Gymnax auto-reset with PyTree-aware graph observation selection."""

        if params is None:
            params = self.default_params
        key_step, key_reset = jax.random.split(key)
        obs_step, state_step, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_reset, state_reset = self.reset_env(key_reset, params)
        state = jax.tree.map(
            lambda reset, stepped: jax.lax.select(done, reset, stepped),
            state_reset,
            state_step,
        )
        obs = jax.tree.map(
            lambda reset, stepped: jax.lax.select(done, reset, stepped),
            obs_reset,
            obs_step,
        )
        return obs, state, reward, done, info

    def graph_observation_template(self) -> GraphObservation:
        return self.graph_builder.empty_observation()

    def action_descriptor(self, action_index: int):
        return self.graph_builder.action_descriptor(action_index)

    def gate_matrix_for_action(self, action_index: int) -> jnp.ndarray:
        if not 0 <= int(action_index) < self.num_actions:
            raise IndexError(f"invalid action index: {action_index}")
        return self.actions[int(action_index)]

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Dict:
        builder = self.graph_builder
        return spaces.Dict(
            {
                "node_features": spaces.Box(
                    0.0, 1.0, (builder.max_nodes, 6), dtype=jnp.float32
                ),
                "edge_features": spaces.Box(
                    0.0, 1.0, (builder.max_edges, 2), dtype=jnp.float32
                ),
                "global_features": spaces.Box(
                    0.0, 1.0, (3,), dtype=jnp.float32
                ),
                "action_mask": spaces.Box(
                    0, 1, (builder.max_actions,), dtype=jnp.bool_
                ),
            }
        )
