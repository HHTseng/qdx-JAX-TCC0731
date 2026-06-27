"""Flax implementation of the GNN-QDX v1 actor-critic."""

from typing import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from qdx.gnn.observation import GraphObservation, TWO_QUBIT_ACTION


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        for index, feature_count in enumerate(self.features):
            x = nn.Dense(feature_count, name=f"dense_{index}")(x)
            if index + 1 < len(self.features):
                x = activation(x)
        return x


def _masked_mean(values, mask):
    weights = mask.astype(values.dtype)[..., :, None]
    return jnp.sum(values * weights, axis=-2) / jnp.maximum(
        jnp.sum(weights, axis=-2), 1.0
    )


def _gather_nodes(nodes, indices):
    """Gather from the node axis while preserving arbitrary leading batch axes."""

    indices = jnp.broadcast_to(indices, nodes.shape[:-2] + indices.shape[-1:])
    gather_indices = jnp.broadcast_to(
        indices[..., :, None], indices.shape + (nodes.shape[-1],)
    )
    return jnp.take_along_axis(nodes, gather_indices, axis=-2)


class GNNQDXActorCritic(nn.Module):
    """Variable-size candidate-scoring actor and global-state critic."""

    num_gate_types: int
    hidden_dim: int = 64
    relation_dim: int = 8
    gate_dim: int = 8
    num_gnn_layers: int = 3
    activation: str = "tanh"

    @nn.compact
    def __call__(self, graph_obs: GraphObservation):
        h = MLP((self.hidden_dim,), self.activation, name="node_embed")(
            graph_obs.node_features
        )
        g = MLP(
            (self.hidden_dim, self.hidden_dim),
            self.activation,
            name="global_embed",
        )(graph_obs.global_features)
        relation_embedder = nn.Embed(3, self.relation_dim, name="relation_embedding")
        gate_embedder = nn.Embed(
            self.num_gate_types, self.gate_dim, name="gate_embedding"
        )

        node_mask_f = graph_obs.node_mask.astype(h.dtype)[..., :, None]
        h = h * node_mask_f
        for layer in range(self.num_gnn_layers):
            sender_h = _gather_nodes(h, graph_obs.senders)
            receiver_h = _gather_nodes(h, graph_obs.receivers)
            relation_h = relation_embedder(graph_obs.relation_ids)
            g_edges = jnp.broadcast_to(
                g[..., None, :], sender_h.shape[:-1] + (g.shape[-1],)
            )
            message_input = jnp.concatenate(
                [
                    sender_h,
                    receiver_h,
                    graph_obs.edge_features,
                    relation_h,
                    g_edges,
                ],
                axis=-1,
            )
            messages = MLP(
                (self.hidden_dim, self.hidden_dim),
                self.activation,
                name=f"edge_mlp_{layer}",
            )(message_input)
            edge_weights = graph_obs.edge_mask.astype(messages.dtype)[..., :, None]
            messages = messages * edge_weights

            receiver_one_hot = jax.nn.one_hot(
                graph_obs.receivers, h.shape[-2], dtype=messages.dtype
            )
            message_sum = jnp.einsum("...ev,...eh->...vh", receiver_one_hot, messages)
            message_count = jnp.einsum(
                "...ev,...e->...v",
                receiver_one_hot,
                graph_obs.edge_mask.astype(messages.dtype),
            )[..., :, None]
            aggregated = message_sum / jnp.maximum(message_count, 1.0)

            g_nodes = jnp.broadcast_to(
                g[..., None, :], h.shape[:-1] + (g.shape[-1],)
            )
            node_delta = MLP(
                (self.hidden_dim, self.hidden_dim),
                self.activation,
                name=f"node_mlp_{layer}",
            )(jnp.concatenate([h, aggregated, g_nodes], axis=-1))
            h = (h + node_delta) * node_mask_f

            q_pool = _masked_mean(h, graph_obs.qubit_mask)
            s_pool = _masked_mean(h, graph_obs.stabilizer_mask)
            global_delta = MLP(
                (self.hidden_dim, self.hidden_dim),
                self.activation,
                name=f"global_mlp_{layer}",
            )(jnp.concatenate([g, q_pool, s_pool], axis=-1))
            g = g + global_delta

        first_h = _gather_nodes(h, graph_obs.action_first)
        second_h = _gather_nodes(h, graph_obs.action_second)
        gate_h = gate_embedder(graph_obs.action_gate_ids)
        g_actions = jnp.broadcast_to(
            g[..., None, :], first_h.shape[:-1] + (g.shape[-1],)
        )
        single_logits = MLP(
            (self.hidden_dim, 1), self.activation, name="single_action_mlp"
        )(jnp.concatenate([first_h, gate_h, g_actions], axis=-1))[..., 0]
        two_logits = MLP(
            (self.hidden_dim, 1), self.activation, name="two_action_mlp"
        )(
            jnp.concatenate([first_h, second_h, gate_h, g_actions], axis=-1)
        )[..., 0]
        logits = jnp.where(
            graph_obs.action_types == TWO_QUBIT_ACTION, two_logits, single_logits
        )
        logits = jnp.where(graph_obs.action_mask, logits, -1.0e9)

        value = MLP((self.hidden_dim, 1), self.activation, name="value_mlp")(g)
        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)
