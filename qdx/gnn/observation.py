"""Padded heterogeneous graph observations for GNN-QDX v1.1."""

from dataclasses import dataclass
from inspect import signature
from typing import Any, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from flax import struct


CHECK_S_TO_Q = 0
CHECK_Q_TO_S = 1
HW_Q_TO_Q = 2
NUM_RELATION_TYPES = 3

SINGLE_ACTION = 0
TWO_QUBIT_ACTION = 1

NODE_FEATURE_DIM = 21
EDGE_FEATURE_DIM = 5
GLOBAL_FEATURE_DIM = 9
EPSILON = 1.0e-6

GATE_NAME_ALIASES = {"CX": "CNOT"}


@struct.dataclass
class GraphObservation:
    """One fixed-shape graph observation; leading batch axes are added by JAX."""

    node_features: jnp.ndarray
    edge_features: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray
    relation_ids: jnp.ndarray
    node_mask: jnp.ndarray
    edge_mask: jnp.ndarray
    qubit_mask: jnp.ndarray
    stabilizer_mask: jnp.ndarray
    global_features: jnp.ndarray
    action_types: jnp.ndarray
    action_gate_ids: jnp.ndarray
    action_first: jnp.ndarray
    action_second: jnp.ndarray
    action_mask: jnp.ndarray
    action_env_indices: jnp.ndarray


@dataclass(frozen=True)
class GraphPadding:
    """Static bucket sizes used only for JIT shapes, never for model parameters."""

    n_max: int
    stabilizers_max: Optional[int] = None
    hardware_edges_max: Optional[int] = None
    actions_max: Optional[int] = None

    @property
    def resolved_stabilizers_max(self) -> int:
        return self.n_max if self.stabilizers_max is None else self.stabilizers_max

    @property
    def resolved_hardware_edges_max(self) -> int:
        if self.hardware_edges_max is None:
            return self.n_max * max(self.n_max - 1, 0)
        return self.hardware_edges_max


@dataclass(frozen=True)
class ActionDescriptor:
    """Host-side description of one environment-compatible candidate action."""

    action_type: str
    gate: str
    qubit: Optional[int] = None
    control: Optional[int] = None
    target: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.action_type, "gate": self.gate}
        if self.action_type == "single":
            result["qubit"] = self.qubit
        else:
            result["control"] = self.control
            result["target"] = self.target
        return result


class GraphObservationBuilder:
    """Turn a stabilizer check matrix and hardware graph into a padded graph."""

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        max_steps: int,
        gates: Sequence[Any],
        hardware_edges: Sequence[Tuple[int, int]],
        padding: Optional[GraphPadding] = None,
    ):
        self.n = int(n)
        self.k = int(k)
        self.d = int(d)
        self.num_stabilizers = self.n - self.k
        self.max_steps = int(max_steps)
        self.gates = tuple(gates)
        self.hardware_edges = tuple((int(i), int(j)) for i, j in hardware_edges)
        self.padding = padding or GraphPadding(n_max=self.n)

        self.n_max = self.padding.n_max
        self.stabilizers_max = self.padding.resolved_stabilizers_max
        self.hardware_edges_max = self.padding.resolved_hardware_edges_max
        if self.n > self.n_max:
            raise ValueError(f"n={self.n} exceeds graph bucket n_max={self.n_max}")
        if self.num_stabilizers > self.stabilizers_max:
            raise ValueError("number of stabilizers exceeds graph bucket capacity")
        if len(self.hardware_edges) > self.hardware_edges_max:
            raise ValueError("hardware edge count exceeds graph bucket capacity")

        self.gate_names = tuple(
            GATE_NAME_ALIASES.get(gate.__name__.upper(), gate.__name__.upper())
            for gate in self.gates
        )
        self.gate_arities = tuple(len(signature(gate).parameters) for gate in self.gates)
        if any(arity not in (1, 2) for arity in self.gate_arities):
            raise ValueError("GNN-QDX v1.1 supports only one- and two-qubit gates")
        self.num_gate_types = len(self.gates)

        self.max_nodes = self.n_max + self.stabilizers_max
        self.max_edges = (
            2 * self.n_max * self.stabilizers_max + self.hardware_edges_max
        )
        default_actions_max = sum(
            self.n_max if arity == 1 else self.hardware_edges_max
            for arity in self.gate_arities
        )
        self.max_actions = (
            default_actions_max
            if self.padding.actions_max is None
            else self.padding.actions_max
        )

        self.action_descriptors = self._build_action_descriptors()
        if len(self.action_descriptors) > self.max_actions:
            raise ValueError("candidate action count exceeds graph bucket capacity")
        self._build_static_arrays()

    def _build_action_descriptors(self) -> Tuple[ActionDescriptor, ...]:
        descriptors = []
        for gate_name, arity in zip(self.gate_names, self.gate_arities):
            if arity == 1:
                descriptors.extend(
                    ActionDescriptor("single", gate_name, qubit=i)
                    for i in range(self.n)
                )
            else:
                descriptors.extend(
                    ActionDescriptor("two", gate_name, control=i, target=j)
                    for i, j in self.hardware_edges
                )
        return tuple(descriptors)

    def _build_static_arrays(self) -> None:
        node_mask = np.zeros(self.max_nodes, dtype=bool)
        qubit_mask = np.zeros(self.max_nodes, dtype=bool)
        stabilizer_mask = np.zeros(self.max_nodes, dtype=bool)
        node_mask[: self.n] = True
        qubit_mask[: self.n] = True
        stab_start = self.n_max
        node_mask[stab_start : stab_start + self.num_stabilizers] = True
        stabilizer_mask[stab_start : stab_start + self.num_stabilizers] = True
        self._node_mask = jnp.asarray(node_mask)
        self._qubit_mask = jnp.asarray(qubit_mask)
        self._stabilizer_mask = jnp.asarray(stabilizer_mask)

        check_pairs = self.n * self.num_stabilizers
        q = np.tile(np.arange(self.n, dtype=np.int32), self.num_stabilizers)
        s = np.repeat(
            np.arange(self.num_stabilizers, dtype=np.int32), self.n
        ) + stab_start
        senders = np.zeros(self.max_edges, dtype=np.int32)
        receivers = np.zeros(self.max_edges, dtype=np.int32)
        relations = np.zeros(self.max_edges, dtype=np.int32)
        senders[:check_pairs], receivers[:check_pairs] = s, q
        relations[:check_pairs] = CHECK_S_TO_Q
        senders[check_pairs : 2 * check_pairs] = q
        receivers[check_pairs : 2 * check_pairs] = s
        relations[check_pairs : 2 * check_pairs] = CHECK_Q_TO_S
        hw_offset = 2 * check_pairs
        for offset, (i, j) in enumerate(self.hardware_edges):
            if not (0 <= i < self.n and 0 <= j < self.n):
                raise ValueError(f"invalid hardware edge {(i, j)} for n={self.n}")
            senders[hw_offset + offset] = i
            receivers[hw_offset + offset] = j
            relations[hw_offset + offset] = HW_Q_TO_Q
        self._senders = jnp.asarray(senders)
        self._receivers = jnp.asarray(receivers)
        self._relation_ids = jnp.asarray(relations)
        self._check_pairs = check_pairs
        self._hw_offset = hw_offset

        neighbors = [set() for _ in range(self.n)]
        for i, j in self.hardware_edges:
            neighbors[i].add(j)
            neighbors[j].add(i)
        hardware_degree = np.asarray(
            [len(node_neighbors) for node_neighbors in neighbors], dtype=np.float32
        )
        self._hardware_degree = jnp.asarray(hardware_degree)
        self._normalized_hw_degree = jnp.asarray(
            hardware_degree / float(max(self.n - 1, 1))
        )

        action_types = np.zeros(self.max_actions, dtype=np.int32)
        gate_ids = np.zeros(self.max_actions, dtype=np.int32)
        first = np.zeros(self.max_actions, dtype=np.int32)
        second = np.zeros(self.max_actions, dtype=np.int32)
        action_mask = np.zeros(self.max_actions, dtype=bool)
        env_indices = np.full(self.max_actions, -1, dtype=np.int32)
        cursor = 0
        for gate_id, arity in enumerate(self.gate_arities):
            if arity == 1:
                for i in range(self.n):
                    gate_ids[cursor] = gate_id
                    first[cursor] = i
                    cursor += 1
            else:
                for i, j in self.hardware_edges:
                    action_types[cursor] = TWO_QUBIT_ACTION
                    gate_ids[cursor] = gate_id
                    first[cursor] = i
                    second[cursor] = j
                    cursor += 1
        action_mask[:cursor] = True
        env_indices[:cursor] = np.arange(cursor, dtype=np.int32)
        self._action_types = jnp.asarray(action_types)
        self._action_gate_ids = jnp.asarray(gate_ids)
        self._action_first = jnp.asarray(first)
        self._action_second = jnp.asarray(second)
        self._action_mask = jnp.asarray(action_mask)
        self._action_env_indices = jnp.asarray(env_indices)

    def build(self, check_matrix: jnp.ndarray, time: jnp.ndarray) -> GraphObservation:
        """Build the graph using only JAX operations, so it is safe under jit."""

        check_matrix = jnp.asarray(check_matrix, dtype=jnp.float32).reshape(
            self.num_stabilizers, 2 * self.n
        )
        h_x = check_matrix[:, : self.n]
        h_z = check_matrix[:, self.n :]

        x_only = jnp.logical_and(h_x != 0, h_z == 0)
        z_only = jnp.logical_and(h_x == 0, h_z != 0)
        y_like = jnp.logical_and(h_x != 0, h_z != 0)
        touched = jnp.logical_or(jnp.logical_or(x_only, z_only), y_like)

        x_only_f = x_only.astype(jnp.float32)
        z_only_f = z_only.astype(jnp.float32)
        y_like_f = y_like.astype(jnp.float32)
        touched_f = touched.astype(jnp.float32)

        stabilizer_denominator = float(max(self.num_stabilizers, 1))
        qubit_denominator = float(max(self.n, 1))

        x_degree = jnp.sum(x_only_f, axis=0)
        z_degree = jnp.sum(z_only_f, axis=0)
        y_degree = jnp.sum(y_like_f, axis=0)
        total_check_degree = jnp.sum(touched_f, axis=0)

        x_weight = jnp.sum(x_only_f, axis=1)
        z_weight = jnp.sum(z_only_f, axis=1)
        y_weight = jnp.sum(y_like_f, axis=1)
        total_weight = jnp.sum(touched_f, axis=1)

        if self.n > 0:
            mean_qubit_check_degree = jnp.mean(total_check_degree)
            mean_x_degree = jnp.mean(x_degree)
            mean_z_degree = jnp.mean(z_degree)
            mean_y_degree = jnp.mean(y_degree)
            std_qubit_check_degree = jnp.std(total_check_degree)
            mean_hardware_degree = jnp.mean(self._hardware_degree)
        else:
            mean_qubit_check_degree = jnp.asarray(0.0, dtype=jnp.float32)
            mean_x_degree = jnp.asarray(0.0, dtype=jnp.float32)
            mean_z_degree = jnp.asarray(0.0, dtype=jnp.float32)
            mean_y_degree = jnp.asarray(0.0, dtype=jnp.float32)
            std_qubit_check_degree = jnp.asarray(0.0, dtype=jnp.float32)
            mean_hardware_degree = jnp.asarray(0.0, dtype=jnp.float32)

        if self.num_stabilizers > 0:
            mean_stabilizer_weight = jnp.mean(total_weight)
            mean_x_weight = jnp.mean(x_weight)
            mean_z_weight = jnp.mean(z_weight)
            mean_y_weight = jnp.mean(y_weight)
            std_stabilizer_weight = jnp.std(total_weight)
        else:
            mean_stabilizer_weight = jnp.asarray(0.0, dtype=jnp.float32)
            mean_x_weight = jnp.asarray(0.0, dtype=jnp.float32)
            mean_z_weight = jnp.asarray(0.0, dtype=jnp.float32)
            mean_y_weight = jnp.asarray(0.0, dtype=jnp.float32)
            std_stabilizer_weight = jnp.asarray(0.0, dtype=jnp.float32)

        node_features = jnp.zeros(
            (self.max_nodes, NODE_FEATURE_DIM), dtype=jnp.float32
        )
        qubit_features = jnp.stack(
            [
                jnp.ones(self.n),
                jnp.zeros(self.n),
                total_check_degree / stabilizer_denominator,
                jnp.sum(h_x, axis=0) / stabilizer_denominator,
                jnp.sum(h_z, axis=0) / stabilizer_denominator,
                x_degree / stabilizer_denominator,
                z_degree / stabilizer_denominator,
                y_degree / stabilizer_denominator,
                jnp.log1p(total_check_degree),
                total_check_degree / (mean_qubit_check_degree + EPSILON),
                x_degree / (mean_x_degree + EPSILON),
                z_degree / (mean_z_degree + EPSILON),
                y_degree / (mean_y_degree + EPSILON),
                self._normalized_hw_degree,
                jnp.log1p(self._hardware_degree),
                x_degree / (total_check_degree + EPSILON),
                z_degree / (total_check_degree + EPSILON),
                y_degree / (total_check_degree + EPSILON),
                (x_degree - z_degree) / (total_check_degree + EPSILON),
                jnp.zeros(self.n),
                jnp.zeros(self.n),
            ],
            axis=-1,
        )
        stabilizer_features = jnp.stack(
            [
                jnp.zeros(self.num_stabilizers),
                jnp.ones(self.num_stabilizers),
                total_weight / qubit_denominator,
                jnp.sum(h_x, axis=1) / qubit_denominator,
                jnp.sum(h_z, axis=1) / qubit_denominator,
                jnp.zeros(self.num_stabilizers),
                jnp.zeros(self.num_stabilizers),
                jnp.zeros(self.num_stabilizers),
                jnp.zeros(self.num_stabilizers),
                jnp.zeros(self.num_stabilizers),
                x_weight / (mean_x_weight + EPSILON),
                z_weight / (mean_z_weight + EPSILON),
                y_weight / (mean_y_weight + EPSILON),
                jnp.zeros(self.num_stabilizers),
                jnp.zeros(self.num_stabilizers),
                x_weight / (total_weight + EPSILON),
                z_weight / (total_weight + EPSILON),
                y_weight / (total_weight + EPSILON),
                (x_weight - z_weight) / (total_weight + EPSILON),
                jnp.log1p(total_weight),
                total_weight / (mean_stabilizer_weight + EPSILON),
            ],
            axis=-1,
        )
        node_features = node_features.at[: self.n].set(qubit_features)
        node_features = node_features.at[
            self.n_max : self.n_max + self.num_stabilizers
        ].set(stabilizer_features)

        edge_features = jnp.zeros((self.max_edges, EDGE_FEATURE_DIM), dtype=jnp.float32)
        edge_mask = jnp.zeros(self.max_edges, dtype=bool)
        check_features = jnp.stack(
            [
                h_x.reshape(-1),
                h_z.reshape(-1),
                x_only_f.reshape(-1),
                z_only_f.reshape(-1),
                y_like_f.reshape(-1),
            ],
            axis=-1,
        )
        touched_flat = touched.reshape(-1)
        edge_features = edge_features.at[: self._check_pairs].set(check_features)
        edge_features = edge_features.at[
            self._check_pairs : 2 * self._check_pairs
        ].set(check_features)
        edge_mask = edge_mask.at[: self._check_pairs].set(touched_flat)
        edge_mask = edge_mask.at[
            self._check_pairs : 2 * self._check_pairs
        ].set(touched_flat)
        edge_mask = edge_mask.at[
            self._hw_offset : self._hw_offset + len(self.hardware_edges)
        ].set(True)

        time_f = jnp.asarray(time, dtype=jnp.float32)
        max_steps = float(max(self.max_steps, 1))
        global_features = jnp.asarray(
            [
                time_f / max_steps,
                (max_steps - time_f) / max_steps,
                float(self.k) / float(max(self.n, 1)),
                float(self.d) / float(max(self.n, 1)),
                jnp.log1p(mean_stabilizer_weight),
                std_stabilizer_weight / (mean_stabilizer_weight + EPSILON),
                jnp.log1p(mean_qubit_check_degree),
                std_qubit_check_degree / (mean_qubit_check_degree + EPSILON),
                jnp.log1p(mean_hardware_degree),
            ],
            dtype=jnp.float32,
        )
        return GraphObservation(
            node_features=node_features,
            edge_features=edge_features,
            senders=self._senders,
            receivers=self._receivers,
            relation_ids=self._relation_ids,
            node_mask=self._node_mask,
            edge_mask=edge_mask,
            qubit_mask=self._qubit_mask,
            stabilizer_mask=self._stabilizer_mask,
            global_features=global_features,
            action_types=self._action_types,
            action_gate_ids=self._action_gate_ids,
            action_first=self._action_first,
            action_second=self._action_second,
            action_mask=self._action_mask,
            action_env_indices=self._action_env_indices,
        )

    def empty_observation(self) -> GraphObservation:
        check = jnp.zeros((self.num_stabilizers, 2 * self.n), dtype=jnp.uint8)
        return self.build(check, jnp.asarray(0, dtype=jnp.int32))

    def action_descriptor(self, action_index: int) -> Dict[str, Any]:
        """Map an actor index to the gate descriptor used by the QDX environment."""

        if not 0 <= int(action_index) < len(self.action_descriptors):
            raise IndexError(f"invalid or padded action index: {action_index}")
        return self.action_descriptors[int(action_index)].as_dict()
