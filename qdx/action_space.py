"""Shared action-space helpers for QDX environments and graph observations."""

from dataclasses import dataclass
from inspect import signature
from typing import Any, Optional, Sequence, Tuple


ACTION_SPACE_VERSION = "v1.4"
GATE_NAME_ALIASES = {"CX": "CNOT"}
SYMMETRIC_TWO_QUBIT_GATES = frozenset({"CZ", "SQRT_XX"})


@dataclass(frozen=True)
class ActionSpec:
    """Host-side description of one executable candidate action."""

    gate_id: int
    gate_name: str
    arity: int
    qubit: Optional[int] = None
    first: Optional[int] = None
    second: Optional[int] = None
    is_symmetric: bool = False

    @property
    def action_type(self) -> str:
        return "single" if self.arity == 1 else "two"

    def gate_args(self) -> Tuple[int, ...]:
        if self.arity == 1:
            return (int(self.qubit),)
        return (int(self.first), int(self.second))


def canonical_gate_name(gate: Any) -> str:
    """Normalize gate function names used by model metadata."""

    name = gate.__name__.upper()
    return GATE_NAME_ALIASES.get(name, name)


def gate_arity(gate: Any) -> int:
    return len(signature(gate).parameters)


def build_action_specs(
    n_qubits: int,
    gates: Sequence[Any],
    hardware_edges: Sequence[Tuple[int, int]],
) -> Tuple[ActionSpec, ...]:
    """Build candidate actions with v1.4 symmetric-gate canonicalization."""

    n_qubits = int(n_qubits)
    edges = tuple((int(src), int(dst)) for src, dst in hardware_edges)
    specs = []

    for gate_id, gate in enumerate(gates):
        arity = gate_arity(gate)
        gate_name = canonical_gate_name(gate)
        if arity == 1:
            specs.extend(
                ActionSpec(gate_id, gate_name, arity, qubit=qubit)
                for qubit in range(n_qubits)
            )
        elif arity == 2:
            if gate_name in SYMMETRIC_TWO_QUBIT_GATES:
                seen_pairs = set()
                for src, dst in edges:
                    _validate_edge(src, dst, n_qubits)
                    if src == dst:
                        raise ValueError(
                            f"symmetric gate {gate_name} requires distinct qubits"
                        )
                    first, second = sorted((src, dst))
                    pair = (first, second)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    specs.append(
                        ActionSpec(
                            gate_id,
                            gate_name,
                            arity,
                            first=first,
                            second=second,
                            is_symmetric=True,
                        )
                    )
            else:
                for src, dst in edges:
                    _validate_edge(src, dst, n_qubits)
                    specs.append(
                        ActionSpec(
                            gate_id,
                            gate_name,
                            arity,
                            first=src,
                            second=dst,
                        )
                    )
        else:
            raise ValueError("QDX supports only one- and two-qubit gates")

    return tuple(specs)


def _validate_edge(src: int, dst: int, n_qubits: int) -> None:
    if not (0 <= src < n_qubits and 0 <= dst < n_qubits):
        raise ValueError(f"invalid hardware edge {(src, dst)} for n={n_qubits}")
