from __future__ import annotations

import hashlib
import json
import os
from itertools import combinations, product
from math import comb
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping

import numpy as np

CACHE_VERSION = 1
RUNTIME_CACHE_ROOT = Path(__file__).resolve().parents[1] / "runtime_cache"


def ensure_runtime_cache_dirs(root: Path | str | None = None) -> dict[str, Path]:
    cache_root = RUNTIME_CACHE_ROOT if root is None else Path(root)
    static_arrays = cache_root / "static_arrays"
    jax_compilation = cache_root / "jax_compilation"
    static_arrays.mkdir(parents=True, exist_ok=True)
    jax_compilation.mkdir(parents=True, exist_ok=True)
    return {
        "root": cache_root,
        "static_arrays": static_arrays,
        "jax_compilation": jax_compilation,
    }


def configure_jax_persistent_cache(root: Path | str | None = None) -> dict[str, Path]:
    cache_dirs = ensure_runtime_cache_dirs(root)
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR", str(cache_dirs["jax_compilation"])
    )
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    return cache_dirs


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _cache_file_path(
    namespace: str,
    key_data: Mapping[str, Any],
    root: Path | str | None = None,
) -> tuple[Path, dict[str, Any]]:
    cache_dirs = ensure_runtime_cache_dirs(root)
    namespace_dir = cache_dirs["static_arrays"] / namespace
    namespace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "namespace": namespace,
        "key": _json_safe(dict(key_data)),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return namespace_dir / f"{digest}.npz", payload


def load_or_build_array_bundle(
    namespace: str,
    key_data: Mapping[str, Any],
    builder: Callable[[], Mapping[str, Any]],
    root: Path | str | None = None,
) -> dict[str, np.ndarray]:
    cache_file, metadata = _cache_file_path(namespace, key_data, root=root)
    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                return {name: data[name] for name in data.files}
        except Exception:
            cache_file.unlink(missing_ok=True)
            cache_file.with_suffix(".json").unlink(missing_ok=True)

    arrays = {name: np.asarray(value) for name, value in builder().items()}

    with NamedTemporaryFile(dir=cache_file.parent, suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, cache_file)
    finally:
        tmp_path.unlink(missing_ok=True)

    metadata_path = cache_file.with_suffix(".json")
    with NamedTemporaryFile(
        dir=metadata_path.parent, suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tmp_meta:
        json.dump(metadata, tmp_meta, indent=2, sort_keys=True)
        tmp_meta_path = Path(tmp_meta.name)
    try:
        os.replace(tmp_meta_path, metadata_path)
    finally:
        tmp_meta_path.unlink(missing_ok=True)

    return arrays


def build_s_structure(num_stabilizers: int, softness: int) -> np.ndarray:
    max_softness = min(int(softness), int(num_stabilizers))
    soft_elements = sum(
        comb(num_stabilizers, weight)
        for weight in range(1, max_softness + 1)
    )
    s_struct = np.zeros((soft_elements, num_stabilizers), dtype=np.uint8)

    start_idx = 0
    for weight in range(1, max_softness + 1):
        for row_offset, indices in enumerate(combinations(range(num_stabilizers), weight)):
            s_struct[start_idx + row_offset, indices] = 1
        start_idx += comb(num_stabilizers, weight)

    if s_struct.size:
        assert np.prod(np.any(s_struct, axis=1)), "There is a row with all zeroes"
    return s_struct


def build_exact_weight_error_operators(
    n_qubits_physical: int,
    weight: int,
) -> np.ndarray:
    weight = int(weight)
    n_qubits_physical = int(n_qubits_physical)
    if weight < 1 or weight > n_qubits_physical:
        return np.zeros((0, 2 * n_qubits_physical), dtype=np.uint8)

    total_errors = comb(n_qubits_physical, weight) * (3 ** weight)
    error_ops = np.zeros((total_errors, 2 * n_qubits_physical), dtype=np.uint8)

    row = 0
    for positions in combinations(range(n_qubits_physical), weight):
        for pauli_types in product((1, 2, 3), repeat=weight):
            row_values = error_ops[row]
            for position, pauli_type in zip(positions, pauli_types):
                if pauli_type != 3:
                    row_values[position] = 1
                if pauli_type != 1:
                    row_values[n_qubits_physical + position] = 1
            row += 1

    return error_ops


def build_error_operators_upto(
    n_qubits_physical: int,
    code_distance: int,
    p_identity: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_qubits_physical = int(n_qubits_physical)
    max_weight = min(int(code_distance) - 1, n_qubits_physical)
    if max_weight < 1:
        return (
            np.zeros((0, 2 * n_qubits_physical), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )

    total_errors = sum(
        comb(n_qubits_physical, weight) * (3 ** weight)
        for weight in range(1, max_weight + 1)
    )
    error_ops = np.zeros((total_errors, 2 * n_qubits_physical), dtype=np.uint8)
    probabilities = np.empty((total_errors,), dtype=np.float32)

    p_single = np.float32((1.0 - p_identity) / 3.0)
    p_identity = np.float32(p_identity)
    row = 0
    for weight in range(1, max_weight + 1):
        weight_probability = np.float32(
            (p_single ** weight) * (p_identity ** (n_qubits_physical - weight))
        )
        for positions in combinations(range(n_qubits_physical), weight):
            for pauli_types in product((1, 2, 3), repeat=weight):
                row_values = error_ops[row]
                for position, pauli_type in zip(positions, pauli_types):
                    if pauli_type != 3:
                        row_values[position] = 1
                    if pauli_type != 1:
                        row_values[n_qubits_physical + position] = 1
                probabilities[row] = weight_probability
                row += 1

    return error_ops, probabilities
