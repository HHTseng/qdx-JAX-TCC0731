# GNN-QDX v1.2 SPEC: Hardware Edge State Features and Edge-Aware Two-Qubit Action Head

## 1. Goal

GNN-QDX v1.2 extends the v1.1 graph observation and actor head to improve two-qubit action scoring under variable-size QEC search.

The main changes are:

1. Add lightweight hardware edge features.
2. Keep all size-related quantities as `log1p(count)` or ratios.
3. Do not add noise features.
4. Do not expose padding or bucket parameters such as `n_max`, `hardware_edges_max`, or `actions_max` to the model.
5. Add hardware-edge embedding directly to the two-qubit action head.

This version is designed for A2A, NN1, and NN2 hardware graphs, where static topology features are mostly redundant, but state-dependent pair features are useful for selecting better two-qubit gates.

---

## 2. Edge Feature Dimension

Update:

```python
CHECK_EDGE_FEATURE_DIM = 5
HW_EDGE_FEATURE_DIM = 8
EDGE_FEATURE_DIM = CHECK_EDGE_FEATURE_DIM + HW_EDGE_FEATURE_DIM  # 13
```

The first 5 dimensions remain the existing stabilizer-check edge features:

```text
0. h_x
1. h_z
2. x_only
3. z_only
4. y_like
```

The new hardware edge feature block occupies dimensions 5--12:

```text
5.  log_deg_src
6.  log_deg_dst
7.  shared_support_jaccard
8.  shared_x_jaccard
9.  shared_y_jaccard
10. shared_z_jaccard
11. directed_load_diff
12. directed_xz_balance_diff
```

---

## 3. Edge Feature Layout

### 3.1 Check edges

For `CHECK_S_TO_Q` and `CHECK_Q_TO_S` edges:

```text
[h_x, h_z, x_only, z_only, y_like, 0, 0, 0, 0, 0, 0, 0, 0]
```

The existing check-edge behavior should remain unchanged.

### 3.2 Hardware edges

For `HW_Q_TO_Q` edges from `src -> dst`:

```text
[0, 0, 0, 0, 0,
 log_deg_src,
 log_deg_dst,
 shared_support_jaccard,
 shared_x_jaccard,
 shared_y_jaccard,
 shared_z_jaccard,
 directed_load_diff,
 directed_xz_balance_diff]
```

---

## 4. Hardware Edge Feature Definitions

For each directed hardware edge:

```text
e = (src, dst)
```

Let:

```text
deg_src = hardware degree of src
deg_dst = hardware degree of dst
```

Use log-scaled degree features:

```text
log_deg_src = log1p(deg_src)
log_deg_dst = log1p(deg_dst)
```

These are the only static topology features in v1.2. They are included because they are cheap and may help distinguish boundary-like qubits in NN1/NN2 graphs, while avoiding raw degree counts.

---

## 5. State-Dependent Pair Features

These features are computed inside `GraphObservationBuilder.build()` from the current check matrix.

Let:

```text
touched[s, q] = 1 if stabilizer s acts as X, Y, or Z on qubit q
x_only[s, q]  = 1 if stabilizer s acts as X-only on qubit q
y_like[s, q]  = 1 if stabilizer s acts as Y-like on qubit q
z_only[s, q]  = 1 if stabilizer s acts as Z-only on qubit q
```

### 5.1 Shared support Jaccard

```text
shared_support_jaccard =
  sum_s touched[s, src] * touched[s, dst]
  / (sum_s max(touched[s, src], touched[s, dst]) + EPSILON)
```

Meaning: how often the two endpoint qubits appear in the same stabilizer support.

### 5.2 Shared X/Y/Z Jaccard

```text
shared_x_jaccard =
  sum_s x_only[s, src] * x_only[s, dst]
  / (sum_s max(x_only[s, src], x_only[s, dst]) + EPSILON)

shared_y_jaccard =
  sum_s y_like[s, src] * y_like[s, dst]
  / (sum_s max(y_like[s, src], y_like[s, dst]) + EPSILON)

shared_z_jaccard =
  sum_s z_only[s, src] * z_only[s, dst]
  / (sum_s max(z_only[s, src], z_only[s, dst]) + EPSILON)
```

Meaning: whether the two endpoint qubits play similar X/Y/Z roles in the current stabilizer structure.

### 5.3 Directed load difference

Let:

```text
total_check_degree[q] = sum_s touched[s, q]
load_frac[q] = total_check_degree[q] / max(num_stabilizers, 1)
```

Then:

```text
directed_load_diff = load_frac[src] - load_frac[dst]
```

Meaning: whether the source/control endpoint is more heavily used in the current stabilizer support than the destination/target endpoint.

### 5.4 Directed XZ-balance difference

Let:

```text
x_degree[q] = sum_s x_only[s, q]
z_degree[q] = sum_s z_only[s, q]

xz_balance[q] =
  (x_degree[q] - z_degree[q])
  / (total_check_degree[q] + EPSILON)
```

Then:

```text
directed_xz_balance_diff = xz_balance[src] - xz_balance[dst]
```

Meaning: direction-aware difference between the X-heavy and Z-heavy roles of the two endpoints. This is especially useful for directed two-qubit gates such as CNOT.

---

## 6. Required ObservationBuilder Changes

### 6.1 Store hardware edge endpoints

In `_build_static_arrays()`, store hardware edge source and destination arrays:

```python
self._hw_src = jnp.asarray([i for i, j in self.hardware_edges], dtype=jnp.int32)
self._hw_dst = jnp.asarray([j for i, j in self.hardware_edges], dtype=jnp.int32)
```

Also store the graph-edge index for each hardware edge:

```python
self._hw_edge_indices = jnp.arange(
    self._hw_offset,
    self._hw_offset + len(self.hardware_edges),
    dtype=jnp.int32,
)
```

### 6.2 Add action-edge mapping

Add a new field to `GraphObservation`:

```python
action_edge_indices: jnp.ndarray
```

For single-qubit actions and padded actions, set a safe default index, e.g. `0`. These values must be ignored by the single-qubit head and by masked padded actions.

For two-qubit actions, set:

```python
action_edge_indices[action_id] = corresponding hardware graph edge index
```

This allows the actor head to gather the learned hardware edge embedding for each two-qubit candidate action.

---

## 7. Required Model Changes

The v1.1 model already computes:

```python
edge_h = edge_embed_mlp(concat(edge_features, relation_onehot))
```

v1.2 keeps this behavior, but also uses `edge_h` in the two-qubit action head.

### 7.1 Gather action edge embeddings

After message passing, add:

```python
action_edge_h = _gather_edges(edge_h, graph_obs.action_edge_indices)
```

Implement `_gather_edges()` analogously to `_gather_nodes()`, but gather along the edge axis.

### 7.2 Update two-qubit action head

Change the two-qubit action head from:

```python
two_logits = MLP_two([
    first_h,
    second_h,
    gate_h,
    g_actions,
])
```

to:

```python
two_logits = MLP_two([
    first_h,
    second_h,
    action_edge_h,
    gate_h,
    g_actions,
])
```

The single-qubit action head remains unchanged:

```python
single_logits = MLP_single([
    first_h,
    gate_h,
    g_actions,
])
```

---

## 8. Invariants and Constraints

v1.2 must satisfy:

```text
No noise features.
No raw size features such as n, n-k, num_edges, or num_actions.
No padding or bucket features such as n_max, hardware_edges_max, actions_max, max_nodes, or max_edges.
All size-related edge features must be log-scaled or ratios.
The existing check-edge feature semantics must remain unchanged.
The actor must still mask invalid and padded actions with action_mask.
```

---

## 9. Expected Benefit

The new hardware edge features give the model explicit pair-level information about the current stabilizer structure. This is especially useful for A2A, NN1, and NN2 settings, where static topology features are often nearly constant across valid two-qubit edges.

The edge-aware two-qubit action head lets the actor directly score:

```text
CNOT(src -> dst)
```

using:

```text
src node embedding
+ dst node embedding
+ hardware edge embedding
+ gate embedding
+ global embedding
```

This should improve two-qubit gate selection and reduce the burden on message passing to indirectly encode pair-specific compatibility into node embeddings.
