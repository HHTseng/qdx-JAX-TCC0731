# GNN-QDX v1.1 SPEC: Variable-Size GNN Policy for QDX Code Discovery

## 1. Goal

GNN-QDX v1.1 extends the v1 variable-size GNN actor-critic with richer QEC-aware observation features and a revised embedding design.

The main goals are:

1. Improve representation of stabilizer code structure.
2. Reduce direct dependence on absolute problem size.
3. Preserve variable-size candidate action scoring.
4. Make edge representation more expressive by combining edge features and relation type information.

---

## 2. Definitions

Let:

```text
n = number of physical qubits
k = number of logical qubits
m = n - k = number of stabilizer generators
eps = 1e-6
```

Given the stabilizer check matrix:

```text
H = [Hx | Hz]
```

Define Pauli-type masks:

```text
x = X-only  = (Hx != 0) and (Hz == 0)
z = Z-only  = (Hx == 0) and (Hz != 0)
y = Y-like  = (Hx != 0) and (Hz != 0)
touched = x or z or y
```

For qubit `i`:

```text
x_degree_i = sum_s x[s, i]
z_degree_i = sum_s z[s, i]
y_degree_i = sum_s y[s, i]
total_check_degree_i = sum_s touched[s, i]
```

For stabilizer `j`:

```text
x_weight_j = sum_i x[j, i]
z_weight_j = sum_i z[j, i]
y_weight_j = sum_i y[j, i]
total_weight_j = sum_i touched[j, i]
```

---

## 3. Node Features

The node feature vector is shared by both qubit nodes and stabilizer nodes.
Fields that do not apply to a node type must be filled with `0`.

### Node Feature Table

| Feature                        | Qubit Node Value                                           | Stabilizer Node Value                                | Purpose                                                                      |
| ------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| `is_qubit`                     | `1`                                                        | `0`                                                  | Node type indicator.                                                         |
| `is_stabilizer`                | `0`                                                        | `1`                                                  | Node type indicator.                                                         |
| `total_touch_frac`             | `total_check_degree_i / m`                                 | `total_weight_j / n`                                 | Normalized connection density to the opposite node type.                     |
| `hx_frac`                      | `sum_s Hx[s,i] / m`                                        | `sum_i Hx[j,i] / n`                                  | Original X-component density feature.                                        |
| `hz_frac`                      | `sum_s Hz[s,i] / m`                                        | `sum_i Hz[j,i] / n`                                  | Original Z-component density feature.                                        |
| `x_degree_frac`                | `x_degree_i / m`                                           | `0`                                                  | Fraction of stabilizers touching this qubit with X-only terms.               |
| `z_degree_frac`                | `z_degree_i / m`                                           | `0`                                                  | Fraction of stabilizers touching this qubit with Z-only terms.               |
| `y_degree_frac`                | `y_degree_i / m`                                           | `0`                                                  | Fraction of stabilizers touching this qubit with Y-like terms.               |
| `qubit_check_load_log1p`       | `log1p(total_check_degree_i)`                              | `0`                                                  | Local qubit stabilizer load with reduced size sensitivity.                   |
| `relative_qubit_load`          | `total_check_degree_i / (mean_qubit_check_degree + eps)`   | `0`                                                  | Whether this qubit is overloaded relative to other qubits in the same graph. |
| `x_relative_load`              | `x_degree_i / (mean_x_degree + eps)`                       | `x_weight_j / (mean_x_weight + eps)`                 | Relative X-only load within the same node type.                              |
| `z_relative_load`              | `z_degree_i / (mean_z_degree + eps)`                       | `z_weight_j / (mean_z_weight + eps)`                 | Relative Z-only load within the same node type.                              |
| `y_relative_load`              | `y_degree_i / (mean_y_degree + eps)`                       | `y_weight_j / (mean_y_weight + eps)`                 | Relative Y-like load within the same node type.                              |
| `hardware_degree_over_nminus1` | `hardware_degree_i / (n - 1)`                              | `0`                                                  | Original hardware connectivity feature.                                      |
| `hardware_degree_log1p`        | `log1p(hardware_degree_i)`                                 | `0`                                                  | Local hardware connectivity with reduced size sensitivity.                   |
| `x_frac`                       | `x_degree_i / (total_check_degree_i + eps)`                | `x_weight_j / (total_weight_j + eps)`                | Fraction of local support that is X-only.                                    |
| `z_frac`                       | `z_degree_i / (total_check_degree_i + eps)`                | `z_weight_j / (total_weight_j + eps)`                | Fraction of local support that is Z-only.                                    |
| `y_frac`                       | `y_degree_i / (total_check_degree_i + eps)`                | `y_weight_j / (total_weight_j + eps)`                | Fraction of local support that is Y-like.                                    |
| `xz_balance`                   | `(x_degree_i - z_degree_i) / (total_check_degree_i + eps)` | `(x_weight_j - z_weight_j) / (total_weight_j + eps)` | Local X/Z imbalance.                                                         |
| `weight_log1p`                 | `0`                                                        | `log1p(total_weight_j)`                              | Stabilizer local Pauli weight with reduced size sensitivity.                 |
| `relative_weight`              | `0`                                                        | `total_weight_j / (mean_stabilizer_weight + eps)`    | Whether this stabilizer is heavier than the average stabilizer.              |

### Node Feature Vector Order

```text
node_features = [
    is_qubit,
    is_stabilizer,

    total_touch_frac,
    hx_frac,
    hz_frac,

    x_degree_frac,
    z_degree_frac,
    y_degree_frac,

    qubit_check_load_log1p,
    relative_qubit_load,

    x_relative_load,
    z_relative_load,
    y_relative_load,

    hardware_degree_over_nminus1,
    hardware_degree_log1p,

    x_frac,
    z_frac,
    y_frac,
    xz_balance,

    weight_log1p,
    relative_weight,
]
```

Node feature dimension:

```text
node_feature_dim = 21
```

---

## 4. Edge Features

The graph keeps the same relation types as v1:

```text
CHECK_S_TO_Q = 0
CHECK_Q_TO_S = 1
HW_Q_TO_Q   = 2
```

In v1.1, `relation_id` must be converted to a one-hot vector before being used by the model:

```text
relation_onehot = one_hot(relation_id, 3)
```

### Edge Feature Table

| Feature | Check Edge Value            | Hardware Edge Value | Purpose                                    |
| ------- | --------------------------- | ------------------- | ------------------------------------------ |
| `hx`    | `Hx[s,i]`                   | `0`                 | Whether the check edge has an X component. |
| `hz`    | `Hz[s,i]`                   | `0`                 | Whether the check edge has a Z component.  |
| `is_x`  | `1 if Hx=1 and Hz=0 else 0` | `0`                 | Explicit X-only edge indicator.            |
| `is_z`  | `1 if Hx=0 and Hz=1 else 0` | `0`                 | Explicit Z-only edge indicator.            |
| `is_y`  | `1 if Hx=1 and Hz=1 else 0` | `0`                 | Explicit Y-like edge indicator.            |

### Edge Feature Vector Order

```text
edge_features = [
    hx,
    hz,
    is_x,
    is_z,
    is_y,
]
```

Edge feature dimension:

```text
edge_feature_dim = 5
```

The model-side edge embedding input is:

```text
edge_embed_input = concat(edge_features, relation_onehot)
```

Therefore:

```text
edge_embed_input_dim = 5 + 3 = 8
```

---

## 5. Global Features

### Global Feature Table

| Feature                    | Formula                                    | Purpose                                                        |
| -------------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| `time_frac`                | `time / max_steps`                         | Current search progress.                                       |
| `remaining_steps_frac`     | `(max_steps - time) / max_steps`           | Remaining circuit construction budget.                         |
| `k_over_n`                 | `k / n`                                    | Code rate information.                                         |
| `d_over_n`                 | `d / n`                                    | Target distance normalized by code size.                       |
| `mean_weight_log1p`        | `log1p(mean_stabilizer_weight)`            | Average stabilizer local weight with reduced size sensitivity. |
| `std_weight_over_mean`     | `std_weight / (mean_weight + eps)`         | Imbalance of stabilizer weights.                               |
| `mean_qubit_load_log1p`    | `log1p(mean_qubit_check_load)`             | Average qubit stabilizer load.                                 |
| `std_qubit_load_over_mean` | `std_qubit_load / (mean_qubit_load + eps)` | Imbalance of qubit loads.                                      |
| `mean_hw_degree_log1p`     | `log1p(mean_hardware_degree)`              | Average local hardware connectivity.                           |

### Global Feature Vector Order

```text
global_features = [
    time_frac,
    remaining_steps_frac,

    k_over_n,
    d_over_n,

    mean_weight_log1p,
    std_weight_over_mean,

    mean_qubit_load_log1p,
    std_qubit_load_over_mean,

    mean_hw_degree_log1p,
]
```

Global feature dimension:

```text
global_feature_dim = 9
```

---

## 6. Embedding Architecture

### 6.1 Node Embedding

In v1.1, `node_embed` must be a 2-layer MLP.

```text
node_embed = MLP(
    input_dim = node_feature_dim,
    hidden_dims = [hidden_dim, hidden_dim],
    output_dim = hidden_dim,
)
```

Example module name:

```text
node_embed_mlp
```

Output:

```text
h_i = node_embed_mlp(node_features_i)
```

---

### 6.2 Global Embedding

In v1.1, `global_embed` must also be a 2-layer MLP.

```text
global_embed = MLP(
    input_dim = global_feature_dim,
    hidden_dims = [hidden_dim, hidden_dim],
    output_dim = hidden_dim,
)
```

Example module name:

```text
global_embed_mlp
```

Output:

```text
g = global_embed_mlp(global_features)
```

---

### 6.3 Edge Embedding

The original `relation_embedding` is removed.

Instead, v1.1 uses `edge_embed`, which embeds the concatenation of edge features and one-hot relation type.

```text
relation_onehot = one_hot(relation_id, 3)

edge_embed_input = concat(
    edge_features,
    relation_onehot,
)
```

`edge_embed` must be a 2-layer MLP.

```text
edge_embed = MLP(
    input_dim = edge_feature_dim + num_relation_types,
    hidden_dims = [hidden_dim, hidden_dim],
    output_dim = hidden_dim,
)
```

Example module name:

```text
edge_embed_mlp
```

Output:

```text
e_ij = edge_embed_mlp(concat(edge_features_ij, relation_onehot_ij))
```

---

### 6.4 Gate Embedding

The original learned `nn.Embed` gate embedding is replaced by a one-hot gate input followed by a one-layer projection.

```text
gate_onehot = one_hot(action_gate_id, num_gate_types)
```

Use one linear layer:

```text
gate_embed = Linear(
    input_dim = num_gate_types,
    output_dim = gate_dim,
)
```

Example module name:

```text
gate_embed_linear
```

Output:

```text
gate_h = gate_embed_linear(gate_onehot)
```

`gate_embed` must remain one layer only.

---

## 7. GNN Message Passing

For each GNN layer, gather sender and receiver node states:

```text
h_sender = gather(h, senders)
h_receiver = gather(h, receivers)
```

Use embedded edge state:

```text
edge_h = edge_embed_mlp(concat(edge_features, relation_onehot))
```

The edge message input becomes:

```text
message_input = concat(
    h_sender,
    h_receiver,
    edge_h,
    g
)
```

The message function remains a 2-layer MLP:

```text
message_ij = edge_message_mlp_l(message_input)
```

Aggregate messages by receiver node using masked mean:

```text
aggregated_v = mean_{u -> v}(message_uv)
```

Update nodes with residual connection:

```text
node_delta_v = node_update_mlp_l(concat(h_v, aggregated_v, g))
h_v = h_v + node_delta_v
```

Update global state with pooled qubit and stabilizer states:

```text
q_pool = masked_mean(h, qubit_mask)
s_pool = masked_mean(h, stabilizer_mask)

global_delta = global_update_mlp_l(concat(g, q_pool, s_pool))
g = g + global_delta
```

---

## 8. Actor Head

The actor keeps the v1 variable-size candidate-scoring design.

### Single-Qubit Action

For action:

```text
a = (gate, q_i)
```

Score with:

```text
single_action_input = concat(
    h_i,
    gate_h,
    g
)

single_logit = single_action_mlp(single_action_input)
```

---

### Two-Qubit Action

For action:

```text
a = (gate, q_i, q_j)
```

Score with:

```text
two_action_input = concat(
    h_i,
    h_j,
    gate_h,
    g
)

two_logit = two_action_mlp(two_action_input)
```

Invalid or padded actions must be masked with:

```text
logit = -1e9
```

The final policy is:

```text
pi(a | s) = Categorical(logits)
```

---

## 9. Critic Head

The critic remains global-state based:

```text
value = value_mlp(g)
```

Output:

```text
V(s)
```

---

## 10. Summary of v1.1 Changes

| Component         | v1                           | v1.1                                                      |
| ----------------- | ---------------------------- | --------------------------------------------------------- |
| Node features     | 6 dims                       | 21 dims, QEC-aware X/Z/Y/load features                    |
| Edge features     | `[hx, hz]`                   | `[hx, hz, is_x, is_z, is_y]`                              |
| Global features   | 3 dims                       | 9 dims                                                    |
| Relation encoding | Learned `relation_embedding` | One-hot `relation_id` included in `edge_embed_input`      |
| Edge embedding    | None before message MLP      | 2-layer `edge_embed_mlp(edge_features + relation_onehot)` |
| Node embedding    | MLP                          | 2-layer `node_embed_mlp`                                  |
| Global embedding  | MLP                          | 2-layer `global_embed_mlp`                                |
| Gate embedding    | Learned embedding table      | One-hot input + one-layer linear projection               |
| Actor             | Candidate action scoring     | Same                                                      |
| Critic            | `value = MLP(g)`             | Same                                                      |
