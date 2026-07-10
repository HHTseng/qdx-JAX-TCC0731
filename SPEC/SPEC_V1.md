# GNN-QDX v1 SPEC: Variable-Size GNN Policy for QDX Code Discovery

## 0. Goal

This SPEC defines a variable-size GNN actor-critic policy for QDX code discovery. The goal is to replace the original fixed-size MLP policy so that the same set of model parameters can handle different qubit counts (N), and provide Level 4 extrapolation capability for training on small (N) and testing on larger (N).

This version does not yet handle the noise-aware setting. It focuses on the stabilizer check matrix, hardware connectivity, gate action scoring, and PPO actor-critic training.

---

## 1. Design Principles

### 1.1 Do not use a fixed input vector

The original QDX used:

$$
\text{flatten}(H) \in {0,1}^{2N(N-K)}
$$

as the MLP input. This makes the model input dimension depend directly on (N).

GNN-QDX v1 instead uses a graph representation:

```text
stabilizer check matrix + hardware connectivity
        ->
heterogeneous graph
        ->
shared GNN message passing
```

The model no longer depends on a fixed (2N(N-K)) input size.

---

### 1.2 Do not use a fixed action output

The original actor's final layer was:

$$
\text{Dense}(n_A)
$$

where (n_A) is the number of all gate actions under the current (N). This makes the actor output dimension depend on (N).

GNN-QDX v1 instead uses candidate-action scoring:

```text
for each valid candidate action:
    logit = shared_action_scorer(action, graph_embedding)
```

So different (N) values can have different numbers of candidate actions, while the scorer parameters are shared.

---

### 1.3 Do not use learned absolute qubit embeddings

Do not use:

```python
embedding[q_index]
```

This avoids memorizing the qubit index.
The model should learn the qubit role, local stabilizer pattern, hardware degree, and global code state.

---

### 1.4 Do not use a virtual global node

This version does not add a virtual global node.
All global information is passed through the global feature (g) into edge messages, node updates, the global update, the actor head, and the critic head.

---

### 1.5 Use mean for all pooling

All aggregation/pooling is unified to mean:

```text
edge messages -> node: mean
qubit nodes -> global: mean
stabilizer nodes -> global: mean
critic/global pooling: mean
```

This reduces the risk that embedding magnitudes grow uncontrollably as (N) increases.

---

## 2. Input Data

Each environment state is converted into a graph observation.

### 2.1 Raw state

The input comes from the current tableau / stabilizer check matrix:

$$
H = [H_X \mid H_Z]
$$

where:

```text
H_X shape = (N-K, N)
H_Z shape = (N-K, N)
H shape   = (N-K, 2N)
```

Let:

```text
N = number of physical qubits
K = number of logical qubits
S = N - K = number of stabilizer generators
D = target distance
t = current episode step
T_max = max episode steps
```

---

## 3. Graph Construction

### 3.1 Node types

There are two node types in the graph:

```text
Qubit nodes:
    q_0, q_1, ..., q_{N-1}

Stabilizer nodes:
    s_0, s_1, ..., s_{S-1}
```

Total number of nodes:

$$
|V| = N + S = N + (N-K)
$$

---

## 4. Node Features

All node features are first passed through a shared input projection:

$$
h_v^{(0)} = \text{MLP}_{\text{node\_embed}}(x_v)
$$

or simply use:

$$
h_v^{(0)} = \text{Dense}(d_{hidden})(x_v)
$$

---

### 4.1 Qubit node feature

For a qubit node (q_i), define:

```text
x_qi = [
    is_qubit,
    is_stabilizer,
    normalized_check_degree,
    normalized_hw_degree,
    normalized_x_count,
    normalized_z_count
]
```

where:

```text
is_qubit = 1
is_stabilizer = 0

check_degree_i = number of stabilizer generators touching q_i
hw_degree_i = hardware graph degree of q_i

x_count_i = number of stabilizer rows with X on q_i
z_count_i = number of stabilizer rows with Z on q_i
```

Normalization：

$$
\text{normalized\_check\_degree}_i = \frac{\text{check\_degree}_i}{S}
$$

$$
\text{normalized\_hw\_degree}_i = \frac{\text{hw\_degree}_i}{N-1}
$$

$$
\text{normalized\_x\_count}_i = \frac{x\_count_i}{S}
$$

$$
\text{normalized\_z\_count}_i = \frac{z\_count_i}{S}
$$

---

### 4.2 Stabilizer node feature

For a stabilizer node (s_a), define:

```text
x_sa = [
    is_qubit,
    is_stabilizer,
    normalized_weight,
    normalized_x_weight,
    normalized_z_weight,
    0.0
]
```

where:

```text
is_qubit = 0
is_stabilizer = 1

weight_a = number of qubits touched by stabilizer s_a
x_weight_a = number of X entries in stabilizer s_a
z_weight_a = number of Z entries in stabilizer s_a
```

Normalization：

$$
\text{normalized\_weight}_a = \frac{\text{weight}_a}{N}
$$

$$
\text{normalized\_x\_weight}_a = \frac{x\_weight_a}{N}
$$

$$
\text{normalized\_z\_weight}_a = \frac{z\_weight_a}{N}
$$

Append a final `0.0` so that the qubit-node feature and stabilizer-node feature have the same dimensionality.

---

## 5. Edge Construction

This version uses three directed relations:

```text
relation 0: CHECK_S_TO_Q
relation 1: CHECK_Q_TO_S
relation 2: HW_Q_TO_Q
```

All edge messages share the same `MLP_edge`.
Differences between edge types are represented by `relation_embedding`.

---

### 5.1 Check edges

For each stabilizer (s_a) and qubit (q_i), if:

$$
H_X[a,i] = 1
$$

or:

$$
H_Z[a,i] = 1
$$

then create two directed check edges:

```text
s_a -> q_i
q_i -> s_a
```

The edge feature is simplified to:

```text
edge_feature = [x_bit, z_bit]
```

where:

```text
x_bit = H_X[a,i]
z_bit = H_Z[a,i]
```

So:

```text
X: [1, 0]
Z: [0, 1]
Y: [1, 1]
```

---

### 5.2 Hardware edges

For each directed edge in the hardware connectivity graph:

```text
q_i -> q_j
```

create one hardware edge.

The hardware-edge feature in this version is fixed as:

```text
edge_feature = [0, 0]
```

The hardware-edge information is represented only through the relation id:

```text
relation_id = HW_Q_TO_Q
```

This version does not add gate error rates, gate durations, distances, coupling strengths, or other hardware features.

---

## 6. Global Feature

This version does not use noise-aware features.
The global feature is defined as:

```text
global_feature = [
    t / T_max,
    K / N,
    D / N
]
```

where:

```text
t / T_max = current step ratio
K / N = code rate
D / N = normalized target distance
```

It does not include:

```text
noise bias
p_X, p_Y, p_Z
hardware noise parameters
absolute N embedding
```

The global feature is first projected to the hidden dimension:

$$
g^{(0)} = \text{MLP}_{\text{global\_embed}}(\text{global\_feature})
$$

---

## 7. GNN Layer

This version uses an (L)-layer shared-message GNN.
Each layer contains:

```text
edge message
mean aggregation
node update with residual connection
global update with residual connection
```

---

### 7.1 Relation embedding

Each edge has a relation id:

```text
CHECK_S_TO_Q = 0
CHECK_Q_TO_S = 1
HW_Q_TO_Q    = 2
```

Use an embedding table:

$$
r_{uv} = \text{Embed}(\text{relation\_id}_{uv})
$$

where:

```text
r_uv shape = relation_dim
```

---

### 7.2 Edge message

For each directed edge (u -> v), the message is defined as:

$$
m_{u \to v}^{(\ell)}
=
\text{MLP}_{\text{edge}}
\left(
[
h_u^{(\ell)},
h_v^{(\ell)},
e_{uv},
r_{uv},
g^{(\ell)}
]
\right)
$$

where:

```text
h_u^(l) = sender node embedding
h_v^(l) = receiver node embedding
e_uv = simplified edge feature
r_uv = relation embedding
g^(l) = global embedding
```

All relations share the same `MLP_edge`.

---

### 7.3 Mean aggregation

For each node (v), collect all incoming messages:

$$
\mathcal{M}_v^{(\ell)}
=
\{m_{u \to v}^{(\ell)} : u \in \mathcal{N}_{\text{in}}(v)\}
$$

Use mean aggregation:

$$
M_v^{(\ell)}
=
\operatorname{mean}_{u \in \mathcal{N}_{\text{in}}(v)}
m_{u \to v}^{(\ell)}
$$

If a node has no incoming messages, then:

$$
M_v^{(\ell)} = 0
$$

---

### 7.4 Node update with residual connection

The node update is defined as:

$$
\Delta h_v^{(\ell)}
=
\text{MLP}_{\text{node}}
\left(
[
h_v^{(\ell)},
M_v^{(\ell)},
g^{(\ell)}
]
\right)
$$

Use a residual connection:

$$
h_v^{(\ell+1)}
=
h_v^{(\ell)}
+
\Delta h_v^{(\ell)}
$$

---

### 7.5 Global mean pooling

Mean-pool the qubit nodes and stabilizer nodes separately:

$$
\bar{h}_Q^{(\ell+1)}
=
\operatorname{mean}_{i=0}^{N-1}
h_{q_i}^{(\ell+1)}
$$

$$
\bar{h}_S^{(\ell+1)}
=
\operatorname{mean}_{a=0}^{S-1}
h_{s_a}^{(\ell+1)}
$$

---

### 7.6 Global update with residual connection

The global update is defined as:

$$
\Delta g^{(\ell)}
=
\text{MLP}_{\text{global}}
\left(
[
g^{(\ell)},
\bar{h}_Q^{(\ell+1)},
\bar{h}_S^{(\ell+1)}
]
\right)
$$

Use a residual connection:

$$
g^{(\ell+1)}
=
g^{(\ell)}
+
\Delta g^{(\ell)}
$$

---

## 8. GNN Encoder Output

After (L) GNN layers, obtain:

```text
h_qi = final embedding of qubit q_i
h_sa = final embedding of stabilizer s_a
g = final global embedding
```

The actor mainly uses:

```text
qubit embeddings h_qi
global embedding g
```

The critic uses only:

```text
global embedding g
```

---

## 9. Actor Head

The actor does not use a fixed-size `Dense(action_dim)`.
Instead, it uses shared candidate-action scorers.

---

### 9.1 Candidate action set

For the current state, build candidate actions:

```text
Single-qubit gates:
    gate(i) for every i in 0 ... N-1

Two-qubit gates:
    gate(i, j) for every valid directed hardware edge i -> j
```

For example, if the gate set is:

```text
single-qubit gates = {H, S}
two-qubit gates = {CNOT}
```

then the candidate actions are:

```text
H(i) for all qubits i
S(i) for all qubits i
CNOT(i,j) for all valid hardware edges i->j
```

---

### 9.2 Gate embedding

Each gate type has a trainable gate embedding:

$$
e_{gate} = \text{Embed}(\text{gate\_id})
$$

For example:

```text
H    -> gate_id 0
S    -> gate_id 1
CNOT -> gate_id 2
```

---

### 9.3 Single-qubit gate logit

For a single-qubit action (gate(i)), the logit is defined as:

$$
\text{logit}(gate, i)
=
\text{MLP}_{\text{single}}
\left(
[
h_{q_i},
e_{gate},
g
]
\right)
$$

where:

```text
h_qi = qubit i embedding
e_gate = gate type embedding
g = global embedding
```

---

### 9.4 Two-qubit gate logit

For a two-qubit action (gate(i,j)), the logit is defined as:

$$
\text{logit}(gate, i, j)
=
\text{MLP}_{\text{two}}
\left(
[
h_{q_i},
h_{q_j},
e_{gate},
g
]
\right)
$$

This version does not add hardware-edge features.
Hardware constraints are controlled only through the candidate action set: only valid hardware edges (i -> j) are listed as candidate actions.

For CNOT, the order must be preserved:

```text
CNOT(i,j) = control i, target j
CNOT(i,j) ≠ CNOT(j,i)
```

So the input order of `MLP_two` is fixed as:

```text
[h_control, h_target, e_CNOT, global]
```

---

### 9.5 Policy distribution

Concatenate all candidate-action logits:

$$
\ell =
[
\ell_{H(0)}, \ldots,
\ell_{H(N-1)},
\ell_{S(0)}, \ldots,
\ell_{S(N-1)},
\ell_{CNOT(i,j)}, \ldots
]
$$

The policy is:

$$
\pi(a \mid s)
=
\text{Categorical}(\text{logits}=\ell)
$$

If padding to a fixed `A_max` is needed in implementation, use an action mask:

```python
masked_logits = jnp.where(action_mask, logits, -1e9)
pi = distrax.Categorical(logits=masked_logits)
```

---

## 10. Critic Head

The critic output uses the final global embedding directly:

$$
V(s) = \text{MLP}_{value}(g)
$$

Do not concatenate the qubit mean pool or stabilizer mean pool, because the global embedding already integrates full-graph information through mean pooling and residual updates at each layer.

The critic output is a scalar:

```text
value shape = ()
```

or in batch form:

```text
value shape = (batch_size,)
```

---

## 11. Model Forward Pass

Full forward pass:

```text
Input:
    check matrix H = [H_X | H_Z]
    hardware directed edges
    current step t
    N, K, D, T_max

Build graph:
    qubit nodes
    stabilizer nodes
    check edges with edge_feature=[x_bit,z_bit]
    hardware edges with edge_feature=[0,0]
    relation ids

Initial embeddings:
    h_nodes = MLP_node_embed(node_features)
    g = MLP_global_embed([t/T_max, K/N, D/N])

For l = 0 ... L-1:
    relation_emb = Embed(relation_ids)
    messages = MLP_edge([h_sender, h_receiver, edge_feature, relation_emb, g])
    aggregated_messages = mean messages by receiver
    h_nodes = h_nodes + MLP_node([h_nodes, aggregated_messages, g])
    q_pool = mean qubit node embeddings
    s_pool = mean stabilizer node embeddings
    g = g + MLP_global([g, q_pool, s_pool])

Actor:
    for each single-qubit candidate gate(i):
        logit = MLP_single([h_qi, gate_emb, g])

    for each two-qubit candidate gate(i,j):
        logit = MLP_two([h_qi, h_qj, gate_emb, g])

    pi = Categorical(masked_logits)

Critic:
    value = MLP_value(g)

Return:
    pi, value
```

---

## 12. Pseudo-code

```python
class GNNQDXActorCritic(nn.Module):
    hidden_dim: int
    relation_dim: int
    gate_dim: int
    num_gnn_layers: int
    num_relations: int = 3
    num_gate_types: int = 3

    @nn.compact
    def __call__(self, graph_obs):
        nodes = graph_obs.nodes
        edges = graph_obs.edges
        senders = graph_obs.senders
        receivers = graph_obs.receivers
        relation_ids = graph_obs.relation_ids

        qubit_mask = graph_obs.qubit_mask
        stabilizer_mask = graph_obs.stabilizer_mask

        global_features = graph_obs.global_features

        single_actions = graph_obs.single_actions
        two_actions = graph_obs.two_actions
        action_mask = graph_obs.action_mask

        h = MLP_node_embed(nodes)
        g = MLP_global_embed(global_features)

        relation_embedder = nn.Embed(
            num_embeddings=self.num_relations,
            features=self.relation_dim,
        )

        gate_embedder = nn.Embed(
            num_embeddings=self.num_gate_types,
            features=self.gate_dim,
        )

        for _ in range(self.num_gnn_layers):
            r = relation_embedder(relation_ids)

            h_sender = h[senders]
            h_receiver = h[receivers]

            g_edge = repeat_global_to_edges(g, edges.shape[0])

            edge_input = concat([
                h_sender,
                h_receiver,
                edges,
                r,
                g_edge,
            ])

            messages = MLP_edge(edge_input)

            aggregated = segment_mean(
                messages,
                receivers,
                num_segments=h.shape[0],
            )

            g_node = repeat_global_to_nodes(g, h.shape[0])

            node_input = concat([
                h,
                aggregated,
                g_node,
            ])

            h = h + MLP_node(node_input)

            q_pool = masked_mean(h, qubit_mask)
            s_pool = masked_mean(h, stabilizer_mask)

            global_input = concat([
                g,
                q_pool,
                s_pool,
            ])

            g = g + MLP_global(global_input)

        qubit_embeddings = h[qubit_mask]

        logits = []

        for action in single_actions:
            gate_id = action.gate_id
            i = action.qubit_index

            gate_emb = gate_embedder(gate_id)

            z = concat([
                qubit_embeddings[i],
                gate_emb,
                g,
            ])

            logit = MLP_single(z)
            logits.append(logit)

        for action in two_actions:
            gate_id = action.gate_id
            i = action.control_or_first
            j = action.target_or_second

            gate_emb = gate_embedder(gate_id)

            z = concat([
                qubit_embeddings[i],
                qubit_embeddings[j],
                gate_emb,
                g,
            ])

            logit = MLP_two(z)
            logits.append(logit)

        logits = stack(logits)

        masked_logits = where(action_mask, logits, -1e9)
        pi = Categorical(logits=masked_logits)

        value = MLP_value(g)

        return pi, value
```

---

## 13. Implementation Requirements

### 13.1 Required masks

To support padding/batching, the following masks are required:

```text
node_mask
edge_mask
qubit_mask
stabilizer_mask
action_mask
```

Purpose:

```text
node_mask:
    exclude padded nodes

edge_mask:
    exclude padded edges

qubit_mask:
    mean pool qubit nodes

stabilizer_mask:
    mean pool stabilizer nodes

action_mask:
    exclude padded or invalid actions
```

---

### 13.2 Padding strategy

For JAX/JIT convenience, bucket padding can be used:

```text
Bucket 1: N = 5, 6, 7      pad to N_max = 7
Bucket 2: N = 8, 9, 10     pad to N_max = 10
Bucket 3: N = 11, 12       pad to N_max = 12
```

Each bucket has fixed:

```text
max_num_nodes
max_num_edges
max_num_actions
```

However, the model parameters do not depend on the bucket size; the bucket size is only for JIT static shapes.

---

### 13.3 Action mapping

Each action index must be reversible back to a QDX environment gate operation:

```text
action_idx -> action descriptor -> Clifford gate matrix -> update tableau
```

Action descriptor format:

```text
Single-qubit:
    {
        "type": "single",
        "gate": "H",
        "qubit": i
    }

Two-qubit:
    {
        "type": "two",
        "gate": "CNOT",
        "control": i,
        "target": j
    }
```

---

## 14. Training Setup

### 14.1 Training tasks

For the first version, recommended:

```text
train N = 5, 6, 7
validation N = 8
test N = 9, 10
```

This can be used to test Level 4 extrapolation.

---

### 14.2 PPO compatibility

The GNN actor-critic still returns:

```text
pi = action distribution
value = scalar value estimate
```

So the PPO loss can reuse the original QDX actor-critic training structure.

The main changes are:

```text
1. observation construction
2. policy network
3. action logits generation
4. action index to gate mapping
5. padding and masks
```

The reward, KL condition, and environment transition can initially remain the same as the original QDX.

---

## 15. Non-goals for v1

GNN-QDX v1 does not do the following:

```text
1. Do not use noise-aware input
2. Do not add noise bias / Pauli error probabilities
3. Do not add hardware-edge features in the two-qubit actor head
4. Do not use a virtual global node
5. Do not use learned absolute qubit embeddings
6. Do not use a fixed Dense(action_dim) actor output
7. Do not change the original KL reward
8. Do not change the original environment tableau update rule
```

---

## 16. Expected Advantages

Compared with the original fixed-size MLP, GNN-QDX v1 has the following advantages:

```text
1. It can accept stabilizer code graphs with different N
2. It uses shared message passing and does not depend on a fixed input size
3. It uses candidate-action scoring and does not depend on a fixed output size
4. It can naturally adapt to different hardware connectivity
5. It has a chance to achieve train on small N, test on larger N
6. Mean pooling improves embedding scale stability across N
7. Residual connections improve PPO + GNN training stability
```

---

## 17. Summary

The final architecture of GNN-QDX v1 is:

```text
check matrix H + hardware graph
        ->
qubit/stabilizer heterogeneous graph
        ->
simplified node features
        ->
simplified edge features:
    check edge: [x_bit, z_bit]
    hardware edge: [0, 0]
        ->
global feature:
    [t/T_max, K/N, D/N]
        ->
L-layer residual GNN:
    message = MLP_edge([h_sender, h_receiver, edge_feature, relation_embedding, global])
    node aggregation = mean
    h_next = h + MLP_node([h, mean_message, global])
    q_pool = mean(qubit nodes)
    s_pool = mean(stabilizer nodes)
    global_next = global + MLP_global([global, q_pool, s_pool])
        ->
actor:
    single logit = MLP_single([h_qi, gate_emb, global])
    two logit = MLP_two([h_qi, h_qj, gate_emb, global])
        ->
masked categorical policy
        ->
critic:
    value = MLP_value(global)
```

This version is the smallest but complete Level 4 GNN policy design.
