# GNN-QDX v1 SPEC：Variable-Size GNN Policy for QDX Code Discovery

## 0. 目標

本 SPEC 定義一個用於 QDX code discovery 的 variable-size GNN actor-critic policy。目標是取代原本 fixed-size MLP policy，使同一組模型參數可以處理不同 qubit 數量 (N)，並具備 train on small (N)、test on larger (N) 的 Level 4 擴展能力。

本版本先不處理 noise-aware setting，專注於 stabilizer check matrix、hardware connectivity、gate action scoring 與 PPO actor-critic 訓練。

---

## 1. 設計原則

### 1.1 不使用 fixed input vector

原本 QDX 使用：

$$
\text{flatten}(H) \in {0,1}^{2N(N-K)}
$$

作為 MLP input。這會讓模型 input dimension 直接依賴 (N)。

GNN-QDX v1 改成 graph representation：

```text
stabilizer check matrix + hardware connectivity
        ↓
heterogeneous graph
        ↓
shared GNN message passing
```

模型不再依賴固定 (2N(N-K)) input size。

---

### 1.2 不使用 fixed action output

原本 actor 最後一層是：

$$
\text{Dense}(n_A)
$$

其中 (n_A) 是目前 (N) 下所有 gate actions 的數量。這會讓 actor output dimension 依賴 (N)。

GNN-QDX v1 改成 candidate-action scoring：

```text
for each valid candidate action:
    logit = shared_action_scorer(action, graph_embedding)
```

所以不同 (N) 可以有不同數量的 candidate actions，但 scorer 的參數是共享的。

---

### 1.3 不使用 learned absolute qubit embedding

不使用：

```python
embedding[q_index]
```

避免模型記住 qubit index。
模型應學習 qubit 的 role、local stabilizer pattern、hardware degree 和 global code state。

---

### 1.4 不使用 virtual global node

本版本不加入 virtual global node。
所有 global information 都透過 global feature (g) 傳入 edge message、node update、global update、actor head 和 critic head。

---

### 1.5 所有 pooling 使用 mean

所有 aggregation / pooling 統一使用 mean：

```text
edge messages → node: mean
qubit nodes → global: mean
stabilizer nodes → global: mean
critic/global pooling: mean
```

這可以降低 embedding magnitude 隨 (N) 增大而失控的風險。

---

## 2. 輸入資料

每個 environment state 轉換成一個 graph observation。

### 2.1 原始 state

輸入來自目前 tableau / stabilizer check matrix：

$$
H = [H_X \mid H_Z]
$$

其中：

```text
H_X shape = (N-K, N)
H_Z shape = (N-K, N)
H shape   = (N-K, 2N)
```

令：

```text
N = physical qubit 數量
K = logical qubit 數量
S = N - K = stabilizer generator 數量
D = target distance
t = current episode step
T_max = max episode steps
```

---

## 3. Graph Construction

### 3.1 Node types

Graph 中有兩種 node：

```text
Qubit nodes:
    q_0, q_1, ..., q_{N-1}

Stabilizer nodes:
    s_0, s_1, ..., s_{S-1}
```

總 node 數：

$$
|V| = N + S = N + (N-K)
$$

---

## 4. Node Features

所有 node feature 最後會先經過 shared input projection：

$$
h_v^{(0)} = \text{MLP}_{\text{node\_embed}}(x_v)
$$

或簡單使用：

$$
h_v^{(0)} = \text{Dense}(d_{hidden})(x_v)
$$

---

### 4.1 Qubit node feature

對 qubit node (q_i)，定義：

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

其中：

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

對 stabilizer node (s_a)，定義：

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

其中：

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

最後補一個 `0.0`，讓 qubit node feature 和 stabilizer node feature 維度一致。

---

## 5. Edge Construction

本版本使用三種 directed relation：

```text
relation 0: CHECK_S_TO_Q
relation 1: CHECK_Q_TO_S
relation 2: HW_Q_TO_Q
```

所有 edge message 共用同一個 `MLP_edge`。
edge 類型差異由 `relation_embedding` 表示。

---

### 5.1 Check edges

對每個 stabilizer (s_a) 和 qubit (q_i)，若：

$$
H_X[a,i] = 1
$$

或：

$$
H_Z[a,i] = 1
$$

則建立兩條 directed check edges：

```text
s_a → q_i
q_i → s_a
```

edge feature 簡化為：

```text
edge_feature = [x_bit, z_bit]
```

其中：

```text
x_bit = H_X[a,i]
z_bit = H_Z[a,i]
```

所以：

```text
X: [1, 0]
Z: [0, 1]
Y: [1, 1]
```

---

### 5.2 Hardware edges

對 hardware connectivity graph 中每條 directed edge：

```text
q_i → q_j
```

建立一條 hardware edge。

本版本的 hardware edge feature 固定為：

```text
edge_feature = [0, 0]
```

hardware edge 的資訊只透過 relation id 表示：

```text
relation_id = HW_Q_TO_Q
```

本版本不加入 gate error rate、gate duration、distance、coupling strength 等 hardware feature。

---

## 6. Global Feature

本版本不使用 noise-aware feature。
global feature 定義為：

```text
global_feature = [
    t / T_max,
    K / N,
    D / N
]
```

其中：

```text
t / T_max = current step ratio
K / N = code rate
D / N = normalized target distance
```

不包含：

```text
noise bias
p_X, p_Y, p_Z
hardware noise parameters
absolute N embedding
```

global feature 會先投影成 hidden dimension：

$$
g^{(0)} = \text{MLP}_{\text{global\_embed}}(\text{global\_feature})
$$

---

## 7. GNN Layer

本版本使用 (L) 層 shared-message GNN。
每一層包含：

```text
edge message
mean aggregation
node update with residual connection
global update with residual connection
```

---

### 7.1 Relation embedding

每條 edge 有一個 relation id：

```text
CHECK_S_TO_Q = 0
CHECK_Q_TO_S = 1
HW_Q_TO_Q    = 2
```

使用 embedding table：

$$
r_{uv} = \text{Embed}(\text{relation\_id}_{uv})
$$

其中：

```text
r_uv shape = relation_dim
```

---

### 7.2 Edge message

對每條 directed edge (u \to v)，message 定義為：

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

其中：

```text
h_u^(l) = sender node embedding
h_v^(l) = receiver node embedding
e_uv = simplified edge feature
r_uv = relation embedding
g^(l) = global embedding
```

所有 relation 共用同一個 `MLP_edge`。

---

### 7.3 Mean aggregation

對每個 node (v)，收集所有 incoming messages：

$$
\mathcal{M}_v^{(\ell)}
=
\{m_{u \to v}^{(\ell)} : u \in \mathcal{N}_{\text{in}}(v)\}
$$

使用 mean aggregation：

$$
M_v^{(\ell)}
=
\operatorname{mean}_{u \in \mathcal{N}_{\text{in}}(v)}
m_{u \to v}^{(\ell)}
$$

若 node 沒有 incoming message，則：

$$
M_v^{(\ell)} = 0
$$

---

### 7.4 Node update with residual connection

node update 定義為：

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

使用 residual connection：

$$
h_v^{(\ell+1)}
=
h_v^{(\ell)}
+
\Delta h_v^{(\ell)}
$$

---

### 7.5 Global mean pooling

將 qubit nodes 和 stabilizer nodes 分開 mean pool：

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

global update 定義為：

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

使用 residual connection：

$$
g^{(\ell+1)}
=
g^{(\ell)}
+
\Delta g^{(\ell)}
$$

---

## 8. GNN Encoder Output

經過 (L) 層 GNN 後，得到：

```text
h_qi = final embedding of qubit q_i
h_sa = final embedding of stabilizer s_a
g = final global embedding
```

其中 actor 主要使用：

```text
qubit embeddings h_qi
global embedding g
```

critic 只使用：

```text
global embedding g
```

---

## 9. Actor Head

actor 不使用 fixed-size `Dense(action_dim)`。
actor 使用 shared candidate-action scorers。

---

### 9.1 Candidate action set

對目前 state，建立 candidate actions：

```text
Single-qubit gates:
    gate(i) for every i in 0 ... N-1

Two-qubit gates:
    gate(i, j) for every valid directed hardware edge i → j
```

例如 gate set 是：

```text
single-qubit gates = {H, S}
two-qubit gates = {CNOT}
```

則 candidate actions 是：

```text
H(i) for all qubits i
S(i) for all qubits i
CNOT(i,j) for all valid hardware edges i→j
```

---

### 9.2 Gate embedding

每種 gate type 有一個 trainable gate embedding：

$$
e_{gate} = \text{Embed}(\text{gate\_id})
$$

例如：

```text
H    → gate_id 0
S    → gate_id 1
CNOT → gate_id 2
```

---

### 9.3 Single-qubit gate logit

對 single-qubit action (gate(i))，logit 定義為：

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

其中：

```text
h_qi = qubit i embedding
e_gate = gate type embedding
g = global embedding
```

---

### 9.4 Two-qubit gate logit

對 two-qubit action (gate(i,j))，logit 定義為：

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

本版本不加入 hardware edge feature。
hardware constraint 只透過 candidate action set 控制：只有 valid hardware edge (i \to j) 才會被列為 candidate action。

對 CNOT 而言，順序必須保留：

```text
CNOT(i,j) = control i, target j
CNOT(i,j) ≠ CNOT(j,i)
```

所以 `MLP_two` 的輸入順序固定為：

```text
[h_control, h_target, e_CNOT, global]
```

---

### 9.5 Policy distribution

將所有 candidate action logits concatenate：

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

policy 為：

$$
\pi(a \mid s)
=
\text{Categorical}(\text{logits}=\ell)
$$

實作上如果需要 padding 到固定 `A_max`，則使用 action mask：

```python
masked_logits = jnp.where(action_mask, logits, -1e9)
pi = distrax.Categorical(logits=masked_logits)
```

---

## 10. Critic Head

critic output 直接使用 final global embedding：

$$
V(s) = \text{MLP}_{value}(g)
$$

不 concat qubit mean pool，也不 concat stabilizer mean pool，因為 global embedding 在每一層已經透過 mean pooling 和 residual update 整合全圖資訊。

critic output 是 scalar：

```text
value shape = ()
```

或 batch version：

```text
value shape = (batch_size,)
```

---

## 11. Model Forward Pass

完整 forward pass：

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

為了支援 padding / batching，需要以下 masks：

```text
node_mask
edge_mask
qubit_mask
stabilizer_mask
action_mask
```

用途：

```text
node_mask:
    排除 padded nodes

edge_mask:
    排除 padded edges

qubit_mask:
    mean pool qubit nodes

stabilizer_mask:
    mean pool stabilizer nodes

action_mask:
    排除 padded actions 或 invalid actions
```

---

### 13.2 Padding strategy

為了 JAX/JIT 方便，可以使用 bucket padding：

```text
Bucket 1: N = 5, 6, 7      pad to N_max = 7
Bucket 2: N = 8, 9, 10     pad to N_max = 10
Bucket 3: N = 11, 12       pad to N_max = 12
```

每個 bucket 內固定：

```text
max_num_nodes
max_num_edges
max_num_actions
```

但是模型參數不依賴 bucket size；bucket size 只是為了 JIT static shape。

---

### 13.3 Action mapping

每個 action index 必須能轉回 QDX environment 的 gate operation：

```text
action_idx → action descriptor → Clifford gate matrix → update tableau
```

action descriptor 格式：

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

第一版建議：

```text
train N = 5, 6, 7
validation N = 8
test N = 9, 10
```

這樣可以測試 Level 4 extrapolation。

---

### 14.2 PPO compatibility

GNN actor-critic 仍然回傳：

```text
pi = action distribution
value = scalar value estimate
```

所以 PPO loss 可以沿用原本 QDX 的 actor-critic training structure。

需要改的主要是：

```text
1. observation construction
2. policy network
3. action logits generation
4. action index to gate mapping
5. padding and masks
```

reward、KL condition、environment transition 可以先沿用原本 QDX。

---

## 15. Non-goals for v1

GNN-QDX v1 不做以下事情：

```text
1. 不使用 noise-aware input
2. 不加入 noise bias / Pauli error probabilities
3. 不在 two-qubit actor head 加 hardware edge feature
4. 不使用 virtual global node
5. 不使用 learned absolute qubit embedding
6. 不使用 fixed Dense(action_dim) actor output
7. 不更改原本 KL reward
8. 不更改原本 environment 的 tableau update rule
```

---

## 16. Expected Advantages

相對於原本 fixed-size MLP，GNN-QDX v1 的優點是：

```text
1. 可以接受不同 N 的 stabilizer code graph
2. 使用 shared message passing，不依賴固定 input size
3. 使用 candidate-action scoring，不依賴固定 output size
4. 可以自然套用不同 hardware connectivity
5. 有機會做到 train on small N, test on larger N
6. mean pooling 提高跨 N 的 embedding scale stability
7. residual connection 提高 PPO + GNN 訓練穩定性
```

---

## 17. Summary

GNN-QDX v1 的最終架構是：

```text
check matrix H + hardware graph
        ↓
qubit/stabilizer heterogeneous graph
        ↓
simplified node features
        ↓
simplified edge features:
    check edge: [x_bit, z_bit]
    hardware edge: [0, 0]
        ↓
global feature:
    [t/T_max, K/N, D/N]
        ↓
L-layer residual GNN:
    message = MLP_edge([h_sender, h_receiver, edge_feature, relation_embedding, global])
    node aggregation = mean
    h_next = h + MLP_node([h, mean_message, global])
    q_pool = mean(qubit nodes)
    s_pool = mean(stabilizer nodes)
    global_next = global + MLP_global([global, q_pool, s_pool])
        ↓
actor:
    single logit = MLP_single([h_qi, gate_emb, global])
    two logit = MLP_two([h_qi, h_qj, gate_emb, global])
        ↓
masked categorical policy
        ↓
critic:
    value = MLP_value(global)
```

這版是最小但完整的 Level 4 GNN policy 設計。
