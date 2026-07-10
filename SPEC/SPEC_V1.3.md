# GNN-QDX v1.3 SPEC: Check/Hardware Edge-Separated Aggregation

## Goal

Separate check-edge and hardware-edge messages before pooling, so node and global updates can distinguish stabilizer-structure information from hardware/action-compatibility information.

This version assumes v1.2 already provides hardware edge features and an edge-aware two-qubit action head.

---

## 1. Edge Groups

Use two edge groups:

```text
check edges:
  CHECK_S_TO_Q
  CHECK_Q_TO_S

hardware edges:
  HW_Q_TO_Q
```

Do not change relation IDs.

---

## 2. Node Message Aggregation

Replace the current single incoming-edge mean:

```text
agg_all = mean(all valid incoming edge messages)
```

with two separate masked means:

```text
agg_check = mean(valid incoming check-edge messages)
agg_hw    = mean(valid incoming hardware-edge messages)
```

Node update input becomes:

```text
node_delta = MLP_node([
    h,
    agg_check,
    agg_hw,
    g_node
])
```

For nodes with no valid incoming edge of a group, use zero aggregation for that group.

---

## 3. Global Edge Pooling

Add edge-level pooling to the global update.

Compute:

```text
check_edge_pool = mean(edge_h for valid check edges)
hw_edge_pool    = mean(edge_h for valid hardware edges)
```

Global update input becomes:

```text
global_delta = MLP_global([
    g,
    q_pool,
    s_pool,
    check_edge_pool,
    hw_edge_pool
])
```

---

## 4. Actor/Critic

Keep the v1.2 actor design:

```text
two-qubit action head uses action_edge_h
```

Critic remains:

```text
value = MLP_value(g)
```

unless separately modified.

---

## 5. Expected Benefit

v1.3 prevents stabilizer-check messages and hardware/action messages from being mixed too early.

```text
check aggregation:
  code/stabilizer structure

hardware aggregation:
  two-qubit action and hardware-pair context
```

This is especially useful after v1.2, where hardware edges carry state-dependent pair features.
