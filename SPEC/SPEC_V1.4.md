# GNN-QDX v1.4 SPEC: Redundant-Action Filtering and Symmetric Two-Qubit Scoring

## 1. Objective

GNN-QDX v1.4 shall:

1. Filter algebraically redundant repeated actions using a runtime `pending_action_mask`.
2. Avoid scanning the full action history or simulating every candidate action.
3. Canonicalize symmetric two-qubit gates to a single action with `qubit_a < qubit_b`.
4. Score symmetric two-qubit gates by averaging both qubit orderings.
5. Preserve ordered control-target scoring for directional gates such as `CX`.

All redundancy rules are defined under the current phase-free binary stabilizer representation.

---

## 2. Gate Classification

### Directional two-qubit gates

```text
CX(control, target)
```

Directional gates shall preserve both hardware directions:

```text
CX(a, b)
CX(b, a)
```

Their action logits shall continue to use ordered inputs:

[
\ell_{CX(a,b)}
=

f_{\mathrm{dir}}
([h_a,h_b,h_{\mathrm{gate}},g]).
]

### Symmetric two-qubit gates

```text
CZ(a, b)
SQRT_XX(a, b)
```

Symmetric gates shall use an unordered qubit pair and only create the canonical action:

[
a<b.
]

If the hardware graph contains either `(a,b)` or `(b,a)`, the canonical symmetric action `(min(a,b), max(a,b))` shall be created once.

---

## 3. Symmetric Action Scoring

For a canonical symmetric action with (a<b), compute both ordered scores:

[
\ell_{a\rightarrow b}
=====================

f_{\mathrm{sym}}
([h_a,h_b,h_{\mathrm{gate}},g]),
]

[
\ell_{b\rightarrow a}
=====================

f_{\mathrm{sym}}
([h_b,h_a,h_{\mathrm{gate}},g]).
]

The final action logit shall be:

[
\ell_{\mathrm{sym}}(a,b)
========================

\frac{
\ell_{a\rightarrow b}+\ell_{b\rightarrow a}
}{2}.
]

Only one environment action and one final logit shall be retained for the canonical pair (a<b).

Directional gates shall not use this averaging.

---

## 4. Precomputed Action-Relation Tables

At environment or observation-builder initialization, precompute two static Boolean tables:

```python
commute_table[A, A]
cancel_table[A, A]
```

where (A) is the padded candidate-action count.

### Commutation table

[
C[i,j]=1
]

when candidate actions (i) and (j) commute under the phase-free binary Clifford representation:

[
G_iG_j=G_jG_i \pmod 2.
]

### Cancellation table

[
R[i,j]=1
]

when the two actions cancel under the same representation:

[
G_iG_j=I \pmod 2.
]

The table shall include the supported redundant repetitions:

```text
H(q)          + H(q)
S(q)          + S(q)
SQRT_X(q)     + SQRT_X(q)
CX(c,t)       + CX(c,t)
CZ(a,b)       + CZ(a,b)
SQRT_XX(a,b)  + SQRT_XX(a,b)
```

For `CX`, control and target must match exactly.

For `CZ` and `SQRT_XX`, qubit order is ignored because only the canonical pair is retained.

---

## 5. Pending Redundant-Action Mask

The environment state shall contain:

```python
pending_action_mask: bool[A]
```

`pending_action_mask[i] = True` means that candidate action (i) would cancel an earlier action after commuting through all intervening actions and must therefore be masked.

At reset:

```python
pending_action_mask = zeros(A, dtype=bool)
```

The valid action mask shall be:

```python
dynamic_action_mask = (
    base_action_mask
    & ~pending_action_mask
)
```

The existing Actor shall continue to assign invalid actions a logit of `-1e9`.

---

## 6. Pending-Mask Update

After executing candidate action `x`, update the complete pending vector without scanning action history:

```python
commutes = commute_table[:, x]
cancels = cancel_table[:, x]

new_pending = where(
    commutes,
    logical_xor(old_pending, cancels),
    False,
)
```

Equivalent mathematical form:

[
p_i'
====

\begin{cases}
p_i\oplus R[i,x], & C[i,x]=1,\
0, & C[i,x]=0.
\end{cases}
]

This preserves a pending redundant action across intervening commuting gates and clears it when a noncommuting gate blocks cancellation.

Runtime complexity shall be:

[
O(A)
]

per environment step, with no full-history scan.

---

## 7. Required Examples

### Independent single-qubit gates

```text
H1 → H2 → S3
```

The next action mask shall exclude:

```text
H1
H2
S3
```

### Noncommuting interruption

```text
H1 → CX(1,2)
```

The next `H1` action shall remain valid.

### Commuting interruption

```text
S1 → CX(1,2)
```

Because `S1` commutes with `CX(1,2)` when qubit 1 is the control, the next `S1` action shall be masked.

### Symmetric action canonicalization

Given bidirectional hardware edges:

```text
(1,2)
(2,1)
```

the candidate set shall contain:

```text
CX(1,2)
CX(2,1)
CZ(1,2)
SQRT_XX(1,2)
```

It shall not contain:

```text
CZ(2,1)
SQRT_XX(2,1)
```

---

## 8. Observation and PPO Integration

`GraphObservation.action_mask` shall contain the dynamic mask rather than only the static padding mask.

The rollout buffer shall preserve the complete observation, including the dynamic action mask, so PPO rollout and update phases evaluate the same action distribution.

No redundancy reward penalty is required for actions already removed by the pending mask.

---

## 9. Non-Goals

v1.4 shall not:

1. Scan the complete circuit history at every step.
2. Simulate every candidate action against the current check matrix.
3. Remove arbitrary repeated actions that do not algebraically cancel.
4. Treat `CX(a,b)` and `CX(b,a)` as equivalent.
5. Track stabilizer phase or full-unitary equivalence.
6. Detect longer rewrite identities beyond commuting repeated-action cancellation.

---

## 10. Expected Benefits

GNN-QDX v1.4 is expected to:

* reduce redundant exploration;
* preserve more effective search steps;
* reduce duplicated symmetric actions;
* enforce symmetric scoring for `CZ` and `SQRT_XX`;
* improve PPO sample efficiency;
* retain correct control-target directionality for `CX`;
* add only (O(A)) Boolean runtime work per step.
