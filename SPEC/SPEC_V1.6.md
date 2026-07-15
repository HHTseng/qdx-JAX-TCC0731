# Reward Update SPEC

## Objective

Update the environment reward to combine the existing physical-error reward, distance-search progress, and a terminal success bonus.

## Reward

[
r_t
===

r_{\text{physical},t}
+
0.1(P_t-P_{t-1})
+
\mathbf{1}[\text{success}]
]

Where:

* (r_{\text{physical},t}): existing physical-error reward.
* (P_t-P_{t-1}): normalized progress made in the current step.
* (\mathbf{1}[\text{success}]): equals 1 when the target distance is reached, otherwise 0.
* The episode terminates immediately after success.

## Progress Score

Let:

* (D): target distance.
* (d_t): current exact distance, capped at (D).
* (c_{d_t}): number of logical violations at weight (d_t).
* (M_{d_t}): total number of Pauli errors at weight (d_t).

Compute the frontier completion score:

[
A_t
===

1-
\frac{\log(1+c_{d_t})}
{\log(1+M_{d_t})}
]

Then compute:

[
P_t=
\begin{cases}
1, & \text{if success}[4pt]
\dfrac{d_t-1+A_t}{D-1}, & \text{otherwise}
\end{cases}
]

## Implementation Requirements

* Use the exact GF(2) verifier to compute (d_t), (c_{d_t}), and success.
* Store (P_t) in the environment state for use as (P_{t-1}) in the next step.
* Initialize (P_0) from the initial circuit state.
* Apply the success bonus only once.
* Set `done = True` immediately when the target distance is reached.
* Do not use reward thresholds to determine success.
