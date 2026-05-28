# LSL Strict Architecture Notes

This repository now treats `GOAL.md` as the contract. The canonical proof is
`benchmark_goal_strict.py`, not the older demo scripts.

## Representation

Tokens are encoded as binary sparse distributed representations (SDRs). For
`d=1024` and `k=20`, the combinatorial capacity is above 130 bits. Semantic
structure comes from a checked-in mini embedding asset, then a fixed random
projection and top-k binarization. The projection is seed-locked and never
learned online.

## Sparse Compute

`LivingSynapseLayer.forward(..., use_sparse=True)` touches only active input
columns. It avoids full effective-matrix construction and avoids dense fatigue
outer products. The strict benchmark measures full layer forward time, including
fatigue bookkeeping, and requires both operation-count and wall-clock speedups
of at least 40x versus dense full forward.

## Predictive Coding

Hidden predictive-coding state uses local transition predictors:

```text
previous local state -> predicted current local state
current - prediction -> local prediction error
```

Updates use only the previous local state, current local state, and the local
prediction error. Output synapses receive direct output error, while hidden
layers do not receive global output-error projections.

Next-token convergence and simple relation reasoning are handled by a local
online token/relation association memory. It updates after each token from the
current token and a short rolling token window.

## Cortical Column Sequence Memory

Each token owns a column with multiple cells. Expected tokens activate predicted
cells only; surprising tokens burst and trigger local learning. Context segments
record recent token contexts locally and are used for next-step prediction and
generation. The forward path does not construct sequence-length-by-sequence-
length matrices.

## Verification Standard

The strict benchmark verifies all 18 goals and a structural scan:

- no global backward pass
- no optimizer state or optimizer calls
- no GPU or deep learning framework dependency
- no DFA feedback matrices
- no strict-path attention mechanism
- online local updates only
