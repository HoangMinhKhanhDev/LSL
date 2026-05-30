# Phase 9 Bio-Compute Closure Contract

Phase 9 closes the biological-mechanism loop across six mechanisms:
predictive coding, SDR, cortical columns, hippocampal two-speed memory,
neuromodulation, and dendritic computation.

The strict constraints remain unchanged: no backpropagation, optimizer state,
GPU requirement, deep-learning framework, attention matrix/QKV/all-pairs token
interaction, global hidden error path, full-context scan, hardcoded eval
answers, or external API inside the strict model path.

## Required Proofs

1. Predictive coding v2: local layer predictors must reduce error at least 90%
   within 10 epochs, suppress at least 95% learned signal, leave at least 80%
   learned tokens with zero update in the standalone proof, learn one-pass
   causal effects at least 10x random, complete 3-hop chains at least 70%, and
   reduce next-token loss immediately after a new fact.
2. SDR v2: virtual sparse SDR must test `d=100000,k=20` without dense
   allocation, report exact `log2(C(d,k))`, complete 70% masked patterns at
   least 80%, preserve morphology/subword structure, and report optional
   bilingual overlap separately from strict-zero claims.
3. Cortical column v2: sequence memory must recall seen sequences at 100%,
   reach at least 95% grammar accuracy after one pass, maintain 200-token topic
   coherence at least 0.5, show constant per-token latency proxy across sequence
   length, and beat dense Transformer energy proxy by at least 100x.
4. Hippocampal memory: fast sparse memory must store 10,000 facts, recall exact
   facts at 100%, recall partial cues through bounded candidates, encode only
   surprising events, and consolidate with replay budget at most 10%.
5. Neuromodulation: dopamine/acetylcholine/serotonin-style gates must place at
   least 95% updates on novel/surprising tokens, adapt tone, keep weight/sparse
   state within finite stress bounds, and select uncertain items better than a
   random pool.
6. Dendritic computation: sparse dendritic branches must perform local
   nonlinear sigmoid computation, specialize with at most 10% branch activity
   overlap, solve AND/OR coincidence detection at least 90%, solve XOR with one
   dendritic neuron, keep average active branches at most 5%, instantiate a
   1,000-branch tree, use branch-level SDR receptive fields, learn with 100%
   branch-local Hebbian updates and zero global error updates, leave at least
   90% familiar branches with zero update, and reach at least 100x dense proxy
   compute-density gain. The legacy role/context and sparse ops checks must
   also remain at least 95%, 90%, and 50x respectively.
7. Integrated proof: one `BioComputeAgent` must combine the mechanisms, and
   ablations that remove each mechanism must degrade its associated metric by
   at least 20%.
8. Mechanisms 1-5 target suite: predictive coding, SDR, cortical columns,
   hippocampal memory, and neuromodulation must pass the moonshot targets as a
   single strict benchmark, including deep suppression, 70% SDR mask recovery,
   cross-lingual overlap, grammar transfer, 10,000-fact retention, sparse
   replay, attention gating, tone adaptation, homeostasis, and curiosity.
9. Model-level LSL proof: one online `BioComputeAgent` instance must behave as
   the claim-bearing LSL language model, not only as isolated mechanism tests.
   It must learn text/facts/chains/context patterns on CPU, adapt after one new
   text observation, retain old facts after interference, generate coherent
   text, answer multi-hop chains, expose sparse/low-parameter proxies, and show
   ablation drops for predictive coding, SDR, cortical columns, hippocampus,
   dendrites, and neuromodulation.

## Canonical Commands

```powershell
python benchmarks/phase9/run_phase9.py --profile quick
python benchmarks/phase9/run_phase9.py --profile claim
python benchmarks/phase9/run_phase9.py --profile full
```

`claim` and `full` are the claim-bearing profiles. They must return non-zero on
any metric failure and must include the model-level LSL proof and strict
scanner.

## Claim Boundary

Passing Phase 9 supports the claim that the strict prototype now has executable
proofs for the six biological mechanisms plus a model-level LSL integration
proof. It does not claim LLM parity, frontier open generation, or broad public
benchmark superiority.
