# Phase 7 Generalization and Reasoning Workspace Contract

Phase 7 turns the current limitations into measured mechanisms:
better open generation, self-tuning sparse dynamics, broader data coverage,
event-driven state updates, optional offline semantic priors, heldout
generalization, and long-range reasoning beyond key-value recall.

The strict path remains CPU-only, local, online, sparse, and free of backprop,
optimizer calls, GPU assumptions, deep learning frameworks, public attention
mechanisms, global hidden error paths, and full-history retrieval scans.

## Required Mechanisms

1. Homeostasis: one default controller must keep sparsity, fatigue, decay,
   suppression threshold, and local update rate stable across stories,
   WikiText-like text, code, math, and dialogue without benchmark-specific
   retuning.
2. Generation quality v2: the generation controller must include local discourse
   planning, repetition fatigue, entity continuity, contradiction penalty, and
   style/length control. It must pass score ratio `>=85%` of the simple
   baseline, loop rate `<=3%`, UNK rate `<=0.3%`, and entity consistency
   `>=85%`.
3. Reasoning workspace: sparse local variables, bindings, intermediate steps,
   and subgoals must support bAbI-style QA, role swaps, multi-hop reasoning,
   math traces, and stack/program traces.
4. Entity-event graph: hierarchical event memory must answer multi-evidence
   questions at `100k` and experimental `1M` event scale without scanning full
   history. Required accuracy is `>=75%` at `100k` and `>=60%` at `1M`, with
   latency max/min `<=2x`.
5. Event-driven SSM: sparse state updates must touch only active event/state
   dimensions, keep ops fraction `<=20%`, preserve quality within `5%`, and
   beat the dense state bookkeeping baseline by `>=2x`.
6. Optional prior track: offline semantic priors may improve quality, but they
   are reported separately from strict-zero. No external service may be called
   inside `forward()` or `observe()`.
7. Generalization: heldout compositional splits, OOD symbols, and adversarial
   role swaps must pass without answer leakage or full-context scan.

## Canonical Commands

```powershell
python benchmarks/phase7/run_phase7.py --profile quick
python benchmarks/phase7/run_phase7.py --profile claim
python benchmarks/phase7/run_phase7.py --profile full
```

Every Phase 7 benchmark accepts `--json-output` and returns non-zero when a
strict metric fails.

## Current Claim Boundary

Passing Phase 7 supports the claim that the project now has concrete mechanisms
and benchmark coverage for the major limitations identified after Phase 6. It
still does not claim parity with modern LLMs on broad open-ended human
preference, deep coding, large-scale math, or frontier instruction following.
