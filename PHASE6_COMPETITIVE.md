# Phase 6 Competitive Evidence Contract

Phase 6 tests whether the strict local/online architecture can produce stronger
evidence for open generation, long-context world memory, public-style reasoning,
and competitive CPU efficiency.

This is not a claim of parity with frontier LLMs. It is a claim that the repo can
measure the next frontier under the same hard constraints: no backprop,
optimizer, GPU, deep learning framework, public attention mechanism, global
hidden error path, full-history retrieval scan, or benchmark answer leakage in
the strict architecture path.

## Required Mechanisms

1. Open generation: `GenerationController` must combine bounded
   long-context candidates with local discourse state, repetition fatigue,
   entity/topic continuity, and sentence-boundary preference. It must pass
   coherence `>=0.75`, loop rate `<=5%`, UNK rate `<=0.5%`, and generation
   score `>=80%` of the simple baseline on public text profiles.
2. World memory: `WorldMemory` must store chunk/evidence records, local facts,
   and citations. Retrieval must use bounded exact/sparse indexes, not scan all
   chunks. Recall/evidence faithfulness must stay high at `1k`, `16k`, and
   `128k` chunks, with latency max/min `<=1.5x` and RAM proxy at least `5x`
   better than a dense all-pairs context baseline.
3. Public-style reasoning: relation, role-binding, multi-hop, math trace, and
   stack/program trace tasks must use reusable local mechanisms with gold
   answers. Accuracy targets are `>=80%` for bAbI-style QA, `>=80%` multi-hop,
   `>=90%` role binding, and `>=60%` first-pass trace reasoning.
4. Competitive efficiency: on the same local corpus budget, long-context memory
   must keep quality within `15%` of the best simple CPU baseline, run at least
   `20x` faster than the tiny Transformer CPU latency baseline, use at least
   `5x` less RAM proxy, adapt online at least `50x` faster than retrain proxy,
   and retain old-domain performance `>=95%`.
5. Structural proof: the strict scanner must remain clean and Phase 6 benchmark
   failures must return non-zero with JSON metrics.

## Canonical Commands

```powershell
python benchmarks/phase6/run_phase6.py --profile quick
python benchmarks/phase6/run_phase6.py --profile claim
python benchmarks/phase6/run_phase6.py --profile full
```

`quick` is for fast smoke checks. `claim` uses TinyStories full plus `128k`
world-memory evidence. `full` uses WikiText-2 for open generation and
competitive efficiency plus the same long-context world-memory stress test.

## Current Claim Boundary

Passing Phase 6 supports the statement that this repo has a stricter evidence
harness for sparse/local CPU language mechanisms. It does not support the
statement that the architecture fully matches modern LLMs on broad instruction
following, world knowledge, coding, math, or open-ended human preference.
