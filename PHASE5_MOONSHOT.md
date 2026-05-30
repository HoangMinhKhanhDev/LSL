# Phase 5 Moonshot v5.0 Contract

Phase 5 tests whether the mechanisms can become a competitive CPU small-model
candidate, not merely pass toy demonstrations.

## Required Mechanisms

1. Semantic SDR scale: `100k` vocab ratio `>=10x`, retrieval `>=95%`; `1M`
   vocab ratio `>=5x`, retrieval `>=90%`; collision sample `<=0.1%`.
2. Sparse physical compute: CPU wall-clock, RAM, memory traffic, cache proxy,
   and energy proxy must beat dense. Real joule/token is optional unless meter
   values are supplied.
3. Interference-free storage: store `100k` SDR patterns with retention `>=95%`,
   false positive `<=1%`, 50% mask completion `>=90%`, 70% mask completion
   `>=75%`, and 10% noisy-bit recognition `>=90%`.
4. Predictive coding and suppression: local error drop per layer `>=80%`,
   suppression `>=70%`, loss degradation `<=5%`, and ablation delta `>=25%`.
5. Long-context retrieval and memory: bounded sparse candidate lookup, no
   full-history scan, recall `>=95%` at `1k`, `>=90%` at `4k`, `>=85%` at
   `16k`, `>=75%` at `64k`, and `>=60%` at `128k`. The integrated memory
   benchmark must also pass sparse fact recall, instruction recall, and
   next-token transition loss improvement. The real-corpus benchmark must run
   on actual TinyStories and WikiText-2 text, support subword tokenization,
   report perplexity/loss against CPU baselines, and pass latency, RAM,
   adaptation, and retention thresholds without synthetic fallback.
6. Compositional reasoning: direct association `>=95%`, multi-hop `>=80%`,
   role binding `>=90%`, causal unseen relation `>=70%`, ablation drop `>=25%`.
   The mini exact-eval suite must also pass QA, reasoning, and coding tasks
   with gold answers stored outside benchmark code. The natural-instruction
   exact-judge suite must pass long-passage QA, instruction formatting,
   multi-step math traces, stack-program traces, and evidence-faithfulness
   scoring.
7. Branching cortical sequence memory: ambiguous-context disambiguation
   `>=90%`, coherence `>=0.75`, loop rate `<=5%`.
8. Continual learning: A -> B -> C protocol, predict-only evaluation,
   retention A/B `>=95%`, new-domain improvement `>=50%`, replay budget
   `<=5%`, consolidation ablation drop `>=20%`.
9. Hierarchy/routing: learned token -> phrase -> topic routing, route sparsity
   `<=10%`, topic accuracy `>=85%`, ablation drop `>=20%`.
10. Integrated scaling law: quality improves monotonically within tolerance,
    sparse advantage remains positive at every tested scale.
11. Baseline competition: quality no worse than the best simple CPU baseline by
    more than 15%, latency at least 20x faster than tiny Transformer where
    attention dominates, online adaptation at least 50x faster than retrain
    proxy, and retention at least 95%.

## Canonical Commands

```powershell
python benchmarks/phase5/run_moonshot.py --profile quick
python benchmarks/phase5/run_moonshot.py --profile full
python benchmarks/phase5/run_moonshot.py --profile claim
```

Every benchmark in the Moonshot runner must write machine-readable JSON when
`--json-output` is supplied and must return non-zero on any strict metric fail.

`claim` is the fairness-hardened profile: it uses bucket-only sparse retrieval,
random-value long-context tests, absent-key false positives, held-out random
symbol reasoning, overlapping-vocabulary continual learning, trained NumPy CPU
Transformer/SSM baselines, and multi-seed scaling-law checks.

Baseline training is allowed only in benchmark comparison code. The strict
architecture path under `lsl/` remains constrained by `GOAL.md`.
