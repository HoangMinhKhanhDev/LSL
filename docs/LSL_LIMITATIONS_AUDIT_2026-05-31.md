# LSL Limitations Audit - 2026-05-31

This report audits the 12 known limitations of the current LSL implementation and records the concrete fixes made in this pass. It separates measured facts from proxy/structural claims.

## Measurement Snapshot

- Strict gate: `benchmark_goal_strict.py` previously passed `25/25` strict targets, with G1.6 native sparse wall speedup above `500x`.
- Bio-native corpus run: `python benchmarks/train_lsl_corpus.py --dataset tinystories --max-tokens 10000 --lsl-profile bio_native`
  - Core train: `85.26 us/token`
  - End-to-end wall throughput: `4871.19 tokens/s`
  - Vocab: `622`
  - Sample: `the little girl named tom was very happy and said," lily and said`
- Competitive smoke: `python benchmarks/competitive/run_lsl_vs_transformer.py --dataset tinystories --tokens 10000 --lsl-profile bio_native`
  - LSL loss/perplexity/accuracy: `5.1928 / 179.98 / 16.85%`
  - Transformer loss/perplexity/accuracy: `6.4326 / 621.79 / 4.67%`
  - LSL train speed: `11868.15 tokens/s`
  - Transformer train speed: `16686.87 tokens/s`
  - LSL inference speed: `12738.85 tokens/s`
  - Transformer inference speed: `29325.51 tokens/s`
  - LSL generation coherence: `0.9396`
  - LSL loop/UNK: `0.00% / 0.00%`
  - LSL fact recall: `100%`
- Real energy probe:
  - Command: `python benchmarks/energy/measure_native_sparse_energy.py --tokens 64 --warmup 4 --dim 256 --active 2`
  - Result: sensor unavailable. `EnergyLib64.dll` was not found and Windows `Power Meter(*)\Power` counter was not present.

## Code Improvements Made In This Pass

- Added bounded maintenance for `bio_native` runtime sidecars in `lsl/core.py`.
- Added pruning for dendritic branches in `lsl/bio.py`.
- Added pruning for cortical temporal/context tables in `lsl/cortical_column.py`.
- Added hippocampal fast/slow memory pruning in `lsl/bio.py`.
- Added generation-context candidate scoring and repetition penalties in `lsl/core.py`, while preserving the fast single-token eval path.
- Added focused regression tests in `test_lsl_limitations_improvements.py`.

## 1. Natural Language Quality

Status: improved, still not solved.

Evidence:
- LSL has better loss than the tiny NumPy Transformer baseline on the 10k TinyStories smoke (`5.1928` vs `6.4326`), but the generated sample is still short and simple.
- LSL generation metrics are clean on loop/UNK (`0.00% / 0.00%`) and coherence proxy is high (`0.9396`), but this is not the same as human-level fluency.

Improvement:
- Generation now uses a wider candidate set from local long-context memory when a real prompt context is available.
- It penalizes repeated tokens, repeated trigrams, `<UNK>`, and immediate self-repetition.
- The fast eval path stays separate, so quality scoring does not slow one-token evaluation.

Remaining limitation:
- LSL still lacks large-scale learned stylistic and instruction-following behavior. It can track local transitions and facts, but it does not yet produce LLM-grade open-ended prose.

Next required work:
- Add human-readable generation eval sets, longer prompts, and multi-domain dialogue corpora.
- Add a stronger decoder that can combine native transition scores, cortical context, world facts, and discourse plan without collapsing into short transition chains.

## 2. Benchmark Scale Is Still Small

Status: partially improved by measurement discipline, not solved.

Evidence:
- Current measured competitive smoke is `10,000` train tokens and `1,200` eval tokens.
- Existing strict targets include synthetic large-scale probes, but full-corpus competitive training is not yet the default claim path.

Improvement:
- The benchmark path now records concrete `bio_native` speed and quality without pretending the smoke run proves general dominance.
- Tokenizer `max_tokens` handling from the previous bottleneck work prevents accidental full-corpus tokenization when only a token slice is requested.

Remaining limitation:
- No completed 1M/10M/100M token LSL-vs-Transformer sweep exists yet.

Next required work:
- Run and store scale sweeps at `100k`, `1M`, `10M` tokens for both TinyStories and WikiText-2.
- Track variance across seeds and checkpoint sizes.

## 3. Strict Goals Include Proxy/Synthetic Measurements

Status: clearer, but not eliminated.

Evidence:
- Strict gate has structural and proxy measurements by design.
- Energy probe failed because this machine lacks a readable real watt sensor.

Improvement:
- This report explicitly separates proxy results from measured results.
- Real energy command fails closed instead of fabricating evidence.

Remaining limitation:
- Proxy energy and synthetic retrieval cannot be advertised as real watt or real-world language intelligence.

Next required work:
- Install Intel Power Gadget or expose a Windows Power Meter counter.
- Run `benchmark_goal_strict.py --require-real-energy --energy-evidence results/energy_evidence.json`.

## 4. SDR Semantics Are Still Shallow

Status: unchanged in this pass, limitation remains.

Evidence:
- SDR capacity and overlap are strong structurally, but semantic richness is mostly produced by deterministic features, related-pair updates, and token identity.
- Numeric token SDRs do not yet capture full lexical meaning or morphology at production level.

Improvement:
- No direct semantic-SDR algorithm change was made in this pass. I did not lower the claim.

Remaining limitation:
- SDR currently remembers and separates patterns better than it learns deep semantic manifolds.

Next required work:
- Train a sparse co-occurrence semantic layer on real corpora and feed lexical/subword strings, not only numeric token IDs, into SDR relation learning.
- Add held-out synonym, analogy, and multilingual semantic tests.

## 5. Open Generation Is Still Weak

Status: improved, still weak.

Evidence:
- LSL loop rate remains `0.00%`; UNK remains `0.00%`.
- Sample still reads like a transition-memory chain, not a robust assistant answer.

Improvement:
- Added generation-context scoring using local next-candidate probabilities plus repetition fatigue.
- Added explicit penalties for repeated trigrams and immediate repeats.

Remaining limitation:
- The generated text is still short and locally plausible rather than globally planned.

Next required work:
- Add paragraph-level planning, answer intent detection, and entity-state constraints.
- Evaluate on open-generation public prompts, not only TinyStories continuations.

## 6. Reasoning Workspace Is Narrow

Status: unchanged in capability, documented.

Evidence:
- Current workspace handles local math traces, stack traces, entity-event chains, and fact lookup.
- It does not yet solve broad multi-hop QA or code reasoning tasks.

Improvement:
- Maintenance/pruning now protects runtime sidecars from unbounded growth, which helps long-running reasoning sessions stay stable.

Remaining limitation:
- Reasoning is mostly symbolic/template-like, not broad neural reasoning.

Next required work:
- Add general trace operators, scratchpad variable lifetimes, branch/merge control flow, and real QA datasets.

## 7. Local Learning Can Struggle With Deep Abstractions

Status: not solved.

Evidence:
- LSL learns transitions and facts online quickly, but deep abstractions require repeated exposure and better consolidation.
- The competitive smoke shows good loss relative to the tiny baseline, but that does not prove high-level abstraction.

Improvement:
- Added periodic maintenance so online learning has a stable memory budget instead of uncontrolled accumulation.

Remaining limitation:
- No backprop means LSL must learn abstractions through local structure, replay, and sparse co-occurrence. That path needs stronger evidence.

Next required work:
- Add abstraction benchmarks: analogy, compositional generalization, held-out grammar, and OOD semantic transfer.

## 8. Memory Growth Can Still Be A Problem

Status: materially improved.

Evidence:
- Before this pass, dendritic branches, cortical context tables, and hippocampal fast/slow maps could grow without explicit runtime caps.

Improvement:
- `DendriticLayer.prune_branches(max_branches)` keeps high-value branches and rebuilds indexes.
- `CorticalColumnSequenceMemory.prune_memory(...)` bounds temporal segments, context keys, and targets per context.
- `HippocampalMemory.prune(...)` bounds fast and slow stores and rebuilds feature buckets.
- `LSLCoreConfig` now exposes maintenance caps:
  - `bio_maintenance_interval`
  - `bio_dendrite_max_branches`
  - `bio_column_max_segments`
  - `bio_column_max_contexts`
  - `bio_column_max_targets_per_context`
  - `bio_hippocampus_max_fast`
  - `bio_hippocampus_max_slow`
- Added regression test proving maintenance prunes runtime sidecars.

Remaining limitation:
- World memory and entity-event graph still need deeper aging policies for very long-running agents.

Next required work:
- Add importance-weighted eviction for world facts and event graph evidence.
- Add checkpoint compaction statistics.

## 9. Native C Coverage Is Incomplete

Status: unchanged in coverage, but fast paths protected.

Evidence:
- Native C covers sparse transition scoring/update and remains enabled in the competitive smoke (`forward_native_ratio=100%`, `update_native_ratio=100%`).
- Python still handles tokenizer, cortical logic, hippocampus, neuromodulation, workspace, and some generation logic.

Improvement:
- The bio-native inference regression was avoided by keeping single-token eval on the fast path.

Remaining limitation:
- Full six-mechanism runtime is not fully native.

Next required work:
- Port dendrite candidate scoring, cortical top prediction, and tokenizer hot paths to C/Rust if scale runs show they dominate again.

## 10. Real Watt Measurement Is Not Available Yet

Status: checked, not solved.

Evidence:
- Probe failed closed:
  - Intel Power Gadget `EnergyLib64.dll` not found.
  - Windows Power Meter performance counter instance not present.

Improvement:
- No fake watt file was written. The strict runner still accepts proxy energy by default and requires real evidence only when requested.

Remaining limitation:
- No real `J/token` or watt claim can be made on this machine today.

Next required work:
- Install Intel Power Gadget or expose OS power counters.
- Rerun `benchmarks/energy/measure_native_sparse_energy.py`.

## 11. Transformer Comparison Is Not Yet Strong Enough

Status: clearer, not solved.

Evidence:
- In one 10k smoke, LSL has better loss than the tiny NumPy Transformer baseline.
- In this latest smoke, the tiny Transformer is faster on p50 inference (`29325.51 tokens/s` vs LSL `12738.85 tokens/s`) and training (`16686.87 tokens/s` vs LSL `11868.15 tokens/s`).
- The baseline is a small NumPy Transformer, not llama.cpp, ONNX, PyTorch CPU, or a tuned production Transformer.

Improvement:
- The benchmark report now makes the failure explicit: latency check is false in this smoke.
- No claim gate is enabled by default.

Remaining limitation:
- We cannot claim same-scale Transformer dominance yet.

Next required work:
- Compare against multiple Transformer baselines and multiple scales.
- Report quality-speed Pareto curves rather than a single smoke run.

## 12. Tokenizer Is Still Not Production Grade

Status: partially improved.

Evidence:
- The tokenizer is a small deterministic BPE-style tokenizer. It is not SentencePiece/tiktoken quality.

Improvement:
- Prior bottleneck work added word-level encode caching and bounded `max_tokens` encoding.
- This pass keeps those changes and adds tests ensuring bounded encode returns the same prefix as full encode.

Remaining limitation:
- Multilingual segmentation, Vietnamese tone handling, code tokenization, and production-grade normalization are still incomplete.

Next required work:
- Add a SentencePiece-compatible strict tokenizer option or improve the existing tokenizer with Unicode normalization and multilingual tests.

## Current Bottom Line

LSL is stronger after this pass in memory safety and generation control, and the report now exposes where it still fails. The largest remaining hard problems are not micro-optimization anymore. They are scale, semantic depth, production tokenizer quality, real watt measurement, and stronger Transformer comparisons.

