# Living Synapse Language Model (LSL)

LSL is a CPU-only NumPy research prototype for local online language learning.
The strict path is designed to satisfy `GOAL.md` without backpropagation,
optimizer state, GPU use, deep learning frameworks, or attention mechanisms.

## Strict Verification

Build the native sparse kernel, then run the canonical extreme strict suite:

```bash
python setup.py build_ext --inplace
python benchmark_goal_strict.py
```

Expected result:

```text
Extreme strict: PASS (25/25)
```

The legacy 18-goal gate remains available only as a smoke profile:

```bash
python benchmark_goal_strict.py --profile smoke
```

To attach real watt evidence to G2.5, capture a hardware sensor run and feed it
back into the strict gate:

```bash
python benchmarks/energy/measure_native_sparse_energy.py --output results/energy_evidence.json
python benchmark_goal_strict.py --require-real-energy --energy-evidence results/energy_evidence.json
```

The energy helper uses Intel Power Gadget `EnergyLib64.dll` on Windows when it
is installed, or the Windows `Power Meter` performance counter when exposed. It
exits non-zero instead of writing fake watt evidence if no real sensor is
available.

The default umbrella command runs unit tests, smoke checks, and the extreme
strict gate:

```bash
python run_all.py
```

For a concise map of the repo, see [docs/PROJECT_LAYOUT.md](/F:/brain/docs/PROJECT_LAYOUT.md).

Moonshot v5.0 extends the strict 18-goal suite into competitive-small-model
mechanism tests:

```bash
python benchmarks/phase5/run_moonshot.py --profile quick
python benchmarks/phase5/run_moonshot.py --profile full
python benchmarks/phase5/run_moonshot.py --profile claim
```

Phase 6 adds competitive-evidence tests for open generation, world/evidence
memory, public-style reasoning, and stricter CPU efficiency:

```bash
python benchmarks/phase6/run_phase6.py --profile quick
python benchmarks/phase6/run_phase6.py --profile claim
python benchmarks/phase6/run_phase6.py --profile full
```

Phase 7 targets the remaining limitations directly: self-tuning, stronger
generation quality, diverse data, event-driven state updates, optional offline
prior, heldout generalization, and 100k/1M long-range reasoning:

```bash
python benchmarks/phase7/run_phase7.py --profile quick
python benchmarks/phase7/run_phase7.py --profile claim
python benchmarks/phase7/run_phase7.py --profile full
```

Phase 8 integrates the mechanisms into one agent and runs external-style gold
checks plus public text profiles. It also caches/parses official public
datasets for the next capability ratchet:

```bash
python benchmarks/phase8/download_public_datasets.py
python benchmarks/phase8/benchmark_public_dataset_adapters.py
python benchmarks/phase8/benchmark_public_integrated_eval.py
python benchmarks/phase8/run_phase8.py --profile quick
python benchmarks/phase8/run_phase8.py --profile claim
python benchmarks/phase8/run_phase8.py --profile full
```

Phase 9 closes the six biological mechanisms with executable proofs for
predictive coding v2, SDR v2, cortical columns, hippocampal two-speed memory,
neuromodulation, dendritic computation, a strict mechanisms 1-5 target suite,
one integrated bio-compute agent, BioComputeAgent dialogue generation, and one
model-level LSL language-model proof:

```bash
python benchmarks/phase9/benchmark_bio_mechanisms_1_5_targets.py
python benchmarks/phase9/benchmark_lsl_model_level.py
python benchmarks/phase9/benchmark_bio_dialogue_generation.py
python benchmarks/phase9/run_phase9.py --profile quick
python benchmarks/phase9/run_phase9.py --profile claim
python benchmarks/phase9/run_phase9.py --profile full
```

The first single-model competitive runner compares one `LSLCoreModel` against a
trainable CPU NumPy Transformer on the same tokenizer and token budget:

```bash
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset tinystories --tokens 100000
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset wikitext2 --tokens 100000
```

By default this is descriptive, not a strict claim gate. Add `--claim` to make
the configured quality/latency/generation thresholds fail the command.

Train and save a single LSLCoreModel checkpoint on a real corpus:

```bash
python benchmarks/train_lsl_corpus.py --dataset tinystories --max-tokens 1000000
```

Then run the interactive demo. On a fresh checkout, `python lsl_chat.py` also
bootstraps a small local TinyStories checkpoint automatically if this file is
missing:

```bash
python lsl_chat.py --checkpoint checkpoints/lsl_tinystories.json
```

The full profile includes 1M-vocabulary semantic SDR scaling, 100k-pattern SDR
memory, 128k-horizon sparse retrieval, long-context fact/instruction/transition
memory, real-corpus TinyStories and WikiText-2 long-context evidence,
exact-answer QA/reasoning/coding checks, predictive-coding suppression, role
binding, continual learning, hierarchy/routing, scaling-law checks, and the
anti-cheat structural scan.

Use `claim` for the fairness-hardened run: bucket-only long-context retrieval,
random-value recall, absent-key false-positive checks, held-out random-symbol
reasoning, overlapping-domain continual learning, trained NumPy CPU
Transformer/SSM baselines, and multi-seed scaling checks.

## Components

- `lsl/sdr.py` and `lsl/semantic_sdr.py`: binary SDR encoding, combinatorial capacity, and checked-in offline semantic priors.
- `lsl/synapse.py`: local living synapse layer with sparse active-index forward/update paths and operation counts.
- `lsl/model.py`: local predictive-coding language model with online token/relation association memory.
- `lsl/memory.py`: sparse key-value memory with bounded candidate lookup for long-context retrieval.
- `lsl/long_context.py`: sparse long-context memory for facts, instructions, and next-token transitions.
- `lsl/generation.py`: local open-generation controller with discourse state and repetition fatigue.
- `lsl/world_memory.py`: bounded world/evidence memory for QA and citation-style retrieval.
- `lsl/reasoning.py`: local relation and role-binding memories for multi-hop and compositional checks.
- `lsl/homeostasis.py`: local self-tuning controller for sparse dynamics.
- `lsl/workspace.py`: reasoning workspace and entity-event graph.
- `lsl/event_ssm.py`: event-driven sparse state memory.
- `lsl/prior.py`: optional offline semantic prior quantized into SDRs.
- `lsl/agent.py`: integrated strict-path agent combining text, memory, reasoning, and generation.
- `lsl/core.py`: unified `LSLCoreModel` facade for train/evaluate/generate/answer/save/load.
- `lsl/bio.py`: Phase 9 bio-compute primitives and integrated bio agent.
- `lsl/hierarchy.py`: learned token-to-phrase-to-topic routing memory.
- `lsl/cortical_column.py`: local cortical-column sequence memory with burst/silent dynamics and context segments.
- `lsl/associative_memory.py`: sparse SDR pattern completion memory.

## Top-Level Layout

- `lsl/`: core package code
- `benchmarks/`: canonical benchmark implementations grouped by phase
- `tests/`: unit and integration test notes
- `examples/`: demos, traces, and exploratory scripts
- `docs/`: project map and architecture notes
- `GOAL.md`: strict success contract
- `benchmark_goal_strict.py`: root compatibility wrapper for the canonical all-in-one verification

## Phase Benchmarks

```bash
python benchmark_sdr_phase1.py
python benchmark_pc_phase2.py
python benchmark_cortical_column_sequence.py
python benchmark_goal_strict.py
```

Phase 4 mechanism-scale checks:

```bash
python benchmark_semantic_sdr_scaling.py
python benchmark_sparse_physical_compute.py
```

Phase 5 Moonshot checks:

```bash
python benchmark_long_context.py
python benchmark_long_context_memory.py
python download_wikitext2.py
python download_tinystories_full.py
python benchmark_long_context_real_corpus.py
python benchmark_long_context_real_corpus.py --dataset wikitext2
python benchmark_long_context_real_corpus.py --dataset wikitext2 --tokenizer subword
python benchmark_long_context_real_corpus.py --dataset tinystories_full --tokenizer subword
python benchmark_mini_exact_eval.py
python benchmark_natural_instruction_eval.py
python benchmark_reasoning.py
python benchmark_continual_learning.py
python benchmark_language_quality.py
python benchmark_sdr_large_memory.py
python benchmark_pc_suppression_ood.py
python benchmark_branching_cortical.py
python benchmark_hierarchy.py
python benchmark_baseline_competition.py
python benchmark_scaling_law.py
python benchmark_moonshot.py --profile full
python benchmark_moonshot.py --profile claim
```

Phase 6 competitive-evidence checks:

```bash
python benchmark_open_generation_public.py
python benchmark_world_memory_qa.py
python benchmark_public_reasoning.py
python benchmark_phase6_competitive.py
python benchmark_phase6.py --profile full
```

Phase 7 generalization checks:

```bash
python benchmark_generation_quality_v2.py
python benchmark_homeostasis.py
python benchmark_diverse_data.py
python benchmark_event_driven_ssm.py
python benchmark_optional_prior.py
python benchmark_generalization_heldout.py
python benchmark_long_range_reasoning.py
python benchmark_phase7.py --profile full
```

Phase 8 external reality checks:

```bash
python benchmark_external_gold.py
python benchmark_public_dataset_adapters.py
python benchmark_public_integrated_eval.py
python benchmark_integrated_agent.py
python benchmark_multievidence_long_context.py
python benchmark_external_scaling.py
python benchmark_phase8.py --profile full
```

Phase 9 bio-compute checks:

```bash
python benchmark_lsl_model_level.py
python benchmark_bio_mechanisms_1_5_targets.py
python benchmark_bio_predictive_coding.py
python benchmark_bio_sdr_semantics.py
python benchmark_bio_cortical_column.py
python benchmark_bio_hippocampus.py
python benchmark_bio_neuromodulation.py
python benchmark_bio_dendritic.py
python benchmark_bio_integrated_agent.py
python benchmarks/phase9/benchmark_bio_dialogue_generation.py
python benchmark_phase9.py --profile full
```

Current extreme strict highlights:

- Phase 1 SDR: semantic overlap ratio >= 30x, exact `log2(C(100000,40)) >= 500`, 100k-pattern sparse recall >= 99%, 20% cue completion >= 95%, and native sparse CPU speedup >= 500x.
- Phase 2 predictive coding: local error drop >= 99%, suppression >= 95%, online loss <= 2.0 within 10 epochs, proxy energy savings >= 98%, and causal true/false relation split >= 0.90 / <= 0.10.
- Phase 3 cortical column: ambiguous context and complex grammar accuracy >= 95%, active-state suppression >= 98%, 20k-token topic coherence >= 0.90, latency max/min <= 1.20, and 50-domain retention >= 99%.
- Phase 4 semantic scaling: verifies that offline semantic priors remain separable as SDRs at 1k, 10k, and 100k vocabulary, with a random-prior ablation that must fail.
- Phase 4 sparse physical compute: measures wall-clock latency, peak allocation, touched synapses, cache/locality sensitivity, and energy proxies under strict claim-ready thresholds.
- Phase 5 Moonshot: validates semantic SDR at 1M vocab, physical sparse compute with and without predictive coding, bounded long-context retrieval to 128k, sparse fact/instruction/transition memory, real-corpus TinyStories/WikiText-2 long-context evidence, exact-answer QA/reasoning/coding checks, natural instruction exact-judge scoring, compositional reasoning, continual learning, hierarchy/routing, branching sequence memory, baseline competition, and integrated scaling behavior.
- Phase 6 Competitive Evidence: validates local open generation on public text profiles, world/evidence memory at 128k chunks without full-history scan, public-style relation/role/multi-hop/trace reasoning, and CPU efficiency against small baselines.
- Phase 7 Generalization: validates self-tuning homeostasis, stronger generation scoring, diverse data, event-driven sparse state updates, optional offline prior reporting, heldout/OOD generalization, and 100k/1M entity-event reasoning.
- Phase 8 External Reality Check: validates one integrated strict-path agent on official public dataset adapters for bAbI/SQuAD/GSM8K/MBPP, gold-answer QA/reasoning/code/math/dialogue checks, public text profiles, 100k/1M multi-evidence chains, and scaling smoke tests.
- Phase 9 Bio-Compute Closure: validates predictive coding v2, exact combinatorial SDR capacity, cortical sequence memory, hippocampal two-speed memory, neuromodulation gates, the combined mechanisms 1-5 target suite, dendritic branch computation with one-neuron XOR plus 1,000-branch sparse trees, integrated ablations, BioComputeAgent dialogue generation speed/coherence, and one online LSL model-level language-model proof.

## Constraints

The strict scanner checks the implementation path for forbidden constructs:

- no backprop calls
- no optimizer calls
- no PyTorch, TensorFlow, or JAX
- no DFA feedback matrices
- no public or strict-path attention mechanism

Offline semantic information lives in `lsl/data/mini_semantic_embeddings.json`; no external API is used during `forward()` or `observe()`.
