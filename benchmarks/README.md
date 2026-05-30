# Benchmarks

This folder contains the canonical benchmark implementations grouped by phase.

Root-level `benchmark_*.py` files are thin compatibility wrappers so older
commands still work, but the maintained code now lives here:

- `phase1/`: SDR, semantic overlap, sparse compute, capacity, pattern completion
- `phase2/`: predictive coding and related reasoning checks
- `phase3/`: cortical column sequence-memory checks
- `phase4/`: scale-oriented mechanism checks, semantic SDR at 100k vocab, and physical sparse-compute validation
- `phase5/`: Moonshot v5.0 checks for 1M semantic scale, sparse memory, long-context retrieval and real-corpus memory, exact-answer QA/reasoning/coding, natural instruction judging, continual learning, hierarchy/routing, baseline competition, and scaling law
- `phase6/`: competitive-evidence checks for open generation, world/evidence memory, public-style reasoning, and CPU efficiency
- `phase7/`: checks for homeostasis, generation quality v2, diverse data, event-driven sparse state, optional prior, heldout generalization, and long-range reasoning
- `phase8/`: integrated agent checks, official public dataset adapters,
  external-style gold tasks, public text profiles, multi-evidence long context,
  and scaling smoke tests
- `phase9/`: biological compute closure checks for predictive coding v2, SDR v2,
  cortical columns, hippocampal memory, neuromodulation, dendritic computation,
  a mechanisms 1-5 target suite, 1,000-branch dendritic moonshot checks,
  integrated ablations, BioComputeAgent dialogue generation, and a model-level
  LSL language-model proof
- `competitive/`: single-model LSLCoreModel vs CPU NumPy Transformer comparisons
- `strict/`: the canonical all-goals suite
- `misc/`: legacy or supporting benchmark experiments

The extreme strict suite remains the source of truth for pass/fail status:

```bash
python setup.py build_ext --inplace
python benchmark_goal_strict.py
```

For real watt evidence on Windows:

```bash
python benchmarks/energy/measure_native_sparse_energy.py --output results/energy_evidence.json
python benchmark_goal_strict.py --require-real-energy --energy-evidence results/energy_evidence.json
```

The measurement path reads Intel Power Gadget `EnergyLib64.dll` or Windows
`Power Meter` counters and fails closed when neither real sensor is readable.

Use the old 18-goal thresholds only as a smoke check:

```bash
python benchmark_goal_strict.py --profile smoke
```

The current Phase 4 mechanism benchmarks are:

```bash
python benchmark_semantic_sdr_scaling.py
python benchmark_sparse_physical_compute.py
```

The Phase 5 Moonshot runner is:

```bash
python download_wikitext2.py
python download_tinystories_full.py
python benchmarks/phase5/run_moonshot.py --profile quick
python benchmarks/phase5/run_moonshot.py --profile full
python benchmarks/phase5/run_moonshot.py --profile claim
```

All Moonshot benchmarks accept `--json-output` and must return non-zero when a
strict metric fails.

`claim` is stricter than `full`: it turns on fairness hardening such as
bucket-only retrieval, held-out random symbols, overlapping continual-learning
domains, trained CPU NumPy baselines, and multi-seed scaling checks.

The Phase 6 runner is:

```bash
python benchmarks/phase6/run_phase6.py --profile quick
python benchmarks/phase6/run_phase6.py --profile claim
python benchmarks/phase6/run_phase6.py --profile full
```

Phase 6 benchmarks also accept `--json-output` and return non-zero on metric
failure. The strict scanner remains part of the runner.

The Phase 7 runner is:

```bash
python benchmarks/phase7/run_phase7.py --profile quick
python benchmarks/phase7/run_phase7.py --profile claim
python benchmarks/phase7/run_phase7.py --profile full
```

Phase 7 keeps strict-zero and optional-prior claims separate.

The Phase 8 runner is:

```bash
python benchmarks/phase8/download_public_datasets.py
python benchmarks/phase8/benchmark_public_dataset_adapters.py
python benchmarks/phase8/benchmark_public_integrated_eval.py
python benchmarks/phase8/run_phase8.py --profile quick
python benchmarks/phase8/run_phase8.py --profile claim
python benchmarks/phase8/run_phase8.py --profile full
```

The public dataset adapter caches bAbI 1-20, SQuAD v1.1, GSM8K, and MBPP under
`benchmarks/data/public/`. It validates parsing plus exact/numeric/code judges.
The integrated public benchmark is a smoke test, not a full public leaderboard
claim.

The Phase 9 runner is:

```bash
python benchmarks/phase9/benchmark_lsl_model_level.py
python benchmarks/phase9/benchmark_bio_mechanisms_1_5_targets.py
python benchmarks/phase9/benchmark_bio_dialogue_generation.py
python benchmarks/phase9/run_phase9.py --profile quick
python benchmarks/phase9/run_phase9.py --profile claim
python benchmarks/phase9/run_phase9.py --profile full
```

Phase 9 is claim-bearing only when `claim` or `full` passes, including the
model-level LSL proof, BioComputeAgent dialogue generation speed/coherence,
strict scanner, and JSON success checks for every sub-benchmark.

The competitive single-model runner is:

```bash
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset tinystories --tokens 100000
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset wikitext2 --tokens 100000
```

It reports language loss/perplexity, accuracy, latency, train time, model-size
proxy, generation metrics, and online adaptation for one unified LSLCoreModel
against a trainable NumPy Transformer on the same token stream. It also reports
context-latency rows, QA/fact recall, and latency/ops energy proxies.

Corpus checkpoint training:

```bash
python benchmarks/train_lsl_corpus.py --dataset tinystories --max-tokens 1000000
python lsl_chat.py --checkpoint checkpoints/lsl_tinystories.json
```

If the default checkpoint is absent, `lsl_chat.py` bootstraps a small local
TinyStories checkpoint before entering the chat loop.
