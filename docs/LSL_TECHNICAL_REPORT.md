# LSL Technical Report

## Overview

Living Synapse Language Model (LSL) is a CPU-first research prototype for
online language learning without backpropagation. The repository organizes the
model around one unified `LSLCoreModel` surface while retaining separate,
testable subsystems for memory, generation, reasoning, and biological-style
mechanisms.

## Core Systems

### 1. Sparse transition core

The strict runtime uses a native sparse transition kernel for the hot path.
This kernel handles local forward/update work and avoids dense all-pairs
attention in the strict path.

### 2. Predictive coding

Local predictive coding is used to track error and suppression at each layer.
The implementation exposes per-layer error curves, confidence, anomaly scores,
and update suppression diagnostics.

### 3. Sparse distributed representations

The SDR path uses large virtual sparse codes and benchmarked semantic probes for
capacity, overlap, noisy recall, reconstruction, multilingual overlap, and
collision behavior.

### 4. Cortical columns

Sequence memory is modeled as a sparse cortical column system with burst and
silent behavior, bounded context tables, and latency diagnostics.

### 5. Hippocampal memory

The memory system separates fast and slow stores, supports consolidation,
conflict resolution, replay, and bounded pruning.

### 6. Neuromodulation

Reward, novelty, and homeostasis gates are tracked explicitly. Diagnostics are
available for update ratios and gating behavior.

### 7. Dendritic computation

Dendritic branches provide sparse routing and native candidate scoring. The
benchmarks report branch utilization, compute-density gains, and native-vs-
fallback agreement.

## Public Interfaces

- `lsl.LSLCoreModel`
- `lsl.BioComputeAgent`
- `lsl_cli.py`
- `lsl_chat.py`
- `lsl_web_demo.py`
- `lsl_report.py`

## Benchmark Coverage

The repository includes:

- strict gate verification
- phase 1 scaling-law and tokenizer benchmarks
- phase 2 predictive-coding checks
- phase 3 cortical-column checks
- phase 4 scale checks
- phase 5 Moonshot checks
- phase 6 to 8 competitive and integrated-agent checks
- phase 9 biological compute closure checks
- competitive LSL-vs-Transformer comparisons
- energy evidence capture from a real sensor when available

## Comparisons and Baselines

The comparison artifacts live in
[docs/LSL_COMPARISONS.md](LSL_COMPARISONS.md). They are produced by the
canonical competitive runner and keep the evaluation conditions explicit:

- same corpus
- same token budget
- CPU-only environment
- measured loss, perplexity, accuracy, latency, train throughput, model size,
  generation quality, fact recall, and online adaptation

The current snapshot pattern is:

- LSL tends to win inference latency on the reported small and 100k-token
  runs
- LSL achieves lower loss than the CPU NumPy Transformer on the reported
  snapshots
- the Transformer often retains a train-throughput advantage on the larger
  100k-token corpus runs
- the dialogue smoke scaling run is the cleanest example where LSL also wins
  train throughput

This is a research comparison layer, not a universal leaderboard.

## Reproducibility

Benchmark results are written with metadata that includes:

- timestamp
- git commit
- platform
- Python version
- benchmark config

The repository also includes an HTML report generator that renders stored
results into a browsable summary.

## Known Limits

The current implementation remains a research prototype. The main remaining
gaps are:

- broad open-ended instruction following
- large-scale semantic abstraction
- production-grade multilingual tokenization in every corner case
- real watt sensor availability on every machine
- decisive same-scale superiority versus a tuned Transformer baseline

## Recommended Verification Sequence

1. Build the native extension.
2. Run the strict gate.
3. Run the phase 9 claim suite.
4. Run the competitive LSL-vs-Transformer benchmark.
5. Generate the HTML report.
6. Attach real energy evidence only when a valid hardware sensor is present.
