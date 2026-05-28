# Living Synapse Language Model (LSL)

LSL is a CPU-only NumPy research prototype for local online language learning.
The strict path is designed to satisfy `GOAL.md` without backpropagation,
optimizer state, GPU use, deep learning frameworks, or attention mechanisms.

## Strict Verification

Run the canonical suite:

```bash
python benchmark_goal_strict.py
```

Expected result:

```text
Goals passed: 18/18
Overall: PASS
Structural PASS clean
```

The default umbrella command runs unit tests plus the strict phase benchmarks:

```bash
python run_all.py
```

For a concise map of the repo, see [docs/PROJECT_LAYOUT.md](/F:/brain/docs/PROJECT_LAYOUT.md).

## Components

- `lsl/sdr.py` and `lsl/semantic_sdr.py`: binary SDR encoding, combinatorial capacity, and checked-in offline semantic priors.
- `lsl/synapse.py`: local living synapse layer with sparse active-index forward/update paths and operation counts.
- `lsl/model.py`: local predictive-coding language model with online token/relation association memory.
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

Current strict highlights:

- Phase 1 SDR: semantic overlap ratio > 3x, capacity > 130 bits, retention >= 90%, completion >= 70%, sparse full-forward compute >= 40x by ops and wall-clock.
- Phase 2 predictive coding: every tracked local prediction error drops >= 50%, `theta=0.02` suppression saves >= 60%, eval loss <= 4.0 after 25 epochs, and `stroke -> aphasia` relation probability >= 0.3.
- Phase 3 cortical column: sequence prediction >= 60%, silent-column processing >= 80%, S-V-O emergence >= 7/10, coherence >= 0.6, constant per-token compute, and old-domain retention >= 85%.

## Constraints

The strict scanner checks the implementation path for forbidden constructs:

- no backprop calls
- no optimizer calls
- no PyTorch, TensorFlow, or JAX
- no DFA feedback matrices
- no public or strict-path attention mechanism

Offline semantic information lives in `lsl/data/mini_semantic_embeddings.json`; no external API is used during `forward()` or `observe()`.
