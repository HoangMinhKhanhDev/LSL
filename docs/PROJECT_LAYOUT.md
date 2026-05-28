# Project Layout

This repo is organized around one strict research contract and a small set of
supporting scripts.

## Root

- `GOAL.md`: the non-negotiable target contract
- `README.md`: quick start and strict verification summary
- `run_all.py`: umbrella runner for the strict suite
- `benchmark_*.py`: thin compatibility wrappers for the benchmark package
- `test_*`: unit and integration coverage
- `demo_*`, `trace_*`, `verify_*`: exploratory scripts and proofs

## `benchmarks/`

Canonical benchmark implementations grouped by purpose:

- `phase1/`: SDR, semantic overlap, sparse compute, capacity, and completion
- `phase2/`: predictive coding and reasoning benchmarks
- `phase3/`: cortical column sequence-memory benchmarks
- `strict/`: canonical 18-goal verification suite
- `misc/`: supporting experiments and older auxiliary benchmarks

## `lsl/`

Core package code:

- `model.py`: local predictive-coding language model
- `synapse.py`: sparse living synapse primitive
- `sdr.py` and `semantic_sdr.py`: sparse binary representation utilities
- `cortical_column.py`: sequence memory and burst/silent dynamics
- `associative_memory.py`: SDR pattern completion memory
- `data/mini_semantic_embeddings.json`: checked-in offline semantic priors

## Recommended reading order

1. `README.md`
2. `GOAL.md`
3. `docs/PROJECT_LAYOUT.md`
4. `benchmarks/README.md`
5. `lsl/__init__.py`
6. `benchmark_goal_strict.py`
