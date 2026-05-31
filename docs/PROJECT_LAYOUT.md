# Project Layout

This repo is organized around one strict research contract and a small set of
supporting scripts.

## Root

- `GOAL.md`: the non-negotiable target contract
- `PHASE5_MOONSHOT.md`: extended Moonshot v5.0 mechanism contract
- `PHASE6_COMPETITIVE.md`: open generation, world memory, reasoning, and competitive-evidence contract
- `PHASE7_GENERALIZATION.md`: self-tuning, generalization, and long-range reasoning contract
- `PHASE8_EXTERNAL.md`: integrated agent and external reality-check contract
- `PHASE9_BIO_COMPUTE.md`: six-mechanism biological compute closure contract
- `README.md`: quick start and strict verification summary
- `run_all.py`: umbrella runner for the strict suite
- `lsl_chat.py`: interactive checkpoint demo for `LSLCoreModel`
- `benchmark_*.py`: thin compatibility wrappers for the benchmark package
- `test_*`: unit and integration coverage
- `demo_*`, `trace_*`, `verify_*`: exploratory scripts and proofs

## `benchmarks/`

Canonical benchmark implementations grouped by purpose:

- `phase1/`: SDR, semantic overlap, sparse compute, capacity, and completion
- `phase2/`: predictive coding and reasoning benchmarks
- `phase3/`: cortical column sequence-memory benchmarks
- `phase4/`: scale-oriented mechanism benchmarks
- `phase5/`: Moonshot v5.0 benchmarks and runner
  (`quick`, `full`, and fairness-hardened `claim` profiles)
- `phase6/`: competitive-evidence benchmarks for open generation, world memory, public-style reasoning, and CPU efficiency
- `phase7/`: generalization, self-tuning, event-driven state, optional prior, and long-range reasoning benchmarks
- `phase8/`: integrated agent, official public dataset adapters, external-style gold, public text, multi-evidence, and scaling checks
- `phase9/`: biological compute closure checks for predictive coding, SDR, cortical columns, hippocampus, neuromodulation, dendrites, and integrated ablations
- `competitive/`: single-model LSLCoreModel vs CPU NumPy Transformer comparisons
- `train_lsl_corpus.py`: real-corpus trainer that writes an `LSLCoreModel` checkpoint
- `strict/`: canonical 18-goal verification suite
- `misc/`: supporting experiments and older auxiliary benchmarks

## `lsl/`

Core package code:

- `model.py`: local predictive-coding language model
- `memory.py`: bounded sparse key-value memory and episodic replay buffer
- `reasoning.py`: local relation and role-binding memory
- `generation.py`: open generation controller with local discourse state
- `world_memory.py`: bounded world/evidence memory
- `homeostasis.py`: self-tuning sparse dynamics
- `workspace.py`: reasoning workspace and entity-event graph
- `event_ssm.py`: event-driven sparse state memory
- `prior.py`: optional offline semantic prior
- `agent.py`: integrated strict-path agent
- `core.py`: unified `LSLCoreModel` API for train/evaluate/generate/save/load,
  including `native_fast` throughput and `bio_native` six-mechanism profiles
- `bio.py`: Phase 9 biological compute primitives and `BioComputeAgent`
- `hierarchy.py`: learned hierarchy/routing memory
- `synapse.py`: sparse living synapse primitive
- `sdr.py` and `semantic_sdr.py`: sparse binary representation utilities
- `cortical_column.py`: sequence memory and burst/silent dynamics
- `associative_memory.py`: SDR pattern completion memory
- `data/mini_semantic_embeddings.json`: checked-in offline semantic priors

## Recommended reading order

1. `README.md`
2. `GOAL.md`
3. `PHASE5_MOONSHOT.md`
4. `PHASE6_COMPETITIVE.md`
5. `PHASE7_GENERALIZATION.md`
6. `PHASE8_EXTERNAL.md`
7. `PHASE9_BIO_COMPUTE.md`
8. `docs/PROJECT_LAYOUT.md`
9. `benchmarks/README.md`
10. `lsl/__init__.py`
11. `benchmark_goal_strict.py`
