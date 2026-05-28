# Benchmarks

This folder contains the canonical benchmark implementations grouped by phase.

Root-level `benchmark_*.py` files are thin compatibility wrappers so older
commands still work, but the maintained code now lives here:

- `phase1/`: SDR, semantic overlap, sparse compute, capacity, pattern completion
- `phase2/`: predictive coding and related reasoning checks
- `phase3/`: cortical column sequence-memory checks
- `strict/`: the canonical all-goals suite
- `misc/`: legacy or supporting benchmark experiments

The strict suite remains the source of truth for pass/fail status:

```bash
python benchmark_goal_strict.py
```
