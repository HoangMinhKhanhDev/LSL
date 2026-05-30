# Phase 4 Benchmarks

Phase 4 contains the CPU mechanism benchmarks that bridge the original
18-goal contract and the Phase 5 Moonshot suite.

## Runner

```bash
python benchmarks/phase4/run_phase4.py
```

The runner returns non-zero when any child benchmark fails.

## Benchmarks

- `benchmark_semantic_sdr_scaling.py`: semantic SDR structure through 100k vocab
- `benchmark_sparse_physical_compute.py`: wall-clock, ops, memory traffic, cache proxy, RAM, and energy proxy
- `benchmark_long_context.py`: bounded sparse key-value retrieval without full-history scan
- `benchmark_reasoning.py`: direct association, multi-hop, role binding, causal unseen relation, and ablation
- `benchmark_continual_learning.py`: A -> B -> C predict-only continual learning protocol
- `benchmark_language_quality.py`: sparse online local model vs NGram CPU baseline
- `benchmark_anti_cheat.py`: strict structural scan
- `baseline_transformer.py`: NumPy tiny Transformer latency/parameter baseline
- `baseline_ssm.py`: NumPy SSM/Mamba-like CPU baseline

## Strict Constraints

- no backprop or autograd
- no optimizer state
- no GPU or deep-learning framework in the strict path
- no attention matrix or all-pairs token interaction
- no global hidden error path
- online/local updates only
- no full-context scan in long-context retrieval

Phase 5 extends these checks to 1M semantic scale, 100k SDR memory, 128k
retrieval horizon, learned hierarchy/routing, and integrated scaling-law tests.
