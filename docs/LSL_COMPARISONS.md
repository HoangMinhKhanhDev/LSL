# LSL Comparisons

This page collects the current comparison snapshots for LSL against a CPU NumPy
Transformer baseline. The goal is to make the evidence easy to inspect without
mixing it into the model card or the technical report.

## Methodology

- Baselines are run on the same corpus, same token budget, and same CPU-only
  environment used by the benchmark runner.
- The canonical runner is
  `benchmarks/competitive/run_lsl_vs_transformer.py`.
- Reported metrics include:
  - loss / perplexity / accuracy
  - p50 latency
  - train tokens per second
  - inference tokens per second
  - model size
  - generation coherence and loop rate
  - fact recall and online adaptation
  - context-latency profile across multiple context lengths
- These are benchmark snapshots, not universal claims about every workload.

## Snapshot: TinyStories, 100k tokens

Source: `results/lsl_vs_transformer_tinystories_100k.json`

- LSL: loss `3.9038`, perplexity `49.59`, accuracy `0.2661`
- Transformer: loss `6.4284`, perplexity `619.19`, accuracy `0.0634`
- Inference latency: `8.3 us` vs `34.1 us`
  - LSL is about `4.11x` faster at p50 inference latency
- Train throughput: `1,299.5 tok/s` vs `15,887.6 tok/s`
  - Transformer trains about `12.2x` faster here
- Model size: `24.91 MB` vs `0.56 MB`
- Generation coherence: LSL `0.994`, Transformer `0.919`
- Fact recall: LSL `1.0`, Transformer `0.0`
- Online adaptation: LSL passes the update/query check

## Snapshot: WikiText-2, 100k tokens

Source: `results/lsl_vs_transformer_wikitext2_100k.json`

- LSL: loss `5.0919`, perplexity `162.69`, accuracy `0.1276`
- Transformer: loss `6.5583`, perplexity `705.08`, accuracy `0.0334`
- Inference latency: `7.9 us` vs `34.3 us`
  - LSL is about `4.34x` faster at p50 inference latency
- Train throughput: `1,322.6 tok/s` vs `13,703.5 tok/s`
  - Transformer trains about `10.4x` faster here
- Model size: `27.16 MB` vs `0.62 MB`
- Generation coherence: LSL `0.925`, Transformer `0.903`
- Loop rate: LSL `0.000`, Transformer `0.017`
- Fact recall: LSL `1.0`, Transformer `0.0`

## Snapshot: Dialogue smoke, phase 1 scaling compare

Source: `results/lsl_vs_transformer_dialogue_small_phase1.json`

- LSL: loss `4.6777`, perplexity `107.52`, accuracy `0.1064`
- Transformer: loss `5.0876`, perplexity `161.99`, accuracy `0.0142`
- Inference latency: `2.7 us` vs `32.7 us`
  - LSL is about `12.11x` faster at p50 inference latency
- Train throughput: `369,772 tok/s` vs `23,457 tok/s`
  - LSL trains about `15.98x` faster here
- Generation coherence: LSL `0.962`, Transformer `0.914`
- Fact recall: LSL `1.0`, Transformer `0.0`
- Native core: available and enabled, with forward/update native ratio `1.0`

## Scaling Smoke

Source: `results/phase1_scaling_compare_smoke.json`

- On a tiny dialogue smoke run, LSL kept the same token budget comparison
  cleanly and reported:
  - loss ratio LSL/Transformer `0.9358`
  - train speedup LSL/Transformer `12.9985x`
  - inference speedup LSL/Transformer `12.0769x`
- Memory-budget checks in the same run showed LSL fitting under both `1 MB`
  and `4 MB` targets in the smoke configuration.

## How To Reproduce

```bash
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset tinystories --tokens 100000 --lsl-profile native_fast --json-output results/lsl_vs_transformer_tinystories_100k.json
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset wikitext2 --tokens 100000 --lsl-profile native_fast --json-output results/lsl_vs_transformer_wikitext2_100k.json
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset dialogue_small --tokens 512 --max-eval-tokens 256 --max-train-chars 20000 --max-eval-chars 10000 --tokenizer-train-chars 10000 --vocab-size 256 --lsl-profile native_fast --transformer-epochs 1 --json-output results/lsl_vs_transformer_dialogue_small_phase1.json
```

For a stricter claim gate, add `--claim`.

