# Phase 8 External Reality Check Contract

Phase 8 moves from mechanism-only verification toward an integrated external
reality check. It combines text ingestion, sparse long-context memory,
world/evidence memory, entity-event graph reasoning, workspace reasoning,
trace execution, and generation in one strict-path agent.

The strict constraints remain unchanged: no backprop, optimizer calls, GPU
requirement, deep learning framework, public attention mechanism, global hidden
error path, full-history retrieval scan, or external API inside `forward()`,
`observe()`, or answer generation.

## Required Mechanisms

1. Integrated agent: one pipeline must ingest text/facts/events, answer QA,
   execute math/program traces, perform event-chain reasoning, and generate
   text from the same memory stack.
2. Official public dataset adapters: bAbI 1-20, SQuAD v1.1, GSM8K, and MBPP
   must download/cache, parse, normalize gold answers, and expose exact,
   numeric, and executable-code judges without calling an external API.
3. Public-data smoke evaluation: the integrated agent must run on at least one
   answerable public bAbI task through the same agent interface, while SQuAD,
   GSM8K, and MBPP judges prove that harder public datasets are wired in for
   honest future scoring.
4. External-style gold checks: QA, dialogue, event reasoning, arithmetic, and
   stack/program tasks must be judged by gold answers stored outside benchmark
   code.
5. Public text profiles: integrated generation must run on TinyStories and
   WikiText-2 profiles using the existing cached/downloaded corpus path.
6. Multi-evidence long context: the integrated agent must answer 3-hop evidence
   chains at `100k` and experimental `1M` scale without full-history scan.
7. Scaling smoke test: accuracy must remain high and proxy loss must improve as
   data size grows across the tested sizes.
8. Structural proof: the strict scanner must remain clean and every Phase 8
   benchmark must emit JSON and return non-zero on strict metric failure.

## Canonical Commands

```powershell
python benchmarks/phase8/download_public_datasets.py
python benchmarks/phase8/benchmark_public_dataset_adapters.py
python benchmarks/phase8/benchmark_public_integrated_eval.py
python benchmarks/phase8/run_phase8.py --profile quick
python benchmarks/phase8/run_phase8.py --profile claim
python benchmarks/phase8/run_phase8.py --profile full
```

`quick` is a fast smoke test. `claim` uses TinyStories full and 1M
multi-evidence context. `full` uses WikiText-2 for the integrated text profile.

## Current Claim Boundary

Passing Phase 8 supports the claim that the mechanisms can run together in a
single strict-path prototype, survive a small external-style gold harness, and
connect to official public benchmark formats with strict local judges.
It still does not claim parity with modern LLMs on broad human preference,
large-scale coding, advanced math, or open-domain instruction following.
The public adapter benchmark is infrastructure proof; full public capability
claims require raising the integrated public-eval targets beyond the current
smoke profile.
