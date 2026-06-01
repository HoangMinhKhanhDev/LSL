# LSL Model Card

## Model

- Name: Living Synapse Language Model (LSL)
- Type: CPU-first local online language model with sparse native kernels and biological-mechanism-inspired components
- Core API: `lsl.LSLCoreModel`
- Primary entrypoints: `lsl_cli.py`, `lsl_chat.py`, `lsl_web_demo.py`

## Intended Use

LSL is intended for:

- local research on online learning without backpropagation
- sparse-memory and CPU-efficient experimentation
- benchmark-driven comparison against small CPU Transformers
- dialogue/fact-recall prototypes in a controlled offline environment

## Not Intended For

LSL is not a production assistant model and should not be used as a substitute for:

- safety-reviewed medical, legal, or financial advice
- large-scale open-domain chat services
- claims of universal language intelligence
- claims of real watt savings unless a valid sensor evidence file is attached

## Architecture Summary

LSL combines:

- local predictive coding
- sparse distributed representations
- cortical-column sequence memory
- hippocampal two-speed memory
- neuromodulation gates
- dendritic computation
- a native sparse transition kernel for CPU fast paths

The strict runtime avoids:

- global backward passes
- global optimizer state
- GPU dependence
- deep learning frameworks in the strict path
- attention matrices in the strict path

## Data

The repository currently includes smoke-scale and benchmark corpora such as:

- TinyStories
- WikiText-2
- small Vietnamese seed text
- small dialogue seed text

Larger corpora are loaded through the benchmark runners and dataset loader.

## Evaluation

The canonical checks are:

- `python benchmark_goal_strict.py`
- `python benchmark_goal_strict.py --profile smoke`
- `python benchmarks/phase9/run_phase9.py --profile claim`
- `python benchmarks/competitive/run_lsl_vs_transformer.py --dataset tinystories --tokens 100000`

The repository also includes:

- tokenizer benchmarks
- native kernel benchmarks
- phase 1 scaling-law runs
- HTML report generation from stored results

## Comparisons

Current LSL-vs-Transformer snapshots are documented in
[docs/LSL_COMPARISONS.md](LSL_COMPARISONS.md). The short version is:

- on TinyStories and WikiText-2 100k-token runs, LSL is faster at inference
  latency and shows lower loss than the CPU NumPy Transformer baseline
- on the dialogue smoke scaling run, LSL also trains faster than the baseline
- the same comparison artifacts keep the tradeoff visible: LSL can be larger in
  model size on some runs even when it is faster at inference

Treat these as benchmark snapshots, not universal dominance claims.

## Current Strengths

- strong bounded sparse memory behavior
- CPU-native sparse kernel support
- local online update path
- structured benchmark/report pipeline
- honest separation of proxy energy from real sensor evidence

## Current Limitations

- open-ended generation is still simpler than frontier LLMs
- large-scale semantic abstraction remains incomplete
- real watt claims require actual sensor evidence
- same-scale Transformer comparisons still need broader sweeps
- multilingual/tokenizer coverage is improving but not solved in every corner

## Responsible Use Notes

- Do not present proxy energy as measured watts.
- Do not present smoke benchmarks as full-scale dominance claims.
- Treat benchmark outputs as experiment records, not universal guarantees.
- Re-run the strict gate after any native or core-memory change.
