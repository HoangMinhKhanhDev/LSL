# LSL Training Guide

This guide is the practical companion to the model card and technical report.
It focuses on one thing: how to train, resume, scale, and validate LSL without
having to remember which script does what.

## Profiles

- `native_fast` is the throughput-first profile. Use it when you want the
  fastest CPU path and the native sparse core.
- `bio_native` and `continual` are the full online-learning profiles. They
  engage predictive coding, SDR, cortical columns, hippocampal replay,
  neuromodulation, and dendritic computation in one stream.
- `continual` is an alias for `bio_native`. It exists so the intent is obvious
  when you are resuming training on new data.

## Quick Start

Train a fresh checkpoint on a small corpus:

```bash
python lsl_cli.py train --dataset tinystories --lsl-profile native_fast
```

Train on your own corpus and save a new checkpoint:

```bash
python lsl_cli.py train ^
  --dataset custom ^
  --corpus-path F:\data\my_corpus.txt ^
  --lsl-profile continual ^
  --max-tokens 200000 ^
  --checkpoint checkpoints\my_domain.json
```

Resume from an existing checkpoint, learn from new data, and save a new model
copy:

```bash
python lsl_cli.py train ^
  --dataset custom ^
  --corpus-path F:\data\new_domain.txt ^
  --load-checkpoint checkpoints\base.json ^
  --lsl-profile continual ^
  --max-tokens 200000 ^
  --checkpoint checkpoints\base_plus_domain.json
```

## Large-Scale Training

For preset large-corpus runs, use the corpus trainer:

```bash
python train_full_corpora.py --corpus tinystories_1m --lsl-profile native_fast
python train_full_corpora.py --corpus tinystories_10m --lsl-profile native_fast
python train_full_corpora.py --corpus wikitext2_full --lsl-profile continual
```

That script now also supports:

- `--load-checkpoint` for continual updates
- reproducible checkpoint naming per corpus and seed
- result metadata in `results/`

For benchmarked single-corpus runs, use:

```bash
python benchmarks/train_lsl_corpus.py --dataset tinystories --max-tokens 1000000 --lsl-profile native_fast
python benchmarks/train_lsl_corpus.py --dataset dialogue_small --max-tokens 50000 --lsl-profile continual
```

## Three-Stage Curriculum

When you want the simplest path from base model to continual learner, use the
curriculum runner. It trains in three explicit stages:

1. grammar bootstrap on a clean starter corpus,
2. broadening on a multi-corpus mix,
3. continual adaptation on your target domain or a smoke fallback corpus.

The default command is:

```bash
python train_curriculum.py
```

The same runner is also available through the unified CLI:

```bash
python lsl_cli.py curriculum
```

The default stage split is:

- bootstrap: TinyStories
- broaden: WikiText-2, Vietnamese seed corpus, dialogue seed corpus
- adapt: `dialogue_small` unless you override it with your own corpus

To point stage 3 at your own data:

```bash
python train_curriculum.py ^
  --load-checkpoint checkpoints\lsl_tinystories.json ^
  --adapt-dataset custom ^
  --adapt-corpus-path F:\data\my_domain.txt ^
  --final-checkpoint checkpoints\lsl_curriculum_final.json
```

For a quick verification run, add `--smoke`. That shrinks token budgets and is
the path used by the smoke test in `run_all.py`.

If you want the "lighter broaden, larger stage 3" experiment we just ran, use:

```bash
python train_curriculum.py --preset grammar_safe --load-checkpoint checkpoints\lsl_tinystories.json
```

That preset keeps stage 2 lighter, switches stage 3 to WikiText-2, and raises
the stage 3 token budget so you can compare grammar retention against a more
expansive adaptation pass.

If you want the first scale-readiness gate, use:

```bash
python train_curriculum.py --preset scale_ready --load-checkpoint checkpoints\lsl_tinystories.json
```

That preset keeps the broaden stage light, pushes stage 3 to a larger
WikiText-2 run, and adds an OOD suite over public corpora so you can read
retention, grammar, throughput, and out-of-domain loss in one report. The
scale lane now splits stage 3 into smaller continual-adaptation sub-stages and
evaluates the promotion gate after each stage. Until the retention / grammar /
throughput / OOD gate passes, this lane exits non-zero and should be treated as
"not yet promotable" rather than a green scale signal.

If you need a different stage-3 shape, pass `--adapt-token-splits` with a
comma-separated list of chunk sizes. That keeps the same resume/checkpoint
flow while making the adaptation lane easier to inspect.

## Scaling and Significance

Use the scaling-law runner when you want token-budget and seed sweeps:

```bash
python benchmarks/phase1/run_scaling_law.py --datasets tinystories,wikitext2 --token-budgets 1000000,10000000 --seeds 42,43,44,45,46
```

Use the seed sweep runner when you want a compact statistical check:

```bash
python benchmarks/seed_sweep.py --dataset tinystories --seeds 42 43 44 45 46 --lsl-profile native_fast
```

Use the competitive runner when you want a fair CPU baseline against a trainable
NumPy Transformer:

```bash
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset tinystories --tokens 100000 --lsl-profile native_fast
python benchmarks/competitive/run_lsl_vs_transformer.py --dataset wikitext2 --tokens 100000 --lsl-profile continual
```

## What Changed Recently

The latest training-oriented changes are:

- a `continual` profile alias for the full bio-native path
- `--load-checkpoint` support in the main train entrypoints
- token-id fast paths in the bio-native sidecars
- stride and warmup controls for heavy biological sidecars
- consistent result metadata in train runs and benchmark runs

That means you can now do a clean loop:

1. train a base model,
2. save a checkpoint,
3. resume from that checkpoint on new data,
4. keep the same tokenizer and memory state,
5. and write a new checkpoint for the next stage.

## Output Artifacts

Training and benchmark runs write:

- checkpoints under `checkpoints/`
- result JSON under `results/`
- HTML summaries with `lsl_report.py`
- benchmark snapshots for comparisons and claims

## Validation Order

When changing training code, the safe order is:

1. `python test_lsl_core_model.py`
2. `python test_lsl_limitations_improvements.py`
3. `python run_all.py`
4. one small corpus train run
5. one resume-from-checkpoint run
6. one competitive benchmark run

## Where To Read Next

- [Model Card](LSL_MODEL_CARD.md)
- [Technical Report](LSL_TECHNICAL_REPORT.md)
- [Comparisons](LSL_COMPARISONS.md)
- [Project Layout](PROJECT_LAYOUT.md)
