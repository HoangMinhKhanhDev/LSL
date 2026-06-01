"""Train LSL on full corpora: TinyStories 1M/10M tokens and WikiText-2 full profile."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lsl import DatasetConfig, DatasetLoader, LSLCoreModel, RUNTIME_PROFILE_CHOICES
from lsl.results_storage import ResultsStorage, RunResult, RunMetadata, RunConfig, RunMetrics


ROOT = Path(__file__).parent.absolute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        choices=["tinystories_1m", "tinystories_10m", "wikitext2_full"],
        required=True,
        help="Which corpus to train on",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument(
        "--lsl-profile",
        choices=list(RUNTIME_PROFILE_CHOICES),
        default="native_fast",
    )
    parser.add_argument("--load-checkpoint", type=str, default=None, help="resume training from an existing checkpoint before saving the new output checkpoint")
    parser.add_argument("--tokenizer-train-chars", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def get_corpus_config(corpus: str) -> dict:
    """Get corpus-specific configuration."""
    configs = {
        "tinystories_1m": {
            "dataset": "tinystories",
            "split": "train",
            "max_tokens": 1_000_000,
            "max_chars": None,
            "repeat": True,
        },
        "tinystories_10m": {
            "dataset": "tinystories",
            "split": "train",
            "max_tokens": 10_000_000,
            "max_chars": None,
            "repeat": True,
        },
        "wikitext2_full": {
            "dataset": "wikitext2",
            "split": "train",
            "max_tokens": None,  # Full corpus
            "max_chars": None,
            "repeat": False,
        },
    }
    return configs[corpus]


def train_corpus(args: argparse.Namespace) -> RunResult:
    """Train on a single corpus and return result."""
    corpus_config = get_corpus_config(args.corpus)
    
    # Load dataset
    loader = DatasetLoader(str(ROOT))
    text = loader.load_text(
        DatasetConfig(
            name=corpus_config["dataset"],
            split=corpus_config["split"],
            max_chars=corpus_config["max_chars"],
            repeat=corpus_config["repeat"],
            seed=args.seed,
        )
    )
    corpus_path = loader.resolve_path(corpus_config["dataset"], corpus_config["split"])
    
    # Initialize model
    if args.load_checkpoint:
        model = LSLCoreModel.load(args.load_checkpoint)
        model.set_runtime_profile(args.lsl_profile)
        loaded_checkpoint = os.path.abspath(args.load_checkpoint)
    else:
        model = LSLCoreModel(
            vocab_size=args.vocab_size,
            seed=args.seed,
            candidate_cap=args.candidate_cap,
            runtime_profile=args.lsl_profile,
        )
        loaded_checkpoint = None
    
    # Train
    started = time.perf_counter()
    metrics = model.train_stream(
        [text],
        tokenizer_text_chars=args.tokenizer_train_chars,
        max_tokens=corpus_config["max_tokens"],
    )
    elapsed = time.perf_counter() - started
    
    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"lsl_{args.corpus}_seed{args.seed}.json"
    model.save(str(checkpoint_path))
    
    # Generate sample
    sample_prompt = " ".join(str(text).split()[:3]) or "the little girl"
    sample = model.generate(sample_prompt, max_new_tokens=48)
    
    # Get diagnostics
    diag = model.diagnostics()
    
    # Create run config
    run_config = RunConfig(
        dataset=corpus_config["dataset"],
        vocab_size=args.vocab_size,
        max_tokens=corpus_config["max_tokens"] or -1,
        lsl_profile=args.lsl_profile,
        candidate_cap=args.candidate_cap,
        tokenizer="subword",
        extra={
            "split": corpus_config["split"],
            "corpus": args.corpus,
            "tokenizer_train_chars": args.tokenizer_train_chars,
            "loaded_checkpoint": loaded_checkpoint,
        },
    )
    
    # Create run metrics
    run_metrics = RunMetrics(
        tokens=float(metrics["tokens"]),
        elapsed_seconds=float(elapsed),
        us_per_token=float(metrics["us_per_token"]),
        tokens_per_second=float(metrics["tokens"] / max(elapsed, 1e-12)),
        vocab_size=model.vocab_size,
        seen_tokens=float(diag.get("seen_tokens", 0.0)),
        extra={
            "wall_seconds": float(elapsed),
        },
    )
    
    # Create run metadata
    storage = ResultsStorage(args.results_dir)
    metadata = storage.create_metadata(
        benchmark_name=f"train_{args.corpus}",
        seed=args.seed,
    )
    
    # Create run result
    result = RunResult(
        metadata=metadata,
        config=run_config,
        metrics=run_metrics,
        success=True,
        sample_prompt=sample_prompt,
        sample=sample,
        checkpoint_path=str(checkpoint_path.absolute()),
        corpus_path=str(corpus_path),
    )
    
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    
    args = parse_args()
    
    print(f"Training LSL on {args.corpus}")
    print("=" * 88)
    
    # Train
    result = train_corpus(args)
    
    # Save result
    storage = ResultsStorage(args.results_dir)
    run_id = storage.save_result(result)
    
    # Print summary
    print(f"Dataset:          {result.config.dataset}")
    print(f"Corpus:           {result.corpus_path}")
    print(f"Checkpoint:       {result.checkpoint_path}")
    loaded_checkpoint = result.config.extra.get("loaded_checkpoint")
    if loaded_checkpoint:
        print(f"Loaded:           {loaded_checkpoint}")
    print(f"Tokens:           {int(result.metrics.tokens):,}")
    print(f"Train tok/s:      {result.metrics.tokens_per_second:.2f}")
    print(f"us/token:         {result.metrics.us_per_token:.2f}")
    print(f"Vocab:            {result.metrics.vocab_size:,}")
    print(f"Profile:          {result.config.lsl_profile}")
    print(f"Sample:           {result.sample[:240]}")
    print(f"Result ID:        {run_id}")
    print(f"Result JSON:      {storage.results_dir / f'{run_id}.json'}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
