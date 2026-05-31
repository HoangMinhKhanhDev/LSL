"""Train one LSLCoreModel checkpoint on a real corpus."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lsl import DatasetConfig, DatasetLoader, LSLCoreModel, write_result


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read_text(args: argparse.Namespace) -> tuple[str, str]:
    loader = DatasetLoader(ROOT)
    if args.dataset == "custom":
        name = args.corpus_path or ""
        if not name:
            raise FileNotFoundError("--corpus-path is required for custom dataset")
    else:
        name = args.dataset
    text = loader.load_text(DatasetConfig(name=name, split=args.split, max_chars=args.max_chars, repeat=args.repeat_small))
    path = loader.resolve_path(name, args.split)
    return text, os.path.abspath(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "wikitext2", "vietnamese_small", "dialogue_small", "custom"],
        default="tinystories",
    )
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "validation", "val", "test"], default="train")
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--max-chars", type=int, default=5000000)
    parser.add_argument("--tokenizer-train-chars", type=int, default=250000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--lsl-profile", choices=["full", "native_long_context", "native_fast", "bio_native"], default="native_fast")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--repeat-small", action="store_true", help="repeat tiny corpora until max chars/tokens are reached")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    text, path = read_text(args)
    checkpoint = args.checkpoint or os.path.join("checkpoints", f"lsl_{args.dataset}.json")
    model = LSLCoreModel(
        vocab_size=args.vocab_size,
        seed=args.seed,
        candidate_cap=args.candidate_cap,
        runtime_profile=args.lsl_profile,
    )
    started = time.perf_counter()
    metrics = model.train_stream([text], tokenizer_text_chars=args.tokenizer_train_chars, max_tokens=args.max_tokens)
    elapsed = time.perf_counter() - started
    model.save(checkpoint)
    sample_prompt = " ".join(str(text).split()[:3]) or "the little girl"
    sample = model.generate(sample_prompt, max_new_tokens=48)
    diag = model.diagnostics()
    payload = {
        "benchmark": "train_lsl_corpus",
        "dataset": args.dataset,
        "split": args.split,
        "corpus_path": path,
        "checkpoint": os.path.abspath(checkpoint),
        "success": True,
        "metrics": {
            **metrics,
            "wall_seconds": float(elapsed),
            "tokens_per_second": float(metrics["tokens"] / max(elapsed, 1e-12)),
            "vocab_size": model.vocab_size,
            "seen_tokens": diag.get("seen_tokens", 0.0),
            "lsl_profile": args.lsl_profile,
        },
        "sample_prompt": sample_prompt,
        "sample": sample,
    }
    config = vars(args)
    print("Train LSLCoreModel Corpus")
    print("=" * 88)
    print(f"Dataset:          {args.dataset}")
    print(f"Corpus:           {path}")
    print(f"Checkpoint:       {os.path.abspath(checkpoint)}")
    print(f"Tokens:           {int(metrics['tokens']):,}")
    print(f"Train tok/s:      {payload['metrics']['tokens_per_second']:.2f}")
    print(f"us/token:         {metrics['us_per_token']:.2f}")
    print(f"Vocab:            {model.vocab_size:,}")
    print(f"Profile:          {args.lsl_profile}")
    print(f"Sample:           {sample[:240]}")
    output = write_result(
        payload,
        benchmark="train_lsl_corpus",
        dataset=args.dataset,
        seed=args.seed,
        config=config,
        output_path=args.json_output,
        results_root=args.results_root,
    )
    print(f"Result JSON:      {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
