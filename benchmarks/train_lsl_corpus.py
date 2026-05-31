"""Train one LSLCoreModel checkpoint on a real corpus."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lsl import LSLCoreModel


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TINYSTORIES = os.path.join(ROOT, "benchmarks", "data", "tinystories", "TinyStoriesV2-GPT4-valid.txt")
TINYSTORIES_SUBSET = os.path.join(ROOT, "benchmarks", "phase4", "tinystories_subset.txt")
WIKITEXT = os.path.join(ROOT, "benchmarks", "data", "wikitext-2-raw-v1", "wiki.train.raw.txt")


def read_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.dataset == "custom":
        if not args.corpus_path:
            raise FileNotFoundError("--corpus-path is required for custom dataset")
        path = args.corpus_path
    elif args.dataset == "wikitext2":
        path = WIKITEXT
    else:
        path = TINYSTORIES if os.path.exists(TINYSTORIES) else TINYSTORIES_SUBSET
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(args.max_chars)
    return text, os.path.abspath(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "wikitext2", "custom"], default="tinystories")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--max-chars", type=int, default=5000000)
    parser.add_argument("--tokenizer-train-chars", type=int, default=250000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--lsl-profile", choices=["full", "native_long_context", "native_fast", "bio_native"], default="native_fast")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
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
    sample_prompt = "the little girl"
    sample = model.generate(sample_prompt, max_new_tokens=48)
    diag = model.diagnostics()
    payload = {
        "benchmark": "train_lsl_corpus",
        "dataset": args.dataset,
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
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
