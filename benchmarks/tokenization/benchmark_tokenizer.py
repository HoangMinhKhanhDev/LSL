"""Benchmark LSL tokenizer quality, UNK rate, cache, and speed by corpus."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import DatasetLoader, SimpleSubwordTokenizer, SimpleWordTokenizer, write_result  # noqa: E402
from lsl.text_normalization import looks_vietnamese  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def timed_call(fn):
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    return value, elapsed


def build_tokenizer(kind: str, vocab_size: int, text: str, cache_dir: str | None, dataset: str):
    kind = str(kind).strip().lower()
    vietnamese = kind in {"subword_vi", "vietnamese"} or (kind == "auto" and looks_vietnamese(text))
    if kind == "word":
        tok = SimpleWordTokenizer(vocab_size=vocab_size)
    else:
        tok = SimpleSubwordTokenizer(
            vocab_size=vocab_size,
            max_merges=800,
            min_pair_count=2,
            vietnamese_normalization=vietnamese,
            normalization_form="NFC",
            byte_fallback=True,
            cache_dir=os.path.join(cache_dir, dataset, kind) if cache_dir else None,
        )
    _, train_seconds = timed_call(lambda: tok.build_vocab(text))
    return tok, train_seconds


def tokenizer_name(tokenizer) -> str:
    if isinstance(tokenizer, SimpleWordTokenizer):
        return "word"
    if getattr(tokenizer, "vietnamese_normalization", False):
        return "subword_vi"
    return "subword"


def bench_one(args: argparse.Namespace, loader: DatasetLoader, dataset: str, kind: str) -> Dict[str, object]:
    splits = loader.load_text_splits(
        dataset,
        max_train_chars=int(args.train_chars),
        max_eval_chars=int(args.eval_chars),
        seed=int(args.seed),
    )
    train_text = splits.train[: int(args.train_chars)]
    eval_text = (splits.validation or splits.test or splits.train)[: int(args.eval_chars)]
    tokenizer, train_seconds = build_tokenizer(kind, int(args.vocab_size), train_text, args.cache_dir, dataset)
    persistent_cache_entries = 0.0
    if hasattr(tokenizer, "cache_stats"):
        persistent_cache_entries = float(tokenizer.cache_stats().get("encode_entries", 0.0))
    if hasattr(tokenizer, "_encode_text_cache"):
        tokenizer._encode_text_cache.clear()
    if hasattr(tokenizer, "_decode_cache"):
        tokenizer._decode_cache.clear()

    ids, encode_seconds = timed_call(lambda: tokenizer.encode(eval_text, max_tokens=int(args.max_tokens)))
    decoded, decode_seconds = timed_call(lambda: tokenizer.decode(ids))
    _, encode_cached_seconds = timed_call(lambda: tokenizer.encode(eval_text, max_tokens=int(args.max_tokens)))
    _, decode_cached_seconds = timed_call(lambda: tokenizer.decode(ids))

    cache_path = None
    if hasattr(tokenizer, "save_cache"):
        cache_path = tokenizer.save_cache()
    tokenizer_path = None
    if dataset == "vietnamese_small" or tokenizer_name(tokenizer) == "subword_vi":
        tokenizer_path = os.path.abspath(os.path.join(args.tokenizer_output_dir, f"{dataset}_{tokenizer_name(tokenizer)}.json"))
        if hasattr(tokenizer, "save"):
            tokenizer.save(tokenizer_path)

    unk_rate = float(tokenizer.unk_rate(ids)) if hasattr(tokenizer, "unk_rate") else 0.0
    cache_stats = tokenizer.cache_stats() if hasattr(tokenizer, "cache_stats") else {}
    return {
        "dataset": dataset,
        "language": splits.language,
        "requested_tokenizer": kind,
        "tokenizer": tokenizer_name(tokenizer),
        "requested_vocab_size": int(args.vocab_size),
        "vocab_size": int(getattr(tokenizer, "vocab_size", args.vocab_size)),
        "train_chars": int(len(train_text)),
        "eval_chars": int(len(eval_text)),
        "tokens": int(len(ids)),
        "unk_rate": unk_rate,
        "train_seconds": float(train_seconds),
        "encode_seconds": float(encode_seconds),
        "decode_seconds": float(decode_seconds),
        "encode_tokens_per_second": float(len(ids) / max(encode_seconds, 1e-12)),
        "decode_tokens_per_second": float(len(ids) / max(decode_seconds, 1e-12)),
        "cached_encode_tokens_per_second": float(len(ids) / max(encode_cached_seconds, 1e-12)),
        "cached_decode_tokens_per_second": float(len(ids) / max(decode_cached_seconds, 1e-12)),
        "roundtrip_chars": int(len(decoded)),
        "persistent_cache_entries_loaded": persistent_cache_entries,
        "cache": cache_stats,
        "cache_path": cache_path,
        "tokenizer_path": tokenizer_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, default="tinystories,wikitext2,vietnamese_small,dialogue_small")
    parser.add_argument("--tokenizers", type=str, default="subword,subword_vi")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--train-chars", type=int, default=250000)
    parser.add_argument("--eval-chars", type=int, default=120000)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--cache-dir", type=str, default=os.path.join("results", "tokenizer_cache"))
    parser.add_argument("--tokenizer-output-dir", type=str, default=os.path.join("results", "tokenizers"))
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    loader = DatasetLoader(ROOT)
    rows = []
    for dataset in parse_csv(args.datasets):
        for kind in parse_csv(args.tokenizers):
            if kind == "subword_vi" and dataset != "vietnamese_small":
                continue
            row = bench_one(args, loader, dataset, kind)
            rows.append(row)
            print(
                f"{dataset:<18} {row['tokenizer']:<10} "
                f"unk={row['unk_rate']:.4f} encode_tps={row['encode_tokens_per_second']:.1f} "
                f"cached={row['cached_encode_tokens_per_second']:.1f}"
            )
    payload = {
        "benchmark": "tokenizer_quality_speed",
        "success": True,
        "config": vars(args),
        "runs": rows,
    }
    output = write_result(
        payload,
        benchmark="tokenizer_quality_speed",
        dataset="_".join(parse_csv(args.datasets)[:3]),
        seed=int(args.seed),
        config=vars(args),
        output_path=args.json_output,
        results_root=args.results_root,
    )
    print(f"Result JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
