"""Phase 6 competitive efficiency check against small CPU baselines."""
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase4.baseline_transformer import TinyTransformer
from benchmarks.phase5.benchmark_long_context_real_corpus import (
    eval_adaptation_retention,
    eval_long_context,
    eval_ngram,
    train_long_context,
    train_ngram,
)
from benchmarks.phase5.benchmark_long_context_real_corpus import read_text, tokenize_splits


def transformer_latency(vocab_size: int, d_model: int, context: int, tokens: List[int], iterations: int, seed: int) -> float:
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        d_ff=d_model * 4,
        n_layers=2,
        max_seq_len=max(context, 512),
        seed=seed,
    )
    times = []
    usable = max(1, min(iterations, len(tokens) - context - 1))
    for i in range(usable):
        t0 = time.perf_counter_ns()
        model.forward(tokens[i:i + context])
        times.append((time.perf_counter_ns() - t0) / 1000.0)
    times.sort()
    return float(times[len(times) // 2]) if times else 0.0


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    train_text, eval_text, corpus_path = read_text(args)
    tokenizer, train_tokens, eval_tokens = tokenize_splits(train_text, eval_text, args)
    train_tokens = train_tokens[: args.max_train_tokens]
    eval_tokens = eval_tokens[: args.max_eval_tokens]
    args.vocab_size = tokenizer.vocab_size

    memory, observe_us, memory_peak_bytes = train_long_context(train_tokens, args)
    lc = eval_long_context(memory, eval_tokens, args.vocab_size)
    ngram = train_ngram(train_tokens, args.vocab_size, sparse=False)
    sparse_ngram = train_ngram(train_tokens, args.vocab_size, sparse=True)
    ng = eval_ngram(ngram, eval_tokens)
    sng = eval_ngram(sparse_ngram, eval_tokens)
    best_baseline_loss = min(ng["loss"], sng["loss"])
    quality_ratio = lc["loss"] / max(best_baseline_loss, 1e-9)

    dense_us = transformer_latency(
        args.vocab_size,
        args.d_model,
        args.latency_context,
        train_tokens + eval_tokens,
        args.latency_iterations,
        args.seed,
    )
    latency_speedup = dense_us / max(lc["p50_latency_us"], 1e-9)
    dense_ram = float(args.latency_context * args.latency_context * args.d_model * 4)
    ram_speedup = dense_ram / max(1.0, float(memory_peak_bytes))

    split = min(args.adaptation_tokens, len(train_tokens), len(eval_tokens))
    adaptation = eval_adaptation_retention(memory, train_tokens[:split], eval_tokens[:split], args)
    energy_proxy_speedup = latency_speedup * max(1.0, ram_speedup)
    checks = {
        "quality": quality_ratio <= args.quality_ratio_target,
        "latency": latency_speedup >= args.latency_speedup_target,
        "ram": ram_speedup >= args.ram_speedup_target,
        "adaptation": adaptation["adaptation_speedup"] >= args.adaptation_speedup_target,
        "retention": adaptation["retention"] >= args.retention_target,
        "energy_proxy": energy_proxy_speedup >= args.energy_proxy_target,
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "corpus_path": corpus_path,
        "train_tokens": len(train_tokens),
        "eval_tokens": len(eval_tokens),
        "vocab_size": args.vocab_size,
        "long_context": lc,
        "ngram": ng,
        "sparse_ngram": sng,
        "quality_ratio": float(quality_ratio),
        "transformer_p50_us": float(dense_us),
        "latency_speedup": float(latency_speedup),
        "memory_peak_bytes": float(memory_peak_bytes),
        "ram_speedup": float(ram_speedup),
        "observe_us_per_token": float(observe_us),
        "adaptation": adaptation,
        "energy_proxy_speedup": float(energy_proxy_speedup),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "tinystories_full", "wikitext2", "custom"], default="tinystories")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--wikitext-cache-dir", type=str, default=None)
    parser.add_argument("--tinystories-cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=2500)
    parser.add_argument("--tokenizer", choices=["word", "subword"], default="subword")
    parser.add_argument("--tokenizer-train-chars", type=int, default=120000)
    parser.add_argument("--max-train-chars", type=int, default=240000)
    parser.add_argument("--max-eval-chars", type=int, default=80000)
    parser.add_argument("--subword-max-merges", type=int, default=500)
    parser.add_argument("--subword-min-pair-count", type=int, default=3)
    parser.add_argument("--max-train-tokens", type=int, default=6000)
    parser.add_argument("--max-eval-tokens", type=int, default=1500)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--context-width", type=int, default=6)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--target-cap", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--context-train", type=int, default=32)
    parser.add_argument("--latency-context", type=int, default=256)
    parser.add_argument("--latency-iterations", type=int, default=8)
    parser.add_argument("--adaptation-tokens", type=int, default=1024)
    parser.add_argument("--baseline-lr", type=float, default=0.15)
    parser.add_argument("--retrain-epochs", type=int, default=50)
    parser.add_argument("--quality-ratio-target", type=float, default=1.15)
    parser.add_argument("--latency-speedup-target", type=float, default=20.0)
    parser.add_argument("--ram-speedup-target", type=float, default=5.0)
    parser.add_argument("--adaptation-speedup-target", type=float, default=50.0)
    parser.add_argument("--retention-target", type=float, default=0.95)
    parser.add_argument("--energy-proxy-target", type=float, default=20.0)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 6: Competitive Efficiency")
    print("=" * 88)
    print(f"Corpus:                {result['corpus_path']}")
    print(f"Quality ratio:         {result['quality_ratio']:.3f}x (target <={args.quality_ratio_target:.2f}x)")
    print(f"Latency speedup:       {result['latency_speedup']:.2f}x (target >={args.latency_speedup_target:.1f}x)")
    print(f"RAM speedup proxy:     {result['ram_speedup']:.2f}x (target >={args.ram_speedup_target:.1f}x)")
    print(f"Adaptation speedup:    {result['adaptation']['adaptation_speedup']:.2f}x")
    print(f"Retention:             {result['adaptation']['retention']:.2%}")
    print(f"Energy proxy speedup:  {result['energy_proxy_speedup']:.2f}x")
    print(f"Overall status:        {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase6_competitive", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
