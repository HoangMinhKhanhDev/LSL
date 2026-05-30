"""Phase 4/5 long-context retrieval without full-history scanning."""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import SparseKeyValueMemory


def target_for_horizon(context_length: int) -> float:
    if context_length <= 1000:
        return 0.95
    if context_length <= 4000:
        return 0.90
    if context_length <= 16000:
        return 0.85
    if context_length <= 64000:
        return 0.75
    return 0.60


def build_memory(context_length: int, args: argparse.Namespace) -> Tuple[SparseKeyValueMemory, Dict[int, int]]:
    memory = SparseKeyValueMemory(
        capacity=context_length + 8,
        sdr_dim=args.sdr_dim,
        sparsity=args.sparsity,
        candidate_cap=args.candidate_cap,
        bucket_probe_bits=args.bucket_probe_bits,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed + 100_003 + int(context_length))
    values = rng.choice(
        np.arange(10_000_000, 10_000_000 + max(context_length * 4, 1024), dtype=np.int64),
        size=context_length,
        replace=False,
    )
    expected: Dict[int, int] = {}
    for key in range(context_length):
        value = int(values[key]) if args.random_values else key + 10_000_000
        expected[int(key)] = value
        memory.add(key, value, vocab_size=args.vocab_size)
    return memory, expected


def evaluate_horizon(context_length: int, args: argparse.Namespace) -> Dict[str, float]:
    rng = np.random.default_rng(args.seed + int(context_length))
    memory, expected = build_memory(context_length, args)
    queries = rng.choice(context_length, size=args.num_trials, replace=args.num_trials > context_length)

    correct = 0
    candidates = []
    full_scans = 0
    times_us = []
    for query in queries:
        t0 = time.perf_counter_ns()
        value, diag = None, {}
        for _ in range(args.lookup_repeats):
            value, diag = memory.lookup(
                int(query),
                vocab_size=args.vocab_size,
                return_diagnostics=True,
                allow_direct_lookup=args.mode != "bucket-only",
            )
        times_us.append((time.perf_counter_ns() - t0) / 1000.0 / max(1, args.lookup_repeats))
        correct += int(value == expected[int(query)])
        candidates.append(diag["candidate_count"])
        full_scans += int(diag["full_scan"])

    recall = correct / max(1, len(queries))
    absent_false_positive = 0.0
    if args.absent_queries > 0:
        false_hits = 0
        for query in range(context_length, context_length + args.absent_queries):
            value, diag = memory.lookup(
                int(query),
                vocab_size=args.vocab_size,
                return_diagnostics=True,
                allow_direct_lookup=args.mode != "bucket-only",
            )
            false_hits += int(value is not None and diag["best_score"] >= args.min_match_bits)
            full_scans += int(diag["full_scan"])
        absent_false_positive = false_hits / max(1, args.absent_queries)
    target = target_for_horizon(context_length)
    p50_us = float(np.percentile(times_us, 50))
    p95_us = float(np.percentile(times_us, 95))
    candidate_p95 = float(np.percentile(candidates, 95)) if candidates else 0.0
    ok = (
        recall >= target
        and full_scans == 0
        and candidate_p95 <= args.candidate_cap
        and absent_false_positive <= args.absent_fp_target
    )

    print(f"\nHorizon {context_length:,}")
    print(f"  recall:        {recall:.2%} (target >= {target:.0%})")
    print(f"  p50 lookup:    {p50_us:.2f} us")
    print(f"  p95 lookup:    {p95_us:.2f} us")
    print(f"  p95 candidates:{candidate_p95:.1f} (cap {args.candidate_cap})")
    if args.absent_queries > 0:
        print(f"  absent FP:     {absent_false_positive:.2%} (target <= {args.absent_fp_target:.2%})")
    print(f"  full scans:    {full_scans}")
    print(f"  status:        {'PASS' if ok else 'FAIL'}")

    return {
        "context_length": float(context_length),
        "recall": float(recall),
        "target": float(target),
        "p50_us": p50_us,
        "p95_us": p95_us,
        "candidate_p95": candidate_p95,
        "candidate_cap": float(args.candidate_cap),
        "absent_false_positive": float(absent_false_positive),
        "absent_fp_target": float(args.absent_fp_target),
        "full_scans": float(full_scans),
        "success": float(ok),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1000, 4000, 16000, 64000, 128000])
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--num-trials", type=int, default=32)
    parser.add_argument("--sdr-dim", type=int, default=2048)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--bucket-probe-bits", type=int, default=8)
    parser.add_argument("--lookup-repeats", type=int, default=16)
    parser.add_argument("--latency-ratio-target", type=float, default=1.5)
    parser.add_argument("--mode", choices=["exact", "bucket-only"], default="exact")
    parser.add_argument("--random-values", action="store_true")
    parser.add_argument("--absent-queries", type=int, default=0)
    parser.add_argument("--absent-fp-target", type=float, default=0.01)
    parser.add_argument("--min-match-bits", type=int, default=10)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Phase 4/5: Long-Context Retrieval")
    print("Sparse key-value memory with bounded candidate retrieval")
    print("No full-context scan is allowed in lookup.")
    print(f"Mode: {args.mode}; random_values={args.random_values}; absent_queries={args.absent_queries}")

    results: List[Dict[str, float]] = [evaluate_horizon(ctx, args) for ctx in args.context_lengths]

    by_ctx = {int(r["context_length"]): r for r in results}
    latency_ok = True
    if 1000 in by_ctx and 64000 in by_ctx:
        ratio = by_ctx[64000]["p50_us"] / max(by_ctx[1000]["p50_us"], 1e-9)
        latency_ok = ratio <= args.latency_ratio_target
    else:
        ratio = 1.0

    ok = all(bool(r["success"]) for r in results) and latency_ok

    print("\n" + "=" * 88)
    print("LONG-CONTEXT SUMMARY")
    print("=" * 88)
    print(f"{'Horizon':>10} {'Recall':>10} {'Target':>10} {'p50_us':>10} {'Cand95':>10} {'Status':>8}")
    print("-" * 88)
    for result in results:
        print(
            f"{int(result['context_length']):>10,} "
            f"{100 * result['recall']:>9.1f}% "
            f"{100 * result['target']:>9.1f}% "
            f"{result['p50_us']:>10.2f} "
            f"{result['candidate_p95']:>10.1f} "
            f"{'PASS' if result['success'] else 'FAIL':>8}"
        )
    print("-" * 88)
    print(f"64k/1k latency ratio: {ratio:.2f}x (target <= {args.latency_ratio_target:.2f}x)")
    print(f"Mechanism #9 Long-Context Retrieval: {'PASS' if ok else 'FAIL'}")

    payload = {
        "benchmark": "long_context",
        "success": bool(ok),
        "latency_ratio_64k_vs_1k": float(ratio),
        "latency_ratio_target": float(args.latency_ratio_target),
        "results": results,
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
