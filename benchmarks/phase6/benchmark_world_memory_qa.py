"""Phase 6 world-memory QA and evidence benchmark."""
import argparse
import json
import os
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import WorldMemory


FIELDS = ["launch code", "handler", "region", "status", "backup key", "route"]


def make_fact(i: int) -> Tuple[str, str, str, str]:
    entity = f"entity-{i:07d}"
    field = FIELDS[i % len(FIELDS)]
    value = f"value_{(i * 7919 + 17) % 1000003}"
    text = f"The {field} for {entity} is {value}. Archive marker {i:07d} closes the record."
    return entity, field, value, text


def build_world(size: int, args: argparse.Namespace) -> Tuple[WorldMemory, List[Tuple[str, str, str]], float, int]:
    memory = WorldMemory(
        capacity=max(size + 128, args.capacity),
        sdr_dim=args.sdr_dim,
        sparsity=args.sparsity,
        candidate_cap=args.candidate_cap,
        seed=args.seed,
    )
    facts = []
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    for i in range(size):
        entity, field, value, text = make_fact(i)
        memory.observe_chunk(text, source=f"synthetic-public:{i}")
        if i % max(1, size // args.query_count) == 0:
            facts.append((entity, field, value))
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return memory, facts[: args.query_count], float(elapsed_us / max(1, size)), int(peak)


def evaluate_size(size: int, args: argparse.Namespace) -> Dict[str, float]:
    memory, facts, ingest_us, peak_bytes = build_world(size, args)
    correct = 0
    faithful = 0
    full_scan = 0
    candidate_counts = []
    latencies = []
    questions = [(f"What is the {field} for {entity}?", value) for entity, field, value in facts]
    for question, _ in questions:
        memory.answer(question)
    for question, value in questions:
        t0 = time.perf_counter_ns()
        answer = memory.answer(question)
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
        correct += int(answer.answer == value)
        faithful += int(answer.evidence is not None and value in answer.evidence.text.lower())
        full_scan += int(answer.diagnostics.get("full_scan", 0.0) > 0.0)
        candidate_counts.append(answer.diagnostics.get("candidate_count", 0.0))

    recall = correct / max(1, len(facts))
    faithfulness = faithful / max(1, len(facts))
    p50_us = float(np.percentile(latencies, 50)) if latencies else 0.0
    attention_bytes = float(max(1, size) * max(1, size) * 96 * 4)
    ram_speedup = attention_bytes / max(1.0, float(peak_bytes))
    return {
        "size": float(size),
        "recall": float(recall),
        "faithfulness": float(faithfulness),
        "p50_latency_us": p50_us,
        "ingest_us_per_chunk": float(ingest_us),
        "peak_bytes": float(peak_bytes),
        "ram_speedup_proxy": float(ram_speedup),
        "full_scan_count": float(full_scan),
        "mean_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }


def target_for_size(size: int) -> float:
    if size <= 1000:
        return 0.95
    if size <= 16000:
        return 0.85
    return 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 16000, 128000])
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--sdr-dim", type=int, default=4096)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--latency-ratio-target", type=float, default=1.5)
    parser.add_argument("--ram-speedup-target", type=float, default=5.0)
    parser.add_argument("--faithfulness-target", type=float, default=0.90)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [evaluate_size(int(size), args) for size in sorted(args.sizes, reverse=True)]
    rows = sorted(rows, key=lambda row: row["size"])
    min_latency = min(row["p50_latency_us"] for row in rows) if rows else 0.0
    max_latency = max(row["p50_latency_us"] for row in rows) if rows else 0.0
    latency_ratio = max_latency / max(min_latency, 1e-9)
    checks = {
        "recall": all(row["recall"] >= target_for_size(int(row["size"])) for row in rows),
        "faithfulness": all(row["faithfulness"] >= args.faithfulness_target for row in rows),
        "latency": latency_ratio <= args.latency_ratio_target,
        "ram": all(row["ram_speedup_proxy"] >= args.ram_speedup_target for row in rows),
        "no_full_scan": all(row["full_scan_count"] == 0.0 for row in rows),
        "candidate_cap": all(row["mean_candidate_count"] <= args.candidate_cap for row in rows),
    }
    ok = all(checks.values())

    print("Phase 6: World Memory QA with Evidence")
    print("=" * 88)
    print(f"{'chunks':>10} {'recall':>10} {'faithful':>10} {'p50_us':>10} {'ram_x':>10} {'scan':>8}")
    print("-" * 88)
    for row in rows:
        print(
            f"{int(row['size']):>10,} {row['recall']:>9.2%} {row['faithfulness']:>9.2%} "
            f"{row['p50_latency_us']:>10.3f} {row['ram_speedup_proxy']:>10.2f} {int(row['full_scan_count']):>8}"
        )
    print("-" * 88)
    print(f"Latency max/min:       {latency_ratio:.2f}x (target <={args.latency_ratio_target:.1f}x)")
    print(f"Overall status:        {'PASS' if ok else 'FAIL'}")

    payload = {
        "benchmark": "phase6_world_memory_qa",
        "success": bool(ok),
        "checks": checks,
        "latency_ratio": float(latency_ratio),
        "rows": rows,
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
