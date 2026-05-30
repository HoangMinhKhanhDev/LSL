"""Phase 7 long-range entity-event reasoning benchmark."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import EntityEventGraph


def build_graph(size: int, args):
    graph = EntityEventGraph(candidate_cap=64)
    r1, r2, r3 = 11, 12, 13
    t0 = time.perf_counter_ns()
    for i in range(size):
        base = i * 4 + 1000
        graph.observe_event(base, r1, base + 1, episode_id=i, evidence_id=i)
        graph.observe_event(base + 1, r2, base + 2, episode_id=i, evidence_id=i)
        graph.observe_event(base + 2, r3, base + 3, episode_id=i, evidence_id=i)
    ingest_us = (time.perf_counter_ns() - t0) / 1000.0 / max(1, size)
    return graph, (r1, r2, r3), ingest_us


def evaluate_size(size: int, args):
    graph, relations, ingest_us = build_graph(size, args)
    stride = max(1, size // args.queries)
    starts = [i * 4 + 1000 for i in range(0, size, stride)][: args.queries]
    for start in starts:
        graph.query_chain(start, relations)
    correct = 0
    latencies = []
    full_scan = 0
    for start in starts:
        t0 = time.perf_counter_ns()
        pred = None
        for _ in range(args.query_repeats):
            pred = graph.query_chain(start, relations)
        latencies.append((time.perf_counter_ns() - t0) / 1000.0 / max(1, args.query_repeats))
        correct += int(pred == start + 3)
        full_scan += int(graph.diagnostics()["last_full_scan"] > 0)
    return {
        "size": float(size),
        "accuracy": correct / max(1, len(starts)),
        "p50_us": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "ingest_us": float(ingest_us),
        "full_scan": float(full_scan),
        "candidate_count": graph.diagnostics()["last_candidate_count"],
    }


def target(size: int) -> float:
    return 0.75 if size <= 100000 else 0.60


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000, 1000000])
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--query-repeats", type=int, default=16)
    parser.add_argument("--latency-ratio-target", type=float, default=2.0)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [evaluate_size(size, args) for size in args.sizes]
    min_latency = min(row["p50_us"] for row in rows)
    max_latency = max(row["p50_us"] for row in rows)
    latency_ratio = max_latency / max(1e-9, min_latency)
    checks = {
        "accuracy": all(row["accuracy"] >= target(int(row["size"])) for row in rows),
        "latency": latency_ratio <= args.latency_ratio_target,
        "no_scan": all(row["full_scan"] == 0.0 for row in rows),
    }
    ok = all(checks.values())
    print("Phase 7: Long-Range Reasoning")
    print("=" * 88)
    for row in rows:
        print(f"{int(row['size']):>10,} accuracy={row['accuracy']:.2%} p50={row['p50_us']:.3f}us scan={int(row['full_scan'])}")
    print(f"Latency max/min: {latency_ratio:.2f}x")
    print(f"Overall status:  {'PASS' if ok else 'FAIL'}")
    payload = {
        "benchmark": "phase7_long_range_reasoning",
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
