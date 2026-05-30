"""Phase 8 long-context multi-evidence QA through the integrated agent."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import IntegratedLSLAgent


def evaluate_size(size: int, args):
    agent = IntegratedLSLAgent(vocab_size=1000, seed=args.seed)
    agent.build_tokenizer("entity relation value node next")
    for i in range(size):
        a = f"entity_{i}_a"
        b = f"entity_{i}_b"
        c = f"entity_{i}_c"
        d = f"entity_{i}_d"
        agent.observe_event(a, "link1", b, episode_id=i, evidence_id=i)
        agent.observe_event(b, "link2", c, episode_id=i, evidence_id=i)
        agent.observe_event(c, "link3", d, episode_id=i, evidence_id=i)
    stride = max(1, size // args.queries)
    starts = [f"entity_{i}_a" for i in range(0, size, stride)][: args.queries]
    for start in starts:
        agent.answer(f"Starting from {start}, follow link1 then link2 then link3?")
    correct = 0
    latencies = []
    full_scan = 0
    for start in starts:
        idx = int(start.split("_")[1])
        t0 = time.perf_counter_ns()
        pred = None
        for _ in range(args.query_repeats):
            pred = agent.answer(f"Starting from {start}, follow link1 then link2 then link3?")
        latencies.append((time.perf_counter_ns() - t0) / 1000.0 / max(1, args.query_repeats))
        correct += int(pred == f"entity_{idx}_d")
        full_scan += int(agent.diagnostics().get("event_last_full_scan", 0.0) > 0.0)
    return {
        "size": float(size),
        "accuracy": correct / max(1, len(starts)),
        "p50_us": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "full_scan": float(full_scan),
    }


def target(size: int) -> float:
    return 0.70 if size <= 100000 else 0.60


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000, 1000000])
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--query-repeats", type=int, default=16)
    parser.add_argument("--latency-ratio-target", type=float, default=2.0)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [evaluate_size(size, args) for size in sorted(args.sizes, reverse=True)]
    rows = sorted(rows, key=lambda row: row["size"])
    min_latency = min(row["p50_us"] for row in rows)
    max_latency = max(row["p50_us"] for row in rows)
    latency_ratio = max_latency / max(min_latency, 1e-9)
    checks = {
        "accuracy": all(row["accuracy"] >= target(int(row["size"])) for row in rows),
        "latency": latency_ratio <= args.latency_ratio_target,
        "no_scan": all(row["full_scan"] == 0.0 for row in rows),
    }
    ok = all(checks.values())
    print("Phase 8: Multi-Evidence Long Context")
    print("=" * 88)
    for row in rows:
        print(f"{int(row['size']):>10,} accuracy={row['accuracy']:.2%} p50={row['p50_us']:.3f}us scan={int(row['full_scan'])}")
    print(f"Latency max/min: {latency_ratio:.2f}x")
    print(f"Overall status:  {'PASS' if ok else 'FAIL'}")
    payload = {
        "benchmark": "phase8_multievidence_long_context",
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
