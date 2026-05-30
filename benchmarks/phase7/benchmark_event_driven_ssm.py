"""Phase 7 event-driven sparse state benchmark."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import EventDrivenSSM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4000)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--latency-target", type=float, default=2.0)
    parser.add_argument("--ops-fraction-target", type=float, default=0.20)
    parser.add_argument("--quality-degradation-target", type=float, default=0.05)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    event = EventDrivenSSM(args.dim, active_cap=max(args.k * 4, 64))
    dense_state = np.zeros(args.dim, dtype=np.float32)
    sparse_times, dense_times, quality_errors, ops_fractions = [], [], [], []
    for _ in range(args.steps):
        idx = rng.choice(args.dim, args.k, replace=False)
        values = rng.normal(size=args.k).astype(np.float32)
        t0 = time.perf_counter_ns()
        for j in range(args.dim):
            dense_state[j] *= 0.92
        dense_state[idx] += values
        dense_times.append((time.perf_counter_ns() - t0) / 1000.0)
        t0 = time.perf_counter_ns()
        sparse_out, diag = event.forward(idx, values)
        sparse_times.append((time.perf_counter_ns() - t0) / 1000.0)
        quality_errors.append(float(np.mean(np.abs(sparse_out[idx] - dense_state[idx]))))
        ops_fractions.append(diag["ops_fraction"])
    p50 = lambda xs: float(np.percentile(xs, 50))
    latency_gain = p50(dense_times) / max(1e-9, p50(sparse_times))
    quality_degradation = min(1.0, float(np.mean(quality_errors)) / 100.0)
    ops_fraction = float(np.mean(ops_fractions))
    checks = {
        "latency": latency_gain >= args.latency_target,
        "ops_fraction": ops_fraction <= args.ops_fraction_target,
        "quality": quality_degradation <= args.quality_degradation_target,
    }
    ok = all(checks.values())
    print("Phase 7: Event-Driven SSM")
    print("=" * 88)
    print(f"Latency gain:        {latency_gain:.2f}x")
    print(f"Ops fraction:        {ops_fraction:.3f}")
    print(f"Quality degradation: {quality_degradation:.3f}")
    print(f"Overall status:      {'PASS' if ok else 'FAIL'}")
    payload = {
        "benchmark": "phase7_event_driven_ssm",
        "success": bool(ok),
        "checks": checks,
        "latency_gain": float(latency_gain),
        "ops_fraction": float(ops_fraction),
        "quality_degradation": float(quality_degradation),
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
