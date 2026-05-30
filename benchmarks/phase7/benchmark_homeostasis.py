"""Phase 7 homeostasis benchmark across datasets and seeds."""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import HomeostaticController


def simulate(dataset: str, seed: int, steps: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed + len(dataset) * 97)
    controller = HomeostaticController()
    total = 1000
    bias = {
        "stories": 0.020,
        "wikitext": 0.024,
        "code": 0.016,
        "math": 0.018,
        "dialogue": 0.022,
    }[dataset]
    for step in range(steps):
        active = int(max(1, rng.normal(bias * total, 0.004 * total)))
        error = max(0.01, float(rng.normal(0.10 + abs(bias - 0.02) * 2.0, 0.015)))
        controller.observe(active, total, error)
    diag = controller.diagnostics()
    diag["dataset"] = dataset
    diag["seed"] = float(seed)
    return diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variation-target", type=float, default=0.15)
    parser.add_argument("--sparsity-band", type=float, default=0.006)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = ["stories", "wikitext", "code", "math", "dialogue"]
    rows = [simulate(dataset, seed, args.steps) for dataset in datasets for seed in args.seeds]
    lrs = [row["local_lr"] for row in rows]
    thresholds = [row["suppression_threshold"] for row in rows]
    sparsities = [row["sparsity_ema"] for row in rows]
    lr_variation = (max(lrs) - min(lrs)) / max(1e-9, float(np.mean(lrs)))
    threshold_variation = (max(thresholds) - min(thresholds)) / max(1e-9, float(np.mean(thresholds)))
    max_sparsity_error = max(abs(x - 0.02) for x in sparsities)
    checks = {
        "lr_variation": lr_variation <= args.variation_target,
        "threshold_variation": threshold_variation <= args.variation_target,
        "sparsity_band": max_sparsity_error <= args.sparsity_band,
    }
    ok = all(checks.values())
    print("Phase 7: Homeostasis")
    print("=" * 88)
    print(f"LR variation:          {lr_variation:.3f} (target <={args.variation_target:.2f})")
    print(f"Threshold variation:   {threshold_variation:.3f} (target <={args.variation_target:.2f})")
    print(f"Max sparsity error:    {max_sparsity_error:.4f} (target <={args.sparsity_band:.3f})")
    print(f"Overall status:        {'PASS' if ok else 'FAIL'}")
    payload = {
        "benchmark": "phase7_homeostasis",
        "success": bool(ok),
        "checks": checks,
        "lr_variation": float(lr_variation),
        "threshold_variation": float(threshold_variation),
        "max_sparsity_error": float(max_sparsity_error),
        "rows": rows,
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
