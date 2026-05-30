"""Phase 9 neuromodulation v2 proof."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import BioNeuromodulator


def evaluate(args):
    mod = BioNeuromodulator()
    weight_min = weight_max = mod.weight_norm
    sparsity_min = sparsity_max = mod.sparsity
    for i in range(args.stress_steps):
        if i % 20 == 0:
            token = f"novel_{i}"
            surprise = 1.0
        else:
            token = f"common_{i % 16}"
            surprise = 0.02
        mod.observe(token, surprise=surprise)
        weight_min = min(weight_min, mod.weight_norm)
        weight_max = max(weight_max, mod.weight_norm)
        sparsity_min = min(sparsity_min, mod.sparsity)
        sparsity_max = max(sparsity_max, mod.sparsity)

    formal = BioNeuromodulator()
    for token in ["please", "therefore", "sincerely", "regards"] * 4:
        formal.observe(token, surprise=0.9)
    formal_tone = formal.tone()
    for token in ["hey", "cool", "thanks", "yep"] * 8:
        formal.observe(token, surprise=0.9)
    casual_tone = formal.tone()

    candidates = [(f"item_{i}", (i * 37) % 101 / 100.0) for i in range(64)]
    pick = mod.curiosity_pick(candidates)
    picked_uncertainty = dict(candidates)[pick]
    random_proxy = sum(v for _, v in candidates) / len(candidates)
    curiosity_gain = picked_uncertainty / max(random_proxy, 1e-9)

    diag = mod.diagnostics()
    metrics = {
        "novel_update_ratio": diag["novel_update_ratio"],
        "weight_norm_min": weight_min,
        "weight_norm_max": weight_max,
        "sparsity_min": sparsity_min,
        "sparsity_max": sparsity_max,
        "formal_tone_pass": float(formal_tone == "formal"),
        "casual_tone_pass": float(casual_tone == "casual"),
        "curiosity_gain": curiosity_gain,
        "stress_steps": float(args.stress_steps),
    }
    checks = {
        "novel_updates": metrics["novel_update_ratio"] >= args.novel_target,
        "weight_stable": metrics["weight_norm_min"] >= 0.90 and metrics["weight_norm_max"] <= 1.10,
        "sparsity_stable": metrics["sparsity_min"] >= 0.018 and metrics["sparsity_max"] <= 0.022,
        "tone": metrics["formal_tone_pass"] == 1.0 and metrics["casual_tone_pass"] == 1.0,
        "curiosity": metrics["curiosity_gain"] > args.curiosity_target,
        "stress_proxy": metrics["stress_steps"] >= args.min_stress_steps,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-steps", type=int, default=1000000)
    parser.add_argument("--min-stress-steps", type=int, default=1000000)
    parser.add_argument("--novel-target", type=float, default=0.95)
    parser.add_argument("--curiosity-target", type=float, default=1.20)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: Bio Neuromodulation")
    print("=" * 88)
    for key, value in result["metrics"].items():
        print(f"{key:<28} {value:.4f}")
    print(f"Overall status:              {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_bio_neuromodulation", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
