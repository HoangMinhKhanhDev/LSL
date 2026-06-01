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
    reward_mod = BioNeuromodulator()
    no_reward_mod = BioNeuromodulator()
    reward_updates = 0
    no_reward_updates = 0
    dopamine_novel = []
    dopamine_repeat = []
    acetylcholine_novel = []
    acetylcholine_repeat = []
    serotonin_values = []
    for i in range(args.stress_steps):
        if i % 20 == 0:
            token = f"novel_{i}"
            surprise = 1.0
            reward = 1.0 if i % 40 == 0 else 0.0
        else:
            token = f"common_{i % 16}"
            surprise = 0.02
            reward = 0.0
        gates = mod.gates(token, surprise=surprise, reward=reward)
        if "novel" in token:
            dopamine_novel.append(gates["dopamine"])
            acetylcholine_novel.append(gates["acetylcholine"])
        else:
            dopamine_repeat.append(gates["dopamine"])
            acetylcholine_repeat.append(gates["acetylcholine"])
        serotonin_values.append(gates["serotonin"])
        mod.observe(token, surprise=surprise, reward=reward)
        reward_updates += int(reward_mod.observe(token, surprise=surprise, reward=reward))
        no_reward_updates += int(no_reward_mod.observe(token, surprise=surprise, reward=0.0))
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
    reward_gain = reward_updates / max(1.0, float(no_reward_updates))
    dopamine_gap = (sum(dopamine_novel) / max(1, len(dopamine_novel))) - (sum(dopamine_repeat) / max(1, len(dopamine_repeat)))
    acetylcholine_gap = (sum(acetylcholine_novel) / max(1, len(acetylcholine_novel))) - (
        sum(acetylcholine_repeat) / max(1, len(acetylcholine_repeat))
    )
    serotonin_stability = max(serotonin_values) - min(serotonin_values)

    diag = mod.diagnostics()
    metrics = {
        "novel_update_ratio": diag["novel_update_ratio"],
        "reward_update_ratio": diag["reward_update_ratio"],
        "weight_norm_min": weight_min,
        "weight_norm_max": weight_max,
        "sparsity_min": sparsity_min,
        "sparsity_max": sparsity_max,
        "formal_tone_pass": float(formal_tone == "formal"),
        "casual_tone_pass": float(casual_tone == "casual"),
        "curiosity_gain": curiosity_gain,
        "reward_gate_gain": reward_gain,
        "dopamine_novelty_gap": dopamine_gap,
        "acetylcholine_novelty_gap": acetylcholine_gap,
        "serotonin_stability": serotonin_stability,
        "mean_dopamine": diag["mean_dopamine"],
        "mean_acetylcholine": diag["mean_acetylcholine"],
        "mean_serotonin": diag["mean_serotonin"],
        "stress_steps": float(args.stress_steps),
    }
    checks = {
        "novel_updates": metrics["novel_update_ratio"] >= args.novel_target,
        "weight_stable": metrics["weight_norm_min"] >= 0.90 and metrics["weight_norm_max"] <= 1.10,
        "sparsity_stable": metrics["sparsity_min"] >= 0.018 and metrics["sparsity_max"] <= 0.022,
        "tone": metrics["formal_tone_pass"] == 1.0 and metrics["casual_tone_pass"] == 1.0,
        "curiosity": metrics["curiosity_gain"] > args.curiosity_target,
        "stress_proxy": metrics["stress_steps"] >= args.min_stress_steps,
        "reward": metrics["reward_gate_gain"] >= args.reward_target,
        "dopamine_gap": metrics["dopamine_novelty_gap"] >= args.dopamine_gap_target,
        "acetylcholine_gap": metrics["acetylcholine_novelty_gap"] >= args.acetylcholine_gap_target,
        "serotonin": metrics["serotonin_stability"] <= args.serotonin_stability_target,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-steps", type=int, default=1000000)
    parser.add_argument("--min-stress-steps", type=int, default=1000000)
    parser.add_argument("--novel-target", type=float, default=0.95)
    parser.add_argument("--curiosity-target", type=float, default=1.20)
    parser.add_argument("--reward-target", type=float, default=0.90)
    parser.add_argument("--dopamine-gap-target", type=float, default=0.10)
    parser.add_argument("--acetylcholine-gap-target", type=float, default=0.10)
    parser.add_argument("--serotonin-stability-target", type=float, default=0.06)
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
