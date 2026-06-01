"""Phase 9 hippocampus / two-speed memory proof."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import HippocampalMemory


def evaluate(args):
    memory = HippocampalMemory(candidate_cap=args.candidate_cap, surprise_threshold=0.5)
    for i in range(args.items):
        memory.observe(["fact", f"entity-{i:05d}", f"group-{i % 17}"], f"value_{i}", surprise=1.0)
    pollution_ignored = 0
    for i in range(args.noise_items):
        pollution_ignored += int(not memory.observe(["fact", f"noise-{i:05d}", "boring"], "ignored", surprise=0.05))
    ignored = memory.observe(["fact", "boring"], "ignored", surprise=0.1)
    replayed = memory.consolidate(replay_fraction=args.replay_fraction)
    memory.observe(["fact", "conflict"], "alpha", surprise=1.0)
    memory.observe(["fact", "conflict"], "beta", surprise=2.0)
    conflict = memory.resolve_conflict(["fact", "conflict"]) or {"value": None, "confidence": 0.0, "candidates": 0.0}
    compact_view = HippocampalMemory(candidate_cap=args.candidate_cap, surprise_threshold=0.5)
    for i in range(min(args.items, 2048)):
        compact_view.observe(["fact", f"entity-{i:05d}", f"group-{i % 17}"], f"value_{i}", surprise=1.0)
    compacted = compact_view.compact(max_fast=max(1, args.candidate_cap * 2), max_slow=max(1, args.items // 2))

    exact_correct = 0
    partial_correct = 0
    full_scan = 0
    max_candidates = 0
    for i in range(args.items):
        exact_correct += int(memory.recall(["fact", f"entity-{i:05d}", f"group-{i % 17}"]) == f"value_{i}")
        partial_correct += int(memory.recall([f"entity-{i:05d}"]) == f"value_{i}")
        diag = memory.diagnostics()
        full_scan += int(diag["last_full_scan"] > 0)
        max_candidates = max(max_candidates, int(diag["last_candidate_count"]))

    diag = memory.diagnostics()
    pollution_ratio = diag["pollution_ratio"]
    replay_mass = diag["replay_surprise_total"] / max(1.0, diag["encoded"])
    metrics = {
        "items": float(args.items),
        "exact_retention": exact_correct / max(1, args.items),
        "partial_cue_recall": partial_correct / max(1, args.items),
        "replay_budget": diag["replay_budget"],
        "pollution_ratio": pollution_ratio,
        "pollution_rejection_rate": pollution_ignored / max(1, args.noise_items),
        "replay_mass": replay_mass,
        "replayed_items": float(replayed),
        "context_gate_ignored": float(not ignored),
        "max_candidate_count": float(max_candidates),
        "full_scan_count": float(full_scan),
        "conflict_keys": diag["conflict_keys"],
        "conflict_confidence": float(conflict["confidence"]),
        "conflict_candidates": float(conflict["candidates"]),
        "compaction_removed": float(compacted),
    }
    checks = {
        "exact_retention": metrics["exact_retention"] >= args.retention_target,
        "partial_cue": metrics["partial_cue_recall"] >= args.partial_target,
        "replay_budget": metrics["replay_budget"] <= args.replay_budget_target,
        "pollution": metrics["pollution_rejection_rate"] >= args.pollution_target,
        "context_gate": metrics["context_gate_ignored"] == 1.0,
        "candidate_cap": metrics["max_candidate_count"] <= args.candidate_cap,
        "no_scan": metrics["full_scan_count"] == 0.0,
        "conflict": metrics["conflict_confidence"] >= 0.50,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=10000)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--replay-fraction", type=float, default=0.10)
    parser.add_argument("--noise-items", type=int, default=512)
    parser.add_argument("--retention-target", type=float, default=1.0)
    parser.add_argument("--partial-target", type=float, default=0.95)
    parser.add_argument("--replay-budget-target", type=float, default=0.10)
    parser.add_argument("--pollution-target", type=float, default=0.95)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: Bio Hippocampus")
    print("=" * 88)
    for key, value in result["metrics"].items():
        print(f"{key:<28} {value:.4f}")
    print(f"Overall status:              {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_bio_hippocampus", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
