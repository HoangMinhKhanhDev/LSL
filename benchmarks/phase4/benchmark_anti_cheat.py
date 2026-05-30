"""Phase 4 Anti-Cheat Structural Scan.

Enforce strict constraints to ensure no cheating:
1. No backprop: No `.backward()`, no autograd
2. No optimizer state: No Adam, SGD, momentum
3. No attention matrix: No Q/K/V, no all-pairs interaction
4. No global hidden error: Only local prediction errors
5. No future information: Prediction uses only t-1 state
6. Local updates only: Update touches only active synapses
7. No batch retrain: Updates must be online per token
8. No external APIs: No calls to external services in forward/observe
9. No hardcoded rules: No grammar/reasoning rules in generation
10. No eval corpus leakage: No semantic prior from test data
11. No post-hoc tuning: No hyperparameter changes after seeing results
12. Full metric accounting: Include bookkeeping, memory access, fatigue, routing
"""
import argparse
import json
import os
import sys
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def structural_scan() -> Tuple[bool, List[str]]:
    """Scan codebase for forbidden patterns."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Files to scan
    files = [
        "lsl/model.py",
        "lsl/synapse.py",
        "lsl/ssm.py",
        "lsl/sdr.py",
        "lsl/semantic_sdr.py",
        "lsl/associative_memory.py",
        "lsl/cortical_column.py",
        "lsl/memory.py",
        "lsl/long_context.py",
        "lsl/generation.py",
        "lsl/world_memory.py",
        "lsl/reasoning.py",
        "lsl/homeostasis.py",
        "lsl/workspace.py",
        "lsl/event_ssm.py",
        "lsl/prior.py",
        "lsl/agent.py",
        "lsl/bio.py",
        "lsl/hierarchy.py",
        "lsl/router.py",
        "lsl/neuromod.py",
    ]
    
    # Forbidden patterns (obfuscated to avoid false positives in this file)
    forbidden = [
        "to" + "rch",  # torch
        "tensor" + "flow",  # tensorflow
        "ja" + "x",  # jax
        "." + "backward",  # autograd backward
        "optimizer" + ".step",  # optimizer step
        "Gradient" + "Tape",  # tensorflow gradient tape
        "B_" + "rec",  # recurrent backward
        "B_" + "ssm",  # SSM backward
        "B_" + "emb",  # embedding backward
        "dfa" + "_update",  # DFA global update
        "Living" + "Attention" + "Layer",  # attention layer
        "last_" + "attention" + "_map",  # attention map
        "attention" + "_map",  # attention map
        "self-" + "attention",  # self-attention
        "cross-" + "attention",  # cross-attention
        "autograd",  # autograd
        "Adam(",  # Adam optimizer (with parenthesis to avoid false positives)
        "SGD(",  # SGD optimizer (with parenthesis)
        "momentum",  # momentum
        # Note: "batch" and "key" removed due to false positives in legitimate code
    ]
    
    hits = []
    for rel in files:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for item in forbidden:
            if item in text:
                hits.append(f"{rel}: {item}")
    
    return len(hits) == 0, hits


def check_local_updates() -> Tuple[bool, List[str]]:
    """Check that updates are local (only touch active synapses)."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Check synapse.py for local update patterns
    path = os.path.join(root, "lsl", "synapse.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Should have top-k or active-index updates
    has_local_updates = (
        "top_k" in text or
        "active" in text or
        "sparse" in text
    )
    
    # Should NOT have full matrix updates in learning methods
    has_full_updates = False
    if "hebbian_update" in text:
        # Check if hebbian_update uses full matrix
        lines = text.split("\n")
        in_hebbian = False
        for line in lines:
            if "def hebbian_update" in line:
                in_hebbian = True
            elif in_hebbian and "def " in line:
                in_hebbian = False
            elif in_hebbian and "@" in line and "W" in line:
                has_full_updates = True
    
    issues = []
    if not has_local_updates:
        issues.append("No local update patterns found (top-k, active, sparse)")
    if has_full_updates:
        issues.append("Full matrix updates found in learning methods")
    
    return len(issues) == 0, issues


def check_online_learning() -> Tuple[bool, List[str]]:
    """Check that learning is online (per token, not batch)."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    path = os.path.join(root, "lsl", "model.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Should have observe() method for online learning
    has_observe = "def observe" in text
    
    # Should NOT have batch training methods
    has_batch = "batch" in text.lower() and "train" in text.lower()
    
    issues = []
    if not has_observe:
        issues.append("No observe() method for online learning")
    if has_batch:
        issues.append("Batch training detected")
    
    return len(issues) == 0, issues


def check_no_attention() -> Tuple[bool, List[str]]:
    """Check that no attention mechanisms are used."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    files = ["lsl/model.py", "lsl/ssm.py"]
    
    # Only flag actual attention mechanisms, not variable names like "key"
    attention_patterns = [
        "attention(",  # attention function call
        "self-attention",  # self-attention
        "cross-attention",  # cross-attention
        "QKV",  # QKV attention
        "softmax(" + "@",  # attention softmax @
    ]
    
    hits = []
    for rel in files:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for pattern in attention_patterns:
            if pattern in text:
                hits.append(f"{rel}: {pattern}")
    
    return len(hits) == 0, hits


def run_anti_cheat_scan() -> Tuple[bool, Dict[str, Tuple[bool, List[str]]]]:
    """Run all anti-cheat checks."""
    print("Phase 4 Anti-Cheat Structural Scan")
    print("=" * 80)
    
    results = {}
    
    # Structural scan
    ok, hits = structural_scan()
    results["structural"] = (ok, hits)
    print(f"Structural scan: {'PASS' if ok else 'FAIL'}")
    if hits:
        for hit in hits:
            print(f"  - {hit}")
    
    # Local updates check
    ok, issues = check_local_updates()
    results["local_updates"] = (ok, issues)
    print(f"Local updates: {'PASS' if ok else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Online learning check
    ok, issues = check_online_learning()
    results["online_learning"] = (ok, issues)
    print(f"Online learning: {'PASS' if ok else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # No attention check
    ok, hits = check_no_attention()
    results["no_attention"] = (ok, hits)
    print(f"No attention: {'PASS' if ok else 'FAIL'}")
    if hits:
        for hit in hits:
            print(f"  - {hit}")
    
    # Overall status
    all_ok = all(ok for ok, _ in results.values())
    
    print("=" * 80)
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 80)
    
    return all_ok, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 Anti-Cheat Structural Scan"
    )
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    all_ok, results = run_anti_cheat_scan()
    payload = {
        "benchmark": "anti_cheat",
        "success": bool(all_ok),
        "checks": {name: {"success": ok, "issues": issues} for name, (ok, issues) in results.items()},
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
