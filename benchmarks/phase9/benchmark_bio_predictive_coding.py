"""Phase 9 predictive coding v2 proof."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import LocalPredictiveStack, OnePassCausalMemory


def evaluate(args):
    stack = LocalPredictiveStack(layers=3, width=256, k=8, theta=0.05)
    sequence = [(i * 7 + 3) % 97 for i in range(args.tokens)]
    epoch_errors = []
    epoch_suppression = []
    epoch_zero_update = []
    for _ in range(args.epochs):
        stack.reset_state()
        updates = 0
        errors = []
        suppressions = []
        zero = 0
        for token in sequence:
            states = [stack.state_for(token + layer * 997, layer) for layer in range(stack.layers)]
            out = stack.observe(states, learn=True)
            updates += int(out["updates"])
            errors.append(out["mean_error"])
            suppressions.append(out["suppression"])
            zero += int(out["updates"] == 0.0)
        epoch_errors.append(sum(errors) / len(errors))
        epoch_suppression.append(sum(suppressions) / len(suppressions))
        epoch_zero_update.append(zero / max(1, len(sequence)))

    drops = []
    for layer_hist in stack.error_history:
        first = sum(layer_hist[: args.tokens]) / max(1, args.tokens)
        last = sum(layer_hist[-args.tokens:]) / max(1, args.tokens)
        drops.append((first - last) / max(first, 1e-9))

    causal = OnePassCausalMemory()
    vocab = 1000
    random_prob = 1.0 / vocab
    causal.observe("stroke", "aphasia")
    one_pass_ratio = causal.probability("stroke", "aphasia", vocab) / random_prob
    for i in range(args.chain_items):
        causal.observe(f"a{i}", f"b{i}")
        causal.observe(f"b{i}", f"c{i}")
        causal.observe(f"c{i}", f"d{i}")
    chain_correct = sum(int(causal.chain(f"a{i}", 3) == f"d{i}") for i in range(args.chain_items))
    chain_accuracy = chain_correct / max(1, args.chain_items)

    before = -1.0 * random_prob
    causal.observe("novel_cause", "novel_effect")
    after = -1.0 * causal.probability("novel_cause", "novel_effect", vocab)
    immediate_loss_drop = before - after

    metrics = {
        "mean_error_drop": sum(drops) / max(1, len(drops)),
        "min_layer_error_drop": min(drops),
        "final_suppression": epoch_suppression[-1],
        "final_zero_update_ratio": epoch_zero_update[-1],
        "one_pass_causal_ratio": one_pass_ratio,
        "chain_accuracy": chain_accuracy,
        "immediate_loss_drop": immediate_loss_drop,
    }
    checks = {
        "error_drop": metrics["min_layer_error_drop"] >= args.error_drop_target,
        "suppression": metrics["final_suppression"] >= args.suppression_target,
        "zero_update": metrics["final_zero_update_ratio"] >= args.zero_update_target,
        "one_pass": metrics["one_pass_causal_ratio"] >= args.one_pass_target,
        "chain": metrics["chain_accuracy"] >= args.chain_target,
        "immediate_loss": metrics["immediate_loss_drop"] > 0.0,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--chain-items", type=int, default=128)
    parser.add_argument("--error-drop-target", type=float, default=0.90)
    parser.add_argument("--suppression-target", type=float, default=0.95)
    parser.add_argument("--zero-update-target", type=float, default=0.80)
    parser.add_argument("--one-pass-target", type=float, default=10.0)
    parser.add_argument("--chain-target", type=float, default=0.70)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: Bio Predictive Coding")
    print("=" * 88)
    for key, value in result["metrics"].items():
        print(f"{key:<28} {value:.4f}")
    print(f"Overall status:              {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_bio_predictive_coding", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
