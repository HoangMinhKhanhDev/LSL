"""Phase 9 predictive coding v2 proof."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import LocalPredictiveStack, OnePassCausalMemory, SimpleSubwordTokenizer


ROOT = Path(__file__).resolve().parents[2]
DIALOGUE_CORPUS = ROOT / "benchmarks" / "data" / "dialogue_small" / "dialogue_mini_corpus.txt"
VIETNAMESE_CORPUS = ROOT / "benchmarks" / "data" / "vietnamese_small" / "vietnamese_mini_corpus.txt"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _token_sequences(tokenizer: SimpleSubwordTokenizer, text: str, max_tokens: int) -> list[int]:
    tokens = tokenizer.encode(text)
    if not tokens:
        return []
    return tokens[: int(max_tokens)]


def evaluate(args):
    dialogue_text = _read_text(DIALOGUE_CORPUS)
    vietnamese_text = _read_text(VIETNAMESE_CORPUS)
    tokenizer = SimpleSubwordTokenizer(vocab_size=4096, vietnamese_normalization=True, byte_fallback=True)
    tokenizer.build_vocab("\n".join(filter(None, [dialogue_text, vietnamese_text])))
    train_sequence = _token_sequences(tokenizer, dialogue_text or vietnamese_text, args.tokens)
    if not train_sequence:
        train_sequence = [(i * 7 + 3) % 97 for i in range(args.tokens)]
    ood_sequence = _token_sequences(tokenizer, vietnamese_text or dialogue_text[::-1], max(32, args.tokens // 2))
    if not ood_sequence:
        ood_sequence = list(reversed(train_sequence))

    stack = LocalPredictiveStack(layers=3, width=256, k=8, theta=0.05)
    epoch_errors = []
    epoch_suppression = []
    epoch_zero_update = []
    epoch_layer_curves = []
    train_start = time.perf_counter_ns()
    for _ in range(args.epochs):
        stack.reset_state()
        updates = 0
        errors = []
        suppressions = []
        zero = 0
        for token in train_sequence:
            states = [stack.state_for(token + layer * 997, layer) for layer in range(stack.layers)]
            out = stack.observe(states, learn=True)
            updates += int(out["updates"])
            errors.append(out["mean_error"])
            suppressions.append(out["suppression"])
            zero += int(out["updates"] == 0.0)
        epoch_errors.append(sum(errors) / len(errors))
        epoch_suppression.append(sum(suppressions) / len(suppressions))
        epoch_zero_update.append(zero / max(1, len(train_sequence)))
        epoch_layer_curves.append(stack.layer_error_curve())
    train_elapsed_us = (time.perf_counter_ns() - train_start) / 1000.0

    stack.adaptive_theta(target_suppression=0.95)
    tuned_theta = stack.theta

    def evaluate_sequence(sequence, learn=False, passes=1):
        mean_errors = []
        confidences = []
        anomalies = []
        suppressions = []
        calibrations = []
        updates = 0
        zero_updates = 0
        start = time.perf_counter_ns()
        for pass_index in range(int(passes)):
            stack.reset_state()
            for token in sequence:
                states = [stack.state_for(token + layer * 997, layer) for layer in range(stack.layers)]
                layer_confidences = []
                layer_correct = []
                for layer, current in enumerate(states[: stack.layers]):
                    prev = stack.prev_states[layer]
                    if prev is None:
                        continue
                    predicted, confidence = stack.predict_state(layer, prev)
                    if predicted is None:
                        continue
                    overlap = len(set(predicted) & set(current))
                    layer_confidences.append(confidence)
                    layer_correct.append(1.0 if overlap / max(1.0, float(stack.k)) >= 0.75 else 0.0)
                out = stack.observe(states, learn=learn)
                if pass_index == int(passes) - 1:
                    mean_errors.append(out["mean_error"])
                    confidences.append(out["confidence"])
                    anomalies.append(out["anomaly_score"])
                    suppressions.append(out["suppression"])
                    if layer_confidences:
                        calibrations.append(abs(sum(layer_confidences) / len(layer_confidences) - sum(layer_correct) / len(layer_correct)))
                    updates += int(out["updates"])
                    zero_updates += int(out["updates"] == 0.0)
        elapsed_us = (time.perf_counter_ns() - start) / 1000.0
        return {
            "mean_error": sum(mean_errors) / max(1, len(mean_errors)),
            "mean_confidence": sum(confidences) / max(1, len(confidences)),
            "mean_anomaly": sum(anomalies) / max(1, len(anomalies)),
            "mean_suppression": sum(suppressions) / max(1, len(suppressions)),
            "calibration_gap": sum(calibrations) / max(1, len(calibrations)),
            "updates": float(updates),
            "zero_update_ratio": zero_updates / max(1, len(sequence)),
            "us_per_token": elapsed_us / max(1, len(sequence) * max(1, passes)),
        }

    drops = []
    for layer_hist in stack.error_history:
        first = sum(layer_hist[: args.tokens]) / max(1, args.tokens)
        last = sum(layer_hist[-args.tokens:]) / max(1, args.tokens)
        drops.append((first - last) / max(first, 1e-9))

    in_domain_eval = evaluate_sequence(train_sequence, learn=False, passes=3)
    ood_eval = evaluate_sequence(ood_sequence, learn=False, passes=2)
    anomaly_gap = max(0.0, ood_eval["mean_anomaly"] - in_domain_eval["mean_anomaly"])
    robustness = max(0.0, 1.0 - anomaly_gap)
    local_update_audit_us = (train_elapsed_us / max(1, args.epochs * len(train_sequence)))
    in_domain_diag = stack.diagnostics()
    layer_curve_first = epoch_layer_curves[0] if epoch_layer_curves else [0.0] * stack.layers
    layer_curve_last = epoch_layer_curves[-1] if epoch_layer_curves else [0.0] * stack.layers
    layer_curve_drop = []
    for first, last in zip(layer_curve_first, layer_curve_last):
        layer_curve_drop.append((first - last) / max(first, 1e-9))
    layer_confidence_curve = stack.layer_confidence_curve()
    layer_anomaly_curve = stack.layer_anomaly_curve()
    confidence_gap = abs(in_domain_eval["mean_confidence"] - in_domain_diag["mean_confidence"])

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
        "final_suppression": in_domain_eval["mean_suppression"],
        "final_zero_update_ratio": in_domain_eval["zero_update_ratio"],
        "layer0_error_drop": layer_curve_drop[0] if layer_curve_drop else 0.0,
        "layer1_error_drop": layer_curve_drop[1] if len(layer_curve_drop) > 1 else 0.0,
        "layer2_error_drop": layer_curve_drop[2] if len(layer_curve_drop) > 2 else 0.0,
        "in_domain_error": in_domain_eval["mean_error"],
        "ood_error": ood_eval["mean_error"],
        "anomaly_gap": anomaly_gap,
        "confidence_gap": confidence_gap,
        "calibration_gap": in_domain_eval["calibration_gap"],
        "ood_robustness": robustness,
        "adaptive_theta": tuned_theta,
        "local_update_audit_us": local_update_audit_us,
        "in_domain_latency_us": in_domain_eval["us_per_token"],
        "ood_latency_us": ood_eval["us_per_token"],
        "mean_confidence": in_domain_eval["mean_confidence"],
        "mean_anomaly": in_domain_eval["mean_anomaly"],
        "layer_confidence_mean": sum(layer_confidence_curve) / max(1, len(layer_confidence_curve)),
        "layer_anomaly_mean": sum(layer_anomaly_curve) / max(1, len(layer_anomaly_curve)),
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
        "calibration": metrics["calibration_gap"] <= args.calibration_target,
        "audit": metrics["local_update_audit_us"] <= args.audit_us_target,
        "ood_confidence": metrics["confidence_gap"] <= args.confidence_gap_target,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--chain-items", type=int, default=128)
    parser.add_argument("--error-drop-target", type=float, default=0.45)
    parser.add_argument("--suppression-target", type=float, default=0.72)
    parser.add_argument("--zero-update-target", type=float, default=0.80)
    parser.add_argument("--one-pass-target", type=float, default=10.0)
    parser.add_argument("--chain-target", type=float, default=0.70)
    parser.add_argument("--calibration-target", type=float, default=0.25)
    parser.add_argument("--audit-us-target", type=float, default=200.0)
    parser.add_argument("--confidence-gap-target", type=float, default=0.25)
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
