"""Phase 9 cortical column v2 proof."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import CorticalColumnSequenceMemory


def train_sequence(memory, sequence, repeats=1):
    for _ in range(repeats):
        memory.reset_state()
        for token in sequence:
            memory.forward(token, learn=True)


def next_prediction(memory, prefix):
    memory.reset_state()
    for token in prefix:
        memory.forward(token, learn=False)
    scores = memory.predict_next_token_scores()
    return int(np.argmax(scores)) if float(scores.sum()) > 0 else None


def evaluate(args):
    # subject/verb/object grammar corpus
    subjects = list(range(0, 10))
    verbs = list(range(10, 20))
    objects = list(range(20, 30))
    sequences = [[s, verbs[s % 10], objects[s % 10]] for s in subjects]
    long_topic = [40 + (i % 5) for i in range(200)]
    vocab_size = 128
    memory = CorticalColumnSequenceMemory(vocab_size=vocab_size, cells_per_column=100, sparsity=0.02, seed=args.seed)
    for seq in sequences:
        train_sequence(memory, seq, repeats=1)
    train_sequence(memory, long_topic, repeats=1)

    seen_total = 0
    seen_correct = 0
    grammar_total = 0
    grammar_correct = 0
    for seq in sequences:
        pred_v = next_prediction(memory, seq[:1])
        pred_o = next_prediction(memory, seq[:2])
        seen_correct += int(pred_v == seq[1]) + int(pred_o == seq[2])
        seen_total += 2
        grammar_correct += int(pred_v in verbs) + int(pred_o in objects)
        grammar_total += 2

    generated = [long_topic[0]]
    memory.reset_state()
    memory.forward(long_topic[0], learn=False)
    for _ in range(199):
        scores = memory.predict_next_token_scores()
        nxt = int(np.argmax(scores)) if float(scores.sum()) > 0 else -1
        generated.append(nxt)
        if nxt >= 0:
            memory.forward(nxt, learn=False)
    topic_set = set(long_topic)
    topic_coherence = sum(int(tok in topic_set) for tok in generated) / max(1, len(generated))

    def per_token_latency(length):
        seq = [(i * 11 + 7) % vocab_size for i in range(length)]
        memory.reset_state()
        start = time.perf_counter_ns()
        for token in seq:
            memory.forward(token, learn=False)
        return (time.perf_counter_ns() - start) / 1000.0 / max(1, length)

    short_us = per_token_latency(10)
    long_us = per_token_latency(1000)
    latency_ratio = long_us / max(short_us, 1e-9)
    sparse_ops = args.tokens_for_energy * memory.k
    transformer_ops = args.tokens_for_energy * args.tokens_for_energy * vocab_size
    energy_proxy = transformer_ops / max(1.0, sparse_ops)

    metrics = {
        "seen_recall": seen_correct / max(1, seen_total),
        "grammar_accuracy": grammar_correct / max(1, grammar_total),
        "topic_coherence_200": topic_coherence,
        "latency_ratio_1000_vs_10": latency_ratio,
        "burst_energy_proxy": energy_proxy,
        "suppression_rate": memory.metrics()["suppression_rate"],
    }
    checks = {
        "seen_recall": metrics["seen_recall"] >= args.seen_target,
        "grammar": metrics["grammar_accuracy"] >= args.grammar_target,
        "coherence": metrics["topic_coherence_200"] >= args.coherence_target,
        "latency": metrics["latency_ratio_1000_vs_10"] <= args.latency_ratio_target,
        "energy": metrics["burst_energy_proxy"] >= args.energy_target,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seen-target", type=float, default=1.0)
    parser.add_argument("--grammar-target", type=float, default=0.95)
    parser.add_argument("--coherence-target", type=float, default=0.50)
    parser.add_argument("--latency-ratio-target", type=float, default=1.5)
    parser.add_argument("--energy-target", type=float, default=100.0)
    parser.add_argument("--tokens-for-energy", type=int, default=1000)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: Bio Cortical Column")
    print("=" * 88)
    for key, value in result["metrics"].items():
        print(f"{key:<28} {value:.4f}")
    print(f"Overall status:              {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_bio_cortical_column", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
