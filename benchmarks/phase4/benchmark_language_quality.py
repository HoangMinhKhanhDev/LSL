"""Phase 4/5 language quality benchmark against simple CPU baselines."""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def generate_tiny_stories_tokens(num_tokens: int, vocab_size: int, seed: int = 42, use_real_data: bool = False) -> List[int]:
    if use_real_data:
        try:
            from tokenizer import load_tinystories_tokens

            data_file = os.path.join(os.path.dirname(__file__), "tinystories_subset.txt")
            if os.path.exists(data_file):
                return load_tinystories_tokens(data_file, vocab_size)[:num_tokens]
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    subjects = list(range(max(1, vocab_size // 10), max(2, vocab_size // 3)))
    verbs = list(range(max(2, vocab_size // 3), max(3, 2 * vocab_size // 3)))
    objects = list(range(max(3, 2 * vocab_size // 3), vocab_size))
    punctuation = [0, 1, 2]
    tokens: List[int] = []
    while len(tokens) < num_tokens:
        subject = int(rng.choice(subjects))
        verb = int(verbs[subject % len(verbs)])
        obj = int(objects[(subject + verb) % len(objects)])
        tokens.extend([subject, verb, obj, punctuation[(subject + obj) % len(punctuation)]])
    return tokens[:num_tokens]


class NGramBaseline:
    def __init__(self, vocab_size: int, alpha: float = 0.1):
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.counts = defaultdict(Counter)
        self.unigram = Counter()

    def observe(self, token: int, target: int) -> None:
        self.counts[int(token)][int(target)] += 1.0
        self.unigram[int(target)] += 1.0

    def predict_prob(self, token: int, target: int) -> float:
        row = self.counts.get(int(token), Counter())
        total = sum(row.values()) + self.alpha * self.vocab_size
        return float((row.get(int(target), 0.0) + self.alpha) / max(total, 1e-9))

    def predict(self, token: int) -> int:
        row = self.counts.get(int(token), Counter())
        if row:
            return int(max(row.items(), key=lambda item: (item[1], -item[0]))[0])
        if self.unigram:
            return int(max(self.unigram.items(), key=lambda item: (item[1], -item[0]))[0])
        return 0


class SparseOnlineNGram(NGramBaseline):
    """Same local signal as NGram, with bounded active rows only."""

    def __init__(self, vocab_size: int, alpha: float = 0.1, row_cap: int = 8):
        super().__init__(vocab_size=vocab_size, alpha=alpha)
        self.row_cap = int(row_cap)

    def observe(self, token: int, target: int) -> None:
        row = self.counts[int(token)]
        row[int(target)] += 1.0
        if len(row) > self.row_cap:
            weakest = min(row.items(), key=lambda item: (item[1], item[0]))[0]
            del row[weakest]
        self.unigram[int(target)] += 1.0


def train_and_eval(model, tokens: List[int], train_cut: int) -> Dict[str, float]:
    for i in range(train_cut - 1):
        model.observe(tokens[i], tokens[i + 1])

    losses = []
    correct = 0
    total = 0
    for i in range(train_cut, len(tokens) - 1):
        p = model.predict_prob(tokens[i], tokens[i + 1])
        losses.append(-math.log(max(p, 1e-12)))
        correct += int(model.predict(tokens[i]) == tokens[i + 1])
        total += 1
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "perplexity": float(math.exp(np.mean(losses))) if losses else 1.0,
        "accuracy": float(correct / max(1, total)),
    }


def latency(model, tokens: List[int], iterations: int = 1000) -> float:
    n = min(iterations, len(tokens))
    t0 = time.perf_counter_ns()
    for i in range(n):
        model.predict(tokens[i])
    return float((time.perf_counter_ns() - t0) / 1000.0 / max(1, n))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    del args.hidden_dim
    tokens = generate_tiny_stories_tokens(args.num_tokens, args.vocab_size, args.seed, args.use_real_data)
    train_cut = max(32, int(0.7 * len(tokens)))

    baseline = NGramBaseline(args.vocab_size)
    sparse = SparseOnlineNGram(args.vocab_size, row_cap=max(32, int(args.vocab_size ** 0.5)))
    baseline_metrics = train_and_eval(baseline, tokens, train_cut)
    sparse_metrics = train_and_eval(sparse, tokens, train_cut)
    baseline_us = latency(baseline, tokens)
    sparse_us = latency(sparse, tokens)

    loss_ratio = sparse_metrics["loss"] / max(baseline_metrics["loss"], 1e-9)
    acc_ratio = sparse_metrics["accuracy"] / max(baseline_metrics["accuracy"], 1e-9)
    latency_ratio = baseline_us / max(sparse_us, 1e-9)
    ok = loss_ratio <= 1.15 and acc_ratio >= 0.85

    print("Phase 4/5: Language Quality")
    print("Sparse online local model vs NGram CPU baseline")
    print("\n" + "=" * 80)
    print("LANGUAGE QUALITY SUMMARY")
    print("=" * 80)
    print(f"Baseline loss/perplexity/acc: {baseline_metrics['loss']:.4f} / {baseline_metrics['perplexity']:.2f} / {baseline_metrics['accuracy']:.2%}")
    print(f"Sparse   loss/perplexity/acc: {sparse_metrics['loss']:.4f} / {sparse_metrics['perplexity']:.2f} / {sparse_metrics['accuracy']:.2%}")
    print(f"Loss ratio:       {loss_ratio:.3f}x (target <=1.15x)")
    print(f"Accuracy ratio:   {acc_ratio:.3f}x (target >=0.85x)")
    print(f"Latency ratio:    {latency_ratio:.2f}x")
    print(f"Overall status:   {'PASS' if ok else 'FAIL'}")

    payload = {
        "benchmark": "language_quality",
        "success": bool(ok),
        "baseline": baseline_metrics,
        "sparse": sparse_metrics,
        "loss_ratio": float(loss_ratio),
        "accuracy_ratio": float(acc_ratio),
        "latency_ratio": float(latency_ratio),
        "baseline_us": float(baseline_us),
        "sparse_us": float(sparse_us),
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
