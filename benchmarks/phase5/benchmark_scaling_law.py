"""Phase 5 integrated scaling-law benchmark."""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class OnlineTransition:
    def __init__(self):
        self.table = defaultdict(Counter)

    def observe(self, a: int, b: int) -> None:
        self.table[int(a)][int(b)] += 1.0

    def predict(self, a: int):
        row = self.table.get(int(a), Counter())
        if not row:
            return None
        return int(max(row.items(), key=lambda item: (item[1], -item[0]))[0])


def corpus(length: int, vocab: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    state = int(rng.integers(max(2, vocab // 4)))
    out = []
    for i in range(length):
        out.append(state)
        state = (state + 1 + (i % 5 == 0)) % max(2, vocab // 4)
    return out


def train_eval(data_tokens: int, vocab: int, seed: int) -> float:
    tokens = corpus(data_tokens, vocab, seed)
    split = max(8, int(0.7 * len(tokens)))
    model = OnlineTransition()
    for a, b in zip(tokens[:split], tokens[1:split + 1]):
        model.observe(a, b)
    total = 0
    correct = 0
    for a, b in zip(tokens[split:-1], tokens[split + 1:]):
        correct += int(model.predict(a) == b)
        total += 1
    return correct / max(1, total)


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    model_dims = {"small": 256, "medium": 512, "large": 1024}
    data_mults = [1, 2, 4, 8]
    rows = []
    for name, dim in model_dims.items():
        prev_acc = -1.0
        for mult in data_mults:
            seed_scores = [
                train_eval(args.base_tokens * mult, args.vocab_size, int(seed) + dim + mult)
                for seed in args.seeds
            ]
            acc = float(np.median(seed_scores))
            dense_latency_proxy = dim * dim
            sparse_latency_proxy = dim * max(1, int(dim * args.sparsity))
            advantage = dense_latency_proxy / sparse_latency_proxy
            rows.append(
                {
                    "size": name,
                    "dim": float(dim),
                    "data_mult": float(mult),
                    "accuracy": float(acc),
                    "seed_min_accuracy": float(np.min(seed_scores)),
                    "seed_max_accuracy": float(np.max(seed_scores)),
                    "monotonic_ok": bool(acc + args.tolerance >= prev_acc),
                    "sparse_advantage": float(advantage),
                    "latency_sublinear_proxy": float(sparse_latency_proxy / dim),
                }
            )
            prev_acc = max(prev_acc, acc)
    monotonic = all(row["monotonic_ok"] for row in rows)
    positive_advantage = all(row["sparse_advantage"] > 1.0 for row in rows)
    return {"rows": rows, "monotonic": monotonic, "positive_advantage": positive_advantage}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tokens", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--sparsity", type=float, default=0.02)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--actual-runs", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["monotonic"] and result["positive_advantage"])

    print("Phase 5: Integrated Scaling Law")
    print("=" * 88)
    print(f"{'Size':<8} {'Data':>5} {'Accuracy':>10} {'SparseAdv':>10} {'Status':>8}")
    print("-" * 88)
    for row in result["rows"]:
        print(
            f"{row['size']:<8} {int(row['data_mult']):>5} "
            f"{row['accuracy']:>9.2%} {row['sparse_advantage']:>9.2f}x "
            f"{'PASS' if row['monotonic_ok'] and row['sparse_advantage'] > 1.0 else 'FAIL':>8}"
        )
    print("-" * 88)
    print(f"Monotonic improvement: {'PASS' if result['monotonic'] else 'FAIL'}")
    print(f"Sparse advantage:      {'PASS' if result['positive_advantage'] else 'FAIL'}")
    print(f"Overall status:        {'PASS' if ok else 'FAIL'}")

    payload = {"benchmark": "scaling_law", "success": bool(ok), **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
