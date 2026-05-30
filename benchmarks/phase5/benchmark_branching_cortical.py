"""Phase 5 branching cortical sequence memory benchmark."""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import CorticalColumnSequenceMemory


VOCAB = {
    "money": 0,
    "cash": 1,
    "river": 2,
    "shore": 3,
    "bank": 4,
    "loan": 5,
    "teller": 6,
    "water": 7,
    "sand": 8,
    "market": 9,
    "trade": 10,
    "forest": 11,
    "trail": 12,
    "pitcher": 13,
    "throws": 14,
    "plant": 15,
    "grows": 16,
}


SEQUENCES = [
    ["money", "bank", "loan", "market", "trade"],
    ["cash", "bank", "teller", "money", "bank"],
    ["river", "bank", "water", "shore", "bank"],
    ["shore", "bank", "sand", "river", "bank"],
    ["forest", "trail", "plant", "grows"],
    ["pitcher", "throws", "plant", "grows"],
]

TESTS = [
    (["money", "bank"], "loan"),
    (["cash", "bank"], "teller"),
    (["river", "bank"], "water"),
    (["shore", "bank"], "sand"),
    (["forest", "trail", "plant"], "grows"),
    (["pitcher", "throws", "plant"], "grows"),
]


def ids(words: List[str]) -> List[int]:
    return [VOCAB[w] for w in words]


def train_model(epochs: int) -> CorticalColumnSequenceMemory:
    model = CorticalColumnSequenceMemory(vocab_size=len(VOCAB), cells_per_column=80, sparsity=0.05, seed=42)
    corpus = [ids(seq) for seq in SEQUENCES]
    for _ in range(epochs):
        for seq in corpus:
            model.reset_state()
            for token in seq:
                model.forward(token, learn=True)
    return model


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    model = train_model(args.epochs)
    correct = 0
    for context_words, target_word in TESTS:
        model.reset_state()
        for token in ids(context_words):
            model.forward(token, learn=False)
        pred = int(np.argmax(model.predict_next_token_scores()))
        correct += int(pred == VOCAB[target_word])
    branching = correct / len(TESTS)

    generated = model.generate(ids(["money", "bank"]), max_steps=24, top_k=3)
    transitions = {(a, b) for seq in SEQUENCES for a, b in zip(ids(seq), ids(seq)[1:])}
    good = 0
    total = 0
    for a, b in zip(generated, generated[1:]):
        good += int((a, b) in transitions)
        total += 1
    coherence = good / max(1, total)
    trigrams = [tuple(generated[i:i + 3]) for i in range(max(0, len(generated) - 2))]
    loop_rate = 1.0 - (len(set(trigrams)) / max(1, len(trigrams)))

    return {
        "branching_accuracy": float(branching),
        "coherence": float(coherence),
        "loop_rate": float(loop_rate),
        "segment_count": float(model.metrics()["segment_count"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    checks = {
        "branching": result["branching_accuracy"] >= 0.90,
        "coherence": result["coherence"] >= 0.75,
        "loop": result["loop_rate"] <= 0.05,
    }
    ok = all(checks.values())

    print("Phase 5: Branching Cortical Sequence Memory")
    print("=" * 80)
    print(f"Branching disambiguation: {result['branching_accuracy']:.2%} (target >=90%)")
    print(f"Generation coherence:     {result['coherence']:.2%} (target >=75%)")
    print(f"Loop rate:                {result['loop_rate']:.2%} (target <=5%)")
    print(f"Segments:                 {int(result['segment_count'])}")
    print(f"Overall status:           {'PASS' if ok else 'FAIL'}")

    payload = {"benchmark": "branching_cortical", "success": bool(ok), "checks": checks, "metrics": result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
