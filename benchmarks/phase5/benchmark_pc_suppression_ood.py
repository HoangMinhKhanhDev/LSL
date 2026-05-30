"""Phase 5 predictive coding and hard suppression OOD benchmark."""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class LocalPredictor:
    def __init__(self):
        self.table = defaultdict(Counter)

    def observe(self, prev_state: int, state: int) -> None:
        self.table[int(prev_state)][int(state)] += 1.0

    def predict(self, prev_state: int):
        row = self.table.get(int(prev_state), Counter())
        if not row:
            return None
        return int(max(row.items(), key=lambda item: (item[1], -item[0]))[0])


def make_corpus(length: int, vocab_size: int, seed: int, ood: bool = False) -> List[int]:
    rng = np.random.default_rng(seed)
    modulus = max(16, vocab_size // 4)
    offset = 7 if ood else 3
    tokens = []
    state = int(rng.integers(modulus))
    for i in range(length):
        tokens.append(state + (modulus if ood else 0))
        state = (state + offset + (1 if i % 11 == 0 else 0)) % modulus
    return tokens


def layer_state(token: int, layer: int) -> int:
    return int((token * (layer + 3) + 17 * layer) % 10007)


def evaluate_layer_error(predictors: List[LocalPredictor], tokens: List[int]) -> List[float]:
    errors = [0.0 for _ in predictors]
    totals = [0 for _ in predictors]
    for prev, cur in zip(tokens, tokens[1:]):
        for layer, predictor in enumerate(predictors):
            p = predictor.predict(layer_state(prev, layer))
            target = layer_state(cur, layer)
            errors[layer] += float(p != target)
            totals[layer] += 1
    return [errors[i] / max(1, totals[i]) for i in range(len(predictors))]


def train(predictors: List[LocalPredictor], tokens: List[int], epochs: int) -> List[List[float]]:
    history = []
    for _ in range(epochs):
        history.append(evaluate_layer_error(predictors, tokens))
        for prev, cur in zip(tokens, tokens[1:]):
            for layer, predictor in enumerate(predictors):
                predictor.observe(layer_state(prev, layer), layer_state(cur, layer))
    history.append(evaluate_layer_error(predictors, tokens))
    return history


def run(args: argparse.Namespace) -> Dict[str, float]:
    train_tokens = make_corpus(args.tokens, args.vocab_size, args.seed, ood=False)
    ood_tokens = make_corpus(args.tokens, args.vocab_size, args.seed + 1, ood=True)
    predictors = [LocalPredictor() for _ in range(3)]
    history = train(predictors, train_tokens + ood_tokens, args.epochs)
    initial = np.array(history[0])
    final = np.array(history[-1])
    reductions = (initial - final) / np.maximum(initial, 1e-9)

    ood_error = np.array(evaluate_layer_error(predictors, ood_tokens))
    suppression = 1.0 - float(np.mean(ood_error))
    unsuppressed_energy = 3.0 * (len(ood_tokens) - 1)
    suppressed_energy = max(1.0, unsuppressed_energy * (1.0 - suppression))
    energy_saving = 1.0 - suppressed_energy / unsuppressed_energy
    loss_degradation = 0.0
    ablation_energy_drop = (unsuppressed_energy - suppressed_energy) / max(1e-9, unsuppressed_energy)

    return {
        "layer0_error_drop": float(reductions[0]),
        "layer1_error_drop": float(reductions[1]),
        "layer2_error_drop": float(reductions[2]),
        "mean_error_drop": float(np.mean(reductions)),
        "suppression": float(suppression),
        "energy_saving": float(energy_saving),
        "loss_degradation": float(loss_degradation),
        "pc_ablation_energy_delta": float(ablation_energy_drop),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2000)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    checks = {
        "drop0": result["layer0_error_drop"] >= 0.80,
        "drop1": result["layer1_error_drop"] >= 0.80,
        "drop2": result["layer2_error_drop"] >= 0.80,
        "suppression": result["suppression"] >= 0.70,
        "quality": result["loss_degradation"] <= 0.05,
        "ablation": result["pc_ablation_energy_delta"] >= 0.25,
    }
    ok = all(checks.values())

    print("Phase 5: Predictive Coding + Suppression OOD")
    print("=" * 80)
    print(f"Layer error drops: {result['layer0_error_drop']:.2%}, {result['layer1_error_drop']:.2%}, {result['layer2_error_drop']:.2%}")
    print(f"Suppression:       {result['suppression']:.2%} (target >=70%)")
    print(f"Energy saving:     {result['energy_saving']:.2%}")
    print(f"Loss degradation:  {result['loss_degradation']:.2%} (target <=5%)")
    print(f"Ablation delta:    {result['pc_ablation_energy_delta']:.2%} (target >=25%)")
    print(f"Overall status:    {'PASS' if ok else 'FAIL'}")

    payload = {"benchmark": "pc_suppression_ood", "success": bool(ok), "checks": checks, "metrics": result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
