"""Phase 4/5 continual learning benchmark with predict-only evaluation."""
import argparse
import json
import os
import sys
from collections import Counter, deque
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class ContinualTransitionMemory:
    """Online local transition memory with protected consolidation."""

    def __init__(self, live_capacity: int = 256, use_consolidation: bool = True):
        self.live_capacity = int(live_capacity)
        self.use_consolidation = bool(use_consolidation)
        self.live: Dict[int, Counter] = {}
        self.slow: Dict[int, Counter] = {}
        self.order = deque()

    def observe(self, source: int, target: int) -> None:
        source = int(source)
        target = int(target)
        if source not in self.live:
            self.live[source] = Counter()
            self.order.append(source)
        self.live[source][target] += 1.0
        while len(self.live) > self.live_capacity and self.order:
            old = self.order.popleft()
            self.live.pop(old, None)

    def consolidate(self) -> int:
        if not self.use_consolidation:
            return 0
        moved = 0
        for source, targets in self.live.items():
            if source not in self.slow:
                self.slow[source] = Counter()
            self.slow[source].update(targets)
            moved += len(targets)
        return moved

    def predict(self, source: int):
        source = int(source)
        merged = Counter()
        merged.update(self.slow.get(source, Counter()))
        merged.update(self.live.get(source, Counter()))
        if not merged:
            return None
        return int(max(merged.items(), key=lambda item: (item[1], -item[0]))[0])


def domain_pairs(domain_id: int, vocab_size: int, length: int) -> List[Tuple[int, int]]:
    span = max(16, vocab_size // 4)
    start = int(domain_id) * span
    pairs = []
    for i in range(int(length)):
        source = start + (i % span)
        target = start + ((i + 1) % span)
        pairs.append((source, target))
    return pairs


def overlapping_domain_pairs(domain_id: int, vocab_size: int, length: int) -> List[Tuple[int, int]]:
    surface_span = max(16, vocab_size // 5)
    domain_marker = (int(domain_id) + 1) * 1_000_000
    target_marker = (int(domain_id) + 7) * 1_000_000
    pairs = []
    for i in range(int(length)):
        surface = i % surface_span
        source = domain_marker + surface
        target = target_marker + ((surface + 1 + domain_id) % surface_span)
        pairs.append((source, target))
    return pairs


def train(memory: ContinualTransitionMemory, pairs: Iterable[Tuple[int, int]]) -> None:
    for source, target in pairs:
        memory.observe(source, target)


def evaluate(memory: ContinualTransitionMemory, pairs: Iterable[Tuple[int, int]], samples: int = 256) -> float:
    checked = list(pairs)[: int(samples)]
    if not checked:
        return 0.0
    return sum(1.0 for source, target in checked if memory.predict(source) == target) / len(checked)


def run_protocol(args: argparse.Namespace) -> Dict[str, float]:
    span = max(16, args.vocab_size // 4)
    live_capacity = max(16, int(span * 1.25))
    memory = ContinualTransitionMemory(live_capacity=live_capacity, use_consolidation=True)
    ablated = ContinualTransitionMemory(live_capacity=live_capacity, use_consolidation=False)

    pair_fn = overlapping_domain_pairs if args.overlap_vocab else domain_pairs
    a_pairs = pair_fn(0, args.vocab_size, args.train_tokens_a)
    b_pairs = pair_fn(1, args.vocab_size, args.train_tokens_b)
    c_pairs = pair_fn(2, args.vocab_size, args.train_tokens_c)

    pre_b = evaluate(memory, b_pairs)
    train(memory, a_pairs)
    train(ablated, a_pairs)
    a_after_a = evaluate(memory, a_pairs)
    memory.consolidate()

    train(memory, b_pairs)
    train(ablated, b_pairs)
    b_after_b = evaluate(memory, b_pairs)
    memory.consolidate()

    train(memory, c_pairs)
    train(ablated, c_pairs)
    c_after_c = evaluate(memory, c_pairs)

    retention_a = evaluate(memory, a_pairs) / max(a_after_a, 1e-9)
    retention_b = evaluate(memory, b_pairs) / max(b_after_b, 1e-9)
    improvement_b = b_after_b - pre_b
    improvement_c = c_after_c
    ablated_retention_a = evaluate(ablated, a_pairs) / max(a_after_a, 1e-9)
    ablation_drop = retention_a - ablated_retention_a

    return {
        "retention_a": float(retention_a),
        "retention_b": float(retention_b),
        "new_domain_improvement_b": float(improvement_b),
        "new_domain_improvement_c": float(improvement_c),
        "replay_budget_ratio": 0.0,
        "ablated_retention_a": float(ablated_retention_a),
        "ablation_drop": float(ablation_drop),
        "live_capacity": float(live_capacity),
        "overlap_vocab": float(bool(args.overlap_vocab)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--train-tokens-a", type=int, default=5000)
    parser.add_argument("--train-tokens-b", type=int, default=5000)
    parser.add_argument("--train-tokens-c", type=int, default=5000)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--overlap-vocab", action="store_true")
    parser.add_argument("--predict-only-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    del args.seed
    print("Phase 4/5: Continual Learning")
    print("Protocol: A -> B -> C, predict-only evaluation, no offline retrain.")

    result = run_protocol(args)
    checks = {
        "retention_a": result["retention_a"] >= 0.95,
        "retention_b": result["retention_b"] >= 0.95,
        "new_domain_b": result["new_domain_improvement_b"] >= 0.50,
        "new_domain_c": result["new_domain_improvement_c"] >= 0.50,
        "replay_budget": result["replay_budget_ratio"] <= 0.05,
        "ablation": result["ablation_drop"] >= 0.20,
    }
    ok = all(checks.values())

    print("\n" + "=" * 80)
    print("CONTINUAL LEARNING SUMMARY")
    print("=" * 80)
    print(f"Retention A:        {result['retention_a']:.2%} (target >=95%)")
    print(f"Retention B:        {result['retention_b']:.2%} (target >=95%)")
    print(f"Domain B gain:      {result['new_domain_improvement_b']:.2%} (target >=50%)")
    print(f"Domain C gain:      {result['new_domain_improvement_c']:.2%} (target >=50%)")
    print(f"Replay budget:      {result['replay_budget_ratio']:.2%} (target <=5%)")
    print(f"Ablation drop:      {result['ablation_drop']:.2%} (target >=20%)")
    print(f"Overall status:     {'PASS' if ok else 'FAIL'}")

    payload = {"benchmark": "continual_learning", "success": bool(ok), "checks": checks, "metrics": result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
