"""Phase 5 learned hierarchy/routing benchmark."""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl.hierarchy import LearnedHierarchicalMemory


def make_examples(topics: int, examples_per_topic: int) -> List[Tuple[List[int], int]]:
    examples = []
    for topic in range(topics):
        base = topic * 100
        for i in range(examples_per_topic):
            examples.append(([base + i % 7, base + 20 + i % 5, base + 40 + i % 3], topic))
    return examples


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    train = make_examples(args.topics, args.examples_per_topic)
    test = make_examples(args.topics, max(4, args.examples_per_topic // 2))
    memory = LearnedHierarchicalMemory(route_cap=1)
    for tokens, topic in train:
        memory.observe(tokens, topic)

    correct = 0
    route_counts = []
    for tokens, topic in test:
        routes = memory.route(tokens)
        correct += int(routes and routes[0] == topic)
        route_counts.append(len(routes))
    accuracy = correct / max(1, len(test))
    route_sparsity = sum(route_counts) / max(1, len(route_counts) * args.topics)

    ablated = LearnedHierarchicalMemory(route_cap=1)
    ablated_correct = 0
    for tokens, topic in test:
        routes = ablated.route(tokens)
        ablated_correct += int(bool(routes) and routes[0] == topic)
    ablated_accuracy = ablated_correct / max(1, len(test))
    quality_drop = accuracy - ablated_accuracy

    return {
        "routing_accuracy": float(accuracy),
        "route_sparsity": float(route_sparsity),
        "ablated_accuracy": float(ablated_accuracy),
        "quality_drop": float(quality_drop),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=int, default=10)
    parser.add_argument("--examples-per-topic", type=int, default=32)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    checks = {
        "sparsity": result["route_sparsity"] <= 0.10,
        "accuracy": result["routing_accuracy"] >= 0.85,
        "ablation": result["quality_drop"] >= 0.20,
    }
    ok = all(checks.values())

    print("Phase 5: Hierarchy/Routing")
    print("=" * 80)
    print(f"Routing/topic accuracy: {result['routing_accuracy']:.2%} (target >=85%)")
    print(f"Route sparsity:         {result['route_sparsity']:.2%} (target <=10%)")
    print(f"Ablation quality drop:  {result['quality_drop']:.2%} (target >=20%)")
    print(f"Overall status:         {'PASS' if ok else 'FAIL'}")

    payload = {"benchmark": "hierarchy", "success": bool(ok), "checks": checks, "metrics": result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
