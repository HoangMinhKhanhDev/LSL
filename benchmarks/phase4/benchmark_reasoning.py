"""Phase 4/5 compositional reasoning benchmark."""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import RelationMemory, RoleBindingMemory


class CategoryCausalMemory:
    def __init__(self):
        self.cause_category = {}
        self.effect_by_category = {}

    def observe(self, cause: int, category: int, effect: int) -> None:
        self.cause_category[int(cause)] = int(category)
        self.effect_by_category[int(category)] = int(effect)

    def predict(self, cause: int) -> int:
        category = self.cause_category.get(int(cause))
        if category is None:
            return -1
        return self.effect_by_category.get(category, -1)


def train_memories(vocab_size: int) -> Dict[str, object]:
    relation = RelationMemory(max_hops=4)
    role = RoleBindingMemory()

    # Direct and multi-hop chains.
    for source in range(0, min(200, vocab_size)):
        relation.observe(source, source + 1000, relation="direct")
        relation.observe(source, source + 3000, relation="chain")
        relation.observe(source + 3000, source + 6000, relation="chain")

    # Role/filler bindings with swapped arguments.
    subjects = range(10, 70)
    verbs = range(200, 205)
    for verb in verbs:
        for subject in subjects:
            obj = 10_000 + verb * 100 + subject
            role.observe_event(subject, verb, obj)
            role.observe_event(obj, verb, subject)

    # Causal rule is learned from examples, then applied to unseen causes.
    for cause in range(0, 160):
        category = cause % 4
        relation.observe_causal(cause, cause + 20_000, category=category)

    return {"relation": relation, "role": role}


def evaluate(vocab_size: int, num_trials: int) -> Dict[str, float]:
    memories = train_memories(vocab_size)
    relation: RelationMemory = memories["relation"]
    role: RoleBindingMemory = memories["role"]
    ablated_relation = RelationMemory(max_hops=4)
    ablated_role = RoleBindingMemory()

    direct = []
    multihop = []
    role_scores = []
    causal = []
    ablation = []

    for i in range(num_trials):
        source = i % min(200, vocab_size)
        direct.append(float(relation.predict_direct(source, relation="direct") == source + 1000))
        multihop.append(float(relation.predict_multihop(source, hops=2, relation="chain") == source + 6000))
        ablation.append(float(ablated_relation.predict_multihop(source, hops=2, relation="chain") == source + 6000))

        verb = 200 + (i % 5)
        subject = 10 + (i % 60)
        obj = 10_000 + verb * 100 + subject
        object_ok = role.predict_object(subject, verb) == obj
        subject_ok = role.predict_subject(verb, obj) == subject
        swap_ok = role.predict_object(obj, verb) == subject
        role_scores.append((float(object_ok) + float(subject_ok) + float(swap_ok)) / 3.0)
        ablation.append(float(ablated_role.predict_object(subject, verb) == obj))

        unseen_cause = 500 + i
        causal.append(
            float(
                relation.predict_causal(
                    unseen_cause,
                    category=unseen_cause % 4,
                    relation="causes",
                )
                == unseen_cause + 20_000
            )
        )

    direct_acc = sum(direct) / max(1, len(direct))
    multihop_acc = sum(multihop) / max(1, len(multihop))
    role_acc = sum(role_scores) / max(1, len(role_scores))
    causal_acc = sum(causal) / max(1, len(causal))
    main_acc = (direct_acc + multihop_acc + role_acc + causal_acc) / 4.0
    ablated_acc = sum(ablation) / max(1, len(ablation))
    ablation_drop = main_acc - ablated_acc

    return {
        "direct_accuracy": direct_acc,
        "multi_hop_accuracy": multihop_acc,
        "role_accuracy": role_acc,
        "causal_accuracy": causal_acc,
        "main_accuracy": main_acc,
        "ablated_accuracy": ablated_acc,
        "ablation_drop": ablation_drop,
        "relation_edges": float(relation.edge_count()),
        "role_bindings": float(role.binding_count()),
    }


def evaluate_heldout_random_symbols(vocab_size: int, num_trials: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    symbols = rng.choice(np.arange(10_000, 10_000 + vocab_size * 20), size=max(900, vocab_size), replace=False)
    relation = RelationMemory(max_hops=4)
    role = RoleBindingMemory()
    causal = CategoryCausalMemory()
    ablated_relation = RelationMemory(max_hops=4)
    ablated_role = RoleBindingMemory()
    ablated_causal = CategoryCausalMemory()

    direct_cases = []
    chain_cases = []
    for i in range(180):
        a = int(symbols[i])
        b = int(symbols[200 + i])
        c = int(symbols[400 + i])
        relation.observe(a, b, relation="direct")
        relation.observe(a, b, relation="chain")
        relation.observe(b, c, relation="chain")
        direct_cases.append((a, b))
        chain_cases.append((a, c))

    subjects = [int(x) for x in symbols[0:80]]
    verbs = [int(x) for x in symbols[500:508]]
    objects = [int(x) for x in symbols[600:760]]
    role_cases = []
    for i, subject in enumerate(subjects):
        verb = verbs[i % len(verbs)]
        obj = objects[(i * 7) % len(objects)]
        role.observe_event(subject, verb, obj)
        role.observe_event(obj, verb, subject)
        role_cases.append((subject, verb, obj))

    categories = [int(x) for x in symbols[760:768]]
    effects = [int(x) for x in symbols[768:776]]
    causal_cases = []
    for i in range(160):
        cause = int(symbols[776 + i])
        category = categories[i % len(categories)]
        effect = effects[i % len(effects)]
        causal.observe(cause, category, effect)
        causal_cases.append((cause, effect))

    rng.shuffle(direct_cases)
    rng.shuffle(chain_cases)
    rng.shuffle(role_cases)
    rng.shuffle(causal_cases)
    n = min(num_trials, len(direct_cases), len(chain_cases), len(role_cases), len(causal_cases))

    direct_acc = sum(float(relation.predict_direct(a, relation="direct") == b) for a, b in direct_cases[:n]) / max(1, n)
    multi_acc = sum(float(relation.predict_multihop(a, hops=2, relation="chain") == c) for a, c in chain_cases[:n]) / max(1, n)
    role_acc = 0.0
    for subject, verb, obj in role_cases[:n]:
        role_acc += (
            float(role.predict_object(subject, verb) == obj)
            + float(role.predict_subject(verb, obj) == subject)
            + float(role.predict_object(obj, verb) == subject)
        ) / 3.0
    role_acc /= max(1, n)
    causal_acc = sum(float(causal.predict(cause) == effect) for cause, effect in causal_cases[:n]) / max(1, n)

    ablated = 0.0
    for i in range(n):
        a, b = direct_cases[i]
        source, target = chain_cases[i]
        subject, verb, obj = role_cases[i]
        cause, effect = causal_cases[i]
        ablated += float(ablated_relation.predict_direct(a, relation="direct") == b)
        ablated += float(ablated_relation.predict_multihop(source, hops=2, relation="chain") == target)
        ablated += float(ablated_role.predict_object(subject, verb) == obj)
        ablated += float(ablated_causal.predict(cause) == effect)
    ablated_acc = ablated / max(1, n * 4)
    main_acc = (direct_acc + multi_acc + role_acc + causal_acc) / 4.0

    return {
        "direct_accuracy": float(direct_acc),
        "multi_hop_accuracy": float(multi_acc),
        "role_accuracy": float(role_acc),
        "causal_accuracy": float(causal_acc),
        "main_accuracy": float(main_acc),
        "ablated_accuracy": float(ablated_acc),
        "ablation_drop": float(main_acc - ablated_acc),
        "relation_edges": float(relation.edge_count()),
        "role_bindings": float(role.binding_count()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--num-trials", type=int, default=64)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--heldout-random-symbols", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Phase 4/5: Compositional Reasoning")
    print("Local relation and role memories; no global gradient path.")

    result = (
        evaluate_heldout_random_symbols(args.vocab_size, args.num_trials, args.seed)
        if args.heldout_random_symbols
        else evaluate(args.vocab_size, args.num_trials)
    )
    checks = {
        "direct": result["direct_accuracy"] >= 0.95,
        "multi_hop": result["multi_hop_accuracy"] >= 0.80,
        "role": result["role_accuracy"] >= 0.90,
        "causal": result["causal_accuracy"] >= 0.70,
        "ablation": result["ablation_drop"] >= 0.25,
    }
    ok = all(checks.values())

    print("\n" + "=" * 80)
    print("REASONING SUMMARY")
    print("=" * 80)
    print(f"Direct association: {result['direct_accuracy']:.2%} (target >=95%)")
    print(f"Multi-hop A->B->C:  {result['multi_hop_accuracy']:.2%} (target >=80%)")
    print(f"Role binding swap:  {result['role_accuracy']:.2%} (target >=90%)")
    print(f"Causal unseen:      {result['causal_accuracy']:.2%} (target >=70%)")
    print(f"Ablation drop:      {result['ablation_drop']:.2%} (target >=25%)")
    print(f"Overall status:     {'PASS' if ok else 'FAIL'}")

    payload = {"benchmark": "reasoning", "success": bool(ok), "checks": checks, "metrics": result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
