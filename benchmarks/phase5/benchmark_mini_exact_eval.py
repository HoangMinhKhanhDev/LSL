"""Mini exact-answer QA/reasoning/coding benchmark."""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import RelationMemory, RoleBindingMemory


DEFAULT_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "mini_qa_reasoning_coding.json")
)


class SymbolTable:
    def __init__(self):
        self.ids: Dict[str, int] = {}
        self.names: Dict[int, str] = {}

    def id(self, value) -> int:
        key = str(value)
        if key not in self.ids:
            idx = len(self.ids) + 1
            self.ids[key] = idx
            self.names[idx] = key
        return self.ids[key]

    def name(self, idx: Optional[int]) -> Optional[str]:
        if idx is None:
            return None
        return self.names.get(int(idx))


class ExactAnswerBaseline:
    def __init__(self):
        self.answers = {}
        self.majority = Counter()

    def observe(self, key: Tuple, answer) -> None:
        self.answers[tuple(key)] = answer
        self.majority[answer] += 1

    def predict(self, key: Tuple):
        if tuple(key) in self.answers:
            return self.answers[tuple(key)]
        if self.majority:
            return self.majority.most_common(1)[0][0]
        return None


class QAExactMemory:
    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols
        self.object_by_sr: Dict[Tuple[int, int], int] = {}
        self.subject_by_ro: Dict[Tuple[int, int], int] = {}

    def observe_fact(self, subject: str, relation: str, obj: str) -> None:
        s = self.symbols.id(subject)
        r = self.symbols.id(relation)
        o = self.symbols.id(obj)
        self.object_by_sr[(s, r)] = o
        self.subject_by_ro[(r, o)] = s

    def answer(self, case: Dict):
        relation = self.symbols.id(case["relation"])
        if "subject" in case:
            value = self.object_by_sr.get((self.symbols.id(case["subject"]), relation))
            return self.symbols.name(value)
        value = self.subject_by_ro.get((relation, self.symbols.id(case["object"])))
        return self.symbols.name(value)


class LocalLinearCodeMemory:
    """Closed-form local rule memory for tiny deterministic code tasks."""

    def __init__(self):
        self.examples: Dict[str, List[Tuple[Tuple[int, ...], int]]] = defaultdict(list)

    def observe(self, function: str, inputs: Iterable[int], output: int) -> None:
        self.examples[str(function)].append((tuple(int(x) for x in inputs), int(output)))

    def _infer_single(self, examples: List[Tuple[Tuple[int, ...], int]], x: int) -> Optional[int]:
        if len(examples) < 2:
            return None
        (x0,), y0 = examples[0]
        (x1,), y1 = examples[1]
        if x1 == x0:
            return None
        slope = Fraction(y1 - y0, x1 - x0)
        intercept = Fraction(y0) - slope * x0
        value = slope * int(x) + intercept
        return int(value) if value.denominator == 1 else None

    def _infer_multi(self, examples: List[Tuple[Tuple[int, ...], int]], inputs: Tuple[int, ...]) -> Optional[int]:
        dim = len(inputs)
        anchor = None
        by_axis = {}
        for xs, y in examples:
            if len(xs) != dim:
                continue
            if all(v == 0 for v in xs):
                anchor = int(y)
            nonzero = [i for i, v in enumerate(xs) if v != 0]
            if len(nonzero) == 1:
                axis = nonzero[0]
                by_axis[axis] = (int(xs[axis]), int(y))
        if anchor is None or len(by_axis) < dim:
            return None
        total = Fraction(anchor)
        for axis, x in enumerate(inputs):
            step_x, step_y = by_axis[axis]
            coeff = Fraction(step_y - anchor, step_x)
            total += coeff * int(x)
        return int(total) if total.denominator == 1 else None

    def predict(self, function: str, inputs: Iterable[int]) -> Optional[int]:
        xs = tuple(int(x) for x in inputs)
        examples = self.examples.get(str(function), [])
        for seen_xs, y in examples:
            if seen_xs == xs:
                return y
        if len(xs) == 1:
            return self._infer_single(examples, xs[0])
        return self._infer_multi(examples, xs)


def load_data(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_qa(data: Dict, baseline: ExactAnswerBaseline) -> Dict[str, float]:
    symbols = SymbolTable()
    memory = QAExactMemory(symbols)
    for fact in data["qa"]["facts"]:
        memory.observe_fact(fact["subject"], fact["relation"], fact["object"])
        baseline.observe(("qa", fact["subject"], fact["relation"]), fact["object"])

    correct = 0
    base_correct = 0
    tests = data["qa"]["tests"]
    for case in tests:
        pred = memory.answer(case)
        base_key = ("qa", case.get("subject", "?"), case["relation"])
        base_pred = baseline.predict(base_key)
        correct += int(pred == case["answer"])
        base_correct += int(base_pred == case["answer"])
    return {
        "accuracy": correct / max(1, len(tests)),
        "baseline_accuracy": base_correct / max(1, len(tests)),
        "count": float(len(tests)),
    }


def evaluate_reasoning(data: Dict, baseline: ExactAnswerBaseline) -> Dict[str, float]:
    symbols = SymbolTable()
    relation = RelationMemory(max_hops=4)
    role = RoleBindingMemory()
    recipient_by_svo = {}

    for chain in data["reasoning"]["chains"]:
        ids = [symbols.id(x) for x in chain]
        for a, b in zip(ids, ids[1:]):
            relation.observe(a, b, relation="chain")
            baseline.observe(("reason", "direct", symbols.name(a)), symbols.name(b))

    for event in data["reasoning"]["role_events"]:
        subject = symbols.id(event["subject"])
        verb = symbols.id(event["verb"])
        obj = symbols.id(event["object"])
        recipient = symbols.id(event["recipient"])
        role.observe_event(subject, verb, obj)
        recipient_by_svo[(subject, verb, obj)] = recipient
        baseline.observe(("reason", "role_object", event["subject"], event["verb"]), event["object"])

    correct = 0
    base_correct = 0
    tests = data["reasoning"]["tests"]
    for case in tests:
        if case["type"] == "multihop":
            pred_id = relation.predict_multihop(symbols.id(case["source"]), hops=case["hops"], relation="chain")
            pred = symbols.name(pred_id)
            base_pred = baseline.predict(("reason", "direct", case["source"]))
        elif case["type"] == "role_object":
            pred = symbols.name(role.predict_object(symbols.id(case["subject"]), symbols.id(case["verb"])))
            base_pred = baseline.predict(("reason", "role_object", case["subject"], case["verb"]))
        elif case["type"] == "role_subject":
            pred = symbols.name(role.predict_subject(symbols.id(case["verb"]), symbols.id(case["object"])))
            base_pred = baseline.predict(("reason", "role_subject", case["verb"], case["object"]))
        else:
            pred = symbols.name(
                recipient_by_svo.get(
                    (symbols.id(case["subject"]), symbols.id(case["verb"]), symbols.id(case["object"]))
                )
            )
            base_pred = baseline.predict(("reason", "recipient", case["subject"], case["verb"], case["object"]))
        correct += int(pred == case["answer"])
        base_correct += int(base_pred == case["answer"])
    return {
        "accuracy": correct / max(1, len(tests)),
        "baseline_accuracy": base_correct / max(1, len(tests)),
        "count": float(len(tests)),
    }


def evaluate_coding(data: Dict, baseline: ExactAnswerBaseline) -> Dict[str, float]:
    memory = LocalLinearCodeMemory()
    for ex in data["coding"]["examples"]:
        memory.observe(ex["function"], ex["input"], ex["output"])
        baseline.observe(("code", ex["function"], tuple(ex["input"])), ex["output"])

    correct = 0
    base_correct = 0
    tests = data["coding"]["tests"]
    for case in tests:
        pred = memory.predict(case["function"], case["input"])
        base_pred = baseline.predict(("code", case["function"], tuple(case["input"])))
        correct += int(pred == case["answer"])
        base_correct += int(base_pred == case["answer"])
    return {
        "accuracy": correct / max(1, len(tests)),
        "baseline_accuracy": base_correct / max(1, len(tests)),
        "count": float(len(tests)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--accuracy-target", type=float, default=0.90)
    parser.add_argument("--baseline-ratio-target", type=float, default=0.80)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_data(args.data)
    baseline = ExactAnswerBaseline()
    qa = evaluate_qa(data, baseline)
    reasoning = evaluate_reasoning(data, baseline)
    coding = evaluate_coding(data, baseline)
    total_correct = qa["accuracy"] * qa["count"] + reasoning["accuracy"] * reasoning["count"] + coding["accuracy"] * coding["count"]
    total_base = (
        qa["baseline_accuracy"] * qa["count"]
        + reasoning["baseline_accuracy"] * reasoning["count"]
        + coding["baseline_accuracy"] * coding["count"]
    )
    total_count = qa["count"] + reasoning["count"] + coding["count"]
    overall = total_correct / max(1.0, total_count)
    baseline_overall = total_base / max(1.0, total_count)
    baseline_ratio = overall / max(baseline_overall, 1e-9)

    checks = {
        "qa": qa["accuracy"] >= args.accuracy_target,
        "reasoning": reasoning["accuracy"] >= args.accuracy_target,
        "coding": coding["accuracy"] >= args.accuracy_target,
        "baseline_ratio": baseline_ratio >= args.baseline_ratio_target,
    }
    ok = all(checks.values())

    print("Phase 5: Mini Exact-Answer QA/Reasoning/Coding")
    print("=" * 88)
    print(f"QA accuracy:         {qa['accuracy']:.2%} (baseline {qa['baseline_accuracy']:.2%})")
    print(f"Reasoning accuracy:  {reasoning['accuracy']:.2%} (baseline {reasoning['baseline_accuracy']:.2%})")
    print(f"Coding accuracy:     {coding['accuracy']:.2%} (baseline {coding['baseline_accuracy']:.2%})")
    print(f"Overall accuracy:    {overall:.2%}")
    print(f"Baseline ratio:      {baseline_ratio:.2f}x (target >={args.baseline_ratio_target:.2f}x)")
    print(f"Overall status:      {'PASS' if ok else 'FAIL'}")

    payload = {
        "benchmark": "mini_exact_eval",
        "success": bool(ok),
        "checks": checks,
        "qa": qa,
        "reasoning": reasoning,
        "coding": coding,
        "overall_accuracy": float(overall),
        "baseline_overall_accuracy": float(baseline_overall),
        "baseline_ratio": float(baseline_ratio),
        "data": os.path.abspath(args.data),
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
