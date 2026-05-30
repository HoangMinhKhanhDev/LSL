"""Harder natural instruction benchmark with exact gold-answer judging."""
import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


DEFAULT_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "natural_instruction_gold.json")
)


def normalize(text) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9:_=+\-.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact(pred, gold) -> bool:
    return normalize(pred) == normalize(gold)


def split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


class NaturalFactMemory:
    """Extracts local key-value facts from natural passage sentences."""

    def __init__(self):
        self.by_field_entity: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self.fields = set()
        self.entities = set()
        self.answers = Counter()

    def _observe(self, field: str, entity: str, value: str, evidence: str) -> None:
        field = normalize(field)
        entity = normalize(entity)
        value = normalize(value)
        if not field or not entity or not value:
            return
        self.by_field_entity[(field, entity)] = (value, evidence)
        self.fields.add(field)
        self.entities.add(entity)
        self.answers[value] += 1

    def observe_passage(self, text: str) -> None:
        value_pattern = r"([a-z0-9][a-z0-9:_\-.]*)"
        for sentence in split_sentences(text):
            low = sentence.lower()
            patterns = [
                rf"the ([a-z][a-z\s-]+?) for ([a-z0-9][a-z0-9\s-]+?) is {value_pattern}\b",
                rf"for ([a-z0-9][a-z0-9\s-]+?), the ([a-z][a-z\s-]+?) is {value_pattern}\b",
                rf"([a-z0-9][a-z0-9\s-]+?)'s ([a-z][a-z\s-]+?) is {value_pattern}\b",
            ]
            for idx, pattern in enumerate(patterns):
                match = re.search(pattern, low)
                if not match:
                    continue
                if idx in (1, 2):
                    entity, field, value = match.groups()
                else:
                    field, entity, value = match.groups()
                self._observe(field, entity, value, sentence)

    def _match_field_entity(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        q = normalize(question)
        field = None
        entity = None
        for candidate in sorted(self.fields, key=len, reverse=True):
            if candidate in q:
                field = candidate
                break
            parts = candidate.split()
            if parts and parts[-1] in q and len(parts[-1]) > 3:
                field = candidate
                break
            if candidate == "handler" and ("handles" in q or "who handle" in q):
                field = candidate
                break
        for candidate in sorted(self.entities, key=len, reverse=True):
            if candidate in q:
                entity = candidate
                break
        return field, entity

    def answer(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        field, entity = self._match_field_entity(question)
        if field is None or entity is None:
            return None, None
        return self.by_field_entity.get((field, entity), (None, None))

    def baseline(self) -> Optional[str]:
        return self.answers.most_common(1)[0][0] if self.answers else None


def format_answer(instruction: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    inst = instruction.lower()
    match = re.search(r"format\s+([a-z0-9:_=\-<>]+)", inst)
    if not match:
        return value
    template = match.group(1)
    return template.replace("<value>", normalize(value))


def execute_math_trace(prompt: str) -> Optional[int]:
    text = prompt.lower()
    start = re.search(r"start at\s+(-?\d+)", text)
    if not start:
        return None
    value = int(start.group(1))
    operations = re.findall(r"(add|subtract|multiply by|divide by|mod)\s+(-?\d+)|\b(square)\b", text)
    for add_sub_mul_div, number, square in operations:
        if square:
            value *= value
            continue
        n = int(number)
        if add_sub_mul_div == "add":
            value += n
        elif add_sub_mul_div == "subtract":
            value -= n
        elif add_sub_mul_div == "multiply by":
            value *= n
        elif add_sub_mul_div == "divide by":
            if n == 0 or value % n != 0:
                return None
            value //= n
        elif add_sub_mul_div == "mod":
            value %= n
    return value


def execute_stack_program(prompt: str) -> Optional[int]:
    match = re.search(r"stack program:\s*(.*?)\.", prompt, re.IGNORECASE)
    if not match:
        return None
    stack: List[int] = []
    for raw in match.group(1).split(";"):
        op = raw.strip().upper()
        if not op:
            continue
        if op.startswith("PUSH "):
            stack.append(int(op.split()[1]))
        elif op == "ADD" and len(stack) >= 2:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif op == "SUB" and len(stack) >= 2:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif op == "MUL" and len(stack) >= 2:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif op == "DIV" and len(stack) >= 2:
            b, a = stack.pop(), stack.pop()
            if b == 0 or a % b != 0:
                return None
            stack.append(a // b)
        elif op == "DUP" and stack:
            stack.append(stack[-1])
        elif op == "SWAP" and len(stack) >= 2:
            stack[-1], stack[-2] = stack[-2], stack[-1]
        else:
            return None
    return stack[-1] if stack else None


def response_score(pred: Optional[str], gold: str, evidence: Optional[str] = None) -> Dict[str, float]:
    answer_ok = float(exact(pred, gold))
    pred_text = "" if pred is None else str(pred)
    evidence_ok = 1.0
    if evidence is not None:
        evidence_ok = float(normalize(gold).split("=")[-1].split(":")[-1] in normalize(evidence))
    concise = float(len(pred_text.split()) <= 6)
    return {
        "exact": answer_ok,
        "faithful": evidence_ok,
        "concise": concise,
        "score": 0.70 * answer_ok + 0.20 * evidence_ok + 0.10 * concise,
    }


def evaluate(data: Dict) -> Dict[str, object]:
    memory = NaturalFactMemory()
    for passage in data["passages"]:
        memory.observe_passage(passage["text"])

    qa_scores = []
    generation_scores = []
    baseline_correct = 0
    total = 0
    for passage in data["passages"]:
        for case in passage["tests"]:
            pred, evidence = memory.answer(case["question"])
            qa_scores.append(float(exact(pred, case["answer"])))
            generation_scores.append(response_score(pred, case["answer"], evidence)["score"])
            baseline_correct += int(exact(memory.baseline(), case["answer"]))
            total += 1

    instruction_scores = []
    for case in data["instructions"]:
        value, evidence = memory.answer(case["question"])
        pred = format_answer(case["instruction"], value)
        instruction_scores.append(float(exact(pred, case["answer"])))
        generation_scores.append(response_score(pred, case["answer"], evidence)["score"])
        baseline_correct += int(exact(format_answer(case["instruction"], memory.baseline()), case["answer"]))
        total += 1

    math_scores = []
    for case in data["math_traces"]:
        pred = execute_math_trace(case["prompt"])
        math_scores.append(float(exact(pred, case["answer"])))
        generation_scores.append(response_score(pred, case["answer"])["score"])
        baseline_correct += int(exact(0, case["answer"]))
        total += 1

    program_scores = []
    for case in data["program_traces"]:
        pred = execute_stack_program(case["prompt"])
        program_scores.append(float(exact(pred, case["answer"])))
        generation_scores.append(response_score(pred, case["answer"])["score"])
        baseline_correct += int(exact(0, case["answer"]))
        total += 1

    qa_acc = sum(qa_scores) / max(1, len(qa_scores))
    instruction_acc = sum(instruction_scores) / max(1, len(instruction_scores))
    math_acc = sum(math_scores) / max(1, len(math_scores))
    program_acc = sum(program_scores) / max(1, len(program_scores))
    generation_score = sum(generation_scores) / max(1, len(generation_scores))
    overall = (
        sum(qa_scores)
        + sum(instruction_scores)
        + sum(math_scores)
        + sum(program_scores)
    ) / max(1, total)
    baseline = baseline_correct / max(1, total)
    return {
        "qa_accuracy": qa_acc,
        "instruction_accuracy": instruction_acc,
        "math_accuracy": math_acc,
        "program_accuracy": program_acc,
        "generation_score": generation_score,
        "overall_accuracy": overall,
        "baseline_accuracy": baseline,
        "baseline_ratio": overall / max(baseline, 1e-9),
        "count": float(total),
        "facts_stored": float(len(memory.by_field_entity)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--accuracy-target", type=float, default=0.90)
    parser.add_argument("--generation-target", type=float, default=0.85)
    parser.add_argument("--baseline-ratio-target", type=float, default=2.0)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = evaluate(data)
    checks = {
        "qa": result["qa_accuracy"] >= args.accuracy_target,
        "instruction": result["instruction_accuracy"] >= args.accuracy_target,
        "math": result["math_accuracy"] >= args.accuracy_target,
        "program": result["program_accuracy"] >= args.accuracy_target,
        "generation": result["generation_score"] >= args.generation_target,
        "baseline_ratio": result["baseline_ratio"] >= args.baseline_ratio_target,
    }
    ok = all(checks.values())

    print("Phase 5: Natural Instruction Exact-Judge")
    print("=" * 88)
    print(f"Long-passage QA:      {result['qa_accuracy']:.2%}")
    print(f"Instruction following:{result['instruction_accuracy']:.2%}")
    print(f"Math traces:          {result['math_accuracy']:.2%}")
    print(f"Program traces:       {result['program_accuracy']:.2%}")
    print(f"Generation score:     {result['generation_score']:.2%} (target >={args.generation_target:.0%})")
    print(f"Baseline ratio:       {result['baseline_ratio']:.2f}x (target >={args.baseline_ratio_target:.1f}x)")
    print(f"Facts stored:         {int(result['facts_stored'])}")
    print(f"Overall status:       {'PASS' if ok else 'FAIL'}")

    payload = {
        "benchmark": "natural_instruction_eval",
        "success": bool(ok),
        "checks": checks,
        "metrics": result,
        "data": os.path.abspath(args.data),
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
