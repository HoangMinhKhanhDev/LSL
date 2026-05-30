"""Phase 7 diverse-data benchmark for code, math, dialogue, and QA."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import ReasoningWorkspace, TraceReasoningMemory, WorldMemory


def evaluate(args):
    trace = TraceReasoningMemory()
    workspace = ReasoningWorkspace()
    world = WorldMemory(capacity=8192)
    correct = {"math": 0, "code": 0, "dialogue": 0, "qa": 0}
    total = {key: args.items for key in correct}
    for i in range(args.items):
        prompt = f"Start at {i % 11}. Add 5, multiply by 2, subtract {i % 3}."
        answer = ((i % 11) + 5) * 2 - (i % 3)
        correct["math"] += int(trace.execute_math(prompt) == answer)

        a, b = i % 9 + 1, i % 7 + 2
        code_prompt = f"Stack program: PUSH {a}; PUSH {b}; ADD; DUP; MUL."
        correct["code"] += int(trace.execute_stack(code_prompt) == (a + b) * (a + b))

        speaker = 1000 + i
        reply = 2000 + (i % 5)
        workspace.bind_pair(speaker, 7, reply)
        correct["dialogue"] += int(workspace.resolve_pair(speaker, 7) == reply)

        entity = f"entity-{i:07d}"
        value = f"value_{i * 13 + 1}"
        world.observe_chunk(f"The launch code for {entity} is {value}.", source=f"qa:{i}")
        correct["qa"] += int(world.answer(f"What is the launch code for {entity}?").answer == value)

    metrics = {f"{key}_accuracy": correct[key] / max(1, total[key]) for key in correct}
    metrics["overall_accuracy"] = sum(metrics.values()) / max(1, len(metrics))
    checks = {
        "math": metrics["math_accuracy"] >= args.accuracy_target,
        "code": metrics["code_accuracy"] >= args.accuracy_target,
        "dialogue": metrics["dialogue_accuracy"] >= args.accuracy_target,
        "qa": metrics["qa_accuracy"] >= args.accuracy_target,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=128)
    parser.add_argument("--accuracy-target", type=float, default=0.90)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    m = result["metrics"]
    ok = bool(result["success"])
    print("Phase 7: Diverse Data")
    print("=" * 88)
    print(f"Math/code/dialogue/QA: {m['math_accuracy']:.2%} / {m['code_accuracy']:.2%} / {m['dialogue_accuracy']:.2%} / {m['qa_accuracy']:.2%}")
    print(f"Overall status:        {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase7_diverse_data", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
