"""Phase 8 external-style gold-answer benchmark."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import IntegratedLSLAgent


DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "phase8_external_gold.json"))


def evaluate(args):
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    corpus = " ".join(p["text"] for p in data["passages"])
    agent = IntegratedLSLAgent(vocab_size=2000, seed=args.seed)
    agent.build_tokenizer(corpus)
    for passage in data["passages"]:
        agent.observe_text(passage["text"], source="external_gold")
    for idx, event in enumerate(data["events"]):
        agent.observe_event(event["subject"], event["relation"], event["object"], episode_id=idx, evidence_id=idx)

    counts = {"qa": [0, 0], "events": [0, 0], "math": [0, 0], "programs": [0, 0], "dialogue": [0, 0]}
    for passage in data["passages"]:
        for case in passage["tests"]:
            counts["qa"][0] += int(agent.answer(case["question"]) == case["answer"])
            counts["qa"][1] += 1
    for case in data["event_tests"]:
        counts["events"][0] += int(agent.answer(case["question"]) == case["answer"])
        counts["events"][1] += 1
    for case in data["math"]:
        counts["math"][0] += int(agent.answer(case["question"]) == case["answer"])
        counts["math"][1] += 1
    for case in data["programs"]:
        counts["programs"][0] += int(agent.answer(case["question"]) == case["answer"])
        counts["programs"][1] += 1
    for case in data["dialogue"]:
        counts["dialogue"][0] += int(agent.answer(case["question"]) == case["answer"])
        counts["dialogue"][1] += 1
    metrics = {f"{key}_accuracy": good / max(1, total) for key, (good, total) in counts.items()}
    metrics["overall_accuracy"] = sum(good for good, _ in counts.values()) / max(1, sum(total for _, total in counts.values()))
    checks = {key: value >= args.accuracy_target for key, value in metrics.items() if key.endswith("_accuracy")}
    ok = all(checks.values())
    return {"success": ok, "checks": checks, "metrics": metrics, "diagnostics": agent.diagnostics()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--accuracy-target", type=float, default=0.80)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 8: External Gold")
    print("=" * 88)
    for key, value in result["metrics"].items():
        print(f"{key:<24} {value:.2%}")
    print(f"Overall status:          {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase8_external_gold", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
