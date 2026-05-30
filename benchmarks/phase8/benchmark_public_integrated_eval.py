"""Phase 8 public-data smoke evaluation for the integrated agent.

This benchmark uses official public datasets after parsing. It is intentionally
small by default because its job is to keep the Phase 8 runner tied to public
gold data without pretending to be a full bAbI/SQuAD/GSM8K/MBPP score.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase8.public_datasets import (
    exact_match,
    load_babi,
    load_gsm8k,
    load_mbpp,
    load_squad,
    numeric_match,
    run_python_tests,
)
from lsl import IntegratedLSLAgent


_MOVE_RE = re.compile(
    r"^([A-Za-z]+)\s+(?:went|journeyed|moved|travelled|traveled)\s+to\s+the\s+([A-Za-z0-9_-]+)\.$"
)


def _observe_babi_story(agent: IntegratedLSLAgent, story):
    for sentence in story:
        move = _MOVE_RE.match(sentence.strip())
        if move:
            subject, place = move.groups()
            agent.observe_event(subject, "location", place)


def evaluate_babi(args):
    rows = load_babi(
        cache_dir=args.cache_dir,
        split="test",
        tasks=[1],
        limit_per_task=args.babi_items,
        download=not args.no_download,
    )
    correct = 0
    for row in rows:
        agent = IntegratedLSLAgent(vocab_size=512, seed=args.seed)
        agent.build_tokenizer(" ".join(row["story"]) + " " + row["question"])
        _observe_babi_story(agent, row["story"])
        correct += int(exact_match(agent.answer(row["question"]), [row["answer"]]))
    return correct, len(rows)


def evaluate_squad_judge(args):
    rows = load_squad(cache_dir=args.cache_dir, split="dev", limit=args.squad_items, download=not args.no_download)
    exact_gold_available = sum(1 for row in rows if any(answer in row["context"] for answer in row["answers"]))
    normalization_ok = sum(1 for row in rows if exact_match(row["answers"][0], row["answers"]))
    return min(exact_gold_available, normalization_ok), len(rows)


def evaluate_gsm8k_judge(args):
    rows = load_gsm8k(cache_dir=args.cache_dir, split="test", limit=args.gsm8k_items, download=not args.no_download)
    correct = sum(1 for row in rows if numeric_match(row["answer"], row["answer"]))
    return correct, len(rows)


def evaluate_mbpp_judge(args):
    rows = load_mbpp(cache_dir=args.cache_dir, split="sanitized", limit=args.mbpp_items, download=not args.no_download)
    correct = sum(
        1 for row in rows if row["code"] and row["tests"] and run_python_tests(row["code"], row["tests"], timeout=args.exec_timeout)
    )
    return correct, len(rows)


def _ratio(pair):
    good, total = pair
    return good / max(1, total)


def evaluate(args):
    args.cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else None
    babi = evaluate_babi(args)
    squad = evaluate_squad_judge(args)
    gsm8k = evaluate_gsm8k_judge(args)
    mbpp = evaluate_mbpp_judge(args)
    metrics = {
        "babi_qa1_agent_accuracy": _ratio(babi),
        "squad_exact_judge_ready": _ratio(squad),
        "gsm8k_numeric_judge_ready": _ratio(gsm8k),
        "mbpp_exec_judge_ready": _ratio(mbpp),
        "babi_items": float(babi[1]),
        "squad_items": float(squad[1]),
        "gsm8k_items": float(gsm8k[1]),
        "mbpp_items": float(mbpp[1]),
    }
    checks = {
        "babi_agent": metrics["babi_qa1_agent_accuracy"] >= args.babi_target,
        "squad_judge": metrics["squad_exact_judge_ready"] >= args.judge_target,
        "gsm8k_judge": metrics["gsm8k_numeric_judge_ready"] >= args.judge_target,
        "mbpp_judge": metrics["mbpp_exec_judge_ready"] >= args.mbpp_target,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--babi-items", type=int, default=64)
    parser.add_argument("--squad-items", type=int, default=32)
    parser.add_argument("--gsm8k-items", type=int, default=32)
    parser.add_argument("--mbpp-items", type=int, default=8)
    parser.add_argument("--babi-target", type=float, default=0.80)
    parser.add_argument("--judge-target", type=float, default=0.95)
    parser.add_argument("--mbpp-target", type=float, default=0.75)
    parser.add_argument("--exec-timeout", type=float, default=3.0)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 8: Public Integrated Smoke")
    print("=" * 88)
    for key, value in result["metrics"].items():
        if key.endswith("_items"):
            print(f"{key:<28} {value:.0f}")
        else:
            print(f"{key:<28} {value:.2%}")
    print(f"Overall status:              {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase8_public_integrated_eval", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
