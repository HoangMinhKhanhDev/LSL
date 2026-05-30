"""Phase 8 official public dataset downloader/parser/judge check."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase8.public_datasets import (
    dataset_card,
    exact_match,
    load_babi,
    load_gsm8k,
    load_mbpp,
    load_squad,
    numeric_match,
    run_python_tests,
)


def evaluate(args):
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else None
    download = not args.no_download
    babi = load_babi(
        cache_dir=cache_dir,
        language=args.babi_language,
        split="test",
        limit_per_task=args.limit_per_task,
        download=download,
    )
    squad = load_squad(cache_dir=cache_dir, split="dev", limit=args.limit, download=download)
    gsm8k = load_gsm8k(cache_dir=cache_dir, split="test", limit=args.limit, download=download)
    mbpp = load_mbpp(cache_dir=cache_dir, split="sanitized", limit=args.mbpp_exec_limit, download=download)

    babi_tasks = sorted({int(row["task_id"]) for row in babi})
    squad_answer_in_context = sum(
        1 for row in squad if any(answer in row["context"] for answer in row["answers"])
    )
    gsm_numeric_ok = sum(1 for row in gsm8k if numeric_match(row["answer"], row["answer"]))
    mbpp_reference_ok = sum(
        1 for row in mbpp if row["code"] and row["tests"] and run_python_tests(row["code"], row["tests"], timeout=args.exec_timeout)
    )

    metrics = {
        "babi_examples": float(len(babi)),
        "babi_tasks": float(len(babi_tasks)),
        "squad_examples": float(len(squad)),
        "squad_answer_in_context": float(squad_answer_in_context),
        "gsm8k_examples": float(len(gsm8k)),
        "gsm8k_numeric_judge_ok": float(gsm_numeric_ok),
        "mbpp_examples": float(len(mbpp)),
        "mbpp_reference_passed": float(mbpp_reference_ok),
        "normalization_exact_match": float(exact_match("The bathroom.", ["bathroom"])),
    }
    checks = {
        "babi_full_tasks": metrics["babi_tasks"] >= args.min_babi_tasks,
        "babi_examples": metrics["babi_examples"] >= args.min_babi_tasks,
        "squad_loaded": metrics["squad_examples"] >= min(args.limit, 1),
        "squad_judge": metrics["squad_answer_in_context"] >= min(args.limit, 1),
        "gsm8k_loaded": metrics["gsm8k_examples"] >= min(args.limit, 1),
        "gsm8k_numeric_judge": metrics["gsm8k_numeric_judge_ok"] >= min(args.limit, 1),
        "mbpp_loaded": metrics["mbpp_examples"] >= min(args.mbpp_exec_limit, 1),
        "mbpp_exec_judge": metrics["mbpp_reference_passed"] >= max(1, int(args.mbpp_exec_limit * args.mbpp_reference_target)),
        "answer_normalization": metrics["normalization_exact_match"] == 1.0,
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "babi_tasks": babi_tasks,
        "sources": dataset_card(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--limit-per-task", type=int, default=4)
    parser.add_argument("--babi-language", choices=["en", "en-10k", "en-valid", "en-valid-10k"], default="en")
    parser.add_argument("--min-babi-tasks", type=int, default=20)
    parser.add_argument("--mbpp-exec-limit", type=int, default=8)
    parser.add_argument("--mbpp-reference-target", type=float, default=0.75)
    parser.add_argument("--exec-timeout", type=float, default=3.0)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 8: Official Public Dataset Adapters")
    print("=" * 88)
    for key, value in result["metrics"].items():
        print(f"{key:<28} {value:.0f}")
    print(f"bAbI tasks:                  {result['babi_tasks']}")
    print(f"Overall status:              {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase8_public_dataset_adapters", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
