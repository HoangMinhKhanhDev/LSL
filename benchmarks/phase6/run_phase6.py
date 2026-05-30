"""Phase 6 competitive-evidence benchmark runner."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE4 = os.path.join(ROOT, "benchmarks", "phase4")
PHASE6 = os.path.join(ROOT, "benchmarks", "phase6")


def run_case(name: str, script: str, args: List[str], tmpdir: str) -> Dict[str, object]:
    json_path = os.path.join(tmpdir, f"{name}.json")
    cmd = [sys.executable, script] + args + ["--json-output", json_path]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    metrics = None
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    success = result.returncode == 0 and (metrics is None or bool(metrics.get("success", False)))
    print(f"{name:<32} {'PASS' if success else 'FAIL'}")
    if not success:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-16:])
        if tail:
            print(tail)
    return {
        "name": name,
        "script": script,
        "args": args,
        "returncode": result.returncode,
        "success": success,
        "metrics": metrics,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def cases(profile: str) -> List[Dict[str, object]]:
    quick = profile == "quick"
    claim = profile == "claim"
    if quick:
        gen_args = ["--dataset", "tinystories", "--max-train-tokens", "2500", "--max-eval-tokens", "800", "--trials", "4"]
        world_args = ["--sizes", "1000", "4000", "--query-count", "64"]
        reason_args = ["--reasoning-items", "64", "--trace-items", "32"]
        comp_args = ["--dataset", "tinystories", "--max-train-tokens", "2500", "--max-eval-tokens", "800", "--latency-iterations", "4"]
    elif claim:
        gen_args = [
            "--dataset", "tinystories_full",
            "--max-train-tokens", "6000",
            "--max-eval-tokens", "1500",
            "--trials", "8",
            "--generate-tokens", "48",
        ]
        world_args = ["--sizes", "1000", "16000", "128000", "--query-count", "128"]
        reason_args = ["--reasoning-items", "160", "--trace-items", "80"]
        comp_args = [
            "--dataset", "tinystories_full",
            "--max-train-tokens", "6000",
            "--max-eval-tokens", "1500",
            "--latency-iterations", "8",
        ]
    else:
        gen_args = [
            "--dataset", "wikitext2",
            "--max-train-tokens", "12000",
            "--max-eval-tokens", "3000",
            "--trials", "10",
            "--generate-tokens", "64",
        ]
        world_args = ["--sizes", "1000", "16000", "128000", "--query-count", "192"]
        reason_args = ["--reasoning-items", "192", "--trace-items", "96"]
        comp_args = [
            "--dataset", "wikitext2",
            "--max-train-tokens", "12000",
            "--max-eval-tokens", "3000",
            "--latency-iterations", "8",
        ]
    return [
        {
            "name": "open_generation_public",
            "script": os.path.join(PHASE6, "benchmark_open_generation_public.py"),
            "args": gen_args,
        },
        {
            "name": "world_memory_qa",
            "script": os.path.join(PHASE6, "benchmark_world_memory_qa.py"),
            "args": world_args,
        },
        {
            "name": "public_reasoning",
            "script": os.path.join(PHASE6, "benchmark_public_reasoning.py"),
            "args": reason_args,
        },
        {
            "name": "phase6_competitive",
            "script": os.path.join(PHASE6, "benchmark_phase6_competitive.py"),
            "args": comp_args,
        },
        {
            "name": "strict_scanner",
            "script": os.path.join(PHASE4, "benchmark_anti_cheat.py"),
            "args": [],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "claim", "full"], default="quick")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Phase 6 competitive-evidence runner ({args.profile})")
    print("=" * 88)
    with tempfile.TemporaryDirectory(prefix="phase6_") as tmpdir:
        results = [run_case(case["name"], case["script"], case["args"], tmpdir) for case in cases(args.profile)]
    ok = all(bool(result["success"]) for result in results)
    print("=" * 88)
    print(f"PHASE 6: {'PASS' if ok else 'FAIL'} ({sum(r['success'] for r in results)}/{len(results)})")
    payload = {
        "benchmark": "phase6_runner",
        "profile": args.profile,
        "success": bool(ok),
        "passed": int(sum(r["success"] for r in results)),
        "total": len(results),
        "results": results,
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
