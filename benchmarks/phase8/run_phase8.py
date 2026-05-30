"""Phase 8 external reality check runner."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE4 = os.path.join(ROOT, "benchmarks", "phase4")
PHASE8 = os.path.join(ROOT, "benchmarks", "phase8")


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
    if profile == "quick":
        integrated = ["--dataset", "tinystories", "--max-train-tokens", "2500", "--max-eval-tokens", "800", "--items", "48"]
        long_args = ["--sizes", "10000", "100000", "--queries", "64"]
        scaling = ["--sizes", "64", "128", "256"]
        public_adapters = ["--limit", "8", "--limit-per-task", "2", "--mbpp-exec-limit", "4"]
        public_integrated = ["--babi-items", "32", "--squad-items", "8", "--gsm8k-items", "8", "--mbpp-items", "4"]
    elif profile == "claim":
        integrated = ["--dataset", "tinystories_full", "--max-train-tokens", "6000", "--max-eval-tokens", "1500", "--items", "96"]
        long_args = ["--sizes", "100000", "1000000", "--queries", "128", "--query-repeats", "64"]
        scaling = ["--sizes", "64", "128", "256", "512"]
        public_adapters = ["--limit", "16", "--limit-per-task", "4", "--mbpp-exec-limit", "8"]
        public_integrated = ["--babi-items", "64", "--squad-items", "16", "--gsm8k-items", "16", "--mbpp-items", "8"]
    else:
        integrated = ["--dataset", "wikitext2", "--max-train-tokens", "10000", "--max-eval-tokens", "2500", "--items", "128"]
        long_args = ["--sizes", "100000", "1000000", "--queries", "160", "--query-repeats", "64"]
        scaling = ["--sizes", "128", "256", "512", "1024"]
        public_adapters = ["--limit", "32", "--limit-per-task", "8", "--mbpp-exec-limit", "12"]
        public_integrated = ["--babi-items", "128", "--squad-items", "32", "--gsm8k-items", "32", "--mbpp-items", "12"]
    return [
        {"name": "public_dataset_adapters", "script": os.path.join(PHASE8, "benchmark_public_dataset_adapters.py"), "args": public_adapters},
        {"name": "public_integrated_smoke", "script": os.path.join(PHASE8, "benchmark_public_integrated_eval.py"), "args": public_integrated},
        {"name": "external_gold", "script": os.path.join(PHASE8, "benchmark_external_gold.py"), "args": []},
        {"name": "integrated_agent", "script": os.path.join(PHASE8, "benchmark_integrated_agent.py"), "args": integrated},
        {"name": "multievidence_long_context", "script": os.path.join(PHASE8, "benchmark_multievidence_long_context.py"), "args": long_args},
        {"name": "external_scaling", "script": os.path.join(PHASE8, "benchmark_external_scaling.py"), "args": scaling},
        {"name": "strict_scanner", "script": os.path.join(PHASE4, "benchmark_anti_cheat.py"), "args": []},
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "claim", "full"], default="quick")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Phase 8 external reality check ({args.profile})")
    print("=" * 88)
    with tempfile.TemporaryDirectory(prefix="phase8_") as tmpdir:
        results = [run_case(case["name"], case["script"], case["args"], tmpdir) for case in cases(args.profile)]
    ok = all(bool(result["success"]) for result in results)
    print("=" * 88)
    print(f"PHASE 8: {'PASS' if ok else 'FAIL'} ({sum(r['success'] for r in results)}/{len(results)})")
    payload = {
        "benchmark": "phase8_runner",
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
