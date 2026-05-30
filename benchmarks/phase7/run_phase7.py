"""Phase 7 generalization and reasoning-workspace runner."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE4 = os.path.join(ROOT, "benchmarks", "phase4")
PHASE7 = os.path.join(ROOT, "benchmarks", "phase7")


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
    if quick:
        gen_args = ["--dataset", "tinystories", "--max-train-tokens", "2500", "--max-eval-tokens", "800", "--trials", "4"]
        home_args = ["--steps", "200", "--seeds", "42", "43"]
        diverse_args = ["--items", "64"]
        general_args = ["--items", "100"]
        ssm_args = ["--dim", "2000", "--steps", "128"]
        long_args = ["--sizes", "10000", "100000", "--queries", "64"]
    elif profile == "claim":
        gen_args = ["--dataset", "tinystories_full", "--max-train-tokens", "7000", "--max-eval-tokens", "1800", "--trials", "8"]
        home_args = ["--steps", "400", "--seeds", "42", "43", "44"]
        diverse_args = ["--items", "128"]
        general_args = ["--items", "200"]
        ssm_args = ["--dim", "4000", "--steps", "256"]
        long_args = ["--sizes", "100000", "1000000", "--queries", "128"]
    else:
        gen_args = ["--dataset", "wikitext2", "--max-train-tokens", "12000", "--max-eval-tokens", "3000", "--trials", "10"]
        home_args = ["--steps", "500", "--seeds", "42", "43", "44"]
        diverse_args = ["--items", "160"]
        general_args = ["--items", "240"]
        ssm_args = ["--dim", "5000", "--steps", "320"]
        long_args = ["--sizes", "100000", "1000000", "--queries", "160"]
    return [
        {"name": "generation_quality_v2", "script": os.path.join(PHASE7, "benchmark_generation_quality_v2.py"), "args": gen_args},
        {"name": "homeostasis", "script": os.path.join(PHASE7, "benchmark_homeostasis.py"), "args": home_args},
        {"name": "diverse_data", "script": os.path.join(PHASE7, "benchmark_diverse_data.py"), "args": diverse_args},
        {"name": "event_driven_ssm", "script": os.path.join(PHASE7, "benchmark_event_driven_ssm.py"), "args": ssm_args},
        {"name": "optional_prior", "script": os.path.join(PHASE7, "benchmark_optional_prior.py"), "args": []},
        {"name": "generalization_heldout", "script": os.path.join(PHASE7, "benchmark_generalization_heldout.py"), "args": general_args},
        {"name": "long_range_reasoning", "script": os.path.join(PHASE7, "benchmark_long_range_reasoning.py"), "args": long_args},
        {"name": "strict_scanner", "script": os.path.join(PHASE4, "benchmark_anti_cheat.py"), "args": []},
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "claim", "full"], default="quick")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Phase 7 runner ({args.profile})")
    print("=" * 88)
    with tempfile.TemporaryDirectory(prefix="phase7_") as tmpdir:
        results = [run_case(case["name"], case["script"], case["args"], tmpdir) for case in cases(args.profile)]
    ok = all(bool(result["success"]) for result in results)
    print("=" * 88)
    print(f"PHASE 7: {'PASS' if ok else 'FAIL'} ({sum(r['success'] for r in results)}/{len(results)})")
    payload = {
        "benchmark": "phase7_runner",
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
