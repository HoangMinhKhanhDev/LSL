"""Phase 9 biological compute closure runner."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE4 = os.path.join(ROOT, "benchmarks", "phase4")
PHASE9 = os.path.join(ROOT, "benchmarks", "phase9")


def run_case(name: str, script: str, args: List[str], tmpdir: str) -> Dict[str, object]:
    json_path = os.path.join(tmpdir, f"{name}.json")
    cmd = [sys.executable, script] + args + ["--json-output", json_path]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    metrics = None
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    success = result.returncode == 0 and metrics is not None and bool(metrics.get("success", False))
    print(f"{name:<32} {'PASS' if success else 'FAIL'}")
    if not success:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-18:])
        if tail:
            print(tail)
    return {
        "name": name,
        "script": script,
        "args": args,
        "returncode": result.returncode,
        "success": bool(success),
        "metrics": metrics,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def cases(profile: str) -> List[Dict[str, object]]:
    if profile == "quick":
        hippocampus = ["--items", "1000"]
        neuromod = ["--stress-steps", "100000", "--min-stress-steps", "100000"]
        integrated = ["--items", "32"]
        dialogue = ["--domains", "4", "--turns-per-domain", "64", "--samples", "8", "--generate-tokens", "48"]
        model_level = ["--items", "512", "--samples", "64", "--interference-items", "64", "--latency-iterations", "32"]
        mechanisms_1_5 = ["--memory-items", "1000", "--stress-steps", "100000", "--sdr-items", "500"]
    elif profile == "claim":
        hippocampus = ["--items", "10000"]
        neuromod = ["--stress-steps", "1000000", "--min-stress-steps", "1000000"]
        integrated = ["--items", "64"]
        dialogue = ["--domains", "8", "--turns-per-domain", "256", "--samples", "24", "--generate-tokens", "64"]
        model_level = ["--items", "10000", "--samples", "256", "--interference-items", "512", "--latency-iterations", "64"]
        mechanisms_1_5 = []
    else:
        hippocampus = ["--items", "10000"]
        neuromod = ["--stress-steps", "1000000", "--min-stress-steps", "1000000"]
        integrated = ["--items", "96"]
        dialogue = ["--domains", "12", "--turns-per-domain", "512", "--samples", "32", "--generate-tokens", "96"]
        model_level = ["--items", "10000", "--samples", "512", "--interference-items", "1000", "--latency-iterations", "96"]
        mechanisms_1_5 = []
    return [
        {"name": "bio_predictive_coding", "script": os.path.join(PHASE9, "benchmark_bio_predictive_coding.py"), "args": []},
        {"name": "bio_sdr_semantics", "script": os.path.join(PHASE9, "benchmark_bio_sdr_semantics.py"), "args": []},
        {"name": "bio_cortical_column", "script": os.path.join(PHASE9, "benchmark_bio_cortical_column.py"), "args": []},
        {"name": "bio_hippocampus", "script": os.path.join(PHASE9, "benchmark_bio_hippocampus.py"), "args": hippocampus},
        {"name": "bio_neuromodulation", "script": os.path.join(PHASE9, "benchmark_bio_neuromodulation.py"), "args": neuromod},
        {"name": "bio_mechanisms_1_5_targets", "script": os.path.join(PHASE9, "benchmark_bio_mechanisms_1_5_targets.py"), "args": mechanisms_1_5},
        {"name": "bio_dendritic", "script": os.path.join(PHASE9, "benchmark_bio_dendritic.py"), "args": []},
        {"name": "bio_integrated_agent", "script": os.path.join(PHASE9, "benchmark_bio_integrated_agent.py"), "args": integrated},
        {"name": "bio_dialogue_generation", "script": os.path.join(PHASE9, "benchmark_bio_dialogue_generation.py"), "args": dialogue},
        {"name": "lsl_model_level", "script": os.path.join(PHASE9, "benchmark_lsl_model_level.py"), "args": model_level},
        {"name": "strict_scanner", "script": os.path.join(PHASE4, "benchmark_anti_cheat.py"), "args": []},
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "claim", "full"], default="quick")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Phase 9 bio-compute closure ({args.profile})")
    print("=" * 88)
    with tempfile.TemporaryDirectory(prefix="phase9_") as tmpdir:
        results = [run_case(case["name"], case["script"], case["args"], tmpdir) for case in cases(args.profile)]
    ok = all(result["success"] for result in results)
    print("=" * 88)
    print(f"PHASE 9: {'PASS' if ok else 'FAIL'} ({sum(r['success'] for r in results)}/{len(results)})")
    payload = {
        "benchmark": "phase9_runner",
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
