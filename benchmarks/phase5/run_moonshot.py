"""Moonshot v5.0 benchmark runner."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE4 = os.path.join(ROOT, "benchmarks", "phase4")
PHASE5 = os.path.join(ROOT, "benchmarks", "phase5")


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
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-12:])
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
    semantic_sizes = ["10000", "100000"] if quick else ["100000", "1000000"]
    long_lengths = ["1000", "4000", "16000"] if quick else ["1000", "4000", "16000", "64000", "128000"]
    memory_horizons = ["1000", "4000"] if quick else ["1000", "4000", "16000", "64000"]
    sdr_patterns = "20000" if quick else "100000"
    semantic_queries = "100" if quick else "200"
    sparse_iters = "20" if quick else "40"
    model_iters = "10"
    real_corpus_args = (
        ["--max-train-tokens", "4000", "--max-eval-tokens", "1000", "--latency-iterations", "6"]
        if quick
        else ["--max-train-tokens", "12000", "--max-eval-tokens", "3000", "--latency-iterations", "12"]
    )
    if claim:
        real_corpus_args = [
            "--max-train-tokens", "6000",
            "--max-eval-tokens", "1500",
            "--latency-iterations", "6",
            "--train-dense-baselines",
        ]
    wikitext_args = (
        ["--dataset", "wikitext2", "--max-train-tokens", "4000", "--max-eval-tokens", "1000", "--latency-iterations", "6"]
        if quick
        else ["--dataset", "wikitext2", "--max-train-tokens", "12000", "--max-eval-tokens", "3000", "--latency-iterations", "12"]
    )
    if claim:
        wikitext_args = [
            "--dataset", "wikitext2",
            "--vocab-size", "8000",
            "--max-train-tokens", "20000",
            "--max-eval-tokens", "5000",
            "--capacity", "4096",
            "--latency-iterations", "8",
        ]
    subword_smoke_args = [
        "--tokenizer", "subword",
        "--vocab-size", "2500",
        "--subword-max-merges", "400",
        "--tokenizer-train-chars", "80000",
        "--max-train-chars", "180000",
        "--max-eval-chars", "70000",
        "--max-train-tokens", "4000",
        "--max-eval-tokens", "1000",
        "--latency-iterations", "4",
        "--capacity", "1024",
    ]

    return [
        {
            "name": "semantic_sdr_scale",
            "script": os.path.join(PHASE4, "benchmark_semantic_sdr_scaling.py"),
            "args": [
                "--sizes", *semantic_sizes,
                "--pairs", "200",
                "--queries", semantic_queries,
                "--collision-sample", "4096",
                "--ablation-size", "10000",
            ],
        },
        {
            "name": "sparse_physical_no_pc",
            "script": os.path.join(PHASE4, "benchmark_sparse_physical_compute.py"),
            "args": [
                "--sizes", "1000",
                "--workloads", "random",
                "--iterations", sparse_iters,
                "--warmup", "5",
                "--model-dim", "1024",
                "--model-vocab", "96",
                "--model-iterations", model_iters,
                "--model-warmup", "2",
                "--no-predictive-coding",
                "--sparsity", "0.01",
                "--end-to-end-latency-target", "20",
                "--observe-latency-target", "5",
                "--end-to-end-ops-target", "25",
                "--sparse-alloc-kb-target", "12",
            ],
        },
        {
            "name": "sparse_physical_with_pc",
            "script": os.path.join(PHASE4, "benchmark_sparse_physical_compute.py"),
            "args": [
                "--sizes", "1000",
                "--workloads", "random",
                "--iterations", "20",
                "--warmup", "5",
                "--model-dim", "512",
                "--model-vocab", "96",
                "--model-iterations", "8",
                "--model-warmup", "2",
                "--sparsity", "0.01",
                "--end-to-end-latency-target", "5",
                "--observe-latency-target", "3",
                "--end-to-end-ops-target", "10",
                "--sparse-alloc-kb-target", "12",
                "--quality-ratio-target", "1.10",
            ],
        },
        {
            "name": "sdr_large_memory",
            "script": os.path.join(PHASE5, "benchmark_sdr_large_memory.py"),
            "args": ["--patterns", sdr_patterns, "--samples", "500" if quick else "1000"],
        },
        {
            "name": "pc_suppression_ood",
            "script": os.path.join(PHASE5, "benchmark_pc_suppression_ood.py"),
            "args": ["--tokens", "1000" if quick else "2000"],
        },
        {
            "name": "long_context",
            "script": os.path.join(PHASE4, "benchmark_long_context.py"),
            "args": [
                "--context-lengths", *long_lengths,
                "--num-trials", "16" if quick else "32",
                *(["--mode", "bucket-only", "--random-values", "--absent-queries", "64"] if claim else []),
            ],
        },
        {
            "name": "long_context_memory",
            "script": os.path.join(PHASE5, "benchmark_long_context_memory.py"),
            "args": [
                "--horizons", *memory_horizons,
                "--trials", "32" if quick else "64",
                "--instructions", "1024" if quick else "4096",
                "--sequence-tokens", "1200" if quick else "2500",
            ],
        },
        {
            "name": "long_context_real_corpus",
            "script": os.path.join(PHASE5, "benchmark_long_context_real_corpus.py"),
            "args": real_corpus_args,
        },
        {
            "name": "long_context_wikitext2",
            "script": os.path.join(PHASE5, "benchmark_long_context_real_corpus.py"),
            "args": wikitext_args,
        },
        {
            "name": "wikitext2_subword",
            "script": os.path.join(PHASE5, "benchmark_long_context_real_corpus.py"),
            "args": ["--dataset", "wikitext2", *subword_smoke_args],
        },
        {
            "name": "tinystories_full_subword",
            "script": os.path.join(PHASE5, "benchmark_long_context_real_corpus.py"),
            "args": ["--dataset", "tinystories_full", *subword_smoke_args],
        },
        {
            "name": "reasoning",
            "script": os.path.join(PHASE4, "benchmark_reasoning.py"),
            "args": [
                "--vocab-size", "1000",
                "--num-trials", "64",
                *(["--heldout-random-symbols"] if claim else []),
            ],
        },
        {
            "name": "mini_exact_eval",
            "script": os.path.join(PHASE5, "benchmark_mini_exact_eval.py"),
            "args": [],
        },
        {
            "name": "natural_instruction_eval",
            "script": os.path.join(PHASE5, "benchmark_natural_instruction_eval.py"),
            "args": [],
        },
        {
            "name": "branching_cortical",
            "script": os.path.join(PHASE5, "benchmark_branching_cortical.py"),
            "args": [],
        },
        {
            "name": "continual_learning",
            "script": os.path.join(PHASE4, "benchmark_continual_learning.py"),
            "args": [
                "--vocab-size", "1000",
                "--train-tokens-a", "1000",
                "--train-tokens-b", "1000",
                "--train-tokens-c", "1000",
                *(["--overlap-vocab", "--predict-only-eval"] if claim else []),
            ],
        },
        {
            "name": "hierarchy_routing",
            "script": os.path.join(PHASE5, "benchmark_hierarchy.py"),
            "args": [],
        },
        {
            "name": "language_quality",
            "script": os.path.join(PHASE4, "benchmark_language_quality.py"),
            "args": ["--vocab-size", "1000", "--num-tokens", "3000" if quick else "10000"],
        },
        {
            "name": "baseline_competition",
            "script": os.path.join(PHASE5, "benchmark_baseline_competition.py"),
            "args": [
                "--context", "512",
                "--iterations", "10",
                *(["--train-baselines", "--baseline-epochs", "2", "--context-train", "16"] if claim else []),
            ],
        },
        {
            "name": "scaling_law",
            "script": os.path.join(PHASE5, "benchmark_scaling_law.py"),
            "args": ["--actual-runs", "--seeds", "42", "43", "44"] if claim else [],
        },
        {
            "name": "anti_cheat",
            "script": os.path.join(PHASE4, "benchmark_anti_cheat.py"),
            "args": [],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "full", "claim"], default="quick")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Moonshot v5.0 runner ({args.profile})")
    print("=" * 88)
    with tempfile.TemporaryDirectory(prefix="moonshot_v5_") as tmpdir:
        results = [run_case(case["name"], case["script"], case["args"], tmpdir) for case in cases(args.profile)]

    ok = all(bool(result["success"]) for result in results)
    print("=" * 88)
    print(f"MOONSHOT V5.0: {'PASS' if ok else 'FAIL'} ({sum(r['success'] for r in results)}/{len(results)})")
    payload = {
        "benchmark": "moonshot_v5",
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
