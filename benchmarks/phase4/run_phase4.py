"""Phase 4 Comprehensive Benchmark Runner.

Integrates all Phase 4 benchmarks:
1. Sparse Physical Compute (efficiency)
2. Language Quality (vs baselines)
3. Long-Context Retrieval (without attention)
4. Continual Learning (domain switching)
5. Reasoning (multi-hop, role binding, causal)
6. Anti-Cheat Structural Scan

This runner executes all benchmarks and generates a comprehensive report.
"""
import argparse
import subprocess
import sys
import os
from typing import Dict, List, Tuple
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def run_benchmark(script_name: str, args: List[str]) -> Tuple[int, str, str]:
    """Run a benchmark script and capture output."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = [sys.executable, script_path] + args
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    )
    
    return result.returncode, result.stdout, result.stderr


def run_sparse_physical_compute() -> Dict:
    """Run sparse physical compute benchmark."""
    print("\n" + "=" * 80)
    print("Running: Sparse Physical Compute Benchmark")
    print("=" * 80)
    
    args = [
        "--sizes", "1000",
        "--workloads", "random",
        "--iterations", "20",
        "--warmup", "5",
        "--model-dim", "1000",
        "--model-vocab", "96",
        "--model-iterations", "10",
        "--model-warmup", "2",
        "--no-predictive-coding",
        "--sparsity", "0.01",
        "--end-to-end-latency-target", "20",
        "--observe-latency-target", "5",
        "--end-to-end-ops-target", "25",
        "--sparse-alloc-kb-target", "12",
    ]
    
    returncode, stdout, stderr = run_benchmark("benchmark_sparse_physical_compute.py", args)
    
    result = {
        "name": "Sparse Physical Compute",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": "PASS" if returncode == 0 else "FAIL",
    }
    
    print(result["status"])
    return result


def run_language_quality() -> Dict:
    """Run language quality benchmark."""
    print("\n" + "=" * 80)
    print("Running: Language Quality Benchmark")
    print("=" * 80)
    
    args = [
        "--vocab-size", "100",
        "--hidden-dim", "64",
        "--num-tokens", "1000",
    ]
    
    returncode, stdout, stderr = run_benchmark("benchmark_language_quality.py", args)
    
    result = {
        "name": "Language Quality",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": "PASS" if returncode == 0 else "FAIL",
    }
    
    print(result["status"])
    return result


def run_long_context() -> Dict:
    """Run long-context retrieval benchmark."""
    print("\n" + "=" * 80)
    print("Running: Long-Context Retrieval Benchmark")
    print("=" * 80)
    
    args = [
        "--context-lengths", "100", "200",
        "--vocab-size", "100",
        "--num-trials", "5",
    ]
    
    returncode, stdout, stderr = run_benchmark("benchmark_long_context.py", args)
    
    result = {
        "name": "Long-Context Retrieval",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": "PASS" if returncode == 0 else "FAIL",
    }
    
    print(result["status"])
    return result


def run_continual_learning() -> Dict:
    """Run continual learning benchmark."""
    print("\n" + "=" * 80)
    print("Running: Continual Learning Benchmark")
    print("=" * 80)
    
    args = [
        "--vocab-size", "100",
        "--train-tokens-a", "500",
        "--train-tokens-b", "500",
    ]
    
    returncode, stdout, stderr = run_benchmark("benchmark_continual_learning.py", args)
    
    result = {
        "name": "Continual Learning",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": "PASS" if returncode == 0 else "FAIL",
    }
    
    print(result["status"])
    return result


def run_reasoning() -> Dict:
    """Run reasoning benchmark."""
    print("\n" + "=" * 80)
    print("Running: Reasoning Benchmark")
    print("=" * 80)
    
    args = [
        "--vocab-size", "100",
        "--num-trials", "5",
    ]
    
    returncode, stdout, stderr = run_benchmark("benchmark_reasoning.py", args)
    
    result = {
        "name": "Reasoning",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": "PASS" if returncode == 0 else "FAIL",
    }
    
    print(result["status"])
    return result


def run_anti_cheat() -> Dict:
    """Run anti-cheat structural scan."""
    print("\n" + "=" * 80)
    print("Running: Anti-Cheat Structural Scan")
    print("=" * 80)
    
    args = []
    
    returncode, stdout, stderr = run_benchmark("benchmark_anti_cheat.py", args)
    
    result = {
        "name": "Anti-Cheat Scan",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": "PASS" if returncode == 0 else "FAIL",
    }
    
    print(result["status"])
    return result


def run_baseline_comparison() -> Dict:
    """Run baseline comparison with tiny Transformer."""
    print("\n" + "=" * 80)
    print("Running: Baseline Comparison (Transformer)")
    print("=" * 80)
    
    args = [
        "--vocab-size", "100",
        "--d-model", "64",
        "--num-tokens", "50",
    ]
    
    returncode, stdout, stderr = run_benchmark("baseline_transformer.py", args)
    
    result = {
        "name": "Baseline Comparison",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": "PASS" if returncode == 0 else "FAIL",
    }
    
    print(result["status"])
    return result


def run_scaling_law() -> Dict:
    """Run scaling law test across multiple sizes."""
    print("\n" + "=" * 80)
    print("Running: Scaling Law Test")
    print("=" * 80)
    
    args = []

    returncode, stdout, stderr = run_benchmark("../phase5/benchmark_scaling_law.py", args)
    
    result = {
        "name": "Scaling Law Test",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": "PASS" if returncode == 0 else "FAIL",
    }
    
    print(result["status"])
    return result


def generate_report(results: List[Dict]) -> str:
    """Generate comprehensive report."""
    report = []
    report.append("=" * 80)
    report.append("PHASE 4 COMPREHENSIVE BENCHMARK REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary table
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"{'Benchmark':<30} {'Status':<10}")
    report.append("-" * 80)
    
    for result in results:
        report.append(f"{result['name']:<30} {result['status']:<10}")
    
    report.append("-" * 80)
    
    # Overall status
    all_pass = all(r["status"] == "PASS" for r in results)
    overall_status = "PASS" if all_pass else "FAIL"
    report.append(f"Overall: {overall_status}")
    report.append("")
    
    # Detailed results
    report.append("=" * 80)
    report.append("DETAILED RESULTS")
    report.append("=" * 80)
    
    for result in results:
        report.append("")
        report.append(f"Benchmark: {result['name']}")
        report.append(f"Status: {result['status']}")
        report.append(f"Return code: {result['returncode']}")
        if result["stderr"]:
            report.append(f"Stderr: {result['stderr']}")
        report.append("")
        report.append("Output:")
        report.append(result["stdout"])
        report.append("-" * 80)
    
    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 Comprehensive Benchmark Runner"
    )
    parser.add_argument("--skip-sparse", action="store_true", help="Skip sparse physical compute")
    parser.add_argument("--skip-lang", action="store_true", help="Skip language quality")
    parser.add_argument("--skip-context", action="store_true", help="Skip long-context")
    parser.add_argument("--skip-continual", action="store_true", help="Skip continual learning")
    parser.add_argument("--skip-reasoning", action="store_true", help="Skip reasoning")
    parser.add_argument("--skip-anti-cheat", action="store_true", help="Skip anti-cheat scan")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip scaling law test")
    parser.add_argument("--output", type=str, default="phase4_report.txt", help="Output report file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    print("Phase 4 Comprehensive Benchmark Runner")
    print("Running all Phase 4 benchmarks...")
    
    results = []
    
    if not args.skip_sparse:
        results.append(run_sparse_physical_compute())
    
    if not args.skip_lang:
        results.append(run_language_quality())
    
    if not args.skip_context:
        results.append(run_long_context())
    
    if not args.skip_continual:
        results.append(run_continual_learning())
    
    if not args.skip_reasoning:
        results.append(run_reasoning())
    
    if not args.skip_anti_cheat:
        results.append(run_anti_cheat())
    
    if not args.skip_baseline:
        results.append(run_baseline_comparison())
    
    if not args.skip_scaling:
        results.append(run_scaling_law())
    
    # Generate report
    report = generate_report(results)
    
    # Print report
    print("\n")
    print(report)
    
    # Save report to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to: {args.output}")
    
    # Return overall status
    all_pass = all(r["status"] == "PASS" for r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
