"""Seed sweep for LSL training across multiple random seeds.

Runs training with 5-10 different seeds to measure statistical significance
and aggregate results with mean/std metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lsl import LSLCoreModel, DatasetLoader, DatasetConfig, RUNTIME_PROFILE_CHOICES, write_result


def run_single_seed(
    dataset: str,
    seed: int,
    max_tokens: int,
    vocab_size: int,
    candidate_cap: int,
    lsl_profile: str,
    tokenizer_train_chars: int,
    max_chars: int,
    split: str = "train",
    repeat_small: bool = False,
) -> Dict:
    """Run training with a single seed.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        max_tokens: Maximum tokens to train
        vocab_size: Vocabulary size
        candidate_cap: Candidate cap for generation
        lsl_profile: LSL runtime profile
        tokenizer_train_chars: Characters for tokenizer training
        max_chars: Maximum characters to load
        split: Dataset split
        repeat_small: Whether to repeat small corpora
        
    Returns:
        Result dictionary with metrics
    """
    print(f"\n{'='*88}")
    print(f"Seed {seed}: Training LSL on {dataset}")
    print(f"{'='*88}")
    
    # Load dataset
    loader = DatasetLoader(Path(__file__).parent.parent)
    config = DatasetConfig(
        name=dataset,
        split=split,
        max_chars=max_chars,
        repeat=repeat_small
    )
    
    try:
        text = loader.load_text(config)
        corpus_path = loader.resolve_path(dataset, split)
    except Exception as e:
        return {
            "seed": seed,
            "success": False,
            "error": str(e),
            "metrics": {}
        }
    
    # Create model
    model = LSLCoreModel(
        vocab_size=vocab_size,
        seed=seed,
        candidate_cap=candidate_cap,
        runtime_profile=lsl_profile,
    )
    
    # Train
    started = time.perf_counter()
    try:
        metrics = model.train_stream(
            [text],
            tokenizer_text_chars=tokenizer_train_chars,
            max_tokens=max_tokens
        )
        elapsed = time.perf_counter() - started
        success = True
        error = None
    except Exception as e:
        elapsed = time.perf_counter() - started
        metrics = {"tokens": 0, "elapsed_seconds": elapsed, "us_per_token": 0}
        success = False
        error = str(e)
    
    # Generate sample
    sample_prompt = "the little girl"
    sample = ""
    if success:
        try:
            sample = model.generate(sample_prompt, max_new_tokens=48)
        except Exception:
            sample = ""
    
    # Get diagnostics
    diag = model.diagnostics() if success else {}
    
    result = {
        "seed": seed,
        "success": success,
        "error": error,
        "metrics": {
            **metrics,
            "wall_seconds": float(elapsed),
            "tokens_per_second": float(metrics.get("tokens", 0) / max(elapsed, 1e-12)),
            "vocab_size": model.vocab_size,
            "seen_tokens": diag.get("seen_tokens", 0.0),
            "lsl_profile": lsl_profile,
        },
        "sample_prompt": sample_prompt,
        "sample": sample,
        "corpus_path": str(corpus_path),
    }
    
    print(f"Seed {seed}: {'SUCCESS' if success else 'FAILED'}")
    if success:
        print(f"  Tokens:           {int(metrics.get('tokens', 0)):,}")
        print(f"  Train tok/s:      {result['metrics']['tokens_per_second']:.2f}")
        print(f"  us/token:         {metrics.get('us_per_token', 0):.2f}")
    else:
        print(f"  Error:            {error}")
    
    return result


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across multiple seeds.
    
    Args:
        results: List of individual seed results
        
    Returns:
        Aggregated statistics
    """
    successful = [r for r in results if r["success"]]
    
    if not successful:
        return {
            "total_runs": len(results),
            "successful_runs": 0,
            "failed_runs": len(results),
            "errors": [r.get("error") for r in results if not r["success"]]
        }
    
    # Extract numeric metrics
    metric_keys = []
    for key in successful[0]["metrics"].keys():
        if isinstance(successful[0]["metrics"][key], (int, float)):
            metric_keys.append(key)
    
    aggregated = {
        "total_runs": len(results),
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
        "metrics": {}
    }
    
    for key in metric_keys:
        values = [r["metrics"][key] for r in successful if key in r["metrics"]]
        if values:
            aggregated["metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "values": [float(v) for v in values]
            }
    
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed sweep for LSL training")
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "wikitext2", "vietnamese_small", "dialogue_small", "custom"],
        default="tinystories"
    )
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "validation", "val", "test"], default="train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--max-chars", type=int, default=5000000)
    parser.add_argument("--tokenizer-train-chars", type=int, default=250000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--lsl-profile", choices=list(RUNTIME_PROFILE_CHOICES), default="native_fast")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--repeat-small", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    
    args = parser.parse_args()
    
    print(f"Seed Sweep: {args.dataset}")
    print(f"Seeds: {args.seeds}")
    print(f"Max tokens: {args.max_tokens:,}")
    print(f"Profile: {args.lsl_profile}")
    
    # Run all seeds
    results = []
    for seed in args.seeds:
        result = run_single_seed(
            dataset=args.dataset,
            seed=seed,
            max_tokens=args.max_tokens,
            vocab_size=args.vocab_size,
            candidate_cap=args.candidate_cap,
            lsl_profile=args.lsl_profile,
            tokenizer_train_chars=args.tokenizer_train_chars,
            max_chars=args.max_chars,
            split=args.split,
            repeat_small=args.repeat_small,
        )
        results.append(result)
        
        # Save individual result
        write_result(
            result,
            benchmark="seed_sweep",
            dataset=args.dataset,
            seed=seed,
            config=vars(args),
            results_root=args.results_root,
        )
    
    # Aggregate results
    aggregated = aggregate_results(results)
    
    # Print summary
    print(f"\n{'='*88}")
    print("Seed Sweep Summary")
    print(f"{'='*88}")
    print(f"Total runs:       {aggregated['total_runs']}")
    print(f"Successful:       {aggregated['successful_runs']}")
    print(f"Failed:           {aggregated['failed_runs']}")
    
    if aggregated['successful_runs'] > 0:
        print(f"\nAggregated Metrics:")
        for key, stats in aggregated['metrics'].items():
            print(f"  {key}:")
            print(f"    mean:     {stats['mean']:.4f}")
            print(f"    std:      {stats['std']:.4f}")
            print(f"    min:      {stats['min']:.4f}")
            print(f"    max:      {stats['max']:.4f}")
            print(f"    median:   {stats['median']:.4f}")
    
    # Save aggregated results
    payload = {
        "benchmark": "seed_sweep",
        "dataset": args.dataset,
        "split": args.split,
        "seeds": args.seeds,
        "individual_results": results,
        "aggregated": aggregated,
        "config": vars(args),
    }
    
    output_path = args.json_output or os.path.join(
        args.results_root,
        "seed_sweep",
        f"{args.dataset}_{args.split}_sweep.json"
    )
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\nAggregated results saved to: {output_path}")
    
    return 0 if aggregated['successful_runs'] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
