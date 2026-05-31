"""Phase 1 training foundation, seed sweep, and scaling-law benchmark.

Examples:
  python benchmarks/phase1/run_scaling_law.py --datasets tinystories,wikitext2 --token-budgets 1000000,10000000
  python benchmarks/phase1/run_scaling_law.py --datasets vietnamese_small,dialogue_small --token-budgets 512 --seeds 1,2 --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.competitive.run_lsl_vs_transformer import (  # noqa: E402
    TrainableTinyTransformerCPU,
    model_size_bytes,
    transformer_eval,
)
from lsl import DatasetLoader, LSLCoreModel, write_result  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_csv_ints(raw: str) -> List[int]:
    return [int(part.strip().replace("_", "")) for part in str(raw).split(",") if part.strip()]


def parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def encode_budget(model: LSLCoreModel, text: str, token_budget: int, repeat: bool) -> List[int]:
    token_budget = int(token_budget)
    tokens = model.encode(text, max_tokens=token_budget)
    if not repeat or not text:
        return tokens[:token_budget]
    while len(tokens) < token_budget:
        needed = token_budget - len(tokens)
        extra = model.encode(text, max_tokens=needed)
        if not extra:
            break
        tokens.extend(extra)
    return tokens[:token_budget]


def train_one(
    args: argparse.Namespace,
    loader: DatasetLoader,
    dataset: str,
    token_budget: int,
    vocab_size: int,
    seed: int,
) -> Dict[str, object]:
    max_train_chars = args.max_train_chars or max(int(token_budget) * int(args.chars_per_token), args.tokenizer_train_chars)
    splits = loader.load_text_splits(
        dataset,
        train_fraction=args.train_fraction,
        max_train_chars=max_train_chars,
        max_eval_chars=args.max_eval_chars,
        seed=seed,
    )
    model = LSLCoreModel(
        vocab_size=int(vocab_size),
        seed=int(seed),
        candidate_cap=int(args.candidate_cap),
        runtime_profile=args.lsl_profile,
    )
    tokenizer_text = splits.train[: int(args.tokenizer_train_chars)]
    if not tokenizer_text:
        raise ValueError(f"Dataset {dataset} produced empty training text")
    model.build_tokenizer(tokenizer_text)
    train_tokens = encode_budget(model, splits.train, int(token_budget), repeat=args.repeat_small)
    eval_source = splits.validation or splits.test or splits.train
    eval_tokens = encode_budget(model, eval_source, int(args.eval_tokens), repeat=args.repeat_small)
    if len(train_tokens) < 2 or len(eval_tokens) < 2:
        raise ValueError(f"Dataset {dataset} produced too few tokens")

    started = time.perf_counter()
    train_metrics = model.fit_tokens(train_tokens, reset=True)
    train_wall = time.perf_counter() - started
    eval_metrics = model.evaluate_tokens(eval_tokens)
    lsl_size = model_size_bytes(model)
    checkpoint = None
    if args.save_checkpoints:
        checkpoint = os.path.abspath(
            os.path.join(
                args.checkpoint_dir,
                f"lsl_{dataset}_{args.lsl_profile}_tok{int(token_budget)}_v{int(vocab_size)}_seed{int(seed)}.json",
            )
        )
        model.save(checkpoint)

    row: Dict[str, object] = {
        "dataset": dataset,
        "dataset_metadata": splits.metadata(),
        "seed": int(seed),
        "token_budget": int(token_budget),
        "vocab_size_requested": int(vocab_size),
        "vocab_size_actual": int(model.vocab_size),
        "profile": args.lsl_profile,
        "train_tokens": int(len(train_tokens)),
        "eval_tokens": int(len(eval_tokens)),
        "train": {
            **train_metrics,
            "wall_seconds": float(train_wall),
            "wall_tokens_per_second": float(len(train_tokens) / max(train_wall, 1e-12)),
        },
        "eval": eval_metrics,
        "model_size_bytes": float(lsl_size),
        "model_size_mb": float(lsl_size / (1024.0 * 1024.0)),
        "checkpoint": checkpoint,
        "diagnostics": model.diagnostics(),
    }

    if args.compare_transformer:
        combined = train_tokens + eval_tokens
        train_cut = len(train_tokens)
        transformer = TrainableTinyTransformerCPU(model.vocab_size, int(args.d_model), int(seed))
        t0 = time.perf_counter()
        transformer.train(combined, train_cut, int(args.context), int(args.transformer_epochs), float(args.lr))
        tf_train_seconds = time.perf_counter() - t0
        tf_eval = transformer_eval(transformer, combined, train_cut, int(args.context), len(eval_tokens) - 1)
        tf_size = model_size_bytes(transformer)
        row["transformer"] = {
            **tf_eval,
            "d_model": int(args.d_model),
            "train_seconds": float(tf_train_seconds),
            "train_tokens_per_second": float(train_cut / max(tf_train_seconds, 1e-12)),
            "model_size_bytes": float(tf_size),
            "model_size_mb": float(tf_size / (1024.0 * 1024.0)),
        }
        row["comparison"] = {
            "same_token_budget": True,
            "loss_ratio_lsl_over_transformer": float(eval_metrics["loss"] / max(tf_eval["loss"], 1e-12)),
            "train_speedup_lsl_over_transformer": float(
                (len(train_tokens) / max(train_wall, 1e-12))
                / max(train_cut / max(tf_train_seconds, 1e-12), 1e-12)
            ),
            "inference_speedup_lsl_over_transformer": float(
                (1_000_000.0 / max(eval_metrics["p50_latency_us"], 1e-12))
                / max(tf_eval["tokens_per_second"], 1e-12)
            ),
        }

    budgets = parse_memory_budgets(args.memory_budgets_mb)
    if budgets:
        row["memory_budget_comparison"] = memory_budget_rows(
            vocab_size=int(model.vocab_size),
            lsl_size_bytes=int(lsl_size),
            seed=int(seed),
            budgets_mb=budgets,
        )

    return row


def memory_budget_rows(vocab_size: int, lsl_size_bytes: int, seed: int, budgets_mb: Iterable[float]) -> List[Dict[str, object]]:
    candidates = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    rows = []
    for budget_mb in budgets_mb:
        budget_bytes = float(budget_mb) * 1024.0 * 1024.0
        best_d: Optional[int] = None
        best_size = 0
        for d_model in candidates:
            transformer = TrainableTinyTransformerCPU(vocab_size, d_model, seed)
            size = model_size_bytes(transformer)
            if size <= budget_bytes:
                best_d = d_model
                best_size = int(size)
        rows.append(
            {
                "budget_mb": float(budget_mb),
                "lsl_fits": bool(lsl_size_bytes <= budget_bytes),
                "lsl_size_mb": float(lsl_size_bytes / (1024.0 * 1024.0)),
                "transformer_best_d_model": best_d,
                "transformer_best_size_mb": float(best_size / (1024.0 * 1024.0)) if best_d is not None else 0.0,
            }
        )
    return rows


def parse_memory_budgets(raw: str) -> List[float]:
    return [float(part.strip()) for part in str(raw or "").split(",") if part.strip()]


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = (row["dataset"], row["token_budget"], row["vocab_size_requested"], row["profile"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (dataset, token_budget, vocab_size, profile), items in sorted(groups.items()):
        losses = [float(item["eval"]["loss"]) for item in items]
        train_tps = [float(item["train"]["wall_tokens_per_second"]) for item in items]
        sizes = [float(item["model_size_mb"]) for item in items]
        summary.append(
            {
                "dataset": dataset,
                "token_budget": int(token_budget),
                "vocab_size": int(vocab_size),
                "profile": profile,
                "seeds": [int(item["seed"]) for item in items],
                "loss_mean": float(np.mean(losses)),
                "loss_std": float(np.std(losses)),
                "train_tps_mean": float(np.mean(train_tps)),
                "train_tps_std": float(np.std(train_tps)),
                "model_size_mb_mean": float(np.mean(sizes)),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, default="tinystories,wikitext2")
    parser.add_argument("--token-budgets", type=str, default="1000000,10000000")
    parser.add_argument("--vocab-sizes", type=str, default="1000,4000")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--lsl-profile", choices=["full", "native_long_context", "native_fast", "bio_native"], default="native_fast")
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--eval-tokens", type=int, default=2000)
    parser.add_argument("--max-train-chars", type=int, default=None)
    parser.add_argument("--max-eval-chars", type=int, default=120000)
    parser.add_argument("--chars-per-token", type=int, default=8)
    parser.add_argument("--tokenizer-train-chars", type=int, default=250000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--repeat-small", action="store_true")
    parser.add_argument("--compare-transformer", action="store_true")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--transformer-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--memory-budgets-mb", type=str, default="")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join("checkpoints", "phase1"))
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--smoke", action="store_true", help="use tiny defaults for CI/smoke verification")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.datasets = "tinystories,vietnamese_small,dialogue_small"
        args.token_budgets = "256"
        args.vocab_sizes = "256"
        args.seeds = "42,43"
        args.eval_tokens = min(int(args.eval_tokens), 128)
        args.max_train_chars = int(args.max_train_chars or 20000)
        args.max_eval_chars = min(int(args.max_eval_chars), 20000)
        args.repeat_small = True

    loader = DatasetLoader(ROOT)
    rows: List[Dict[str, object]] = []
    datasets = parse_csv(args.datasets)
    token_budgets = parse_csv_ints(args.token_budgets)
    vocab_sizes = parse_csv_ints(args.vocab_sizes)
    seeds = parse_csv_ints(args.seeds)
    max_runs = int(args.max_runs)

    run_count = 0
    for dataset in datasets:
        for token_budget in token_budgets:
            for vocab_size in vocab_sizes:
                for seed in seeds:
                    if max_runs and run_count >= max_runs:
                        break
                    row = train_one(args, loader, dataset, token_budget, vocab_size, seed)
                    rows.append(row)
                    run_count += 1
                    print(
                        f"{dataset} tokens={token_budget} vocab={vocab_size} seed={seed} "
                        f"loss={row['eval']['loss']:.4f} train_tps={row['train']['wall_tokens_per_second']:.1f}"
                    )
                if max_runs and run_count >= max_runs:
                    break
            if max_runs and run_count >= max_runs:
                break
        if max_runs and run_count >= max_runs:
            break

    payload = {
        "benchmark": "phase1_scaling_law",
        "success": True,
        "config": vars(args),
        "runs": rows,
        "summary": summarize(rows),
        "available_datasets": loader.list_available_datasets(),
    }
    dataset_label = "_".join(datasets[:3])
    output = write_result(
        payload,
        benchmark="phase1_scaling_law",
        dataset=dataset_label,
        seed=seeds[0] if seeds else None,
        config=vars(args),
        output_path=args.json_output,
        results_root=args.results_root,
    )
    print(f"Result JSON: {output}")
    print(f"Runs: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
