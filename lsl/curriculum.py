"""Three-stage continual-learning curriculum for LSL."""
from __future__ import annotations

import argparse
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import DatasetLoader, LSLCoreModel, RUNTIME_PROFILE_CHOICES, write_result


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = "results"
DEFAULT_CHECKPOINT_DIR = os.path.join("checkpoints", "curriculum")
SCALE_READY_THRESHOLDS = {
    "retention_mean_loss_ratio": 0.99,
    "grammar_coherence": 0.85,
    "train_tps": 4000.0,
    "ood_loss": 4.35,
    "ood_accuracy": 0.13,
}


def _csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _csv_ints(raw: str) -> List[int]:
    return [int(part.strip().replace("_", "")) for part in str(raw).split(",") if part.strip()]


def _safe_name(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip().lower()]
    out = "".join(chars).strip("_")
    return out or "run"


def _default_repeat(dataset: str, explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return str(dataset) in {"vietnamese_small", "dialogue_small"}


def _budget_from_tokens(tokens: Optional[int], chars_per_token: int, fallback: int = 0) -> Optional[int]:
    if tokens is None:
        return None if fallback <= 0 else int(fallback)
    return max(int(tokens) * max(1, int(chars_per_token)), fallback)


def _normalize_public_text(parts: Sequence[object]) -> str:
    cleaned = [str(part).strip() for part in parts if str(part).strip()]
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class CurriculumCorpus:
    dataset: str
    split: str = "train"
    max_tokens: Optional[int] = None
    tokenizer_train_chars: Optional[int] = None
    max_train_chars: Optional[int] = None
    max_eval_chars: Optional[int] = None
    profile: str = "native_fast"
    repeat_small: Optional[bool] = None
    corpus_path: Optional[str] = None
    description: str = ""


@dataclass
class CurriculumStage:
    name: str
    description: str
    corpora: List[CurriculumCorpus] = field(default_factory=list)
    profile: str = "native_fast"


def build_curriculum_plan(args: argparse.Namespace) -> List[CurriculumStage]:
    broaden_datasets = _csv(getattr(args, "broaden_datasets", "wikitext2,vietnamese_small,dialogue_small"))
    broaden_tokens = _csv_ints(getattr(args, "broaden_tokens", "100000,50000,50000"))
    if len(broaden_tokens) == 1 and len(broaden_datasets) > 1:
        broaden_tokens = broaden_tokens * len(broaden_datasets)
    if len(broaden_tokens) != len(broaden_datasets):
        raise ValueError("--broaden-datasets and --broaden-tokens must have the same length, or one broaden token value")

    bootstrap_tokens = int(getattr(args, "bootstrap_tokens", 100000))
    adapt_tokens = int(getattr(args, "adapt_tokens", 100000))
    chars_per_token = max(1, int(getattr(args, "chars_per_token", 8)))
    tokenizer_train_chars = int(getattr(args, "tokenizer_train_chars", 250000))
    eval_chars = max(int(getattr(args, "eval_tokens", 1200)) * chars_per_token, 8000)

    bootstrap_dataset = str(getattr(args, "bootstrap_dataset", "tinystories"))
    bootstrap_stage = CurriculumStage(
        name="bootstrap_grammar",
        description="Grammar bootstrapping on a clean starter corpus.",
        profile=str(getattr(args, "bootstrap_profile", "native_fast")),
        corpora=[
            CurriculumCorpus(
                dataset=bootstrap_dataset,
                split="train",
                max_tokens=bootstrap_tokens,
                tokenizer_train_chars=tokenizer_train_chars,
                max_train_chars=_budget_from_tokens(bootstrap_tokens, chars_per_token),
                max_eval_chars=eval_chars,
                profile=str(getattr(args, "bootstrap_profile", "native_fast")),
                repeat_small=False,
                description="Grammar bootstrap corpus",
            )
        ],
    )

    broaden_stage = CurriculumStage(
        name="broaden_language",
        description="Broaden the language model across real corpora.",
        profile=str(getattr(args, "broaden_profile", "continual")),
        corpora=[],
    )
    for dataset, token_budget in zip(broaden_datasets, broaden_tokens):
        broaden_stage.corpora.append(
            CurriculumCorpus(
                dataset=dataset,
                split="train",
                max_tokens=int(token_budget),
                tokenizer_train_chars=None,
                max_train_chars=_budget_from_tokens(int(token_budget), chars_per_token),
                max_eval_chars=eval_chars,
                profile=str(getattr(args, "broaden_profile", "continual")),
                repeat_small=_default_repeat(dataset, None),
                description="Corpus broadening stage",
            )
        )

    adapt_dataset = str(getattr(args, "adapt_dataset", "dialogue_small"))
    adapt_corpus_path = getattr(args, "adapt_corpus_path", None)
    adapt_token_splits_raw = getattr(args, "adapt_token_splits", None)
    adapt_token_splits = _csv_ints(adapt_token_splits_raw) if adapt_token_splits_raw else []
    adapt_stage = CurriculumStage(
        name="continual_adaptation",
        description="Continual adaptation and consolidation on a target domain.",
        profile=str(getattr(args, "adapt_profile", "continual")),
        corpora=[],
    )

    if adapt_token_splits:
        stages: List[CurriculumStage] = [bootstrap_stage, broaden_stage]
        for split_index, split_tokens in enumerate(adapt_token_splits, start=1):
            stages.append(
                CurriculumStage(
                    name=f"continual_adaptation_{split_index}",
                    description=f"Continual adaptation split {split_index}.",
                    profile=str(getattr(args, "adapt_profile", "continual")),
                    corpora=[
                        CurriculumCorpus(
                            dataset=adapt_dataset,
                            split="train",
                            max_tokens=int(split_tokens),
                            tokenizer_train_chars=None,
                            max_train_chars=_budget_from_tokens(int(split_tokens), chars_per_token),
                            max_eval_chars=eval_chars,
                            profile=str(getattr(args, "adapt_profile", "continual")),
                            repeat_small=_default_repeat(adapt_dataset, getattr(args, "adapt_repeat_small", None)),
                            corpus_path=adapt_corpus_path,
                            description=f"Continual adaptation corpus split {split_index}",
                        )
                    ],
                )
            )
        return stages

    adapt_stage.corpora = [
        CurriculumCorpus(
            dataset=adapt_dataset,
            split="train",
            max_tokens=adapt_tokens,
            tokenizer_train_chars=None,
            max_train_chars=_budget_from_tokens(adapt_tokens, chars_per_token),
            max_eval_chars=eval_chars,
            profile=str(getattr(args, "adapt_profile", "continual")),
            repeat_small=_default_repeat(adapt_dataset, getattr(args, "adapt_repeat_small", None)),
            corpus_path=adapt_corpus_path,
            description="Continual adaptation corpus",
        )
    ]
    return [bootstrap_stage, broaden_stage, adapt_stage]


def apply_curriculum_preset(args: argparse.Namespace) -> None:
    preset = str(getattr(args, "preset", "standard")).strip().lower().replace("-", "_")
    if preset in {"standard", ""}:
        return
    if preset == "grammar_safe":
        args.bootstrap_tokens = int(getattr(args, "bootstrap_tokens", 100000))
        args.broaden_datasets = "wikitext2,vietnamese_small"
        args.broaden_tokens = "25000,25000"
        args.broaden_profile = getattr(args, "broaden_profile", "continual")
        args.adapt_dataset = "wikitext2"
        args.adapt_tokens = 300000
        args.adapt_profile = getattr(args, "adapt_profile", "continual")
        args.adapt_repeat_small = False
        return
    if preset == "scale_ready":
        args.bootstrap_tokens = int(getattr(args, "bootstrap_tokens", 100000))
        args.broaden_datasets = "wikitext2,vietnamese_small"
        args.broaden_tokens = "25000,25000"
        args.broaden_profile = getattr(args, "broaden_profile", "continual")
        args.adapt_dataset = "wikitext2"
        args.adapt_tokens = 1000000
        args.adapt_token_splits = getattr(args, "adapt_token_splits", None) or "250000,250000,250000,250000"
        args.adapt_profile = getattr(args, "adapt_profile", "continual")
        args.adapt_repeat_small = False
        args.ood_items = int(getattr(args, "ood_items", 4))
        args.ood_eval_tokens = int(getattr(args, "ood_eval_tokens", 256))
        return
    if preset == "light_broaden":
        args.bootstrap_tokens = int(getattr(args, "bootstrap_tokens", 100000))
        args.broaden_datasets = "wikitext2,vietnamese_small"
        args.broaden_tokens = "12000,12000"
        args.adapt_dataset = str(getattr(args, "adapt_dataset", "dialogue_small"))
        args.adapt_tokens = int(getattr(args, "adapt_tokens", 100000))
        return
    if preset == "large_stage3":
        args.adapt_dataset = "wikitext2"
        args.adapt_tokens = 300000
        args.adapt_repeat_small = False
        return
    raise ValueError(f"Unsupported curriculum preset: {preset}")


def _resolve_corpus_name(corpus: CurriculumCorpus) -> str:
    if corpus.dataset == "custom":
        if not corpus.corpus_path:
            raise FileNotFoundError("--adapt-corpus-path is required when adapt-dataset=custom")
        return corpus.corpus_path
    return corpus.dataset


def _load_corpus(
    loader: DatasetLoader,
    corpus: CurriculumCorpus,
    seed: int,
    train_fraction: float,
    chars_per_token: int,
) -> Dict[str, object]:
    name = _resolve_corpus_name(corpus)
    max_train_chars = corpus.max_train_chars
    if max_train_chars is None and corpus.max_tokens is not None:
        max_train_chars = _budget_from_tokens(corpus.max_tokens, chars_per_token)
    max_eval_chars = corpus.max_eval_chars
    if max_eval_chars is None:
        max_eval_chars = max(8000, int((corpus.max_tokens or 0) * chars_per_token // 2))

    splits = loader.load_text_splits(
        name,
        train_fraction=float(train_fraction),
        max_train_chars=max_train_chars,
        max_eval_chars=max_eval_chars,
        seed=seed,
    )
    eval_text = splits.validation or splits.test or splits.train
    corpus_name = splits.dataset
    source_path = splits.train_path
    metadata = loader.dataset_metadata(name)
    return {
        "name": corpus_name,
        "source_name": name,
        "train_text": splits.train,
        "eval_text": eval_text,
        "train_path": splits.train_path,
        "eval_path": splits.validation_path if splits.validation else splits.test_path,
        "metadata": metadata,
        "language": splits.language,
        "train_chars": len(splits.train),
        "eval_chars": len(eval_text),
        "source_path": source_path,
    }


def _load_ood_examples(limit_per_source: int) -> List[Tuple[str, str]]:
    try:
        from benchmarks.phase8.public_datasets import load_babi, load_gsm8k, load_mbpp, load_squad
    except Exception:
        return []

    cache_dir = str(ROOT / "benchmarks" / "data" / "public")
    limit_per_source = max(1, int(limit_per_source))
    examples: List[Tuple[str, str]] = []

    try:
        for row in load_babi(
            cache_dir=cache_dir,
            split="test",
            language="en",
            tasks=None,
            limit_per_task=limit_per_source,
            download=False,
        ):
            examples.append(
                (
                    "babi",
                    _normalize_public_text(list(row.get("story", [])) + [row.get("question", ""), row.get("answer", "")]),
                )
            )
    except Exception:
        pass

    try:
        for row in load_squad(cache_dir=cache_dir, split="dev", limit=limit_per_source, download=False):
            examples.append(
                (
                    "squad",
                    _normalize_public_text(
                        [
                            row.get("title", ""),
                            row.get("context", ""),
                            row.get("question", ""),
                            row.get("answers", [""])[0] if row.get("answers") else "",
                        ]
                    ),
                )
            )
    except Exception:
        pass

    try:
        for row in load_gsm8k(cache_dir=cache_dir, split="test", limit=limit_per_source, download=False):
            examples.append(
                (
                    "gsm8k",
                    _normalize_public_text([row.get("question", ""), row.get("solution", ""), row.get("answer", "")]),
                )
            )
    except Exception:
        pass

    try:
        for row in load_mbpp(cache_dir=cache_dir, split="sanitized", limit=limit_per_source, download=False):
            examples.append(
                (
                    "mbpp",
                    _normalize_public_text([row.get("prompt", ""), row.get("code", ""), " ".join(row.get("tests", []))]),
                )
            )
    except Exception:
        pass

    return examples


def _evaluate_ood_suite(model: LSLCoreModel, limit_per_source: int, eval_tokens: int) -> Dict[str, object]:
    examples = _load_ood_examples(limit_per_source)
    rows: List[Dict[str, object]] = []
    if not examples:
        return {
            "limit_per_source": int(limit_per_source),
            "eval_tokens": int(eval_tokens),
            "row_count": 0,
            "rows": [],
            "summary": {
                "mean_loss": 0.0,
                "mean_accuracy": 0.0,
                "mean_p50_latency_us": 0.0,
                "mean_tokens_per_second": 0.0,
                "datasets": [],
            },
        }

    for dataset_name, text in examples:
        started = time.perf_counter()
        metrics = model.evaluate_text(text, max_tokens=eval_tokens)
        wall_seconds = time.perf_counter() - started
        tokens = float(metrics.get("tokens", 0.0))
        rows.append(
            {
                "dataset": dataset_name,
                "text_chars": len(text),
                "loss": float(metrics.get("loss", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "p50_latency_us": float(metrics.get("p50_latency_us", 0.0)),
                "tokens": tokens,
                "wall_seconds": float(wall_seconds),
                "tokens_per_second": float(tokens / max(wall_seconds, 1e-12)),
            }
        )

    datasets = sorted({row["dataset"] for row in rows})
    summary = {
        "mean_loss": float(sum(row["loss"] for row in rows) / max(1, len(rows))),
        "mean_accuracy": float(sum(row["accuracy"] for row in rows) / max(1, len(rows))),
        "mean_p50_latency_us": float(sum(row["p50_latency_us"] for row in rows) / max(1, len(rows))),
        "mean_tokens_per_second": float(sum(row["tokens_per_second"] for row in rows) / max(1, len(rows))),
        "datasets": datasets,
    }
    return {
        "limit_per_source": int(limit_per_source),
        "eval_tokens": int(eval_tokens),
        "row_count": len(rows),
        "rows": rows,
        "summary": summary,
    }


def evaluate_scale_readiness(stage_payloads: Sequence[Dict[str, object]], final_ood: Dict[str, object]) -> Dict[str, object]:
    retention_values = [_safe_float(stage.get("retention", {}).get("mean_loss_ratio", 0.0)) for stage in stage_payloads]
    grammar_values = [_safe_float(stage.get("metrics", {}).get("stage_grammar_coherence", 0.0)) for stage in stage_payloads]
    throughput_values = [_safe_float(stage.get("metrics", {}).get("stage_train_tps", 0.0)) for stage in stage_payloads]
    final_summary = dict(final_ood.get("summary", {}) or {})
    observed = {
        "retention_mean_loss_ratio": min(retention_values) if retention_values else 0.0,
        "grammar_coherence": min(grammar_values) if grammar_values else 0.0,
        "train_tps": min(throughput_values) if throughput_values else 0.0,
        "ood_loss": _safe_float(final_summary.get("mean_loss", 0.0)),
        "ood_accuracy": _safe_float(final_summary.get("mean_accuracy", 0.0)),
        "executed_stage_count": len(stage_payloads),
        "executed_stage_names": [str(stage.get("name", "")) for stage in stage_payloads],
        "final_stage_name": str(stage_payloads[-1]["name"]) if stage_payloads else "",
    }
    thresholds = dict(SCALE_READY_THRESHOLDS)
    failed: List[str] = []
    if observed["retention_mean_loss_ratio"] < thresholds["retention_mean_loss_ratio"]:
        failed.append(
            f"retention {observed['retention_mean_loss_ratio']:.3f} < {thresholds['retention_mean_loss_ratio']:.2f}"
        )
    if observed["grammar_coherence"] < thresholds["grammar_coherence"]:
        failed.append(f"grammar {observed['grammar_coherence']:.3f} < {thresholds['grammar_coherence']:.2f}")
    if observed["train_tps"] < thresholds["train_tps"]:
        failed.append(f"throughput {observed['train_tps']:.1f} < {thresholds['train_tps']:.1f}")
    if observed["ood_loss"] > thresholds["ood_loss"]:
        failed.append(f"ood_loss {observed['ood_loss']:.3f} > {thresholds['ood_loss']:.2f}")
    if observed["ood_accuracy"] < thresholds["ood_accuracy"]:
        failed.append(f"ood_accuracy {observed['ood_accuracy']:.3f} < {thresholds['ood_accuracy']:.2f}")
    passed = not failed
    return {
        "pass": passed,
        "thresholds": thresholds,
        "observed": observed,
        "failed_checks": failed,
        "detail": "all four-way gate thresholds met" if passed else "; ".join(failed),
    }


def _train_one_corpus(
    model: LSLCoreModel,
    corpus: CurriculumCorpus,
    corpus_data: Dict[str, object],
    tokenizer_train_chars: int,
    max_eval_tokens: int,
    generate_tokens: int,
) -> Dict[str, object]:
    train_text = str(corpus_data["train_text"])
    eval_text = str(corpus_data["eval_text"])

    started = time.perf_counter()
    train_metrics = model.train_stream(
        [train_text],
        tokenizer_text_chars=int(corpus.tokenizer_train_chars or tokenizer_train_chars),
        max_tokens=corpus.max_tokens,
    )
    wall_seconds = time.perf_counter() - started
    eval_metrics = model.evaluate_text(eval_text, max_tokens=max_eval_tokens)
    sample_prompt = " ".join(train_text.split()[:3]) or "the little girl"
    sample = model.generate(sample_prompt, max_new_tokens=generate_tokens)
    sample_metrics = model.generation_metrics(sample)

    return {
        "dataset": corpus_data["name"],
        "source_name": corpus_data["source_name"],
        "source_path": corpus_data["source_path"],
        "description": corpus.description,
        "profile": corpus.profile,
        "train_chars": corpus_data["train_chars"],
        "eval_chars": corpus_data["eval_chars"],
        "max_tokens": corpus.max_tokens,
        "repeat_small": bool(corpus.repeat_small),
        "train": {
            **train_metrics,
            "wall_seconds": float(wall_seconds),
            "wall_tokens_per_second": float(train_metrics["tokens"] / max(wall_seconds, 1e-12)),
        },
        "eval": eval_metrics,
        "generation": {
            "prompt": sample_prompt,
            "sample": sample,
            "metrics": sample_metrics,
        },
        "dataset_metadata": corpus_data["metadata"],
    }


def run_curriculum(args: argparse.Namespace) -> Dict[str, object]:
    apply_curriculum_preset(args)
    if getattr(args, "smoke", False):
        apply_smoke_defaults(args)
    loader = DatasetLoader(str(ROOT))
    stages = build_curriculum_plan(args)
    chars_per_token = max(1, int(getattr(args, "chars_per_token", 8)))
    tokenizer_train_chars = int(getattr(args, "tokenizer_train_chars", 250000))
    eval_tokens = int(getattr(args, "eval_tokens", 1200))
    generate_tokens = int(getattr(args, "generate_tokens", 48))
    ood_items = int(getattr(args, "ood_items", 4))
    ood_eval_tokens = int(getattr(args, "ood_eval_tokens", 256))
    train_fraction = float(getattr(args, "train_fraction", 0.70))

    checkpoint_dir = Path(getattr(args, "checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = getattr(args, "final_checkpoint", None) or os.path.join("checkpoints", "lsl_curriculum_final.json")
    scale_ready_mode = str(getattr(args, "preset", "standard")).strip().lower().replace("-", "_") == "scale_ready"

    model: Optional[LSLCoreModel] = None
    base_checkpoint = getattr(args, "load_checkpoint", None)
    if base_checkpoint:
        model = LSLCoreModel.load(base_checkpoint)
    else:
        model = LSLCoreModel(
            vocab_size=int(getattr(args, "vocab_size", 8000)),
            seed=int(getattr(args, "seed", 42)),
            candidate_cap=int(getattr(args, "candidate_cap", 128)),
            runtime_profile=str(getattr(args, "bootstrap_profile", "native_fast")),
        )

    seen_corpora: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    stage_payloads: List[Dict[str, object]] = []
    evaluation_history: List[Dict[str, object]] = []
    gate_fail_stage: Optional[str] = None

    for stage_index, stage in enumerate(stages, start=1):
        model.set_runtime_profile(stage.profile)
        stage_started = time.perf_counter()
        stage_runs: List[Dict[str, object]] = []

        for corpus_index, corpus in enumerate(stage.corpora, start=1):
            corpus_seed = int(getattr(args, "seed", 42)) + (stage_index - 1) * 101 + corpus_index
            corpus_data = _load_corpus(
                loader,
                corpus,
                seed=corpus_seed,
                train_fraction=train_fraction,
                chars_per_token=chars_per_token,
            )
            corpus_key = f"{stage_index}:{corpus_index}:{_safe_name(corpus_data['name'])}"
            corpus_result = _train_one_corpus(
                model,
                corpus,
                corpus_data,
                tokenizer_train_chars=tokenizer_train_chars,
                max_eval_tokens=eval_tokens,
                generate_tokens=generate_tokens,
            )
            seen_corpora[corpus_key] = {
                "name": corpus_data["name"],
                "source_name": corpus_data["source_name"],
                "source_path": corpus_data["source_path"],
                "eval_text": corpus_data["eval_text"],
                "dataset_metadata": corpus_data["metadata"],
                "initial_eval": corpus_result["eval"],
            }
            stage_runs.append(
                {
                    "corpus_key": corpus_key,
                    "stage_index": stage_index,
                    "corpus_index": corpus_index,
                    **corpus_result,
                }
            )

        stage_checkpoint = os.path.abspath(
            os.path.join(
                checkpoint_dir,
                f"lsl_curriculum_stage{stage_index:02d}_{_safe_name(stage.name)}.json",
            )
        )
        model.save(stage_checkpoint)

        stage_eval_after: Dict[str, Dict[str, float]] = {}
        for corpus_key, info in seen_corpora.items():
            eval_metrics = model.evaluate_text(str(info["eval_text"]), max_tokens=eval_tokens)
            stage_eval_after[corpus_key] = eval_metrics
            evaluation_history.append(
                {
                    "stage_index": stage_index,
                    "stage": stage.name,
                    "corpus_key": corpus_key,
                    "eval": eval_metrics,
                }
            )

        stage_ood = _evaluate_ood_suite(model, limit_per_source=ood_items, eval_tokens=ood_eval_tokens)
        stage_payload = {
            "name": stage.name,
            "description": stage.description,
            "profile": stage.profile,
            "checkpoint_path": stage_checkpoint,
            "elapsed_seconds": float(time.perf_counter() - stage_started),
            "train_runs": stage_runs,
            "evaluation_after_stage": stage_eval_after,
            "ood": stage_ood,
            "success": True,
        }
        stage_payload["benchmark"] = f"lsl_curriculum_{stage.name}"
        stage_payload["dataset"] = stage.name
        stage_payload["metrics"] = {
            "stage_tokens": float(sum(float(run["train"]["tokens"]) for run in stage_runs)),
            "stage_wall_seconds": float(sum(float(run["train"]["wall_seconds"]) for run in stage_runs)),
            "stage_train_tps": float(
                sum(float(run["train"]["tokens"]) for run in stage_runs)
                / max(1e-12, sum(float(run["train"]["wall_seconds"]) for run in stage_runs))
            ),
            "stage_eval_loss": float(
                sum(float(run["eval"]["loss"]) * float(run["eval"]["tokens"]) for run in stage_runs)
                / max(1e-12, sum(float(run["eval"]["tokens"]) for run in stage_runs))
            ),
            "stage_eval_accuracy": float(
                sum(float(run["eval"]["accuracy"]) * float(run["eval"]["tokens"]) for run in stage_runs)
                / max(1e-12, sum(float(run["eval"]["tokens"]) for run in stage_runs))
            ),
            "stage_grammar_coherence": float(
                sum(float(run["generation"]["metrics"]["coherence"]) for run in stage_runs) / max(1, len(stage_runs))
            ),
            "stage_grammar_loop_rate": float(
                sum(float(run["generation"]["metrics"]["loop_rate"]) for run in stage_runs) / max(1, len(stage_runs))
            ),
            "stage_grammar_unk_rate": float(
                sum(float(run["generation"]["metrics"]["unk_rate"]) for run in stage_runs) / max(1, len(stage_runs))
            ),
            "stage_ood_loss": float(stage_ood["summary"]["mean_loss"]),
            "stage_ood_accuracy": float(stage_ood["summary"]["mean_accuracy"]),
            "stage_ood_latency_us": float(stage_ood["summary"]["mean_p50_latency_us"]),
            "stage_ood_tps": float(stage_ood["summary"]["mean_tokens_per_second"]),
        }
        retention_rows: List[Dict[str, float]] = []
        for corpus_key, info in seen_corpora.items():
            current = stage_eval_after[corpus_key]
            initial = info["initial_eval"]
            initial_loss = float(initial["loss"])
            current_loss = float(current["loss"])
            retention_rows.append(
                {
                    "corpus_key": corpus_key,
                    "loss_ratio": float(current_loss / max(initial_loss, 1e-12)),
                    "accuracy_delta": float(float(current.get("accuracy", 0.0)) - float(initial.get("accuracy", 0.0))),
                }
            )
        stage_payload["retention"] = {
            "mean_loss_ratio": float(sum(row["loss_ratio"] for row in retention_rows) / max(1, len(retention_rows))),
            "mean_accuracy_delta": float(sum(row["accuracy_delta"] for row in retention_rows) / max(1, len(retention_rows))),
            "rows": retention_rows,
        }
        stage_output = write_result(
            stage_payload,
            benchmark=f"lsl_curriculum_{stage.name}",
            dataset=stage.name,
            seed=int(getattr(args, "seed", 42)),
            config=vars(args),
            results_root=str(getattr(args, "results_root", DEFAULT_RESULTS_ROOT)),
        )
        stage_payload["result_path"] = stage_output
        stage_payloads.append(stage_payload)

        if scale_ready_mode:
            current_scale_readiness = evaluate_scale_readiness(stage_payloads, stage_ood)
            if not current_scale_readiness["pass"]:
                gate_fail_stage = stage.name
                break

    final_checkpoint = os.path.abspath(final_checkpoint)
    model.save(final_checkpoint)

    final_eval: Dict[str, Dict[str, float]] = {}
    for corpus_key, info in seen_corpora.items():
        final_eval[corpus_key] = model.evaluate_text(str(info["eval_text"]), max_tokens=eval_tokens)
    final_ood = _evaluate_ood_suite(model, limit_per_source=ood_items, eval_tokens=ood_eval_tokens)

    retention: Dict[str, Dict[str, float]] = {}
    for corpus_key, info in seen_corpora.items():
        initial = info["initial_eval"]
        final = final_eval[corpus_key]
        initial_loss = float(initial["loss"])
        final_loss = float(final["loss"])
        retention[corpus_key] = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_delta": float(final_loss - initial_loss),
            "loss_ratio": float(final_loss / max(initial_loss, 1e-12)),
            "initial_accuracy": float(initial.get("accuracy", 0.0)),
            "final_accuracy": float(final.get("accuracy", 0.0)),
            "accuracy_delta": float(final.get("accuracy", 0.0) - initial.get("accuracy", 0.0)),
        }

    summary = {
        "stage_count": len(stage_payloads),
        "planned_stage_count": len(stages),
        "corpus_count": len(seen_corpora),
        "mean_final_loss": float(sum(float(m["loss"]) for m in final_eval.values()) / max(1, len(final_eval))),
        "mean_final_accuracy": float(sum(float(m["accuracy"]) for m in final_eval.values()) / max(1, len(final_eval))),
        "mean_final_ood_loss": float(final_ood["summary"]["mean_loss"]),
        "mean_final_ood_accuracy": float(final_ood["summary"]["mean_accuracy"]),
        "final_checkpoint": final_checkpoint,
    }

    scale_readiness = evaluate_scale_readiness(stage_payloads, final_ood)
    scale_readiness["mode"] = "promotion" if scale_ready_mode else "diagnostic"
    scale_readiness["failed_stage"] = gate_fail_stage
    scale_readiness["stopped_early"] = gate_fail_stage is not None
    summary["scale_ready_pass"] = bool(scale_readiness["pass"])
    summary["scale_ready_failed_checks"] = list(scale_readiness["failed_checks"])
    summary["scale_ready_failed_stage"] = gate_fail_stage

    payload = {
        "benchmark": "lsl_curriculum",
        "success": bool(scale_readiness["pass"]) if scale_ready_mode else True,
        "config": vars(args),
        "base_checkpoint": os.path.abspath(base_checkpoint) if base_checkpoint else None,
        "final_checkpoint": final_checkpoint,
        "stages": stage_payloads,
        "final_evaluation": final_eval,
        "final_ood": final_ood,
        "retention": retention,
        "evaluation_history": evaluation_history,
        "scale_readiness": scale_readiness,
        "summary": summary,
        "native_core": model.diagnostics().get("native_core_enabled", 0.0),
    }
    output_path = getattr(args, "json_output", None)
    payload["result_path"] = write_result(
        payload,
        benchmark="lsl_curriculum",
        dataset=_safe_name(str(getattr(args, "bootstrap_dataset", "tinystories"))),
        seed=int(getattr(args, "seed", 42)),
        config=vars(args),
        output_path=output_path,
        results_root=str(getattr(args, "results_root", DEFAULT_RESULTS_ROOT)),
    )
    return payload


def add_curriculum_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--preset", choices=["standard", "grammar_safe", "light_broaden", "large_stage3", "scale_ready"], default="standard")
    parser.add_argument("--bootstrap-dataset", choices=["tinystories", "wikitext2"], default="tinystories")
    parser.add_argument("--bootstrap-tokens", type=int, default=100000)
    parser.add_argument("--bootstrap-profile", choices=list(RUNTIME_PROFILE_CHOICES), default="native_fast")
    parser.add_argument("--broaden-datasets", type=str, default="wikitext2,vietnamese_small,dialogue_small")
    parser.add_argument("--broaden-tokens", type=str, default="100000,50000,50000")
    parser.add_argument("--broaden-profile", choices=list(RUNTIME_PROFILE_CHOICES), default="continual")
    parser.add_argument("--adapt-dataset", choices=["tinystories", "wikitext2", "vietnamese_small", "dialogue_small", "custom"], default="dialogue_small")
    parser.add_argument("--adapt-corpus-path", type=str, default=None)
    parser.add_argument("--adapt-tokens", type=int, default=100000)
    parser.add_argument(
        "--adapt-token-splits",
        type=str,
        default=None,
        help="comma-separated stage-3 token chunks; use this to split the adaptation lane into smaller sub-stages",
    )
    parser.add_argument("--adapt-profile", choices=list(RUNTIME_PROFILE_CHOICES), default="continual")
    parser.add_argument("--adapt-repeat-small", action="store_true")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="resume the curriculum from an existing checkpoint before stage 1")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--final-checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer-train-chars", type=int, default=250000)
    parser.add_argument("--eval-tokens", type=int, default=1200)
    parser.add_argument("--generate-tokens", type=int, default=48)
    parser.add_argument("--chars-per-token", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--ood-items", type=int, default=4)
    parser.add_argument("--ood-eval-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="use tiny budgets for quick verification")
    return parser


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    args.bootstrap_tokens = 2048
    broaden_count = max(1, len(_csv(getattr(args, "broaden_datasets", "wikitext2,vietnamese_small,dialogue_small"))))
    if broaden_count == 1:
        args.broaden_tokens = "1024"
    else:
        args.broaden_tokens = "1024," + ",".join(["512"] * (broaden_count - 1))
    adapt_splits_raw = getattr(args, "adapt_token_splits", None)
    if adapt_splits_raw:
        split_count = max(1, len(_csv(adapt_splits_raw)))
        split_budget = max(64, 512 // split_count)
        args.adapt_token_splits = ",".join([str(split_budget)] * split_count)
        args.adapt_tokens = split_budget * split_count
    else:
        args.adapt_tokens = 512
    args.tokenizer_train_chars = 8000
    args.eval_tokens = 128
    args.generate_tokens = 16
    args.chars_per_token = 4
    args.adapt_repeat_small = True
    args.ood_items = 1
    args.ood_eval_tokens = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_curriculum_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "smoke", False):
        apply_smoke_defaults(args)
    payload = run_curriculum(args)
    print("LSL curriculum")
    print("=" * 72)
    print(f"Base checkpoint:  {payload['base_checkpoint'] or '(fresh model)'}")
    print(f"Final checkpoint: {payload['final_checkpoint']}")
    for stage in payload["stages"]:
        ood = stage.get("ood", {}).get("summary", {})
        print(
            f"Stage {stage['name']}: tokens={stage['metrics']['stage_tokens']:.0f} "
            f"tps={stage['metrics']['stage_train_tps']:.1f} "
            f"grammar={stage['metrics']['stage_grammar_coherence']:.3f} "
            f"retention={stage['retention']['mean_loss_ratio']:.3f} "
            f"ood={float(ood.get('mean_loss', 0.0)):.3f} "
            f"checkpoint={stage['checkpoint_path']}"
        )
    print(f"Mean final loss:   {payload['summary']['mean_final_loss']:.4f}")
    print(f"Mean final acc:    {payload['summary']['mean_final_accuracy']:.4f}")
    print(f"Mean final OOD loss:{payload['summary']['mean_final_ood_loss']:.4f}")
    print(f"Mean final OOD acc: {payload['summary']['mean_final_ood_accuracy']:.4f}")
    scale = payload.get("scale_readiness", {})
    observed = scale.get("observed", {})
    print(
        "Scale gate:      "
        f"{'PASS' if scale.get('pass') else 'FAIL'} "
        f"(retention={float(observed.get('retention_mean_loss_ratio', 0.0)):.3f}, "
        f"grammar={float(observed.get('grammar_coherence', 0.0)):.3f}, "
        f"throughput={float(observed.get('train_tps', 0.0)):.1f}, "
        f"ood_loss={float(observed.get('ood_loss', 0.0)):.3f}, "
        f"ood_acc={float(observed.get('ood_accuracy', 0.0)):.3f})"
    )
    print(f"Result JSON:       {payload['result_path']}")
    return 0 if payload.get("success", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
