"""Unified CLI for LSL train / eval / chat / report commands."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from . import DatasetConfig, DatasetLoader, LSLCoreModel, RUNTIME_PROFILE_CHOICES, write_html_report, write_result
from .curriculum import add_curriculum_arguments, run_curriculum


ROOT = Path(__file__).resolve().parents[1]


def _dataset_name(args: argparse.Namespace) -> str:
    if args.dataset == "custom":
        if not args.corpus_path:
            raise FileNotFoundError("--corpus-path is required for custom dataset")
        return args.corpus_path
    return args.dataset


def _load_text(args: argparse.Namespace) -> tuple[str, str]:
    loader = DatasetLoader(str(ROOT))
    name = _dataset_name(args)
    text = loader.load_text(
        DatasetConfig(
            name=name,
            split=getattr(args, "split", "train"),
            max_chars=getattr(args, "max_chars", None),
            repeat=bool(getattr(args, "repeat_small", False)),
        )
    )
    path = loader.resolve_path(name, getattr(args, "split", "train"))
    return text, os.path.abspath(path)


def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", choices=["tinystories", "wikitext2", "vietnamese_small", "dialogue_small", "custom"], default="tinystories")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "validation", "val", "test"], default="train")
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--max-chars", type=int, default=5000000)
    parser.add_argument("--tokenizer-train-chars", type=int, default=250000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--lsl-profile", choices=list(RUNTIME_PROFILE_CHOICES), default="native_fast")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="resume training from an existing checkpoint before saving the new output checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--repeat-small", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def add_eval_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", choices=["tinystories", "wikitext2", "vietnamese_small", "dialogue_small", "custom"], default="tinystories")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--max-train-chars", type=int, default=120000)
    parser.add_argument("--max-eval-chars", type=int, default=40000)
    parser.add_argument("--tokenizer-train-chars", type=int, default=100000)
    parser.add_argument("--tokens", type=int, default=None, help="Alias for --max-train-tokens")
    parser.add_argument("--max-train-tokens", type=int, default=6000)
    parser.add_argument("--max-eval-tokens", type=int, default=1600)
    parser.add_argument("--eval-tokens", type=int, default=1200)
    parser.add_argument("--vocab-size", type=int, default=4000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--lsl-profile", choices=list(RUNTIME_PROFILE_CHOICES), default="native_fast")
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--context-lengths", type=str, default="16,32,64,128")
    parser.add_argument("--context-latency-iterations", type=int, default=32)
    parser.add_argument("--transformer-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--prompt-tokens", type=int, default=12)
    parser.add_argument("--generate-tokens", type=int, default=48)
    parser.add_argument("--fact-items", type=int, default=8)
    parser.add_argument("--claim", action="store_true")
    parser.add_argument("--quality-ratio-target", type=float, default=1.25)
    parser.add_argument("--latency-speedup-target", type=float, default=4.0)
    parser.add_argument("--loop-rate-target", type=float, default=0.03)
    parser.add_argument("--fact-recall-target", type=float, default=0.95)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def add_chat_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "lsl_tinystories.json"))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-dataset", choices=["tinystories", "wikitext2"], default="tinystories")
    parser.add_argument("--bootstrap-corpus-path", type=str, default=None)
    parser.add_argument("--bootstrap-tokens", type=int, default=5000)
    parser.add_argument("--bootstrap-chars", type=int, default=250000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lsl-profile", choices=list(RUNTIME_PROFILE_CHOICES), default="bio_native")
    parser.add_argument("--no-save-native-upgrade", action="store_true")
    return parser


def add_report_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output", type=str, default=os.path.join("results", "lsl_report.html"))
    parser.add_argument("--title", type=str, default="LSL Benchmark Report")
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    text, source_path = _load_text(args)
    checkpoint = args.checkpoint or os.path.join("checkpoints", f"lsl_{args.dataset}.json")
    if args.load_checkpoint:
        model = LSLCoreModel.load(args.load_checkpoint)
        model.set_runtime_profile(args.lsl_profile)
        loaded_checkpoint = os.path.abspath(args.load_checkpoint)
    else:
        model = LSLCoreModel(
            vocab_size=args.vocab_size,
            seed=args.seed,
            candidate_cap=args.candidate_cap,
            runtime_profile=args.lsl_profile,
        )
        loaded_checkpoint = None
    metrics = model.train_stream([text], tokenizer_text_chars=args.tokenizer_train_chars, max_tokens=args.max_tokens)
    model.save(checkpoint)
    sample_prompt = " ".join(str(text).split()[:3]) or "the little girl"
    sample = model.generate(sample_prompt, max_new_tokens=48)
    payload = {
        "benchmark": "lsl_cli_train",
        "dataset": args.dataset,
        "split": getattr(args, "split", "train"),
        "corpus_path": source_path,
        "checkpoint": os.path.abspath(checkpoint),
        "loaded_checkpoint": loaded_checkpoint,
        "success": True,
        "metrics": {
            **metrics,
            "tokens_per_second": float(metrics["tokens"] / max(metrics["elapsed_seconds"], 1e-12)),
            "vocab_size": model.vocab_size,
            "lsl_profile": args.lsl_profile,
        },
        "sample_prompt": sample_prompt,
        "sample": sample,
    }
    output = write_result(
        payload,
        benchmark="lsl_cli_train",
        dataset=args.dataset,
        seed=args.seed,
        config=vars(args),
        output_path=args.json_output,
        results_root=args.results_root,
    )
    print("LSL train")
    print("=" * 72)
    print(f"Dataset:    {args.dataset}")
    print(f"Corpus:     {source_path}")
    print(f"Checkpoint: {os.path.abspath(checkpoint)}")
    if loaded_checkpoint:
        print(f"Loaded:     {loaded_checkpoint}")
    print(f"Tokens:     {int(metrics['tokens']):,}")
    print(f"Tok/s:      {payload['metrics']['tokens_per_second']:.2f}")
    print(f"Sample:     {sample[:240]}")
    print(f"Result JSON: {output}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from benchmarks.competitive.run_lsl_vs_transformer import run

    if args.tokens is not None:
        args.max_train_tokens = int(args.tokens)
        args.max_train_chars = max(int(args.max_train_chars), int(args.tokens) * 12)
        args.max_eval_chars = max(int(args.max_eval_chars), int(args.eval_tokens) * 12)
    result = run(args)
    metrics = result["metrics"]
    comp = metrics["comparison"]
    print("LSL eval")
    print("=" * 72)
    print(f"Dataset:    {metrics['dataset']}")
    print(f"Loss ratio: {comp['loss_ratio_lsl_over_transformer']:.3f}x")
    print(f"Speedup:    {comp['latency_speedup_transformer_over_lsl']:.2f}x")
    print(f"Native:     forward={metrics['native_core']['forward_native_ratio']:.2%} update={metrics['native_core']['update_native_ratio']:.2%}")
    print(f"Result JSON: {result.get('result_path') or args.json_output or '(not written)'}")
    return 0 if result["success"] else 1


def _chat_loop(model: LSLCoreModel, checkpoint: str, max_new_tokens: int, once: bool) -> int:
    import lsl_chat

    print("LSL chat. Commands: /exit, /diag, /remember SUBJECT RELATION OBJECT")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/diag":
            diag = model.diagnostics()
            keys = [key for key in sorted(diag) if key.startswith("native_core_")]
            keys += [key for key in sorted(diag) if not key.startswith("native_core_")]
            for key in keys[:40]:
                print(f"{key}: {diag[key]}")
            continue
        if prompt.startswith("/remember "):
            parts = prompt.split(maxsplit=3)
            if len(parts) != 4:
                print("usage: /remember SUBJECT RELATION OBJECT")
                continue
            _, subject, relation, obj = parts
            model.agent.observe_event(subject, relation, obj, episode_id=int(model.seen_tokens), evidence_id=0)
            print("remembered")
            continue
        print("lsl>", lsl_chat.respond(model, prompt, max_new_tokens))
        if once:
            return 0


def cmd_chat(args: argparse.Namespace) -> int:
    import lsl_chat

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checkpoint = args.checkpoint
    if not os.path.exists(checkpoint):
        if args.no_bootstrap:
            print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
            print("Create one with: python lsl_cli.py train --dataset tinystories", file=sys.stderr)
            return 2
        model = lsl_chat.bootstrap_checkpoint(args)
        lsl_chat.ensure_native_chat_path(model, checkpoint, save_upgrade=not args.no_save_native_upgrade)
    else:
        model = LSLCoreModel.load(checkpoint)
        model.set_runtime_profile(args.lsl_profile)
        lsl_chat.ensure_native_chat_path(model, checkpoint, save_upgrade=not args.no_save_native_upgrade)
    if args.prompt is not None:
        print(lsl_chat.respond(model, args.prompt, args.max_new_tokens))
        return 0
    return _chat_loop(model, checkpoint, args.max_new_tokens, args.once)


def cmd_report(args: argparse.Namespace) -> int:
    output = write_html_report(args.output, results_root=args.results_root, title=args.title)
    print(f"Wrote HTML report: {output}")
    return 0


def cmd_curriculum(args: argparse.Namespace) -> int:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    add_train_arguments(sub.add_parser("train", help="train a checkpoint on a named corpus"))
    add_eval_arguments(sub.add_parser("eval", help="run the competitive evaluation benchmark"))
    add_chat_arguments(sub.add_parser("chat", help="open the interactive chat loop"))
    add_report_arguments(sub.add_parser("report", help="render the latest HTML report"))
    add_curriculum_arguments(sub.add_parser("curriculum", help="run the 3-stage continual-learning curriculum"))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    if args.command == "train":
        return cmd_train(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "chat":
        return cmd_chat(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "curriculum":
        return cmd_curriculum(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
