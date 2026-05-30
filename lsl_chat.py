"""Interactive chat/demo loop for an LSLCoreModel checkpoint."""
from __future__ import annotations

import argparse
import os
import sys

from lsl import LSLCoreModel


ROOT = os.path.abspath(os.path.dirname(__file__))
TINYSTORIES = os.path.join(ROOT, "benchmarks", "data", "tinystories", "TinyStoriesV2-GPT4-valid.txt")
TINYSTORIES_SUBSET = os.path.join(ROOT, "benchmarks", "phase4", "tinystories_subset.txt")
WIKITEXT = os.path.join(ROOT, "benchmarks", "data", "wikitext-2-raw-v1", "wiki.train.raw.txt")


def respond(model: LSLCoreModel, prompt: str, max_new_tokens: int) -> str:
    answer = model.answer(prompt)
    if answer is not None:
        return str(answer)
    return model.generate(prompt, max_new_tokens=max_new_tokens)


def read_bootstrap_text(dataset: str, corpus_path: str | None, max_chars: int) -> tuple[str, str]:
    if corpus_path:
        path = corpus_path
    elif dataset == "wikitext2":
        path = WIKITEXT
    else:
        path = TINYSTORIES if os.path.exists(TINYSTORIES) else TINYSTORIES_SUBSET
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read(max_chars), os.path.abspath(path)


def bootstrap_checkpoint(args: argparse.Namespace) -> LSLCoreModel:
    text, source_path = read_bootstrap_text(args.bootstrap_dataset, args.bootstrap_corpus_path, args.bootstrap_chars)
    print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
    print(
        f"Bootstrapping a small {args.bootstrap_dataset} checkpoint "
        f"({args.bootstrap_tokens:,} tokens) from {source_path}",
        file=sys.stderr,
    )
    model = LSLCoreModel(vocab_size=args.vocab_size, candidate_cap=args.candidate_cap, seed=args.seed)
    metrics = model.train_stream([text], tokenizer_text_chars=args.bootstrap_chars, max_tokens=args.bootstrap_tokens)
    model.save(args.checkpoint)
    print(
        f"Saved {args.checkpoint} | tokens={int(metrics['tokens']):,} "
        f"| us/token={metrics['us_per_token']:.2f}",
        file=sys.stderr,
    )
    return model


def ensure_native_chat_path(model: LSLCoreModel, checkpoint: str, save_upgrade: bool = True) -> None:
    diag = model.diagnostics()
    if diag.get("native_core_enabled", 0.0) < 1.0:
        return
    if diag.get("native_core_update_calls", 0.0) > 0.0:
        return
    packed = model.rebuild_native_core_from_memory()
    if packed.get("rebuilt_sources", 0.0) <= 0.0:
        return
    print(
        "Packed chat transitions into native C core: "
        f"sources={int(packed['rebuilt_sources'])}, edges={int(packed['rebuilt_edges'])}",
        file=sys.stderr,
    )
    if save_upgrade:
        model.save(checkpoint)
        print(f"Saved native-upgraded checkpoint: {checkpoint}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "lsl_tinystories.json"))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-bootstrap", action="store_true", help="fail if the checkpoint is missing")
    parser.add_argument("--bootstrap-dataset", choices=["tinystories", "wikitext2"], default="tinystories")
    parser.add_argument("--bootstrap-corpus-path", type=str, default=None)
    parser.add_argument("--bootstrap-tokens", type=int, default=5000)
    parser.add_argument("--bootstrap-chars", type=int, default=250000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-native-upgrade", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        if args.no_bootstrap:
            print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            print("Create one with: python benchmarks/train_lsl_corpus.py --dataset tinystories", file=sys.stderr)
            return 2
        model = bootstrap_checkpoint(args)
    else:
        model = LSLCoreModel.load(args.checkpoint)
        ensure_native_chat_path(model, args.checkpoint, save_upgrade=not args.no_save_native_upgrade)
    if args.prompt is not None:
        print(respond(model, args.prompt, args.max_new_tokens))
        return 0
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
        print("lsl>", respond(model, prompt, args.max_new_tokens))
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
