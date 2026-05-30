"""Interactive chat/demo loop for an LSLCoreModel checkpoint."""
from __future__ import annotations

import argparse
import os
import sys

from lsl import LSLCoreModel


def respond(model: LSLCoreModel, prompt: str, max_new_tokens: int) -> str:
    answer = model.answer(prompt)
    if answer is not None:
        return str(answer)
    return model.generate(prompt, max_new_tokens=max_new_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "lsl_tinystories.json"))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        print("Create one with: python benchmarks/train_lsl_corpus.py --dataset tinystories", file=sys.stderr)
        return 2
    model = LSLCoreModel.load(args.checkpoint)
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
            for key in sorted(diag)[:40]:
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
