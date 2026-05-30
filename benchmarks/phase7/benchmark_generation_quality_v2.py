"""Phase 7 stronger open-generation scoring."""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase4.benchmark_language_quality import NGramBaseline
from benchmarks.phase5.benchmark_long_context_real_corpus import read_text, tokenize_splits
from lsl import DiscoursePlan, GenerationController, LongContextMemory


def maps(tokenizer):
    if hasattr(tokenizer, "word_to_id"):
        return tokenizer.word_to_id, tokenizer.id_to_word
    return tokenizer.token_to_id, tokenizer.id_to_token


def train(tokens: List[int], vocab_size: int, args: argparse.Namespace) -> LongContextMemory:
    memory = LongContextMemory(
        capacity=args.capacity,
        vocab_size=vocab_size,
        context_width=args.context_width,
        candidate_cap=args.candidate_cap,
        store_transition_index=False,
        target_cap=args.target_cap,
        seed=args.seed,
    )
    for i in range(len(tokens) - 1):
        memory.observe_transition(tokens[i], tokens[i + 1], vocab_size=vocab_size)
    return memory


def train_ngram(tokens: List[int], vocab_size: int) -> NGramBaseline:
    model = NGramBaseline(vocab_size)
    for i in range(len(tokens) - 1):
        model.observe(tokens[i], tokens[i + 1])
    return model


def gen_ngram(model: NGramBaseline, prompt: List[int], n: int) -> List[int]:
    out = list(prompt)
    current = int(out[-1])
    seen = set()
    for _ in range(n):
        nxt = int(model.predict(current))
        tri = tuple((out[-2], out[-1], nxt)) if len(out) >= 2 else None
        if tri is not None and tri in seen:
            break
        if tri is not None:
            seen.add(tri)
        out.append(nxt)
        current = nxt
    return out


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    train_text, eval_text, corpus_path = read_text(args)
    tokenizer, train_tokens, eval_tokens = tokenize_splits(train_text, eval_text, args)
    train_tokens = train_tokens[: args.max_train_tokens]
    eval_tokens = eval_tokens[: args.max_eval_tokens]
    vocab_size = tokenizer.vocab_size
    to_id, _ = maps(tokenizer)
    unk_id = int(to_id.get("<UNK>", 1))
    entity_ids = tuple(int(to_id[x]) for x in ("lily", "john", "mary", "alice", "bob") if x in to_id)
    plan = DiscoursePlan(target_length=args.generate_tokens, entity_ids=entity_ids, style_tokens=entity_ids)
    memory = train(train_tokens, vocab_size, args)
    controller = GenerationController(
        memory=memory,
        vocab_size=vocab_size,
        candidate_limit=args.candidate_limit,
        unk_id=unk_id,
        sentence_end_ids=[to_id[x] for x in [".", "!", "?"] if x in to_id],
        plan=plan,
        seed=args.seed,
    )
    baseline = train_ngram(train_tokens, vocab_size)
    rng = np.random.default_rng(args.seed + 7)
    max_start = max(1, len(eval_tokens) - args.prompt_tokens - args.generate_tokens)
    starts = rng.choice(np.arange(max_start), size=min(args.trials, max_start), replace=False)
    ours = []
    bases = []
    samples = []
    for start in starts:
        prompt = eval_tokens[int(start): int(start) + args.prompt_tokens]
        generated = controller.generate(prompt, args.generate_tokens, plan=plan)
        base = gen_ngram(baseline, prompt, args.generate_tokens)
        ours.append(GenerationController.generation_metrics(generated, unk_id, entity_ids))
        bases.append(GenerationController.generation_metrics(base, unk_id, entity_ids))
        if len(samples) < 2:
            samples.append({"prompt": tokenizer.decode(prompt), "generated": tokenizer.decode(generated), "baseline": tokenizer.decode(base)})
    mean = lambda key, rows: float(np.mean([row[key] for row in rows])) if rows else 0.0
    metrics = {
        "score": mean("coherence", ours),
        "baseline_score": mean("coherence", bases),
        "loop_rate": mean("loop_rate", ours),
        "unk_rate": mean("unk_rate", ours),
        "entity_consistency": mean("entity_consistency", ours),
        "distinct2": mean("distinct2", ours),
    }
    metrics["score_ratio"] = metrics["score"] / max(metrics["baseline_score"], 1e-9)
    checks = {
        "score_ratio": metrics["score_ratio"] >= args.score_ratio_target,
        "loop": metrics["loop_rate"] <= args.loop_target,
        "unk": metrics["unk_rate"] <= args.unk_target,
        "entity": metrics["entity_consistency"] >= args.entity_target,
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "samples": samples,
        "corpus_path": corpus_path,
        "vocab_size": vocab_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "tinystories_full", "wikitext2", "custom"], default="tinystories")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--wikitext-cache-dir", type=str, default=None)
    parser.add_argument("--tinystories-cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=2500)
    parser.add_argument("--tokenizer", choices=["word", "subword"], default="subword")
    parser.add_argument("--tokenizer-train-chars", type=int, default=140000)
    parser.add_argument("--max-train-chars", type=int, default=260000)
    parser.add_argument("--max-eval-chars", type=int, default=90000)
    parser.add_argument("--subword-max-merges", type=int, default=600)
    parser.add_argument("--subword-min-pair-count", type=int, default=3)
    parser.add_argument("--max-train-tokens", type=int, default=7000)
    parser.add_argument("--max-eval-tokens", type=int, default=1800)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--context-width", type=int, default=6)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--target-cap", type=int, default=24)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=10)
    parser.add_argument("--generate-tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--score-ratio-target", type=float, default=0.85)
    parser.add_argument("--loop-target", type=float, default=0.03)
    parser.add_argument("--unk-target", type=float, default=0.003)
    parser.add_argument("--entity-target", type=float, default=0.85)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    metrics = result["metrics"]
    ok = bool(result["success"])
    print("Phase 7: Generation Quality v2")
    print("=" * 88)
    print(f"Score ratio:       {metrics['score_ratio']:.3f}x")
    print(f"Loop rate:         {metrics['loop_rate']:.3%}")
    print(f"UNK rate:          {metrics['unk_rate']:.3%}")
    print(f"Entity consistency:{metrics['entity_consistency']:.2%}")
    print(f"Overall status:    {'PASS' if ok else 'FAIL'}")
    if result["samples"]:
        print("-" * 88)
        print(f"Sample: {result['samples'][0]['generated'][:320]}")
    payload = {"benchmark": "phase7_generation_quality_v2", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
