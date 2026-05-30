"""Phase 6 open-generation benchmark on real text corpora."""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase4.benchmark_language_quality import NGramBaseline
from benchmarks.phase5.benchmark_long_context_real_corpus import read_text, tokenize_splits
from lsl import GenerationController, LongContextMemory


def token_maps(tokenizer):
    if hasattr(tokenizer, "word_to_id"):
        return tokenizer.word_to_id, tokenizer.id_to_word
    return tokenizer.token_to_id, tokenizer.id_to_token


def train_memory(tokens: List[int], vocab_size: int, args: argparse.Namespace) -> LongContextMemory:
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


def generate_ngram(model: NGramBaseline, prompt: List[int], max_new: int) -> List[int]:
    out = list(prompt)
    current = int(out[-1])
    seen_trigrams = set()
    for _ in range(max_new):
        nxt = int(model.predict(current))
        if len(out) >= 2:
            trigram = (out[-2], out[-1], nxt)
            if trigram in seen_trigrams:
                break
            seen_trigrams.add(trigram)
        out.append(nxt)
        current = nxt
    return out


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    train_text, eval_text, corpus_path = read_text(args)
    tokenizer, train_tokens, eval_tokens = tokenize_splits(train_text, eval_text, args)
    train_tokens = train_tokens[: args.max_train_tokens]
    eval_tokens = eval_tokens[: args.max_eval_tokens]
    vocab_size = tokenizer.vocab_size
    to_id, _ = token_maps(tokenizer)
    unk_id = int(to_id.get("<UNK>", 1))
    sentence_ids = [to_id[x] for x in [".", "!", "?"] if x in to_id]

    memory = train_memory(train_tokens, vocab_size, args)
    controller = GenerationController(
        memory=memory,
        vocab_size=vocab_size,
        candidate_limit=args.candidate_limit,
        unk_id=unk_id,
        sentence_end_ids=sentence_ids,
        seed=args.seed,
    )
    ngram = train_ngram(train_tokens, vocab_size)

    rng = np.random.default_rng(args.seed + 600)
    max_start = max(1, len(eval_tokens) - args.prompt_tokens - args.generate_tokens)
    starts = rng.choice(
        np.arange(max_start),
        size=min(args.trials, max_start),
        replace=False,
    )
    controller_metrics = []
    baseline_metrics = []
    samples = []
    for start in starts:
        prompt = eval_tokens[int(start): int(start) + args.prompt_tokens]
        generated = controller.generate(prompt, max_new_tokens=args.generate_tokens)
        baseline = generate_ngram(ngram, prompt, args.generate_tokens)
        cm = GenerationController.generation_metrics(generated, unk_id=unk_id)
        bm = GenerationController.generation_metrics(baseline, unk_id=unk_id)
        controller_metrics.append(cm)
        baseline_metrics.append(bm)
        if len(samples) < 3:
            samples.append(
                {
                    "prompt": tokenizer.decode(prompt),
                    "generated": tokenizer.decode(generated),
                    "baseline": tokenizer.decode(baseline),
                    "metrics": cm,
                }
            )

    def mean_metric(name: str, rows: List[Dict[str, float]]) -> float:
        return float(np.mean([row[name] for row in rows])) if rows else 0.0

    metrics = {
        "coherence": mean_metric("coherence", controller_metrics),
        "baseline_coherence": mean_metric("coherence", baseline_metrics),
        "unk_rate": mean_metric("unk_rate", controller_metrics),
        "loop_rate": mean_metric("loop_rate", controller_metrics),
        "distinct2": mean_metric("distinct2", controller_metrics),
    }
    metrics["coherence_ratio"] = metrics["coherence"] / max(metrics["baseline_coherence"], 1e-9)
    checks = {
        "coherence": metrics["coherence"] >= args.coherence_target,
        "baseline_ratio": metrics["coherence_ratio"] >= args.baseline_ratio_target,
        "unk": metrics["unk_rate"] <= args.unk_target,
        "loop": metrics["loop_rate"] <= args.loop_target,
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "samples": samples,
        "corpus_path": corpus_path,
        "train_tokens": len(train_tokens),
        "eval_tokens": len(eval_tokens),
        "vocab_size": vocab_size,
        "tokenizer": args.tokenizer,
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
    parser.add_argument("--tokenizer-train-chars", type=int, default=120000)
    parser.add_argument("--max-train-chars", type=int, default=240000)
    parser.add_argument("--max-eval-chars", type=int, default=80000)
    parser.add_argument("--subword-max-merges", type=int, default=500)
    parser.add_argument("--subword-min-pair-count", type=int, default=3)
    parser.add_argument("--max-train-tokens", type=int, default=6000)
    parser.add_argument("--max-eval-tokens", type=int, default=1500)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--context-width", type=int, default=6)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--target-cap", type=int, default=24)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=10)
    parser.add_argument("--generate-tokens", type=int, default=48)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--coherence-target", type=float, default=0.75)
    parser.add_argument("--baseline-ratio-target", type=float, default=0.80)
    parser.add_argument("--unk-target", type=float, default=0.005)
    parser.add_argument("--loop-target", type=float, default=0.05)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    metrics = result["metrics"]
    print("Phase 6: Open Generation Public-Corpus Benchmark")
    print("=" * 88)
    print(f"Corpus:             {result['corpus_path']}")
    print(f"Tokenizer/vocab:    {result['tokenizer']} / {result['vocab_size']:,}")
    print(f"Coherence:          {metrics['coherence']:.3f} (target >={args.coherence_target:.2f})")
    print(f"Baseline ratio:     {metrics['coherence_ratio']:.3f}x (target >={args.baseline_ratio_target:.2f}x)")
    print(f"UNK rate:           {metrics['unk_rate']:.3%} (target <={args.unk_target:.2%})")
    print(f"Loop rate:          {metrics['loop_rate']:.3%} (target <={args.loop_target:.1%})")
    print(f"Distinct-2:         {metrics['distinct2']:.3f}")
    print(f"Overall status:     {'PASS' if ok else 'FAIL'}")
    if result["samples"]:
        sample = result["samples"][0]
        print("-" * 88)
        print(f"Prompt:     {sample['prompt'][:180]}")
        print(f"Generated:  {sample['generated'][:320]}")
        print(f"Baseline:   {sample['baseline'][:320]}")

    payload = {"benchmark": "phase6_open_generation_public", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
