"""Compare one unified LSLCoreModel against a trainable CPU NumPy Transformer.

This benchmark is intentionally descriptive by default: it reports quality,
latency, memory, generation, and online-adaptation metrics without pretending
that LSL already beats Transformer on language quality. Pass ``--claim`` to
turn the comparison into a thresholded pass/fail gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase5.benchmark_baseline_competition import TrainableTinyTransformerCPU, softmax
from lsl import GenerationController, LSLCoreModel


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TINYSTORIES_FULL = os.path.join(ROOT, "benchmarks", "data", "tinystories", "TinyStoriesV2-GPT4-valid.txt")
TINYSTORIES_SUBSET = os.path.join(ROOT, "benchmarks", "phase4", "tinystories_subset.txt")
WIKITEXT_TRAIN = os.path.join(ROOT, "benchmarks", "data", "wikitext-2-raw-v1", "wiki.train.raw.txt")
WIKITEXT_TEST = os.path.join(ROOT, "benchmarks", "data", "wikitext-2-raw-v1", "wiki.test.raw.txt")


def p50(values: List[float]) -> float:
    return float(np.percentile(values, 50)) if values else 0.0


def parse_lengths(raw: str) -> List[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def read_dataset(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.dataset == "wikitext2":
        if not os.path.exists(WIKITEXT_TRAIN) or not os.path.exists(WIKITEXT_TEST):
            raise FileNotFoundError("WikiText-2 files are missing under benchmarks/data/wikitext-2-raw-v1")
        with open(WIKITEXT_TRAIN, "r", encoding="utf-8") as f:
            train = f.read(args.max_train_chars)
        with open(WIKITEXT_TEST, "r", encoding="utf-8") as f:
            test = f.read(args.max_eval_chars)
        return train, test, WIKITEXT_TRAIN

    if args.dataset == "custom":
        if not args.corpus_path:
            raise FileNotFoundError("--corpus-path is required for custom dataset")
        with open(args.corpus_path, "r", encoding="utf-8") as f:
            text = f.read(args.max_train_chars + args.max_eval_chars)
        split = max(1, min(len(text) - 1, int(len(text) * args.train_fraction)))
        return text[:split], text[split:], os.path.abspath(args.corpus_path)

    path = TINYSTORIES_FULL if os.path.exists(TINYSTORIES_FULL) else TINYSTORIES_SUBSET
    if not os.path.exists(path):
        raise FileNotFoundError("TinyStories corpus is missing")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(args.max_train_chars + args.max_eval_chars)
    split = max(1, min(len(text) - 1, int(len(text) * args.train_fraction)))
    return text[:split], text[split:], path


def transformer_eval(
    model: TrainableTinyTransformerCPU,
    tokens: List[int],
    train_cut: int,
    context: int,
    max_eval: int,
) -> Dict[str, float]:
    losses = []
    correct = 0
    times = []
    start = max(train_cut, context)
    stop = min(len(tokens) - 1, start + int(max_eval))
    for i in range(start, stop):
        window = tokens[max(0, i - context + 1):i + 1]
        target = int(tokens[i + 1]) % model.vocab_size
        t0 = time.perf_counter_ns()
        logits = model.logits(window)
        probs = softmax(logits)
        times.append((time.perf_counter_ns() - t0) / 1000.0)
        losses.append(-math.log(max(float(probs[target]), 1e-12)))
        correct += int(np.argmax(probs) == target)
    loss = float(np.mean(losses)) if losses else float("inf")
    return {
        "loss": loss,
        "perplexity": float(math.exp(min(20.0, loss))),
        "accuracy": float(correct / max(1, len(losses))),
        "p50_latency_us": p50(times),
        "tokens_per_second": float(1_000_000.0 / max(p50(times), 1e-12)),
        "tokens": float(len(losses) + 1),
    }


def transformer_generate(
    model: TrainableTinyTransformerCPU,
    prompt: List[int],
    context: int,
    max_new: int,
) -> List[int]:
    out = [int(token) % model.vocab_size for token in prompt]
    if not out:
        return []
    seen_trigrams = {tuple(out[i:i + 3]) for i in range(max(0, len(out) - 2))}
    for _ in range(int(max_new)):
        window = out[-int(context):]
        probs = softmax(model.logits(window))
        order = np.argsort(-probs)[:16]
        nxt = int(order[0])
        for candidate in order:
            candidate = int(candidate)
            if len(out) < 2 or tuple([out[-2], out[-1], candidate]) not in seen_trigrams:
                nxt = candidate
                break
        out.append(nxt)
        if len(out) >= 3:
            tri = tuple(out[-3:])
            if tri in seen_trigrams:
                out.pop()
                break
            seen_trigrams.add(tri)
    return out


def transformer_next_token_accuracy(
    model: TrainableTinyTransformerCPU,
    prompt: List[int],
    target: int,
    context: int,
) -> bool:
    logits = model.logits(prompt[-int(context):])
    return int(np.argmax(logits)) == int(target) % model.vocab_size


def model_size_bytes(obj) -> int:
    seen = set()

    def walk(value) -> int:
        ident = id(value)
        if ident in seen:
            return 0
        seen.add(ident)
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(walk(k) + walk(v) for k, v in value.items())
        if isinstance(value, (list, tuple, set)):
            return sum(walk(v) for v in value)
        if hasattr(value, "__dict__"):
            return walk(vars(value))
        return sys.getsizeof(value)

    return int(walk(obj))


def build_fact_items(count: int) -> List[Tuple[str, str, str]]:
    subjects = [
        "aurora", "basil", "cedar", "dahlia", "ember", "fable", "ginger", "harbor",
        "iris", "juniper", "kiwi", "laurel", "maple", "nova", "opal", "prairie",
    ]
    objects = [
        "amber", "blue", "coral", "denim", "emerald", "frost", "gold", "hazel",
        "ivory", "jade", "khaki", "linen", "mint", "navy", "olive", "pearl",
    ]
    return [(subjects[i % len(subjects)], "links", objects[i % len(objects)]) for i in range(int(count))]


def fact_block(facts: List[Tuple[str, str, str]]) -> str:
    return " ".join(f"{subject} {relation} {obj}." for subject, relation, obj in facts)


def eval_fact_recall(
    lsl: LSLCoreModel,
    transformer: TrainableTinyTransformerCPU,
    facts: List[Tuple[str, str, str]],
    context: int,
) -> Dict[str, float]:
    lsl_correct = 0
    tf_correct = 0
    for subject, relation, obj in facts:
        lsl.agent.observe_event(subject, relation, obj, episode_id=1, evidence_id=1)
        lsl_correct += int(lsl.answer(f"What does {subject} {relation}?") == obj)
        prompt = lsl.encode(f"{subject} {relation}")
        target = lsl.encode(obj)
        if prompt and target:
            tf_correct += int(transformer_next_token_accuracy(transformer, prompt, target[0], context))
    total = max(1, len(facts))
    return {
        "lsl_fact_recall": float(lsl_correct / total),
        "transformer_next_token_fact_accuracy": float(tf_correct / total),
        "items": float(len(facts)),
    }


def context_latency_profile(
    lsl: LSLCoreModel,
    transformer: TrainableTinyTransformerCPU,
    eval_tokens: List[int],
    lengths: List[int],
    iterations: int,
) -> Dict[str, object]:
    rows = []
    sample = eval_tokens[: max(2, int(iterations) + 1)]
    for length in lengths:
        tf_times = []
        usable = min(int(iterations), max(1, len(eval_tokens) - int(length) - 1))
        for i in range(usable):
            window = eval_tokens[i:i + int(length)]
            t0 = time.perf_counter_ns()
            transformer.logits(window)
            tf_times.append((time.perf_counter_ns() - t0) / 1000.0)
        lsl_metrics = lsl.evaluate_tokens(sample, update_context=True)
        rows.append({
            "context": int(length),
            "lsl_p50_us": float(lsl_metrics["p50_latency_us"]),
            "transformer_p50_us": p50(tf_times),
            "speedup": float(p50(tf_times) / max(lsl_metrics["p50_latency_us"], 1e-12)),
        })
    return {"rows": rows}


def run(args: argparse.Namespace) -> Dict[str, object]:
    train_text, eval_text, corpus_path = read_dataset(args)
    facts = build_fact_items(args.fact_items)
    facts_text = fact_block(facts)
    if facts_text:
        train_text = train_text + "\n" + facts_text
    lsl = LSLCoreModel(vocab_size=args.vocab_size, seed=args.seed, candidate_cap=args.candidate_cap)
    lsl.build_tokenizer(train_text[: args.tokenizer_train_chars])
    train_tokens = lsl.encode(train_text)[: args.max_train_tokens]
    eval_tokens = lsl.encode(eval_text)[: args.max_eval_tokens]
    if len(train_tokens) < args.context + 4 or len(eval_tokens) < args.context + 4:
        raise ValueError("Not enough tokens for the requested context/eval window")

    tracemalloc.start()
    t0 = time.perf_counter()
    lsl_train = lsl.fit_tokens(train_tokens)
    lsl_train_seconds = time.perf_counter() - t0
    _, lsl_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lsl_eval = lsl.evaluate_tokens(eval_tokens[: args.eval_tokens])
    lsl_size = model_size_bytes(lsl)

    combined = train_tokens + eval_tokens[: args.eval_tokens]
    train_cut = len(train_tokens)
    transformer = TrainableTinyTransformerCPU(lsl.vocab_size, args.d_model, args.seed)
    t0 = time.perf_counter()
    transformer.train(combined, train_cut, args.context, args.transformer_epochs, args.lr)
    transformer_train_seconds = time.perf_counter() - t0
    tf_eval = transformer_eval(transformer, combined, train_cut, args.context, args.eval_tokens)
    tf_size = model_size_bytes(transformer)
    fact_recall = eval_fact_recall(lsl, transformer, facts, args.context)
    latency_profile = context_latency_profile(
        lsl,
        transformer,
        eval_tokens,
        parse_lengths(args.context_lengths),
        args.context_latency_iterations,
    )

    prompt = eval_tokens[: args.prompt_tokens]
    lsl_text = lsl.generate(lsl.decode(prompt), max_new_tokens=args.generate_tokens)
    lsl_gen_tokens = lsl.encode(lsl_text)
    tf_gen_tokens = transformer_generate(transformer, prompt, args.context, args.generate_tokens)
    unk_id = getattr(lsl.tokenizer, "word_to_id", getattr(lsl.tokenizer, "token_to_id", {})).get("<UNK>", 1)
    lsl_gen_metrics = GenerationController.generation_metrics(lsl_gen_tokens, unk_id=unk_id)
    tf_gen_metrics = GenerationController.generation_metrics(tf_gen_tokens, unk_id=unk_id)

    lsl_answer_before = lsl.answer("What does alpha link?")
    lsl.agent.observe_event("alpha", "link", "omega", episode_id=1, evidence_id=1)
    lsl_answer_after = lsl.answer("What does alpha link?")
    online_adaptation_us = 0.0
    t0 = time.perf_counter_ns()
    lsl.agent.observe_event("beta", "link", "sigma", episode_id=2, evidence_id=2)
    lsl.answer("What does beta link?")
    online_adaptation_us = (time.perf_counter_ns() - t0) / 1000.0

    metrics = {
        "dataset": args.dataset,
        "corpus_path": corpus_path,
        "vocab_size": lsl.vocab_size,
        "train_tokens": len(train_tokens),
        "eval_tokens": min(len(eval_tokens), args.eval_tokens),
        "context": args.context,
        "lsl": {
            **lsl_eval,
            "train_seconds": float(lsl_train_seconds),
            "train_us_per_token": float(lsl_train["us_per_token"]),
            "train_tokens_per_second": float(len(train_tokens) / max(lsl_train_seconds, 1e-12)),
            "inference_tokens_per_second": float(1_000_000.0 / max(lsl_eval["p50_latency_us"], 1e-12)),
            "model_size_bytes": float(lsl_size),
            "model_size_mb": float(lsl_size / (1024.0 * 1024.0)),
            "peak_train_bytes": float(lsl_peak),
            "peak_train_mb": float(lsl_peak / (1024.0 * 1024.0)),
        },
        "transformer": {
            **tf_eval,
            "train_seconds": float(transformer_train_seconds),
            "train_us_per_token": 1_000_000.0 * transformer_train_seconds / max(1, train_cut),
            "train_tokens_per_second": float(train_cut / max(transformer_train_seconds, 1e-12)),
            "inference_tokens_per_second": float(1_000_000.0 / max(tf_eval["p50_latency_us"], 1e-12)),
            "model_size_bytes": float(tf_size),
            "model_size_mb": float(tf_size / (1024.0 * 1024.0)),
            "epochs": int(args.transformer_epochs),
        },
        "comparison": {
            "loss_ratio_lsl_over_transformer": float(lsl_eval["loss"] / max(tf_eval["loss"], 1e-12)),
            "latency_speedup_transformer_over_lsl": float(tf_eval["p50_latency_us"] / max(lsl_eval["p50_latency_us"], 1e-12)),
            "train_speedup_transformer_over_lsl": float(
                (1_000_000.0 * transformer_train_seconds / max(1, train_cut)) / max(lsl_train["us_per_token"], 1e-12)
            ),
            "size_ratio_transformer_over_lsl": float(tf_size / max(1, lsl_size)),
            "latency_energy_proxy_saving": float(1.0 - lsl_eval["p50_latency_us"] / max(tf_eval["p50_latency_us"], 1e-12)),
            "ops_energy_proxy_saving": float(
                1.0 - min(1.0, args.candidate_cap / max(1.0, float(args.context * args.context * args.d_model)))
            ),
        },
        "context_latency": latency_profile,
        "generation": {
            "prompt": lsl.decode(prompt),
            "lsl_text": lsl_text,
            "transformer_text": lsl.decode(tf_gen_tokens),
            "lsl_metrics": lsl_gen_metrics,
            "transformer_metrics": tf_gen_metrics,
        },
        "online_adaptation": {
            "alpha_before": lsl_answer_before,
            "alpha_after": lsl_answer_after,
            "beta_update_and_query_us": float(online_adaptation_us),
            "works": bool(lsl_answer_after == "omega"),
        },
        "fact_recall": fact_recall,
    }

    checks = {
        "quality_ratio": metrics["comparison"]["loss_ratio_lsl_over_transformer"] <= args.quality_ratio_target,
        "latency": metrics["comparison"]["latency_speedup_transformer_over_lsl"] >= args.latency_speedup_target,
        "generation_loop": lsl_gen_metrics["loop_rate"] <= args.loop_rate_target,
        "fact_recall": fact_recall["lsl_fact_recall"] >= args.fact_recall_target,
        "online_adaptation": metrics["online_adaptation"]["works"],
    }
    success = all(checks.values()) if args.claim else True
    return {"benchmark": "lsl_core_vs_transformer", "success": bool(success), "claim": bool(args.claim), "checks": checks, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "wikitext2", "custom"], default="tinystories")
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.tokens is not None:
        args.max_train_tokens = int(args.tokens)
        args.max_train_chars = max(int(args.max_train_chars), int(args.tokens) * 12)
        args.max_eval_chars = max(int(args.max_eval_chars), int(args.eval_tokens) * 12)
    result = run(args)
    metrics = result["metrics"]
    comp = metrics["comparison"]
    print("LSLCoreModel vs Trainable CPU Transformer")
    print("=" * 88)
    print(f"Dataset:                 {metrics['dataset']} ({metrics['corpus_path']})")
    print(f"Train/eval tokens:       {metrics['train_tokens']:,} / {metrics['eval_tokens']:,}")
    print(f"Vocab/context:           {metrics['vocab_size']:,} / {metrics['context']}")
    print("-" * 88)
    print(f"LSL loss/ppl/acc:        {metrics['lsl']['loss']:.4f} / {metrics['lsl']['perplexity']:.2f} / {metrics['lsl']['accuracy']:.2%}")
    print(f"TF loss/ppl/acc:         {metrics['transformer']['loss']:.4f} / {metrics['transformer']['perplexity']:.2f} / {metrics['transformer']['accuracy']:.2%}")
    print(f"Loss ratio LSL/TF:       {comp['loss_ratio_lsl_over_transformer']:.3f}x")
    print(f"Latency speedup TF/LSL:  {comp['latency_speedup_transformer_over_lsl']:.2f}x")
    print(f"Train speedup TF/LSL:    {comp['train_speedup_transformer_over_lsl']:.2f}x")
    print(f"Size ratio TF/LSL:       {comp['size_ratio_transformer_over_lsl']:.3f}x")
    print(f"Train tok/s LSL / TF:    {metrics['lsl']['train_tokens_per_second']:.1f} / {metrics['transformer']['train_tokens_per_second']:.1f}")
    print(f"Infer tok/s LSL / TF:    {metrics['lsl']['inference_tokens_per_second']:.1f} / {metrics['transformer']['inference_tokens_per_second']:.1f}")
    print(f"Memory MB LSL / TF:      {metrics['lsl']['model_size_mb']:.2f} / {metrics['transformer']['model_size_mb']:.2f}")
    print(f"Energy proxy saving:     latency={comp['latency_energy_proxy_saving']:.2%}, ops={comp['ops_energy_proxy_saving']:.2%}")
    print(f"LSL generation coherence:{metrics['generation']['lsl_metrics']['coherence']:.3f}")
    print(f"LSL loop/UNK:            {metrics['generation']['lsl_metrics']['loop_rate']:.2%} / {metrics['generation']['lsl_metrics']['unk_rate']:.2%}")
    print(f"Fact recall LSL / TF:    {metrics['fact_recall']['lsl_fact_recall']:.2%} / {metrics['fact_recall']['transformer_next_token_fact_accuracy']:.2%}")
    print(f"Online adaptation:       {metrics['online_adaptation']['works']} ({metrics['online_adaptation']['beta_update_and_query_us']:.2f} us)")
    print(f"Claim gate:              {'ON' if result['claim'] else 'OFF'}")
    print(f"Overall status:          {'PASS' if result['success'] else 'FAIL'}")
    print("-" * 88)
    print(f"Prompt: {metrics['generation']['prompt'][:160]}")
    print(f"LSL:    {metrics['generation']['lsl_text'][:260]}")
    print(f"TF:     {metrics['generation']['transformer_text'][:260]}")
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
