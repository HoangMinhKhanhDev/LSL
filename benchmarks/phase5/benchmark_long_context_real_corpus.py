"""Real-corpus long-context memory benchmark.

This benchmark is intentionally stricter than the synthetic mechanism tests:
it evaluates long-context memory on an actual text corpus file, compares
against simple CPU baselines, and reports which competitive-small-model claims
are currently supported.
"""
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

from benchmarks.phase4.baseline_transformer import TinyTransformer
from benchmarks.phase4.baseline_ssm import TinySSM
from benchmarks.phase4.benchmark_language_quality import NGramBaseline, SparseOnlineNGram
from benchmarks.phase5.benchmark_baseline_competition import (
    TrainableTinySSMCPU,
    TrainableTinyTransformerCPU,
)
from benchmarks.phase5.download_wikitext2 import ensure_wikitext2
from benchmarks.phase5.download_tinystories_full import ensure_tinystories_valid
from lsl import LongContextMemory, SimpleSubwordTokenizer, SimpleWordTokenizer


DEFAULT_TINYSTORIES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "phase4", "tinystories_subset.txt")
)


def p50(values: List[float]) -> float:
    return float(np.percentile(values, 50)) if values else 0.0


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / max(float(np.sum(exp)), 1e-12)


def read_text(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.dataset == "wikitext2":
        paths = ensure_wikitext2(args.wikitext_cache_dir, download=not args.no_download)
        with open(paths["train"], "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(paths["test"], "r", encoding="utf-8") as f:
            eval_text = f.read()
        return train_text, eval_text, os.path.abspath(paths["train"])

    if args.dataset == "tinystories_full":
        path = ensure_tinystories_valid(args.tinystories_cache_dir, download=not args.no_download)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        split = max(1, min(len(text) - 1, int(len(text) * args.train_fraction)))
        return text[:split], text[split:], os.path.abspath(path)

    path = args.corpus_path
    if path is None and args.dataset == "tinystories" and os.path.exists(DEFAULT_TINYSTORIES):
        path = DEFAULT_TINYSTORIES
    if path is None:
        raise FileNotFoundError(
            "No real corpus found. Pass --corpus-path, or place TinyStories at "
            f"{DEFAULT_TINYSTORIES}"
        )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    split = max(1, min(len(text) - 1, int(len(text) * args.train_fraction)))
    return text[:split], text[split:], os.path.abspath(path)


def tokenize_splits(train_text: str, eval_text: str, args: argparse.Namespace):
    vocab_text = train_text[: args.tokenizer_train_chars] if args.tokenizer_train_chars > 0 else train_text
    train_source = train_text[: args.max_train_chars] if args.max_train_chars > 0 else train_text
    eval_source = eval_text[: args.max_eval_chars] if args.max_eval_chars > 0 else eval_text
    if args.tokenizer == "subword":
        tokenizer = SimpleSubwordTokenizer(
            vocab_size=args.vocab_size,
            max_merges=args.subword_max_merges,
            min_pair_count=args.subword_min_pair_count,
        )
    else:
        tokenizer = SimpleWordTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(vocab_text)
    train_tokens = tokenizer.encode(train_source)
    eval_tokens = tokenizer.encode(eval_source)
    return tokenizer, train_tokens, eval_tokens


def train_long_context(tokens: List[int], args: argparse.Namespace) -> Tuple[LongContextMemory, float, int]:
    memory = LongContextMemory(
        capacity=args.capacity,
        vocab_size=args.vocab_size,
        context_width=args.context_width,
        candidate_cap=args.candidate_cap,
        store_transition_index=False,
        target_cap=args.target_cap,
        seed=args.seed,
    )
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    for i in range(len(tokens) - 1):
        memory.observe_transition(tokens[i], tokens[i + 1], vocab_size=args.vocab_size)
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return memory, float(elapsed_us / max(1, len(tokens) - 1)), int(peak)


def eval_long_context(memory: LongContextMemory, tokens: List[int], vocab_size: int) -> Dict[str, float]:
    memory.reset_state()
    losses = []
    correct = 0
    times = []
    for i in range(len(tokens) - 1):
        t0 = time.perf_counter_ns()
        prob = memory.target_probability(tokens[i], tokens[i + 1], vocab_size=vocab_size, update_context=True)
        times.append((time.perf_counter_ns() - t0) / 1000.0)
        losses.append(-math.log(max(float(prob), 1e-12)))
        pred = memory.top_next(tokens[i], vocab_size=vocab_size)
        correct += int(pred == tokens[i + 1])
    loss = float(np.mean(losses)) if losses else float("inf")
    return {
        "loss": loss,
        "perplexity": float(math.exp(min(20.0, loss))),
        "accuracy": float(correct / max(1, len(losses))),
        "p50_latency_us": p50(times),
    }


def train_ngram(tokens: List[int], vocab_size: int, sparse: bool = False):
    if sparse:
        model = SparseOnlineNGram(vocab_size, row_cap=max(32, int(vocab_size ** 0.5)))
    else:
        model = NGramBaseline(vocab_size)
    for i in range(len(tokens) - 1):
        model.observe(tokens[i], tokens[i + 1])
    return model


def eval_ngram(model, tokens: List[int]) -> Dict[str, float]:
    losses = []
    correct = 0
    times = []
    for i in range(len(tokens) - 1):
        t0 = time.perf_counter_ns()
        prob = model.predict_prob(tokens[i], tokens[i + 1])
        pred = model.predict(tokens[i])
        times.append((time.perf_counter_ns() - t0) / 1000.0)
        losses.append(-math.log(max(float(prob), 1e-12)))
        correct += int(pred == tokens[i + 1])
    loss = float(np.mean(losses)) if losses else float("inf")
    return {
        "loss": loss,
        "perplexity": float(math.exp(min(20.0, loss))),
        "accuracy": float(correct / max(1, len(losses))),
        "p50_latency_us": p50(times),
    }


def eval_trainable_baselines(
    train_tokens: List[int],
    eval_tokens: List[int],
    args: argparse.Namespace,
) -> Dict[str, float]:
    combined = train_tokens + eval_tokens
    train_cut = len(train_tokens)
    result = {
        "transformer_loss": float("inf"),
        "transformer_perplexity": float("inf"),
        "ssm_loss": float("inf"),
        "ssm_perplexity": float("inf"),
    }
    if not args.train_dense_baselines:
        return result
    transformer = TrainableTinyTransformerCPU(args.vocab_size, args.d_model, args.seed)
    ssm = TrainableTinySSMCPU(args.vocab_size, args.d_model, args.seed + 17)
    transformer.train(combined, train_cut, args.context_train, args.baseline_epochs, args.baseline_lr)
    ssm.train(combined, train_cut, args.baseline_epochs, args.baseline_lr)
    transformer_loss = transformer.eval_loss(combined, train_cut, args.context_train)
    ssm_loss = ssm.eval_loss(combined, train_cut)
    result.update(
        {
            "transformer_loss": float(transformer_loss),
            "transformer_perplexity": float(math.exp(min(20.0, transformer_loss))),
            "ssm_loss": float(ssm_loss),
            "ssm_perplexity": float(math.exp(min(20.0, ssm_loss))),
        }
    )
    return result


def generate_memory(memory: LongContextMemory, prompt: List[int], vocab_size: int, max_new: int) -> List[int]:
    if not prompt:
        return []
    memory.reset_state()
    for token in prompt[:-1]:
        memory.advance_context(token)
    out = list(prompt)
    current = int(prompt[-1])
    seen_trigrams = {tuple(out[i:i + 3]) for i in range(max(0, len(out) - 2))}
    for _ in range(max_new):
        candidates = memory.next_candidates(current, limit=8, update_context=True)
        if not candidates:
            break
        nxt = int(candidates[0])
        for candidate in candidates:
            if len(out) < 2 or tuple([out[-2], out[-1], int(candidate)]) not in seen_trigrams:
                nxt = int(candidate)
                break
        out.append(int(nxt))
        if len(out) >= 3:
            seen_trigrams.add(tuple(out[-3:]))
        current = int(nxt)
    return out


def generate_ngram(model, prompt: List[int], max_new: int) -> List[int]:
    if not prompt:
        return []
    out = list(prompt)
    current = int(prompt[-1])
    for _ in range(max_new):
        nxt = int(model.predict(current))
        out.append(nxt)
        current = nxt
    return out


def generation_score(tokens: List[int], unk_id: int = 1) -> float:
    if len(tokens) < 4:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))
    loop_rate = 1.0 - len(set(trigrams)) / max(1, len(trigrams))
    unk_rate = sum(1 for token in tokens if int(token) == unk_id) / max(1, len(tokens))
    length_score = min(1.0, len(tokens) / 48.0)
    return float(
        0.40 * distinct2
        + 0.30 * (1.0 - loop_rate)
        + 0.20 * (1.0 - unk_rate)
        + 0.10 * length_score
    )


def eval_generation(
    memory: LongContextMemory,
    baseline,
    eval_tokens: List[int],
    tokenizer,
    args: argparse.Namespace,
) -> Dict[str, float]:
    rng = np.random.default_rng(args.seed + 900)
    starts = rng.choice(
        np.arange(0, max(1, len(eval_tokens) - args.prompt_tokens - args.generate_tokens)),
        size=min(args.generation_trials, max(1, len(eval_tokens) // 128)),
        replace=False,
    )
    memory_scores = []
    baseline_scores = []
    samples = []
    for start in starts:
        prompt = eval_tokens[int(start): int(start) + args.prompt_tokens]
        mem_out = generate_memory(memory, prompt, args.vocab_size, args.generate_tokens)
        base_out = generate_ngram(baseline, prompt, args.generate_tokens)
        memory_scores.append(generation_score(mem_out))
        baseline_scores.append(generation_score(base_out))
        if len(samples) < 2:
            samples.append(
                {
                    "prompt": tokenizer.decode(prompt),
                    "memory": tokenizer.decode(mem_out),
                    "baseline": tokenizer.decode(base_out),
                }
            )
    memory_score = float(np.mean(memory_scores)) if memory_scores else 0.0
    baseline_score = float(np.mean(baseline_scores)) if baseline_scores else 0.0
    return {
        "memory_score": memory_score,
        "baseline_score": baseline_score,
        "score_ratio": float(memory_score / max(baseline_score, 1e-9)),
        "samples": samples,
    }


def measure_dense_latency(args: argparse.Namespace, tokens: List[int]) -> Dict[str, float]:
    transformer = TinyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=4,
        d_ff=args.d_model * 4,
        n_layers=2,
        max_seq_len=max(args.context_train, args.latency_context),
        seed=args.seed,
    )
    ssm = TinySSM(args.vocab_size, args.d_model, args.seed)
    windows = []
    transformer_times = []
    ssm_times = []
    usable = max(1, min(args.latency_iterations, len(tokens) - args.latency_context - 1))
    for i in range(usable):
        windows.append(tokens[i:i + args.latency_context])
    for window in windows:
        t0 = time.perf_counter_ns()
        transformer.forward(window)
        transformer_times.append((time.perf_counter_ns() - t0) / 1000.0)
    for token in tokens[:usable]:
        t0 = time.perf_counter_ns()
        ssm.forward(int(token))
        ssm_times.append((time.perf_counter_ns() - t0) / 1000.0)

    transformer_param_bytes = int(transformer.get_num_params() * 8)
    attention_activation_bytes = int(2 * 4 * args.latency_context * args.latency_context * 8)
    ssm_param_bytes = int(ssm.get_num_params() * 4)
    return {
        "transformer_p50_us": p50(transformer_times),
        "ssm_p50_us": p50(ssm_times),
        "transformer_ram_bytes": float(transformer_param_bytes + attention_activation_bytes),
        "ssm_ram_bytes": float(ssm_param_bytes),
    }


def eval_adaptation_retention(
    memory: LongContextMemory,
    domain_a: List[int],
    domain_b: List[int],
    args: argparse.Namespace,
) -> Dict[str, float]:
    before = eval_long_context(memory, domain_a, args.vocab_size)["loss"]
    t0 = time.perf_counter_ns()
    for i in range(len(domain_b) - 1):
        memory.observe_transition(domain_b[i], domain_b[i + 1], vocab_size=args.vocab_size)
    adapt_us = (time.perf_counter_ns() - t0) / 1000.0
    after = eval_long_context(memory, domain_a, args.vocab_size)["loss"]
    b_loss = eval_long_context(memory, domain_b, args.vocab_size)["loss"]

    transformer = TrainableTinyTransformerCPU(args.vocab_size, args.d_model, args.seed + 123)
    combined = domain_a + domain_b
    train_cut = len(combined)
    t0 = time.perf_counter_ns()
    transformer.train(combined, train_cut, args.context_train, 1, args.baseline_lr)
    one_epoch_us = (time.perf_counter_ns() - t0) / 1000.0
    retrain_us = one_epoch_us * max(1, args.retrain_epochs)
    return {
        "retention": float(before / max(after, 1e-9)),
        "domain_a_loss_before": float(before),
        "domain_a_loss_after": float(after),
        "domain_b_loss_after": float(b_loss),
        "online_adapt_us": float(adapt_us),
        "retrain_proxy_us": float(retrain_us),
        "adaptation_speedup": float(retrain_us / max(adapt_us, 1e-9)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "tinystories_full", "wikitext2", "custom"], default="tinystories")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--wikitext-cache-dir", type=str, default=None)
    parser.add_argument("--tinystories-cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--tokenizer", choices=["word", "subword"], default="word")
    parser.add_argument("--tokenizer-train-chars", type=int, default=250000)
    parser.add_argument("--max-train-chars", type=int, default=800000)
    parser.add_argument("--max-eval-chars", type=int, default=250000)
    parser.add_argument("--subword-max-merges", type=int, default=1000)
    parser.add_argument("--subword-min-pair-count", type=int, default=3)
    parser.add_argument("--max-train-tokens", type=int, default=12000)
    parser.add_argument("--max-eval-tokens", type=int, default=3000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--context-width", type=int, default=6)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--target-cap", type=int, default=24)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--context-train", type=int, default=32)
    parser.add_argument("--baseline-epochs", type=int, default=1)
    parser.add_argument("--baseline-lr", type=float, default=0.15)
    parser.add_argument("--train-dense-baselines", action="store_true")
    parser.add_argument("--latency-context", type=int, default=256)
    parser.add_argument("--latency-iterations", type=int, default=12)
    parser.add_argument("--adaptation-tokens", type=int, default=1024)
    parser.add_argument("--retrain-epochs", type=int, default=50)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--generate-tokens", type=int, default=48)
    parser.add_argument("--generation-trials", type=int, default=8)
    parser.add_argument("--quality-ratio-target", type=float, default=1.15)
    parser.add_argument("--generation-ratio-target", type=float, default=0.80)
    parser.add_argument("--latency-speedup-target", type=float, default=20.0)
    parser.add_argument("--ram-speedup-target", type=float, default=5.0)
    parser.add_argument("--adaptation-speedup-target", type=float, default=50.0)
    parser.add_argument("--retention-target", type=float, default=0.95)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_text, eval_text, corpus_path = read_text(args)
    tokenizer, train_tokens, eval_tokens = tokenize_splits(train_text, eval_text, args)
    train_tokens = train_tokens[:args.max_train_tokens]
    eval_tokens = eval_tokens[:args.max_eval_tokens]
    args.vocab_size = tokenizer.vocab_size
    unk_id = getattr(tokenizer, "word_to_id", getattr(tokenizer, "token_to_id", {})).get("<UNK>", 1)
    eval_unk_rate = sum(1 for token in eval_tokens if int(token) == int(unk_id)) / max(1, len(eval_tokens))
    if len(train_tokens) < 64 or len(eval_tokens) < 64:
        raise ValueError("Corpus split is too small for real-corpus evaluation")

    memory, observe_us, memory_peak_bytes = train_long_context(train_tokens, args)
    memory_metrics = eval_long_context(memory, eval_tokens, args.vocab_size)

    ngram = train_ngram(train_tokens, args.vocab_size, sparse=False)
    sparse_ngram = train_ngram(train_tokens, args.vocab_size, sparse=True)
    ngram_metrics = eval_ngram(ngram, eval_tokens)
    sparse_ngram_metrics = eval_ngram(sparse_ngram, eval_tokens)
    dense_quality = eval_trainable_baselines(train_tokens, eval_tokens, args)

    baseline_losses = [
        ngram_metrics["loss"],
        sparse_ngram_metrics["loss"],
        dense_quality["transformer_loss"],
        dense_quality["ssm_loss"],
    ]
    best_baseline_loss = float(min(baseline_losses))
    quality_ratio = float(memory_metrics["loss"] / max(best_baseline_loss, 1e-9))

    generation = eval_generation(memory, ngram, eval_tokens, tokenizer, args)
    dense_latency = measure_dense_latency(args, train_tokens + eval_tokens)
    latency_speedup = dense_latency["transformer_p50_us"] / max(memory_metrics["p50_latency_us"], 1e-9)
    ram_speedup = dense_latency["transformer_ram_bytes"] / max(float(memory_peak_bytes), 1.0)

    domain_a = train_tokens[: min(args.adaptation_tokens, len(train_tokens))]
    domain_b = eval_tokens[: min(args.adaptation_tokens, len(eval_tokens))]
    adaptation = eval_adaptation_retention(memory, domain_a, domain_b, args)

    checks = {
        "quality": quality_ratio <= args.quality_ratio_target,
        "generation": generation["score_ratio"] >= args.generation_ratio_target,
        "latency": latency_speedup >= args.latency_speedup_target,
        "ram": ram_speedup >= args.ram_speedup_target,
        "adaptation": adaptation["adaptation_speedup"] >= args.adaptation_speedup_target,
        "retention": adaptation["retention"] >= args.retention_target,
    }
    ok = all(checks.values())

    print("Phase 5: Real-Corpus Long-Context Memory")
    print("=" * 88)
    print(f"Corpus:                  {corpus_path}")
    print(f"Train/eval tokens:       {len(train_tokens):,} / {len(eval_tokens):,}")
    print(f"Tokenizer/vocab:         {args.tokenizer} / {args.vocab_size:,}")
    print(f"Eval UNK rate:           {eval_unk_rate:.2%}")
    print("-" * 88)
    print(f"Long-context loss/ppl:   {memory_metrics['loss']:.4f} / {memory_metrics['perplexity']:.2f}")
    print(f"NGram loss/ppl:          {ngram_metrics['loss']:.4f} / {ngram_metrics['perplexity']:.2f}")
    print(f"Sparse NGram loss/ppl:   {sparse_ngram_metrics['loss']:.4f} / {sparse_ngram_metrics['perplexity']:.2f}")
    if args.train_dense_baselines:
        print(f"Transformer loss/ppl:    {dense_quality['transformer_loss']:.4f} / {dense_quality['transformer_perplexity']:.2f}")
        print(f"SSM loss/ppl:            {dense_quality['ssm_loss']:.4f} / {dense_quality['ssm_perplexity']:.2f}")
    print(f"Quality ratio:           {quality_ratio:.3f}x (target <={args.quality_ratio_target:.2f}x)")
    print(f"Generation score ratio:  {generation['score_ratio']:.3f}x (target >={args.generation_ratio_target:.2f}x)")
    print(f"Latency speedup:         {latency_speedup:.2f}x (target >={args.latency_speedup_target:.1f}x)")
    print(f"RAM speedup proxy:       {ram_speedup:.2f}x (target >={args.ram_speedup_target:.1f}x)")
    print(f"Observe latency:         {observe_us:.2f} us/token")
    print(f"Adaptation speedup:      {adaptation['adaptation_speedup']:.2f}x (target >={args.adaptation_speedup_target:.1f}x)")
    print(f"Retention:               {adaptation['retention']:.2%} (target >={args.retention_target:.0%})")
    print(f"Overall status:          {'PASS' if ok else 'FAIL'}")
    if generation["samples"]:
        print("-" * 88)
        print("Generation sample:")
        sample = generation["samples"][0]
        print(f"Prompt:   {sample['prompt'][:180]}")
        print(f"Memory:   {sample['memory'][:260]}")
        print(f"Baseline: {sample['baseline'][:260]}")

    payload = {
        "benchmark": "long_context_real_corpus",
        "success": bool(ok),
        "checks": checks,
        "corpus_path": corpus_path,
        "train_tokens": len(train_tokens),
        "eval_tokens": len(eval_tokens),
        "vocab_size": args.vocab_size,
        "tokenizer": args.tokenizer,
        "eval_unk_rate": float(eval_unk_rate),
        "long_context": memory_metrics,
        "ngram": ngram_metrics,
        "sparse_ngram": sparse_ngram_metrics,
        "dense_quality": dense_quality,
        "quality_ratio": quality_ratio,
        "generation": generation,
        "dense_latency": dense_latency,
        "latency_speedup": float(latency_speedup),
        "memory_peak_bytes": float(memory_peak_bytes),
        "ram_speedup": float(ram_speedup),
        "observe_us_per_token": float(observe_us),
        "adaptation": adaptation,
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
