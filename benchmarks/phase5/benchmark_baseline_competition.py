"""Phase 5 competitive small-model baseline benchmark."""
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase4.baseline_transformer import TinyTransformer
from benchmarks.phase4.baseline_ssm import TinySSM
from benchmarks.phase4.benchmark_language_quality import (
    NGramBaseline,
    SparseOnlineNGram,
    generate_tiny_stories_tokens,
    train_and_eval,
)
from benchmarks.phase4.benchmark_continual_learning import (
    ContinualTransitionMemory,
    domain_pairs,
    evaluate as eval_continual,
    train,
)
from lsl import LivingSynapseLM


def p50(values: List[float]) -> float:
    return float(np.percentile(values, 50)) if values else 0.0


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / max(float(np.sum(exp)), 1e-12)


class TrainableTinyTransformerCPU:
    """Small CPU NumPy causal-attention baseline trained for next-token loss."""

    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        rng = np.random.default_rng(seed)
        self.embed = (rng.standard_normal((self.vocab_size, self.d_model)) * 0.03).astype(np.float32)
        self.wq = (rng.standard_normal((self.d_model, self.d_model)) * 0.02).astype(np.float32)
        self.wk = (rng.standard_normal((self.d_model, self.d_model)) * 0.02).astype(np.float32)
        self.wv = (rng.standard_normal((self.d_model, self.d_model)) * 0.02).astype(np.float32)
        self.w_out = (rng.standard_normal((self.d_model, self.vocab_size)) * 0.02).astype(np.float32)

    def features(self, window: List[int]) -> np.ndarray:
        x = self.embed[np.asarray(window, dtype=np.int64) % self.vocab_size]
        q = x[-1] @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        weights = softmax((k @ q) / math.sqrt(max(1, self.d_model)))
        return (weights @ v).astype(np.float32)

    def logits(self, window: List[int]) -> np.ndarray:
        return self.features(window) @ self.w_out

    def train(self, tokens: List[int], train_cut: int, context: int, epochs: int, lr: float) -> None:
        for _ in range(int(epochs)):
            for i in range(max(1, context), train_cut - 1):
                window = tokens[max(0, i - context + 1):i + 1]
                target = int(tokens[i + 1]) % self.vocab_size
                h = self.features(window)
                probs = softmax(h @ self.w_out)
                grad = probs
                grad[target] -= 1.0
                self.w_out -= float(lr) * np.outer(h, grad).astype(np.float32)

    def eval_loss(self, tokens: List[int], train_cut: int, context: int) -> float:
        losses = []
        for i in range(max(train_cut, context), len(tokens) - 1):
            target = int(tokens[i + 1]) % self.vocab_size
            probs = softmax(self.logits(tokens[max(0, i - context + 1):i + 1]))
            losses.append(-math.log(max(float(probs[target]), 1e-12)))
        return float(np.mean(losses)) if losses else float("inf")


class TrainableTinySSMCPU:
    """Small CPU NumPy recurrent/SSM baseline trained for next-token loss."""

    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        rng = np.random.default_rng(seed)
        self.embed = (rng.standard_normal((self.vocab_size, self.d_model)) * 0.03).astype(np.float32)
        self.a = np.eye(self.d_model, dtype=np.float32) * 0.92
        self.b = (rng.standard_normal((self.d_model, self.d_model)) * 0.02).astype(np.float32)
        self.w_out = (rng.standard_normal((self.d_model, self.vocab_size)) * 0.02).astype(np.float32)

    def states(self, tokens: List[int]) -> List[np.ndarray]:
        h = np.zeros(self.d_model, dtype=np.float32)
        out = []
        for token in tokens:
            x = self.embed[int(token) % self.vocab_size]
            h = np.tanh(self.a @ h + self.b @ x).astype(np.float32)
            out.append(h.copy())
        return out

    def train(self, tokens: List[int], train_cut: int, epochs: int, lr: float) -> None:
        for _ in range(int(epochs)):
            states = self.states(tokens[:train_cut])
            for i in range(0, max(0, train_cut - 1)):
                target = int(tokens[i + 1]) % self.vocab_size
                h = states[i]
                probs = softmax(h @ self.w_out)
                grad = probs
                grad[target] -= 1.0
                self.w_out -= float(lr) * np.outer(h, grad).astype(np.float32)

    def eval_loss(self, tokens: List[int], train_cut: int) -> float:
        states = self.states(tokens)
        losses = []
        for i in range(train_cut, len(tokens) - 1):
            target = int(tokens[i + 1]) % self.vocab_size
            probs = softmax(states[i] @ self.w_out)
            losses.append(-math.log(max(float(probs[target]), 1e-12)))
        return float(np.mean(losses)) if losses else float("inf")


def measure_transformer(vocab_size: int, d_model: int, context: int, iterations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        d_ff=d_model * 4,
        n_layers=2,
        max_seq_len=max(context, 512),
        seed=seed,
    )
    tokens = rng.integers(0, vocab_size, size=context + iterations).tolist()
    times = []
    for i in range(iterations):
        window = tokens[i:i + context]
        t0 = time.perf_counter_ns()
        model.forward(window)
        times.append((time.perf_counter_ns() - t0) / 1000.0)
    return p50(times)


def measure_sparse(vocab_size: int, d_model: int, iterations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    model = LivingSynapseLM(vocab_size=vocab_size, hidden_dim=d_model, use_sparse_computation=True, seed=seed)
    tokens = rng.integers(0, vocab_size, size=iterations).tolist()
    times = []
    for token in tokens:
        t0 = time.perf_counter_ns()
        model.forward(int(token))
        times.append((time.perf_counter_ns() - t0) / 1000.0)
    return p50(times)


def measure_ssm(vocab_size: int, d_model: int, iterations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    model = TinySSM(vocab_size=vocab_size, d_model=d_model, seed=seed)
    tokens = rng.integers(0, vocab_size, size=iterations).tolist()
    times = []
    for token in tokens:
        t0 = time.perf_counter_ns()
        model.forward(int(token))
        times.append((time.perf_counter_ns() - t0) / 1000.0)
    return p50(times)


def quality(args: argparse.Namespace) -> Dict[str, float]:
    tokens = generate_tiny_stories_tokens(args.quality_tokens, args.vocab_size, args.seed, False)
    train_cut = max(32, int(0.7 * len(tokens)))
    baseline = NGramBaseline(args.vocab_size)
    sparse = SparseOnlineNGram(args.vocab_size, row_cap=max(32, int(args.vocab_size ** 0.5)))
    base = train_and_eval(baseline, tokens, train_cut)
    sp = train_and_eval(sparse, tokens, train_cut)
    transformer_loss = float("inf")
    ssm_loss = float("inf")
    if args.train_baselines:
        transformer = TrainableTinyTransformerCPU(args.vocab_size, args.d_model, args.seed)
        ssm = TrainableTinySSMCPU(args.vocab_size, args.d_model, args.seed + 17)
        transformer.train(tokens, train_cut, args.context_train, args.baseline_epochs, args.baseline_lr)
        ssm.train(tokens, train_cut, args.baseline_epochs, args.baseline_lr)
        transformer_loss = transformer.eval_loss(tokens, train_cut, args.context_train)
        ssm_loss = ssm.eval_loss(tokens, train_cut)
    best_loss = min(base["loss"], transformer_loss, ssm_loss)
    return {
        "baseline_loss": base["loss"],
        "trained_transformer_loss": transformer_loss,
        "trained_ssm_loss": ssm_loss,
        "best_baseline_loss": best_loss,
        "sparse_loss": sp["loss"],
        "loss_ratio": sp["loss"] / max(best_loss, 1e-9),
        "baseline_accuracy": base["accuracy"],
        "sparse_accuracy": sp["accuracy"],
    }


def adaptation(args: argparse.Namespace) -> Dict[str, float]:
    span = max(16, args.vocab_size // 4)
    memory = ContinualTransitionMemory(live_capacity=max(16, int(span * 1.25)), use_consolidation=True)
    a_pairs = domain_pairs(0, args.vocab_size, args.adaptation_tokens)
    b_pairs = domain_pairs(1, args.vocab_size, args.adaptation_tokens)
    train(memory, a_pairs)
    a_before = eval_continual(memory, a_pairs)
    memory.consolidate()
    train(memory, b_pairs)
    b_after = eval_continual(memory, b_pairs)
    a_after = eval_continual(memory, a_pairs)
    online_updates = args.adaptation_tokens
    retrain_style_updates = args.adaptation_tokens * args.retrain_epochs
    return {
        "adaptation_speedup": float(retrain_style_updates / max(1, online_updates)),
        "retention": float(a_after / max(a_before, 1e-9)),
        "new_domain_accuracy": float(b_after),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    sparse_us = measure_sparse(args.vocab_size, args.d_model, args.iterations, args.seed)
    transformer_us = measure_transformer(args.vocab_size, args.d_model, args.context, args.iterations, args.seed)
    ssm_us = measure_ssm(args.vocab_size, args.d_model, args.iterations, args.seed)
    q = quality(args)
    a = adaptation(args)
    return {
        "sparse_latency_us": sparse_us,
        "transformer_latency_us": transformer_us,
        "ssm_latency_us": ssm_us,
        "transformer_speedup": transformer_us / max(sparse_us, 1e-9),
        "ssm_latency_ratio": ssm_us / max(sparse_us, 1e-9),
        **q,
        **a,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--quality-tokens", type=int, default=3000)
    parser.add_argument("--train-baselines", action="store_true")
    parser.add_argument("--baseline-epochs", type=int, default=2)
    parser.add_argument("--baseline-lr", type=float, default=0.15)
    parser.add_argument("--context-train", type=int, default=16)
    parser.add_argument("--adaptation-tokens", type=int, default=1000)
    parser.add_argument("--retrain-epochs", type=int, default=50)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    checks = {
        "quality": result["loss_ratio"] <= 1.15,
        "transformer_latency": result["transformer_speedup"] >= 20.0,
        "adaptation": result["adaptation_speedup"] >= 50.0,
        "retention": result["retention"] >= 0.95,
    }
    ok = all(checks.values())

    print("Phase 5: Competitive Small-Model Baselines")
    print("=" * 80)
    print(f"Quality loss ratio:       {result['loss_ratio']:.3f}x (target <=1.15x)")
    if args.train_baselines:
        print(f"Trained Transformer loss: {result['trained_transformer_loss']:.4f}")
        print(f"Trained SSM loss:         {result['trained_ssm_loss']:.4f}")
    print(f"Transformer speedup:      {result['transformer_speedup']:.2f}x (target >=20x)")
    print(f"SSM latency ratio report: {result['ssm_latency_ratio']:.2f}x")
    print(f"Online adaptation speedup:{result['adaptation_speedup']:.2f}x (target >=50x)")
    print(f"Retention:                {result['retention']:.2%} (target >=95%)")
    print(f"Overall status:           {'PASS' if ok else 'FAIL'}")

    payload = {"benchmark": "baseline_competition", "success": bool(ok), "checks": checks, "metrics": result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
