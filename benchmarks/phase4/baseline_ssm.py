"""Tiny NumPy SSM/Mamba-like CPU baseline for Phase 5 comparison."""
import argparse
import time
from typing import Dict, List

import numpy as np


class TinySSM:
    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        rng = np.random.default_rng(seed)
        self.embed = rng.standard_normal((vocab_size, d_model)).astype(np.float32) * 0.02
        self.A = np.eye(d_model, dtype=np.float32) * 0.95
        self.B = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.C = rng.standard_normal((d_model, vocab_size)).astype(np.float32) * 0.02
        self.state = np.zeros(d_model, dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(0.0)

    def forward(self, token: int) -> np.ndarray:
        x = self.embed[int(token) % self.vocab_size]
        self.state = np.tanh(self.A @ self.state + self.B @ x)
        return self.state @ self.C

    def get_num_params(self) -> int:
        return int(self.embed.size + self.A.size + self.B.size + self.C.size)


def compare(vocab_size: int, d_model: int, num_tokens: int, seed: int) -> Dict[str, float]:
    model = TinySSM(vocab_size, d_model, seed)
    rng = np.random.default_rng(seed)
    tokens: List[int] = rng.integers(0, vocab_size, size=num_tokens).tolist()
    times = []
    for token in tokens:
        t0 = time.perf_counter_ns()
        model.forward(token)
        times.append((time.perf_counter_ns() - t0) / 1000.0)
    return {
        "vocab_size": float(vocab_size),
        "d_model": float(d_model),
        "params": float(model.get_num_params()),
        "mean_latency_us": float(np.mean(times)),
        "p50_latency_us": float(np.percentile(times, 50)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare(args.vocab_size, args.d_model, args.num_tokens, args.seed)
    print("Tiny SSM/Mamba-like CPU baseline")
    print("=" * 80)
    print(f"Params:      {int(result['params']):,}")
    print(f"Mean latency:{result['mean_latency_us']:.2f} us")
    print(f"p50 latency: {result['p50_latency_us']:.2f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
