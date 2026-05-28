"""Benchmark 3: Hippocampus-to-neocortex consolidation.

Measures:
- Score after consolidation + reset_live() vs control (no consolidation)
- Number of synapses transferred
- Slow weight norm changes
"""
import numpy as np
from lsl import LivingSynapseLM


def benchmark_consolidation(n_learn=30, n_replay=5, replay_n=8, threshold=0.1, fraction=0.2, seed=2):
    np.random.seed(seed)
    vocab = [chr(ord('A') + i) for i in range(10)]
    id_map = {sym: i for i, sym in enumerate(vocab)}

    # Model with consolidation
    model_consolidated = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=seed)

    # Control model (no consolidation)
    model_control = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=seed)

    # Select a pair
    src, tgt = 0, 1

    print("=" * 60)
    print("CONSOLIDATION BENCHMARK")
    print("=" * 60)
    print(f"Pair to learn: {vocab[src]} -> {vocab[tgt]}")
    print(f"Learning repetitions: {n_learn}")
    print(f"Replay rounds: {n_replay}")
    print(f"Consolidation threshold: {threshold}, fraction: {fraction}")
    print()

    # Learn in both models
    for _ in range(n_learn):
        model_consolidated.observe(src, tgt, reward=0.2)
        model_control.observe(src, tgt, reward=0.2)

    probs_before_consolidated = model_consolidated.predict(src)
    score_before_consolidated = float(probs_before_consolidated[tgt])

    live_norm_before = model_consolidated.live_norm()
    slow_norm_before = model_consolidated.slow_norm()

    print(f"Before consolidation:")
    print(f"  P(target|src): {score_before_consolidated:.4f}")
    print(f"  Live norm: {live_norm_before:.4f}")
    print(f"  Slow norm: {slow_norm_before:.4f}")
    print()

    # Replay + consolidation
    total_consolidated = 0
    for i in range(n_replay):
        model_consolidated.replay(n=replay_n, lr_factor=0.5)
        n = model_consolidated.consolidate(threshold=threshold, fraction=fraction)
        total_consolidated += n
        print(f"  Round {i+1}: consolidated {n} synapses")

    print(f"\nTotal synapses consolidated: {total_consolidated}")

    slow_norm_after_consolidation = model_consolidated.slow_norm()
    print(f"Slow norm after consolidation: {slow_norm_after_consolidation:.4f}")

    # Reset live weights in both models
    model_consolidated.reset_live()
    model_control.reset_live()

    # Measure final scores
    probs_consolidated = model_consolidated.predict(src)
    score_consolidated = float(probs_consolidated[tgt])

    probs_control = model_control.predict(src)
    score_control = float(probs_control[tgt])

    print()
    print("After reset_live():")
    print(f"  Consolidated model P(target|src): {score_consolidated:.4f}")
    print(f"  Control model P(target|src):      {score_control:.4f}")
    print()

    improvement = score_consolidated - score_control
    print(f"Improvement over control: {improvement:.4f}")
    print()

    # Success criteria: consolidated model > control
    success = score_consolidated > score_control
    print(f"Success (consolidated > control): {success}")
    print()

    return {
        "score_consolidated": score_consolidated,
        "score_control": score_control,
        "improvement": improvement,
        "total_consolidated": total_consolidated,
        "success": success,
    }


if __name__ == "__main__":
    benchmark_consolidation()
