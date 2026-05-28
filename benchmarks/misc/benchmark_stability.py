"""Benchmark 2: Self-stabilization against drift.

Measures:
- Retention ratio after random noise transitions
- Live weight norm stability
- Effective weight norm changes
"""
import numpy as np
from lsl import LivingSynapseLM


def benchmark_stability(n_pairs=2, n_learn=30, n_noise=30, seed=1):
    np.random.seed(seed)
    vocab = [chr(ord('A') + i) for i in range(10)]
    id_map = {sym: i for i, sym in enumerate(vocab)}

    model = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=seed)

    # Select pairs
    pairs = []
    for _ in range(n_pairs):
        src = np.random.randint(len(vocab))
        tgt = np.random.randint(len(vocab))
        while src == tgt:
            tgt = np.random.randint(len(vocab))
        pairs.append((src, tgt))

    print("=" * 60)
    print("STABILITY BENCHMARK")
    print("=" * 60)
    print(f"Pairs to learn: {[(vocab[s], vocab[t]) for s, t in pairs]}")
    print(f"Learning repetitions: {n_learn}")
    print(f"Noise transitions: {n_noise}")
    print()

    # Phase 1: Learn
    for src, tgt in pairs:
        for _ in range(n_learn):
            model.observe(src, tgt, reward=0.3)

    # Measure after learning
    scores_learn = {}
    for src, tgt in pairs:
        probs = model.predict(src)
        scores_learn[(src, tgt)] = float(probs[tgt])

    live_norm_learn = model.live_norm()
    w_eff_learn = model.effective_weight_norms()

    # Phase 2: Noise
    for _ in range(n_noise):
        src = np.random.randint(len(vocab))
        tgt = np.random.randint(len(vocab))
        model.observe(src, tgt, reward=0.0)

    # Measure after noise
    scores_noise = {}
    for src, tgt in pairs:
        probs = model.predict(src)
        scores_noise[(src, tgt)] = float(probs[tgt])

    live_norm_noise = model.live_norm()
    w_eff_noise = model.effective_weight_norms()

    # Calculate retention ratios
    retention_ratios = {}
    for src, tgt in pairs:
        retention = scores_noise[(src, tgt)] / max(scores_learn[(src, tgt)], 1e-10)
        retention_ratios[(src, tgt)] = retention

    print("Results per pair:")
    for src, tgt in pairs:
        print(f"  {vocab[src]} -> {vocab[tgt]}:")
        print(f"    Score after learning: {scores_learn[(src, tgt)]:.4f}")
        print(f"    Score after noise:    {scores_noise[(src, tgt)]:.4f}")
        print(f"    Retention ratio:      {retention_ratios[(src, tgt)]:.2%}")
    print()

    print(f"Live norm after learning: {live_norm_learn:.4f}")
    print(f"Live norm after noise:    {live_norm_noise:.4f}")
    print(f"W_eff diff (embed):       {w_eff_noise['embed'] - w_eff_learn['embed']:.6f}")
    print(f"W_eff diff (recurrent):   {w_eff_noise['recurrent'] - w_eff_learn['recurrent']:.6f}")
    print(f"W_eff diff (output):      {w_eff_noise['output'] - w_eff_learn['output']:.6f}")
    print()

    avg_retention = np.mean(list(retention_ratios.values()))
    print(f"Average retention ratio: {avg_retention:.2%}")
    print()

    # Success criteria: average retention >= 70%
    success = avg_retention >= 0.7
    print(f"Success (average retention >= 70%): {success}")
    print()

    return retention_ratios, success


if __name__ == "__main__":
    benchmark_stability()
