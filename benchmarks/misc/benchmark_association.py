"""Benchmark 1: Instant association learning with quantitative metrics.

Measures:
- P(target|input) before/after learning
- Relative improvement percentage
- Live weight norm changes
- Effective weight norm changes
- Router routing changes
"""
import numpy as np
import time
from lsl import LivingSynapseLM


def benchmark_association(n_pairs=3, n_repetitions=40, seed=42):
    np.random.seed(seed)
    vocab = [chr(ord('A') + i) for i in range(10)]
    id_map = {sym: i for i, sym in enumerate(vocab)}

    model = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=seed)

    # Select random pairs
    pairs = []
    for _ in range(n_pairs):
        src = np.random.randint(len(vocab))
        tgt = np.random.randint(len(vocab))
        while src == tgt:
            tgt = np.random.randint(len(vocab))
        pairs.append((src, tgt))

    print("=" * 60)
    print("ASSOCIATION LEARNING BENCHMARK")
    print("=" * 60)
    print(f"Pairs to learn: {[(vocab[s], vocab[t]) for s, t in pairs]}")
    print(f"Repetitions per pair: {n_repetitions}")
    print()

    results = []

    for src, tgt in pairs:
        # Baseline
        probs_before = model.predict(src)
        p_before = float(probs_before[tgt])

        # Learn
        start = time.time()
        for _ in range(n_repetitions):
            model.observe(src, tgt, reward=0.3)
        elapsed = time.time() - start

        # After learning
        probs_after = model.predict(src)
        p_after = float(probs_after[tgt])

        # Metrics
        relative_improvement = (p_after - p_before) / max(p_before, 1e-10) * 100
        live_norm_before = model.live_norm()
        w_eff_before = model.effective_weight_norms()
        router_active_before = model.router.active_indices()

        model.reset_live()

        live_norm_after = model.live_norm()
        w_eff_after = model.effective_weight_norms()
        router_active_after = model.router.active_indices()

        result = {
            "pair": (vocab[src], vocab[tgt]),
            "p_before": p_before,
            "p_after": p_after,
            "relative_improvement_pct": relative_improvement,
            "elapsed_sec": elapsed,
            "live_norm_before": live_norm_before,
            "live_norm_after": live_norm_after,
            "w_eff_diff": {
                k: w_eff_after[k] - w_eff_before[k]
                for k in w_eff_before
            },
            "router_active_before": len(router_active_before),
            "router_active_after": len(router_active_after),
        }
        results.append(result)

        print(f"Pair {vocab[src]} -> {vocab[tgt]}:")
        print(f"  P(target|src) before: {p_before:.4f}")
        print(f"  P(target|src) after:  {p_after:.4f}")
        print(f"  Relative improvement: {relative_improvement:+.2f}%")
        print(f"  Live norm before reset: {live_norm_before:.4f}")
        print(f"  Live norm after reset:  {live_norm_after:.4f}")
        print(f"  W_eff diff (embed): {result['w_eff_diff']['embed']:.6f}")
        print(f"  W_eff diff (recurrent): {result['w_eff_diff']['recurrent']:.6f}")
        print(f"  W_eff diff (output): {result['w_eff_diff']['output']:.6f}")
        print(f"  Time: {elapsed:.4f}s for {n_repetitions} observations")
        print()

    # Summary
    avg_improvement = np.mean([r["relative_improvement_pct"] for r in results])
    print(f"Average relative improvement: {avg_improvement:.2f}%")
    print()

    # Success criteria
    success = all(r["relative_improvement_pct"] >= 20 for r in results)
    print(f"Success (all pairs >= 20% improvement): {success}")
    print()

    return results, success


if __name__ == "__main__":
    benchmark_association()
