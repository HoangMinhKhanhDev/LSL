"""Demo 4: Mini-token demo closer to a tiny language model.

Uses a small vocabulary of 16 tokens (e.g., letters or short words) to show
that the same mechanisms work at a slightly larger scale.
"""
import numpy as np
from lsl import LivingSynapseLM


def main():
    np.random.seed(3)
    vocab = [f"T{i}" for i in range(16)]
    id_map = {sym: i for i, sym in enumerate(vocab)}

    model = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=32, k_ratio=0.3, seed=0)

    # Define a simple pattern: T0 -> T1 -> T2 -> T3 -> T0 (cycle)
    pattern = [0, 1, 2, 3, 0]

    print("Training on the cycle pattern T0->T1->T2->T3->T0 (30 cycles)")
    for _ in range(30):
        for src, tgt in zip(pattern, pattern[1:]):
            model.observe(src, tgt, reward=0.1)

    # Test prediction along the cycle.
    print("\nPredictions along the cycle:")
    for src, tgt in zip(pattern, pattern[1:]):
        probs = model.predict(src)
        print(f"  P({vocab[tgt]}|{vocab[src]}) = {probs[tgt]:.3f}")

    # Test a non-pattern transition.
    print("\nNon-pattern transition T4->T5 (never seen):")
    probs = model.predict(4)
    print(f"  P(T5|T4) = {probs[5]:.3f}")

    # Consolidate and test again.
    print("\nReplay + consolidation (3 rounds)")
    for i in range(3):
        model.replay(n=12, lr_factor=0.5)
        n = model.consolidate(threshold=0.1, fraction=0.2)
        print(f"  Round {i+1}: consolidated {n} synapses")

    model.reset_live()
    print("\nAfter reset_live() (only slow weights):")
    for src, tgt in zip(pattern, pattern[1:]):
        probs = model.predict(src)
        print(f"  P({vocab[tgt]}|{vocab[src]}) = {probs[tgt]:.3f}")


if __name__ == "__main__":
    main()
