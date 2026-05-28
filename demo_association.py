"""Demo 1: Instant association learning with living weights.

Goal: show that the model can learn a new association in a single session
without any backprop or optimizer, using only local Hebbian/predictive-coding
updates. The association is stored in W_live and can be forgotten by reset_live().
"""
import numpy as np
from lsl import LivingSynapseLM


def main():
    np.random.seed(42)
    vocab = ["A", "B", "C", "D", "E"]
    id_map = {sym: i for i, sym in enumerate(vocab)}
    rev_map = {i: sym for sym, i in id_map.items()}

    model = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=0)

    # Before training: model has no prior knowledge of the A->B association.
    inp = id_map["A"]
    probs_before = model.predict(inp)
    print("Before any exposure:")
    for sym in vocab:
        print(f"  P({sym}|A) = {probs_before[id_map[sym]]:.3f}")

    # Observe the pair A->B repeatedly within a session.
    # Each call to observe() triggers local Hebbian + predictive-coding updates.
    print("\nObserving A->B 40 times in the same session...")
    for _ in range(40):
        model.observe(inp, id_map["B"], reward=0.3)

    # After exposure: the association should be stronger.
    probs_after = model.predict(inp)
    print("\nAfter exposure (still in the same session):")
    for sym in vocab:
        print(f"  P({sym}|A) = {probs_after[id_map[sym]]:.3f}")

    # Show that the model does NOT learn associations it never sees.
    print("\nModel never saw A->D, so P(D|A) should stay low:")
    print(f"  P(D|A) = {probs_after[id_map['D']]:.3f}")

    live_before = model.live_norm()
    print("\nLive weight norm before reset:", live_before)

    # Reset live weights only: slow weights unchanged.
    model.reset_live()
    live_after = model.live_norm()
    print("Live weight norm after reset:", live_after)
    print("Slow weight norm unchanged:", model.slow_norm())

    probs_reset = model.predict(inp)
    print("\nAfter reset_live() (clears short-term plastic memory):")
    for sym in vocab:
        print(f"  P({sym}|A) = {probs_reset[id_map[sym]]:.3f}")


if __name__ == "__main__":
    main()
