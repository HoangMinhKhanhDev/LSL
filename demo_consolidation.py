"""Demo 3: Hippocampus-to-neocortex style consolidation.

Goal: show that repeated replay + consolidation can transfer short-term
plastic memory into long-term storage, surviving reset_live().
"""
import numpy as np
from lsl import LivingSynapseLM


def main():
    np.random.seed(2)
    vocab = ["A", "B", "C", "D", "E"]
    id_map = {sym: i for i, sym in enumerate(vocab)}

    model = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=0)

    # Learn A->B during a session.
    print("Learning A->B in a session (30 repetitions)")
    for _ in range(30):
        model.observe(id_map["A"], id_map["B"], reward=0.2)

    probs_before = model.predict(id_map["A"])
    print(f"\nBefore consolidation: P(B|A) = {probs_before[id_map['B']]:.3f}")
    print(f"Live norm = {model.live_norm():.3f}, Slow norm = {model.slow_norm():.3f}")

    # Perform consolidation: replay episodic buffer and transfer stable synapses.
    print("\nReplay + consolidation (5 rounds)")
    for i in range(5):
        model.replay(n=8, lr_factor=0.5)
        n = model.consolidate(threshold=0.1, fraction=0.2)
        print(f"  Round {i+1}: consolidated {n} synapses")

    # After consolidation, reset live weights.
    model.reset_live()
    print("\nAfter reset_live() (only slow weights remain):")
    probs_after = model.predict(id_map["A"])
    print(f"  P(B|A) = {probs_after[id_map['B']]:.3f}")
    print(f"  Live norm = {model.live_norm():.3f}, Slow norm = {model.slow_norm():.3f}")

    # Compare with a control model that never consolidated.
    model_control = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=0)
    for _ in range(12):
        model_control.observe(id_map["A"], id_map["B"])
    model_control.reset_live()
    probs_control = model_control.predict(id_map["A"])
    print(f"\nControl (no consolidation) after reset_live(): P(B|A) = {probs_control[id_map['B']]:.3f}")


if __name__ == "__main__":
    main()
