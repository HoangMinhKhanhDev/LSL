"""Demo 2: Self-stabilization against catastrophic drift.

Goal: show that the model can learn new associations without completely
forgetting previously learned ones, thanks to:
- fatigue and decay on W_live,
- neuromodulator gating (low surprise = low plasticity),
- and the separation between W_slow and W_live.
"""
import numpy as np
from lsl import LivingSynapseLM


def main():
    np.random.seed(1)
    vocab = ["A", "B", "C", "D", "E"]
    id_map = {sym: i for i, sym in enumerate(vocab)}
    rev_map = {i: sym for sym, i in id_map.items()}

    model = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=0)

    # Phase 1: learn A->B and C->D.
    print("Phase 1: learn A->B and C->D (30 repetitions each)")
    for _ in range(30):
        model.observe(id_map["A"], id_map["B"], reward=0.3)
        model.observe(id_map["C"], id_map["D"], reward=0.3)

    def report(phase):
        probs_A = model.predict(id_map["A"])
        probs_C = model.predict(id_map["C"])
        print(f"\n{phase}:")
        print(f"  P(B|A) = {probs_A[id_map['B']]:.3f}")
        print(f"  P(D|C) = {probs_C[id_map['D']]:.3f}")
        print(f"  Live norm = {model.live_norm():.3f}")

    report("After Phase 1")

    # Phase 2: observe many random transitions that do NOT include A->B or C->D.
    # The model should NOT completely forget the original associations.
    print("\nPhase 2: observe 30 random transitions (no A->B or C->D)")
    for _ in range(30):
        src = np.random.randint(len(vocab))
        tgt = np.random.randint(len(vocab))
        # Avoid reinforcing the original pairs
        if (src == id_map["A"] and tgt == id_map["B"]) or (src == id_map["C"] and tgt == id_map["D"]):
            continue
        model.observe(src, tgt)

    report("After Phase 2 (random noise)")

    # Phase 3: a few more repetitions of the original pairs to refresh.
    print("\nPhase 3: refresh A->B and C->D (8 repetitions each)")
    for _ in range(8):
        model.observe(id_map["A"], id_map["B"], reward=0.2)
        model.observe(id_map["C"], id_map["D"], reward=0.2)

    report("After Phase 3 (refresh)")

    # Phase 4: consolidation to slow weights.
    print("\nPhase 4: consolidate (transfer part of W_live to W_slow)")
    n_consolidated = model.consolidate(threshold=0.1, fraction=0.2)
    print(f"  Synapses consolidated: {n_consolidated}")
    report("After consolidation")

    # Reset live weights and see that slow weights still retain some memory.
    model.reset_live()
    report("After reset_live() (only slow weights remain)")


if __name__ == "__main__":
    main()
