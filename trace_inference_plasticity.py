"""Trace Inference Plasticity - verification script.

Demonstrates that W_live changes *during pure inference* (when calling predict()
without any targets or observe() calls). This verifies true activity-dependent 
unsupervised plasticity.
"""
import numpy as np
from lsl import LivingSynapseLM


def main():
    np.random.seed(42)
    vocab = ["A", "B", "C", "D", "E"]
    id_map = {sym: i for i, sym in enumerate(vocab)}
    
    # Initialize the model
    model = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=8, k_ratio=0.5, seed=42)
    
    print("=" * 70)
    print(" UNSUPERVISED INFERENCE PLASTICITY VERIFICATION ")
    print("=" * 70)
    print("Initial State:")
    print(f"  Live weight norm (W_live): {model.live_norm():.6f}")
    
    # Check baseline prediction for token A
    probs_start = model.predict(id_map["A"])
    print(f"  Initial predictions for input 'A':")
    for sym in vocab:
        print(f"    P({sym}|A) = {probs_start[id_map[sym]]:.4f}")
    
    # Check that calling predict() has already modified W_live slightly!
    print(f"\nAfter 1 forward pass (predict('A')):")
    print(f"  Live weight norm (W_live): {model.live_norm():.6f}")
    
    # Run pure predict() multiple times to show continuous unsupervised adaptation
    print("\nCalling predict('A') 15 times (pure inference, NO targets, NO observe)...")
    for _ in range(15):
        model.predict(id_map["A"])
        
    print(f"\nState after 15 pure inference passes:")
    print(f"  Live weight norm (W_live): {model.live_norm():.6f}")
    
    # Check predictions now
    probs_end = model.predict(id_map["A"])
    print(f"  Final predictions for input 'A':")
    for sym in vocab:
        print(f"    P({sym}|A) = {probs_end[id_map[sym]]:.4f}")
        
    # Check probability changes
    diffs = probs_end - probs_start
    print(f"\nProbability Shifts (Inference adaptation):")
    for sym in vocab:
        print(f"    delta P({sym}|A) = {diffs[id_map[sym]]:+.4f}")
        
    # Check reset_live
    model.reset_live()
    live_after_reset = model.live_norm()
    print(f"\nAfter reset_live():")
    print(f"  Live weight norm (W_live): {live_after_reset:.6f}")
    
    probs_reset = model.predict(id_map["A"])
    print(f"  P(B|A) reset value:        {probs_reset[id_map['B']]:.4f} (Should be near baseline {probs_start[id_map['B']]:.4f})")
    
    # Verification assertions
    assert live_after_reset == 0.0, "FAIL: W_live did not clear on reset!"
    assert np.any(np.abs(diffs) > 1e-4), "FAIL: Predictions did not adapt during inference!"
    print("\nSUCCESS: Synaptic weights (W_live) mutated and predictions adapted in-stride during pure inference!")
    print("=" * 70)


if __name__ == "__main__":
    main()
