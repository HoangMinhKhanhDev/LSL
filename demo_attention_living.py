"""Demo 5: Sequence Learning with Living Attention.

Demonstrates how LivingAttentionLayer helps the model resolve context-dependent
ambiguities (e.g., 'T0 -> T1 -> T2' vs 'T3 -> T1 -> T4').
Here, 'T1' is ambiguous; the correct target depends on whether the prefix was
'T0' or 'T3'. A simple Markovian association model cannot solve this, but
self-attention over the working memory buffer resolves it.
"""
import numpy as np
from lsl import LivingSynapseLM


def draw_attention_grid(attn_map, labels):
    """Draw a text-based attention map using standard ASCII."""
    print("      " + "  ".join(f"{l:<4}" for l in labels))
    for r_idx, row in enumerate(attn_map):
        row_str = " ".join(f"{v:.2f}" for v in row)
        bars = "".join("#" if v > 0.4 else "=" if v > 0.15 else "-" for v in row)
        print(f"  {labels[r_idx]:<4} [{row_str}]  {bars}")


def main():
    np.random.seed(42)
    
    # 8 tokens
    vocab = [f"T{i}" for i in range(8)]
    id_map = {sym: i for i, sym in enumerate(vocab)}
    
    # Initialize LSL with hidden_dim=16
    model = LivingSynapseLM(vocab_size=len(vocab), hidden_dim=16, k_ratio=0.4, seed=42)
    
    # Define two context-dependent sequences:
    # Seq 1: T0 -> T1 -> T2  (T1 should predict T2 because it saw T0)
    # Seq 2: T3 -> T1 -> T4  (T1 should predict T4 because it saw T3)
    
    print("=" * 70)
    print(" CONTEXT-DEPENDENT SEQUENCE LEARNING WITH LIVING ATTENTION ")
    print("=" * 70)
    print("Task definition:")
    print("  Sequence 1: T0 -> T1 -> T2")
    print("  Sequence 2: T3 -> T1 -> T4")
    print("  Ambuguity: T1 is shared! Target depends on the context (T0 vs T3).")
    print()
    
    # 1. Before training: check predictions
    print("Predictions BEFORE training:")
    # We must feed the sequence step by step so the buffer builds up
    
    # Sequence 1 test
    model.reset_state()
    model.forward(id_map["T0"])
    probs1 = model.predict(id_map["T1"])
    print(f"  Given sequence T0 -> T1: P(T2) = {probs1[id_map['T2']]:.3f}, P(T4) = {probs1[id_map['T4']]:.3f}")
    
    # Sequence 2 test
    model.reset_state()
    model.forward(id_map["T3"])
    probs2 = model.predict(id_map["T1"])
    print(f"  Given sequence T3 -> T1: P(T2) = {probs2[id_map['T2']]:.3f}, P(T4) = {probs2[id_map['T4']]:.3f}")
    print()
    
    # 2. Train online
    print("Training online (observing alternating sequences 40 times)...")
    for epoch in range(40):
        # Present Seq 1: T0 -> T1 -> T2
        model.reset_state()
        model.observe(id_map["T0"], id_map["T1"], reward=0.2)
        model.observe(id_map["T1"], id_map["T2"], reward=0.3)
        
        # Present Seq 2: T3 -> T1 -> T4
        model.reset_state()
        model.observe(id_map["T3"], id_map["T1"], reward=0.2)
        model.observe(id_map["T1"], id_map["T4"], reward=0.3)
        
    print("Training completed.")
    print()
    
    # 3. Test after training
    print("Predictions AFTER training:")
    
    # Sequence 1 test
    model.reset_state()
    model.forward(id_map["T0"])
    probs1_after = model.predict(id_map["T1"])
    p_t2_seq1 = probs1_after[id_map['T2']]
    p_t4_seq1 = probs1_after[id_map['T4']]
    print(f"  Given sequence T0 -> T1:")
    print(f"    P(T2|T0, T1) = {p_t2_seq1:.4f} [Target]")
    print(f"    P(T4|T0, T1) = {p_t4_seq1:.4f}")
    
    # Capture attention map for Seq 1
    attn_map1 = model.attn.last_attention_map.copy()
    
    # Sequence 2 test
    model.reset_state()
    model.forward(id_map["T3"])
    probs2_after = model.predict(id_map["T1"])
    p_t2_seq2 = probs2_after[id_map['T2']]
    p_t4_seq2 = probs2_after[id_map['T4']]
    print(f"  Given sequence T3 -> T1:")
    print(f"    P(T2|T3, T1) = {p_t2_seq2:.4f}")
    print(f"    P(T4|T3, T1) = {p_t4_seq2:.4f} [Target]")
    
    # Capture attention map for Seq 2
    attn_map2 = model.attn.last_attention_map.copy()
    print()
    
    # Check success criteria
    success1 = p_t2_seq1 > p_t4_seq1
    success2 = p_t4_seq2 > p_t2_seq2
    print(f"Success in resolving Seq 1: {success1}")
    print(f"Success in resolving Seq 2: {success2}")
    print(f"Overall Attention-driven recall success: {success1 and success2}")
    print()
    
    # 4. Visualize attention maps
    print("Visualizing Attention Maps (How the model routes information):")
    print("\nAttention map for 'T0 -> T1':")
    draw_attention_grid(attn_map1, ["T0", "T1"])
    
    print("\nAttention map for 'T3 -> T1':")
    draw_attention_grid(attn_map2, ["T3", "T1"])
    print()
    print("Note: In both cases, when processing the second token (T1), the model attends heavily")
    print("to the historical context token (T0 or T3, first column) to route the correct prediction!")
    print("=" * 70)


if __name__ == "__main__":
    main()
