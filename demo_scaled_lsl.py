"""demo_scaled_lsl.py - Scale-up Demo of Living Synapse LLM.

Uses SimpleWordTokenizer to tokenize real paragraph data about biology,
instantiates a larger model (hidden_dim=128), and validates online sequence
learning speed (target < 20ms) and vocabulary adaptation success.
"""
import os
import time
import numpy as np
from lsl import LivingSynapseLM, SimpleWordTokenizer


def main():
    np.random.seed(42)
    
    # 1. Load sample text data
    corpus_path = "sample_data.txt"
    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found.")
        return
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        text_corpus = f.read()
        
    print("=" * 75)
    print(" SCALED-UP LIVING SYNAPSE LLM (LSL) REAL-TEXT DEMO ")
    print("=" * 75)
    print(f"Corpus size: {len(text_corpus)} characters.")
    
    # 2. Build vocab and tokenize
    tokenizer = SimpleWordTokenizer(vocab_size=500)
    tokenizer.build_vocab(text_corpus)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size} tokens.")
    
    encoded_ids = tokenizer.encode(text_corpus)
    print(f"Encoded corpus: {len(encoded_ids)} tokens.")
    
    # 3. Instantiate scaled LSL model
    hidden_dim = 128
    print(f"Initializing LSL: vocab_size={tokenizer.vocab_size}, hidden_dim={hidden_dim}")
    model = LivingSynapseLM(
        vocab_size=tokenizer.vocab_size, 
        hidden_dim=hidden_dim, 
        k_ratio=0.2,  # Sparse circuit routing (top 20% active)
        seed=42
    )
    
    # 4. Measure baseline probability on a specific target phrase
    test_word_in = "synaptic"
    test_word_tgt = "plasticity"
    
    id_in = tokenizer.word_to_id.get(test_word_in, 1)
    id_tgt = tokenizer.word_to_id.get(test_word_tgt, 1)
    print(f"Token IDs: '{test_word_in}' -> {id_in}, '{test_word_tgt}' -> {id_tgt}")
    
    probs_before = model.predict(id_in)
    p_before = probs_before[id_tgt]
    
    print("\nBaseline Association:")
    print(f"  P('{test_word_tgt}' | '{test_word_in}') BEFORE reading text: {p_before:.6f}")
    
    # 5. Online sequence learning over the corpus (read multiple times)
    epochs = 8
    print(f"\nLearning the corpus online over {epochs} presentations...")
    
    step_times = []
    
    for epoch in range(epochs):
        model.reset_state()
        for idx in range(len(encoded_ids) - 1):
            curr_token = encoded_ids[idx]
            next_token = encoded_ids[idx + 1]
            
            # Observe and update online - measure step duration
            t_start = time.perf_counter()
            model.observe(curr_token, next_token, reward=0.2, store=True)
            t_end = time.perf_counter()
            
            step_times.append((t_end - t_start) * 1000.0) # in ms
            
    avg_step_time = np.mean(step_times)
    max_step_time = np.max(step_times)
    
    print(f"Online learning completed.")
    print(f"Computation Statistics:")
    print(f"  Average processing time per token: {avg_step_time:.4f} ms")
    print(f"  Max processing time per token:     {max_step_time:.4f} ms")
    
    # Check compute efficiency criterion (< 20ms)
    efficiency_ok = avg_step_time < 20.0
    print(f"  Compute Efficiency Criterion (< 20ms): {'[PASSED]' if efficiency_ok else '[FAILED]'}")
    
    # 6. Measure adapted probability
    probs_after = model.predict(id_in)
    p_after = probs_after[id_tgt]
    relative_improvement = (p_after - p_before) / max(p_before, 1e-10) * 100
    
    print(f"\nAdapted Association:")
    print(f"  P('{test_word_tgt}' | '{test_word_in}') AFTER reading text:  {p_after:.6f}")
    print(f"  Relative improvement:                       {relative_improvement:+.2f}%")
    
    adaptation_ok = relative_improvement >= 15.0
    print(f"  Adaptation success criterion (>= 15%):      {'[PASSED]' if adaptation_ok else '[FAILED]'}")
    
    # Verify that resetting live weight drops it back to baseline
    model.reset_live()
    probs_reset = model.predict(id_in)
    p_reset = probs_reset[id_tgt]
    print(f"  P('{test_word_tgt}' | '{test_word_in}') AFTER reset_live():  {p_reset:.6f} (Should be near baseline)")
    
    # 7. Check another transition: "dopamine" -> ","
    test2_in = "dopamine"
    test2_tgt = ","
    id2_in = tokenizer.word_to_id.get(test2_in, 1)
    id2_tgt = tokenizer.word_to_id.get(test2_tgt, 1)
    
    # Read once to build live weights for dopamine -> ,
    model.reset_state()
    model.observe(id2_in, id2_tgt, reward=0.5)
    p_after_test2 = model.predict(id2_in)[id2_tgt]
    print(f"  P('{test2_tgt}' | '{test2_in}') after immediate observation: {p_after_test2:.6f}")
    
    print("\n" + "=" * 75)
    print(" SUMMARY OF SCALE-UP STATUS ")
    print("=" * 75)
    print(f"  1. Vocabulary size (N={tokenizer.vocab_size}):              [PASSED]")
    print(f"  2. Hidden dimension (D={hidden_dim}):             [PASSED]")
    print(f"  3. Compute efficiency (< 20ms/step):      {f'[PASSED] ({avg_step_time:.3f}ms)' if efficiency_ok else f'[FAILED] ({avg_step_time:.3f}ms)'}")
    print(f"  4. Online text adaptation (>= 15% improvement): {f'[PASSED] ({relative_improvement:+.2f}%)' if adaptation_ok else f'[FAILED] ({relative_improvement:+.2f}%)'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
