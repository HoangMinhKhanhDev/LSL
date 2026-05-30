"""Quick test of LSL text generation using Phase 6 GenerationController."""
import sys
import numpy as np
from lsl import GenerationController, LongContextMemory
from benchmarks.phase5.benchmark_long_context_real_corpus import read_text, tokenize_splits

def main():
    np.random.seed(42)
    
    # Use TinyStories corpus for testing
    print("Loading TinyStories corpus...")
    train_text, eval_text, corpus_path = read_text(type('Args', (), {
        'dataset': 'tinystories',
        'corpus_path': None,
        'max_train_chars': 120000,
        'max_eval_chars': 30000,
        'train_fraction': 0.70,
    })())
    
    print(f"Loaded {len(train_text)} train chars, {len(eval_text)} eval chars")
    
    # Tokenize
    print("Tokenizing...")
    tokenizer, train_tokens, eval_tokens = tokenize_splits(train_text, eval_text, type('Args', (), {
        'tokenizer': 'subword',
        'vocab_size': 1000,
        'tokenizer_train_chars': 80000,
        'max_train_chars': 60000,
        'max_eval_chars': 20000,
        'subword_max_merges': 300,
        'subword_min_pair_count': 3,
    })())
    
    # Test with different training sizes
    test_sizes = [4000, 10000, 20000]
    for train_size in test_sizes:
        train_subset = train_tokens[:train_size]
        vocab_size = tokenizer.vocab_size
        print(f"\n--- Testing with {train_size} tokens ---")
        print(f"Vocab size: {vocab_size}")
        
        # Train memory with timing
        import time
        memory = LongContextMemory(
            capacity=2048,
            vocab_size=vocab_size,
            context_width=6,
            candidate_cap=64,
            seed=42,
        )
        
        start_time = time.time()
        for i in range(len(train_subset) - 1):
            memory.observe_transition(train_subset[i], train_subset[i + 1], vocab_size=vocab_size)
        end_time = time.time()
        
        training_time = end_time - start_time
        tokens_per_sec = len(train_subset) / training_time
        print(f"Training time: {training_time:.2f}s ({tokens_per_sec:.0f} tokens/sec)")
    
    print("\n" + "="*60)
    print("Speed benchmark complete")
    print("="*60)

if __name__ == "__main__":
    main()
