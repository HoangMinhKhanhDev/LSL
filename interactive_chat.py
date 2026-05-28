"""Interactive LSL v2 - CLI Playground.

Allows you to interact with the Living Synapse Language Model directly,
generating completions and teaching it new sentences online.
"""
import sys
import numpy as np
from lsl import LivingSynapseLM, SimpleWordTokenizer
from benchmark_real_task import CORPUS, train_epoch, evaluate

def print_help():
    print("\n" + "=" * 60)
    print(" LSL v2 Interactive CLI - Commands:")
    print("=" * 60)
    print("  <any text>        : Enter a prefix and LSL will complete it.")
    print("  /learn <text>     : Teach LSL a new sentence instantly (online learning).")
    print("  /metrics          : Show current synapse metrics (W_slow, W_live, fatigue).")
    print("  /help             : Show this help message.")
    print("  /exit or /quit    : Exit the playground.")
    print("=" * 60 + "\n")

def main():
    np.random.seed(42)
    
    # 1. Setup tokenizer
    tokenizer = SimpleWordTokenizer(vocab_size=300)
    tokenizer.build_vocab(CORPUS)
    tokens = tokenizer.encode(CORPUS)
    
    print("Initializing LSL v2...")
    # hidden_dim = 64
    model = LivingSynapseLM(vocab_size=tokenizer.vocab_size, hidden_dim=64, seed=42)
    
    # Disable unsupervised plasticity during pre-training to keep it clean
    model.inference_plasticity_enabled = False
    
    # 2. Pre-train model for basic language modeling (20 epochs)
    print("Pre-training LSL on Neurology Corpus (20 epochs)...")
    for ep in range(1, 21):
        train_epoch(model, tokens)
        model.consolidate()
        model.reset_live()
        model.reset_state()
    
    # Enable unsupervised plasticity for live playground
    model.inference_plasticity_enabled = True
    
    print("Pre-training completed successfully!")
    print_help()
    
    while True:
        try:
            user_input = input("LSL Playground > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["/exit", "/quit"]:
            print("Goodbye!")
            break
            
        elif user_input.lower() == "/help":
            print_help()
            continue
            
        elif user_input.lower() == "/metrics":
            m = model.metrics()
            print("\n--- Current Biological Metrics ---")
            print(f"  Long-term memory norm (W_slow): {m['slow_norm']:.4f}")
            print(f"  Short-term plastic norm (W_live): {m['live_norm']:.4f}")
            print(f"  Global state norm:               {m['global_state_norm']:.4f}")
            print(f"  Step count:                      {m['step_count']}")
            print("----------------------------------\n")
            continue
            
        elif user_input.startswith("/learn "):
            learn_text = user_input[7:].strip()
            if not learn_text:
                print("Please provide some text to learn. Example: /learn stroke affects blood flow.")
                continue
                
            # Expand vocabulary dynamically if there are new words
            # (Or map unknown words to <UNK> if they are not in tokenizer)
            learn_tokens = tokenizer.encode(learn_text)
            
            print(f"\nLearning sentence: '{learn_text}'")
            print(f"Tokens: {learn_tokens}")
            
            # Observe the sequence multiple times (e.g. 5 repetitions) to consolidate
            model.reset_state()
            for rep in range(5):
                for i in range(len(learn_tokens) - 1):
                    model.observe(learn_tokens[i], learn_tokens[i+1], reward=0.5, store=False)
            
            model.consolidate()
            model.reset_live()
            model.reset_state()
            print("Learned! The new association has been consolidated into W_slow.")
            print("Type a prefix of the learned sentence to test it!\n")
            continue
            
        # Complete prefix
        prefix_tokens = tokenizer.encode(user_input)
        # Check if all tokens are <UNK>
        unk_id = tokenizer.word_to_id.get("<UNK>", 1)
        if all(tid == unk_id for tid in prefix_tokens):
            print(f"Warning: All words in '{user_input}' are out-of-vocabulary (<UNK>).")
            
        model.reset_state()
        
        # Warm up global state with the prefix
        for tid in prefix_tokens[:-1]:
            model.predict(tid)
            
        current_token = prefix_tokens[-1]
        generated_tokens = list(prefix_tokens)
        
        # Auto-regressive generation loop
        max_generate = 10
        print(f"\nContext: '{user_input}'")
        print("Generated completion: ", end="", flush=True)
        print(user_input, end=" ", flush=True)
        
        for _ in range(max_generate):
            # predict next token
            probs = model.predict(current_token)
            
            # Get top predictions
            top_ids = np.argsort(probs)[::-1][:3]
            next_token = top_ids[0]
            
            if next_token == tokenizer.word_to_id.get("<PAD>", 0):
                break
                
            word = tokenizer.id_to_word.get(next_token, "<UNK>")
            print(word, end=" ", flush=True)
            
            generated_tokens.append(next_token)
            current_token = next_token
            
            if word in [".", "?", "!"]:
                break
                
        print("\n")
        
if __name__ == "__main__":
    main()
