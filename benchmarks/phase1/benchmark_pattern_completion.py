"""Benchmark Pattern Completion using AssociativeMemory.

Tests the ability to recover full SDR patterns from partial inputs.
"""
import numpy as np
from lsl import AssociativeMemory, SemanticSDREncoder, SimpleWordTokenizer

CORPUS = (
    "Ischemic stroke occurs when an artery to the brain is blocked. "
    "This blockage reduces blood flow and oxygen to brain tissues, leading to cell death. "
    "A common symptom of ischemic stroke is aphasia, which affects speech and language comprehension."
)

def benchmark_pattern_completion():
    """Test pattern completion with associative memory."""
    print("=" * 70)
    print("  PATTERN COMPLETION BENCHMARK")
    print("=" * 70)
    print()

    # Build vocabulary
    tokenizer = SimpleWordTokenizer(vocab_size=50)
    tokenizer.build_vocab(CORPUS)

    # Create semantic SDR encoder
    encoder = SemanticSDREncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=1000,
        sparsity=0.02,
        embedding_dim=300,
        use_pretrained=False,
        seed=42
    )

    # Create associative memory
    assoc = AssociativeMemory(dim=1000, capacity=1000, seed=42)

    # Store some patterns
    words_to_store = ["stroke", "brain", "aphasia", "artery"]
    patterns = {}
    print("Storing patterns in associative memory:")
    for word in words_to_store:
        if word in tokenizer.word_to_id:
            token_id = tokenizer.word_to_id[word]
            pattern = encoder.encode(token_id)
            patterns[word] = pattern
            assoc.store(pattern)
            print(f"  {word}: {int(pattern.sum())} active bits")

    print("\nTesting pattern completion from partial inputs:")
    subsample_rate = 0.3  # Keep only 30% of bits

    recovery_rates = []
    for word, full_pattern in patterns.items():
        # Create partial pattern (keep only 30% of active bits)
        active_indices = np.where(full_pattern > 0.5)[0]
        n_keep = max(1, int(len(active_indices) * subsample_rate))
        keep_indices = np.random.choice(active_indices, n_keep, replace=False)

        partial = np.zeros_like(full_pattern)
        partial[keep_indices] = 1.0

        # Retrieve from associative memory
        retrieved = assoc.retrieve(partial)

        # Calculate recovery rate (overlap / original active bits)
        overlap = np.sum(retrieved * full_pattern)
        original_active = int(full_pattern.sum())
        recovery_rate = overlap / original_active if original_active > 0 else 0.0
        recovery_rates.append(recovery_rate)

        print(f"  {word}:")
        print(f"    Original: {original_active} active bits")
        print(f"    Partial: {n_keep} bits ({subsample_rate*100:.0f}%)")
        print(f"    Retrieved overlap: {overlap}/{original_active}")
        print(f"    Recovery rate: {recovery_rate*100:.1f}%")

    avg_recovery = np.mean(recovery_rates)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"Average recovery rate: {avg_recovery*100:.1f}%")

    # Success criteria: average recovery > 50%
    success = avg_recovery > 0.5
    print(f"Success (avg recovery > 50%): {success}")

    if success:
        print("✓ Pattern completion demonstrated")
    else:
        print("✗ Pattern completion failed (need more patterns or different parameters)")

    print("=" * 70)

    return {
        "avg_recovery": float(avg_recovery),
        "success": success,
    }

if __name__ == "__main__":
    benchmark_pattern_completion()
