"""Benchmark Semantic SDR - Verify semantic union property and interference-free storage.

This script tests Phase 2 semantic SDR enhancements:
1. Semantic Union Property: Related concepts share bits
2. Interference-Free Storage: Large dimension with 2% sparsity
"""
import numpy as np
from lsl import SemanticSDREncoder, SimpleWordTokenizer, semantic_overlap

CORPUS = (
    "Ischemic stroke occurs when an artery to the brain is blocked. "
    "This blockage reduces blood flow and oxygen to brain tissues, leading to cell death. "
    "A common symptom of ischemic stroke is aphasia, which affects speech and language comprehension. "
    "Patients with aphasia struggle to find words and understand spoken language. "
    "Prompt treatment of stroke can restore artery blood flow and minimize brain damage. "
    "Rehabilitation helps patients recover from stroke and manage aphasia symptoms over time. "
    "Stroke rehabilitation focuses on restoring lost brain functions through repeated exercises. "
    "Brain plasticity allows damaged regions to be compensated by adjacent healthy brain areas. "
    "Early intervention in ischemic stroke significantly improves patient outcomes. "
    "Aphasia therapy targets language deficits caused by stroke lesions in the brain."
)


def benchmark_semantic_union_property():
    """Test that related concepts have higher SDR overlap than unrelated ones."""
    print("=" * 70)
    print("  SEMANTIC UNION PROPERTY BENCHMARK")
    print("=" * 70)
    print()

    # Build vocabulary
    tokenizer = SimpleWordTokenizer(vocab_size=100)
    tokenizer.build_vocab(CORPUS)
    vocab = tokenizer.word_to_id  # token -> token_id

    # Create semantic SDR encoder (with pre-trained for semantic structure)
    encoder = SemanticSDREncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=1000,
        sparsity=0.02,
        embedding_dim=300,
        use_pretrained=True,  # Use pre-trained for semantic structure
        seed=42
    )

    # Load pre-trained embeddings
    print("Loading pre-trained embeddings (this may take a moment)...")
    encoder.load_embeddings_from_gensim("", tokenizer.word_to_id)

    # Define related and unrelated word pairs (using words from corpus)
    # Get words from vocabulary
    words_in_vocab = list(vocab.keys())

    related_pairs = [
        ("stroke", "brain"),
        ("stroke", "artery"),
        ("aphasia", "speech"),
        ("aphasia", "language"),
        ("brain", "plasticity"),
    ]

    unrelated_pairs = [
        ("stroke", "intervention"),
        ("brain", "rehabilitation"),
        ("aphasia", "intervention"),
        ("artery", "treatment"),
        ("plasticity", "lesions"),
    ]

    print("Encoding SDRs for related pairs:")
    related_overlaps = []
    for word1, word2 in related_pairs:
        if word1 in vocab and word2 in vocab:
            sdr1 = encoder.encode(vocab[word1])
            sdr2 = encoder.encode(vocab[word2])
            overlap = semantic_overlap(sdr1, sdr2)
            related_overlaps.append(overlap)
            print(f"  {word1} <-> {word2}: overlap = {overlap:.2f}")

    print("\nEncoding SDRs for unrelated pairs:")
    unrelated_overlaps = []
    for word1, word2 in unrelated_pairs:
        if word1 in vocab and word2 in vocab:
            sdr1 = encoder.encode(vocab[word1])
            sdr2 = encoder.encode(vocab[word2])
            overlap = semantic_overlap(sdr1, sdr2)
            unrelated_overlaps.append(overlap)
            print(f"  {word1} <-> {word2}: overlap = {overlap:.2f}")

    if related_overlaps and unrelated_overlaps:
        avg_related = np.mean(related_overlaps)
        avg_unrelated = np.mean(unrelated_overlaps)
        ratio = avg_related / max(avg_unrelated, 0.001)

        print(f"\nAverage overlap (related): {avg_related:.2f}")
        print(f"Average overlap (unrelated): {avg_unrelated:.2f}")
        print(f"Ratio (related/unrelated): {ratio:.2f}")

        # Success criteria: related overlap > unrelated overlap
        success = avg_related > avg_unrelated
        print(f"\nSuccess (related > unrelated): {success}")

        if success:
            print("✓ Semantic Union Property demonstrated")
        else:
            print("✗ Semantic Union Property not demonstrated (random initialization)")

    return {
        "avg_related": float(np.mean(related_overlaps)) if related_overlaps else 0.0,
        "avg_unrelated": float(np.mean(unrelated_overlaps)) if unrelated_overlaps else 0.0,
        "success": success if related_overlaps and unrelated_overlaps else False,
    }


def benchmark_interference_free_storage():
    """Test interference-free storage with large dimension (d=1000)."""
    print("\n" + "=" * 70)
    print("  INTERFERENCE-FREE STORAGE BENCHMARK")
    print("=" * 70)
    print()

    # Build vocabulary
    tokenizer = SimpleWordTokenizer(vocab_size=50)
    tokenizer.build_vocab(CORPUS)

    # Create semantic SDR encoder with large dimension
    encoder = SemanticSDREncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=1000,
        sparsity=0.02,
        embedding_dim=300,
        use_pretrained=False,
        seed=42
    )

    # Generate SDRs for all words
    vocab = tokenizer.word_to_id
    sdrs = {}
    for word, token_id in vocab.items():
        sdrs[word] = encoder.encode(token_id)

    # Calculate pairwise overlaps
    words = list(sdrs.keys())
    n_words = len(words)
    overlaps = []

    print(f"Calculating pairwise overlaps for {n_words} words...")
    for i in range(n_words):
        for j in range(i + 1, n_words):
            overlap = semantic_overlap(sdrs[words[i]], sdrs[words[j]])
            overlaps.append(overlap)

    avg_overlap = np.mean(overlaps)
    max_overlap = np.max(overlaps)
    k = int(1000 * 0.02)  # Expected active bits

    print(f"Average pairwise overlap: {avg_overlap:.2f} / {k}")
    print(f"Max pairwise overlap: {max_overlap:.2f} / {k}")
    print(f"Expected active bits per SDR: {k}")

    # Success criteria: low overlap indicates interference-free storage
    success = avg_overlap < (k * 0.1)  # Less than 10% of k
    print(f"\nSuccess (avg overlap < 10% of k): {success}")

    if success:
        print("✓ Interference-free storage demonstrated")
    else:
        print("✗ High overlap detected (may cause interference)")

    return {
        "avg_overlap": float(avg_overlap),
        "max_overlap": float(max_overlap),
        "k": k,
        "success": success,
    }


def main():
    print("\n" + "=" * 70)
    print("  PHASE 2: SEMANTIC SDR BENCHMARK SUITE")
    print("=" * 70)
    print()

    # Benchmark 1: Semantic Union Property
    semantic_results = benchmark_semantic_union_property()

    # Benchmark 2: Interference-Free Storage
    storage_results = benchmark_interference_free_storage()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"Semantic Union Property: {'PASS' if semantic_results['success'] else 'FAIL'}")
    print(f"Interference-Free Storage: {'PASS' if storage_results['success'] else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
