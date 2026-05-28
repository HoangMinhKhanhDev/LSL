"""Test semantic SDR integration with LivingSynapseLM."""
import numpy as np
from lsl import LivingSynapseLM, SimpleWordTokenizer

CORPUS = (
    "Ischemic stroke occurs when an artery to the brain is blocked. "
    "This blockage reduces blood flow and oxygen to brain tissues, leading to cell death. "
    "A common symptom of ischemic stroke is aphasia, which affects speech and language comprehension."
)

def test_semantic_sdr_integration():
    """Test that LivingSynapseLM can use semantic SDR."""
    print("=" * 70)
    print("  SEMANTIC SDR INTEGRATION TEST")
    print("=" * 70)
    print()

    # Build vocabulary
    tokenizer = SimpleWordTokenizer(vocab_size=50)
    tokenizer.build_vocab(CORPUS)

    # Create model with semantic SDR
    print("Creating model with use_semantic_sdr=True...")
    model = LivingSynapseLM(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64,  # Will be overridden to 1000 by semantic SDR
        use_sdr=True,
        use_semantic_sdr=True,
        sdr_sparsity=0.02,
        semantic_hidden_dim=1000,
        embedding_dim=300,
        use_pretrained=False,  # Use random init for quick test
        seed=42
    )

    print(f"Model hidden_dim: {model.hidden_dim}")
    print(f"Expected semantic_hidden_dim: 1000")
    assert model.hidden_dim == 1000, "hidden_dim should be 1000 when using semantic SDR"
    print("OK hidden_dim overridden correctly")

    # Load embeddings (only if using pre-trained)
    if model.use_semantic_sdr and not model.sdr_encoder.use_pretrained:
        print("\nUsing random embeddings (use_pretrained=False)")
    else:
        print("\nLoading semantic embeddings...")
        model.load_semantic_embeddings(tokenizer.word_to_id)

    # Test forward pass
    print("\nTesting forward pass...")
    token_id = tokenizer.word_to_id.get("stroke", 1)
    probs = model.predict(token_id)
    print(f"Predictions shape: {probs.shape}")
    print(f"Predictions sum: {probs.sum():.4f}")
    assert probs.shape[0] == tokenizer.vocab_size
    print("OK Forward pass successful")

    # Test observe
    print("\nTesting observe...")
    target_id = tokenizer.word_to_id.get("brain", 1)
    info = model.observe(token_id, target_id)
    print(f"Observe info keys: {list(info.keys())}")
    print("OK Observe successful")

    # Check metrics
    print("\nChecking metrics...")
    metrics = model.metrics()
    print(f"Metrics keys: {list(metrics.keys())}")
    if "sdr_k" in metrics:
        print(f"SDR k: {metrics['sdr_k']}")
        print(f"SDR capacity (log2): {metrics.get('sdr_capacity_log2', 'N/A')}")
    print("OK Metrics available")

    print("\n" + "=" * 70)
    print("  INTEGRATION TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    test_semantic_sdr_integration()
