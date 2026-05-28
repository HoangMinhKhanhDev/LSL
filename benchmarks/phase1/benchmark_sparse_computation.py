"""Benchmark sparse computation speed improvement.

Compares dense matrix multiplication vs sparse index-based computation
for SDR inputs with 2% sparsity.
"""
import numpy as np
import time
from lsl import LivingSynapseLM, SimpleWordTokenizer

CORPUS = (
    "Ischemic stroke occurs when an artery to the brain is blocked. "
    "This blockage reduces blood flow and oxygen to brain tissues, leading to cell death. "
    "A common symptom of ischemic stroke is aphasia, which affects speech and language comprehension."
)

def benchmark_sparse_speed():
    """Benchmark dense vs sparse computation speed."""
    print("=" * 70)
    print("  SPARSE COMPUTATION SPEED BENCHMARK")
    print("=" * 70)
    print()

    # Build vocabulary
    tokenizer = SimpleWordTokenizer(vocab_size=50)
    tokenizer.build_vocab(CORPUS)

    # Test with semantic SDR (sparse binary, 2% sparsity)
    n_iterations = 1000

    print("Testing with use_sparse_computation=False (dense)...")
    model_dense = LivingSynapseLM(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=1000,
        use_sdr=True,
        use_semantic_sdr=True,
        sdr_sparsity=0.02,
        semantic_hidden_dim=1000,
        embedding_dim=300,
        use_pretrained=False,
        use_sparse_computation=False,
        seed=42
    )

    # Warmup
    for _ in range(10):
        model_dense.forward(0)

    # Benchmark
    start = time.time()
    for _ in range(n_iterations):
        model_dense.forward(0)
    dense_time = time.time() - start

    print(f"  Time for {n_iterations} iterations: {dense_time:.4f}s")
    print(f"  Average time per iteration: {dense_time / n_iterations * 1000:.4f}ms")

    print("\nTesting with use_sparse_computation=True (sparse)...")
    model_sparse = LivingSynapseLM(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=1000,
        use_sdr=True,
        use_semantic_sdr=True,
        sdr_sparsity=0.02,
        semantic_hidden_dim=1000,
        embedding_dim=300,
        use_pretrained=False,
        use_sparse_computation=True,
        seed=42
    )

    # Warmup
    for _ in range(10):
        model_sparse.forward(0)

    # Benchmark
    start = time.time()
    for _ in range(n_iterations):
        model_sparse.forward(0)
    sparse_time = time.time() - start

    print(f"  Time for {n_iterations} iterations: {sparse_time:.4f}s")
    print(f"  Average time per iteration: {sparse_time / n_iterations * 1000:.4f}ms")

    # Calculate speedup
    speedup = dense_time / sparse_time if sparse_time > 0 else 0
    speedup_pct = ((dense_time - sparse_time) / dense_time * 100) if dense_time > 0 else 0

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"Dense computation time:  {dense_time:.4f}s")
    print(f"Sparse computation time: {sparse_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time savings: {speedup_pct:.1f}%")

    if speedup > 1.0:
        print("✓ Sparse computation is faster")
    elif speedup < 1.0:
        print("✗ Sparse computation is slower (Python overhead)")
    else:
        print("= No significant difference")

    print("=" * 70)

    return {
        "dense_time": dense_time,
        "sparse_time": sparse_time,
        "speedup": speedup,
        "speedup_pct": speedup_pct,
    }

if __name__ == "__main__":
    benchmark_sparse_speed()
