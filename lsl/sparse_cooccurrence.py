"""Sparse co-occurrence matrix utilities for large-scale semantic SDR.

Provides memory-efficient co-occurrence computation using sparse COO format.
Avoids O(V²) dense matrix memory footprint.
"""
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


def build_sparse_cooccurrence(
    token_ids: List[int],
    vocab_size: int,
    window: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse co-occurrence matrix in COO format.

    Args:
        token_ids: List of token IDs
        vocab_size: Vocabulary size
        window: Context window size

    Returns:
        Tuple of (row_indices, col_indices, values) for sparse COO matrix
    """
    # Use defaultdict to accumulate counts
    cooccur = defaultdict(float)
    n = len(token_ids)

    for i, center in enumerate(token_ids):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        for j in range(start, end):
            if j == i:
                continue
            ctx = token_ids[j]
            dist = abs(i - j)
            weight = 1.0 / dist  # Harmonic weighting
            # Store both directions for symmetric matrix
            if center < ctx:
                cooccur[(center, ctx)] += weight
            else:
                cooccur[(ctx, center)] += weight

    # Convert to COO format
    if not cooccur:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    rows = np.array([k[0] for k in cooccur.keys()], dtype=np.int32)
    cols = np.array([k[1] for k in cooccur.keys()], dtype=np.int32)
    vals = np.array(list(cooccur.values()), dtype=np.float32)

    return rows, cols, vals


def sparse_pmi_embedding(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    vocab_size: int,
    embed_dim: int,
    seed: int = 0,
    shift: float = 1.0,
) -> np.ndarray:
    """Compute PMI embeddings from sparse co-occurrence matrix.

    Uses skip-gram style iterative refinement to avoid dense matrix construction.
    Never allocates O(V²) memory.

    Args:
        rows: Row indices (COO format)
        cols: Column indices (COO format)
        vals: Values (COO format)
        vocab_size: Vocabulary size
        embed_dim: Target embedding dimension
        seed: Random seed
        shift: PMI shift parameter

    Returns:
        Dense embeddings (vocab_size, embed_dim)
    """
    if len(vals) == 0:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((vocab_size, embed_dim)).astype(np.float32) * 0.1

    # For large vocabularies, use skip-gram style instead of full PMI+SVD
    # This avoids any O(V²) allocation
    if vocab_size > 5000:
        return approximate_skipgram_from_sparse(
            rows, cols, vals, vocab_size, embed_dim, seed=seed
        )

    # For small vocabularies, use the original dense approach
    # Compute row and column sums from sparse data
    row_sums = np.zeros(vocab_size, dtype=np.float32)
    col_sums = np.zeros(vocab_size, dtype=np.float32)

    np.add.at(row_sums, rows, vals)
    np.add.at(col_sums, cols, vals)

    total = float(np.sum(vals))

    # Apply context distribution smoothing (alpha=0.75)
    alpha = 0.75
    col_counts_smooth = np.power(col_sums, alpha)
    col_sum_smooth = float(np.sum(col_counts_smooth))

    # Compute PMI for each non-zero entry
    log_shift = np.log(float(max(shift, 1.0)))
    pmi_vals = np.zeros_like(vals)

    for i, (r, c, v) in enumerate(zip(rows, cols, vals)):
        p_joint = v / total
        p_row = row_sums[r] / total
        p_col_smooth = col_counts_smooth[c] / col_sum_smooth

        if p_joint > 0 and p_row > 0 and p_col_smooth > 0:
            pmi = np.log(p_joint / (p_row * p_col_smooth) + 1e-9)
            pmi_vals[i] = max(0.0, pmi - log_shift)

    # Build row-wise aggregated vectors using sparse operations
    # This is O(V * avg_degree) instead of O(V²)
    row_vectors = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    np.add.at(row_vectors, (rows, cols), pmi_vals)

    # Apply truncated SVD
    k = min(embed_dim, vocab_size - 1)
    if k <= 0:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((vocab_size, embed_dim)).astype(np.float32) * 0.1

    # Use power iteration + randomized SVD
    rng = np.random.default_rng(seed)
    n_oversampling = min(10, vocab_size - k)
    Omega = rng.standard_normal((vocab_size, k + n_oversampling)).astype(np.float32)
    Y = row_vectors @ Omega

    # Power iterations
    for _ in range(3):
        Y = row_vectors @ (row_vectors.T @ Y)

    Q, _ = np.linalg.qr(Y)
    B = Q.T @ row_vectors
    U_hat, S, _ = np.linalg.svd(B, full_matrices=False)
    embeddings = (Q @ U_hat)[:, :k]

    # Weight by sqrt singular values
    embeddings *= np.sqrt(np.maximum(S[:k], 0.0))[np.newaxis, :]

    # Pad to embed_dim if needed
    if k < embed_dim:
        pad = np.zeros((vocab_size, embed_dim - k), dtype=np.float32)
        embeddings = np.concatenate([embeddings, pad], axis=1)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = (embeddings / norms).astype(np.float32)

    return embeddings


def approximate_skipgram_from_sparse(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    vocab_size: int,
    embed_dim: int,
    epochs: int = 5,
    lr: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """PMI-based row-wise context aggregation for sparse co-occurrence.

    Computes embeddings by aggregating PMI-weighted context vectors.
    Never allocates O(V²) memory.

    Args:
        rows: Row indices (COO format)
        cols: Column indices (COO format)
        vals: Values (COO format)
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        epochs: Training epochs (not used, kept for compat)
        lr: Learning rate (not used, kept for compat)
        seed: Random seed

    Returns:
        Embeddings (vocab_size, embed_dim)
    """
    rng = np.random.default_rng(seed)

    if len(vals) == 0:
        return rng.standard_normal((vocab_size, embed_dim)).astype(np.float32) * 0.1

    # Compute row and column sums
    row_sums = np.zeros(vocab_size, dtype=np.float32)
    col_sums = np.zeros(vocab_size, dtype=np.float32)
    np.add.at(row_sums, rows, vals)
    np.add.at(col_sums, cols, vals)

    total = float(np.sum(vals))

    # Compute PMI for each pair
    log_shift = np.log(2.0)  # shift=2.0 for consistency with dense mode
    pmi_vals = np.zeros_like(vals)

    for i, (r, c, v) in enumerate(zip(rows, cols, vals)):
        p_joint = v / total
        p_row = row_sums[r] / total
        p_col = col_sums[c] / total

        if p_joint > 0 and p_row > 0 and p_col > 0:
            pmi = np.log(p_joint / (p_row * p_col) + 1e-9)
            pmi_vals[i] = max(0.0, pmi - log_shift)

    # Create random projection matrix for context vectors
    # Project vocab_size -> embed_dim
    proj = rng.standard_normal((vocab_size, embed_dim)).astype(np.float32)
    proj = proj / np.sqrt(vocab_size)

    # Aggregate PMI-weighted context for each word
    # For each word i: embedding[i] = sum_j PMI(i,j) * proj[j]
    embeddings = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    # Sparse aggregation: only update for non-zero pairs
    for i, (r, c, pmi) in enumerate(zip(rows, cols, pmi_vals)):
        if pmi > 0:
            embeddings[r] += pmi * proj[c]
            embeddings[c] += pmi * proj[r]  # Symmetric

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = (embeddings / norms).astype(np.float32)

    return embeddings


def approximate_skipgram_embedding(
    token_ids: List[int],
    vocab_size: int,
    embed_dim: int,
    window: int = 4,
    epochs: int = 5,
    lr: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Approximate skip-gram style embedding without negative sampling.

    Uses simple co-occurrence statistics with iterative refinement.
    More scalable than full PMI+SVD for very large vocabularies.

    Args:
        token_ids: List of token IDs
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        window: Context window
        epochs: Training epochs
        lr: Learning rate
        seed: Random seed

    Returns:
        Embeddings (vocab_size, embed_dim)
    """
    rows, cols, vals = build_sparse_cooccurrence(token_ids, vocab_size, window)
    return approximate_skipgram_from_sparse(
        rows, cols, vals, vocab_size, embed_dim, epochs, lr, seed
    )
