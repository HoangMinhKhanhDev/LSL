"""Sparse Distributed Representations (SDR) primitives.

Provides deterministic SDR encoding, binarization, overlap metrics, and capacity
calculations for living synapse models.
"""
import numpy as np


class SDREncoder:
    """Deterministic SDR encoder with top-k sparse binary codes.

    Converts dense vectors to sparse binary activations where exactly k bits are
    active (1) and the rest are inactive (0). Uses top-k selection by magnitude.
    """
    def __init__(self, dim, sparsity=0.2, seed=None):
        self.dim = int(dim)
        self.sparsity = float(sparsity)
        self.k = max(1, int(self.dim * self.sparsity))
        self.rng = np.random.default_rng(seed)

    def encode(self, x):
        """Convert dense vector to sparse binary code.

        Args:
            x: Dense vector of shape (dim,)

        Returns:
            Sparse binary vector of shape (dim,) with exactly k active bits.
        """
        x = np.asarray(x, dtype=np.float32)
        # Select top-k by absolute magnitude
        if self.k >= self.dim:
            return np.ones(self.dim, dtype=np.float32)
        thresh = np.partition(np.abs(x), -self.k)[-self.k]
        mask = (np.abs(x) >= thresh).astype(np.float32)
        # Ensure exactly k active (handle ties)
        if mask.sum() > self.k:
            # Keep only top-k among those at threshold
            indices = np.where(mask)[0]
            values = np.abs(x[indices])
            top_k_indices = indices[np.argpartition(values, -self.k)[-self.k:]]
            mask[:] = 0.0
            mask[top_k_indices] = 1.0
        elif mask.sum() < self.k:
            # If too few, add random from inactive
            inactive = np.where(mask < 0.5)[0]
            needed = self.k - int(mask.sum())
            if len(inactive) >= needed:
                add = self.rng.choice(inactive, size=needed, replace=False)
                mask[add] = 1.0
        return mask

    def encode_batch(self, X):
        """Encode batch of vectors.

        Args:
            X: Dense vectors of shape (batch, dim)

        Returns:
            Sparse binary codes of shape (batch, dim)
        """
        return np.stack([self.encode(x) for x in X])


def hamming_overlap(a, b):
    """Compute Hamming overlap between two binary vectors.

    Args:
        a, b: Binary vectors (0/1) of same shape

    Returns:
        Number of positions where both are 1 (intersection)
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sum(a * b))


def pairwise_overlap_matrix(codes):
    """Compute pairwise overlap matrix for a set of SDR codes.

    Args:
        codes: Binary codes of shape (n, dim)

    Returns:
        Overlap matrix of shape (n, n) where entry [i,j] is overlap between codes[i] and codes[j]
    """
    codes = np.asarray(codes, dtype=np.float32)
    return (codes @ codes.T).astype(np.float32)


def combinatorial_capacity(dim, k):
    """Compute combinatorial capacity C(dim, k) = binomial(dim, k).

    This is the number of distinct sparse binary codes with exactly k active bits.

    Args:
        dim: Total dimension
        k: Number of active bits per code

    Returns:
        Number of distinct codes (may be large)
    """
    from math import comb
    return comb(int(dim), int(k))


def log2_capacity(dim, k):
    """Compute log2 of combinatorial capacity.

    Useful for comparing to vocabulary size in bits.

    Args:
        dim: Total dimension
        k: Number of active bits per code

    Returns:
        log2(C(dim, k))
    """
    cap = float(combinatorial_capacity(dim, k))
    return np.log2(max(cap, 1.0))


def sparsity_ratio(code):
    """Compute actual sparsity ratio of a binary code.

    Args:
        code: Binary vector

    Returns:
        Fraction of active bits (0 to 1)
    """
    code = np.asarray(code, dtype=np.float32)
    return float(np.mean(code))


def active_indices(code):
    """Get indices of active bits in a binary code.

    Args:
        code: Binary vector

    Returns:
        List of indices where code == 1
    """
    code = np.asarray(code, dtype=np.float32)
    return [i for i, v in enumerate(code) if v > 0.5]


def capacity_stats(dim, k):
    """Compute capacity statistics for given dimensions.

    Args:
        dim: Total dimension
        k: Number of active bits per code

    Returns:
        Dict with capacity, log2_capacity, log10_capacity
    """
    cap = float(combinatorial_capacity(dim, k))
    return {
        "capacity": cap,
        "log2_capacity": float(np.log2(max(cap, 1.0))),
        "log10_capacity": float(np.log10(max(cap, 1.0))),
    }
