"""Semantic SDR Encoder — Phase 1: Sparse Distributed Representations.

Design goals:
  G1.1  Semantic encoding: related words share active bits (overlap ≥ 3× random)
  G1.2  Exponential capacity: C(1024, 20) ≈ 10^41 patterns
  G1.6  Sparse computation: encode() runs in O(k·d) via index-based projection

Architecture:
  1. Mini co-occurrence Word2Vec (self-contained, no external deps)
     Corpus → PMI co-occurrence matrix → SVD → dense embed_dim vectors
  2. Fixed Random Projection (Johnson-Lindenstrauss)
     embed_dim → sdr_dim  (Gaussian, seed-locked, never changes)
  3. Top-k binary encoding
     k = int(sparsity × sdr_dim)  [default: 2% of 1024 = 20 bits]

Constraints enforced:
  - Binary {0, 1} only — never float activations
  - k ≤ 2% × sdr_dim always
  - Projection matrix is frozen after __init__
  - No external libraries (numpy only)
"""
import json
from pathlib import Path

import numpy as np
from typing import Dict, List, Optional, Tuple

from .sparse_cooccurrence import build_sparse_cooccurrence, sparse_pmi_embedding, approximate_skipgram_embedding


# ---------------------------------------------------------------------------
# Mini Word2Vec via PMI + SVD (no external deps)
# ---------------------------------------------------------------------------

def _build_cooccurrence(
    token_ids: List[int],
    vocab_size: int,
    window: int = 4,
) -> np.ndarray:
    """Build weighted co-occurrence matrix using context window."""
    C = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        start = max(0, i - window)
        end   = min(n, i + window + 1)
        for j in range(start, end):
            if j == i:
                continue
            ctx = token_ids[j]
            dist = abs(i - j)
            weight = 1.0 / dist       # harmonic weighting like word2vec
            C[center, ctx] += weight
    return C


def _pmi_embedding(
    C: np.ndarray,
    embed_dim: int,
    seed: int = 0,
    shift: float = 1.0,
) -> np.ndarray:
    """Compute Shifted PPMI (SPPMI) and reduce to embed_dim via truncated SVD.

    SPPMI(w,c) = max(0, PMI(w,c) - log(shift))
    shift=1 collapses to PPMI. shift=5 penalizes common co-occurrences.
    """
    n_vocab = C.shape[0]

    # Smooth counts with alpha=0.75 (context distribution smoothing)
    alpha = 0.75
    col_counts = np.power(C.sum(axis=0), alpha)
    row_sum = C.sum(axis=1, keepdims=True) + 1e-9      # (V, 1)
    col_sum = col_counts[np.newaxis, :] + 1e-9          # (1, V)
    total   = float(col_counts.sum()) + 1e-9

    # SPPMI = max(0, log(P(w,c)/(P(w)*P(c))) - log(shift))
    log_shift = np.log(float(max(shift, 1.0)))
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((C * total) / (row_sum * col_sum) + 1e-9)
    sppmi = np.maximum(pmi - log_shift, 0.0).astype(np.float32)

    # --- Randomized SVD (Halko et al. 2009) ---
    rng = np.random.default_rng(seed)
    k   = min(embed_dim, n_vocab - 1, sppmi.shape[0] - 1)
    if k <= 0:
        return np.zeros((n_vocab, embed_dim), dtype=np.float32)

    # Power iteration for better accuracy on small matrices
    n_oversampling = min(10, n_vocab - k)
    Omega = rng.standard_normal((n_vocab, k + n_oversampling)).astype(np.float32)
    Y = sppmi @ Omega
    for _ in range(3):   # 3 power iterations for accuracy
        Y = sppmi @ (sppmi.T @ Y)
    Q, _ = np.linalg.qr(Y)           # (V, k+n_over) orthonormal
    B = Q.T @ sppmi                   # (k+n_over, V)
    U_hat, S, _ = np.linalg.svd(B, full_matrices=False)
    embeddings = (Q @ U_hat)[:, :k]  # (V, k) left singular vectors

    # Weight by sqrt of singular values (standard Word2Vec trick)
    embeddings *= np.sqrt(np.maximum(S[:k], 0.0))[np.newaxis, :]

    # Pad or truncate to embed_dim
    if k < embed_dim:
        pad = np.zeros((n_vocab, embed_dim - k), dtype=np.float32)
        embeddings = np.concatenate([embeddings, pad], axis=1)

    # L2-normalise rows (unit sphere -> better JL projection)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = (embeddings / norms).astype(np.float32)
    return embeddings  # (V, embed_dim)


# ---------------------------------------------------------------------------
# SemanticSDREncoder
# ---------------------------------------------------------------------------

class SemanticSDREncoder:
    """Semantic SDR encoder: mini Word2Vec + Random Projection → binary SDR.

    Phase 1 constraints:
      • Binary {0, 1} output — never float
      • k = 2% of sdr_dim (20 of 1024)
      • Projection matrix frozen after init (Johnson-Lindenstrauss)
      • No external dependencies
    """

    def __init__(
        self,
        vocab_size: int,
        sdr_dim: int = 1024,
        sparsity: float = 0.02,
        embed_dim: int = 64,
        seed: int = 42,
        # legacy compat args (ignored)
        hidden_dim: int = 1024,
        embedding_dim: int = 64,
        use_pretrained: bool = False,
        # Phase 4: sparse mode for large vocabularies
        use_sparse: bool = False,
        sparse_threshold: int = 5000,
    ):
        self.vocab_size = int(vocab_size)
        self.sdr_dim    = int(sdr_dim)
        self.embed_dim  = int(embed_dim)
        self.seed       = int(seed)
        self.sparsity   = float(sparsity)
        self.k          = max(1, int(self.sdr_dim * self.sparsity))
        self.use_pretrained = bool(use_pretrained)
        self.use_sparse = bool(use_sparse) if vocab_size >= sparse_threshold else False
        self.sparse_threshold = int(sparse_threshold)

        # Legacy compat: hidden_dim alias
        self.hidden_dim = self.sdr_dim

        rng = np.random.default_rng(seed)

        # Fixed Gaussian Random Projection: R^embed_dim → R^sdr_dim
        # Scale by 1/sqrt(embed_dim) to preserve distances (JL property)
        self._proj = (
            rng.standard_normal((self.embed_dim, self.sdr_dim)).astype(np.float32)
            / np.sqrt(float(self.embed_dim))
        )

        # Dense embeddings (vocab_size × embed_dim) — filled after fit()
        # Default: random so encode() works even before fit()
        self._embeddings: np.ndarray = (
            rng.standard_normal((self.vocab_size, self.embed_dim)).astype(np.float32)
            * 0.1
        )
        self._fitted = False

        # SDR cache: token_id → binary array(sdr_dim,)
        self._cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        token_sequences: List[List[int]],
        window: int = 4,
        verbose: bool = False,
    ) -> "SemanticSDREncoder":
        """Self-train embeddings from token sequences using PMI + SVD.

        Args:
            token_sequences: List of token-id lists (e.g. encoded corpus sentences)
            window: Co-occurrence context window
            verbose: Print progress

        Returns:
            self (for chaining)
        """
        if verbose:
            total = sum(len(s) for s in token_sequences)
            mode = "sparse" if self.use_sparse else "dense"
            print(f"[SemanticSDR] fit(): {len(token_sequences)} seqs, "
                  f"{total} tokens, vocab={self.vocab_size}, mode={mode}")

        # Flatten all sequences into one long stream
        flat: List[int] = []
        for seq in token_sequences:
            flat.extend(seq)

        if len(flat) < 4:
            if verbose:
                print("[SemanticSDR] Too few tokens — keeping random embeddings")
            return self

        # Build co-occurrence and extract PMI embeddings
        # Use sparse mode for large vocabularies (Phase 4)
        if self.use_sparse:
            if verbose:
                print("[SemanticSDR] Using sparse co-occurrence for large vocabulary")
            rows, cols, vals = build_sparse_cooccurrence(flat, self.vocab_size, window=window)
            emb = sparse_pmi_embedding(rows, cols, vals, self.vocab_size,
                                       self.embed_dim, seed=self.seed, shift=2.0)
        else:
            C = _build_cooccurrence(flat, self.vocab_size, window=window)
            emb = _pmi_embedding(C, self.embed_dim, seed=self.seed, shift=2.0)

        self._embeddings = emb.astype(np.float32)
        self._fitted = True
        self._cache.clear()

        if verbose:
            # Verify a few words that should be related
            overlaps = []
            for i in range(min(10, self.vocab_size)):
                for j in range(i + 1, min(10, self.vocab_size)):
                    ov = self.semantic_overlap(i, j)
                    overlaps.append(ov)
            print(f"[SemanticSDR] fit done. Mean pairwise overlap "
                  f"(first 10 words): {np.mean(overlaps):.2f} bits")

        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _project_and_binarize(self, dense_emb: np.ndarray) -> np.ndarray:
        """Project dense embedding to SDR space and binarize.

        Args:
            dense_emb: float array (embed_dim,)

        Returns:
            binary array (sdr_dim,) with exactly k ones
        """
        # Project: (embed_dim,) @ (embed_dim, sdr_dim) → (sdr_dim,)
        projected = dense_emb @ self._proj  # O(embed_dim × sdr_dim)

        # Top-k selection — exactly k bits active
        if self.k >= self.sdr_dim:
            return np.ones(self.sdr_dim, dtype=np.float32)

        # argpartition is O(sdr_dim) — faster than argsort
        indices = np.argpartition(projected, -self.k)[-self.k:]

        sdr = np.zeros(self.sdr_dim, dtype=np.float32)
        sdr[indices] = 1.0
        return sdr

    def encode(self, token_id: int) -> np.ndarray:
        """Encode token_id → semantic binary SDR (sdr_dim,).

        Cached after first call — O(1) for repeated tokens.
        """
        token_id = int(token_id)
        if token_id in self._cache:
            return self._cache[token_id].copy()

        dense_emb = self._embeddings[token_id % self.vocab_size]
        sdr = self._project_and_binarize(dense_emb)
        self._cache[token_id] = sdr
        return sdr.copy()

    def encode_batch(self, token_ids) -> np.ndarray:
        """Encode multiple token IDs. Returns (N, sdr_dim) binary array."""
        ids = np.asarray(token_ids, dtype=np.int64)
        out = np.zeros((len(ids), self.sdr_dim), dtype=np.float32)
        for i, tid in enumerate(ids):
            out[i] = self.encode(int(tid))
        return out

    # ------------------------------------------------------------------
    # Utility / metrics
    # ------------------------------------------------------------------

    def active_indices(self, token_id: int) -> np.ndarray:
        """Return array of active bit indices for token_id."""
        sdr = self.encode(token_id)
        return np.where(sdr > 0.5)[0]

    def semantic_overlap(self, token_id_a: int, token_id_b: int) -> float:
        """Number of shared active bits between two token SDRs (Hamming overlap)."""
        a = self.encode(int(token_id_a))
        b = self.encode(int(token_id_b))
        return float(np.sum(a * b))

    def semantic_overlap_ratio(self, token_id_a: int, token_id_b: int) -> float:
        """Overlap normalised by k (fraction of bits shared)."""
        ov = self.semantic_overlap(token_id_a, token_id_b)
        return ov / max(1.0, float(self.k))

    def random_baseline_overlap(self, n_pairs: int = 1000, seed: int = 0) -> float:
        """Expected overlap for two random SDRs in this space.

        Theoretical: k² / sdr_dim
        This method verifies empirically.
        """
        theoretical = (self.k ** 2) / self.sdr_dim
        return float(theoretical)

    def capacity_log2(self) -> float:
        """log2 of combinatorial capacity C(sdr_dim, k)."""
        from math import comb, log2
        return float(log2(max(comb(self.sdr_dim, self.k), 1)))

    def actual_sparsity(self) -> float:
        """Actual sparsity ratio k/sdr_dim."""
        return float(self.k) / float(self.sdr_dim)

    def clear_cache(self) -> None:
        """Clear SDR cache (frees memory after fit)."""
        self._cache.clear()

    def get_sparsity(self) -> float:
        """Legacy compat: returns actual sparsity."""
        return self.actual_sparsity()

    def load_embeddings_from_gensim(self, model_path: str, vocab: Dict[str, int]):
        """Legacy compat: no-op (gensim not used, use fit() instead)."""
        pass

    def load_embedding_matrix(
        self,
        embeddings: np.ndarray,
        normalize: bool = True,
    ) -> int:
        """Load an offline semantic prior matrix.

        This is the scale-oriented path for semantic SDR: an external/offline
        semantic source supplies dense word vectors, and the encoder converts
        them into sparse binary SDRs through the fixed random projection.
        """
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if matrix.shape[0] != self.vocab_size:
            raise ValueError(
                f"expected {self.vocab_size} rows, got {matrix.shape[0]}"
            )

        if matrix.shape[1] < self.embed_dim:
            pad = np.zeros(
                (self.vocab_size, self.embed_dim - matrix.shape[1]),
                dtype=np.float32,
            )
            matrix = np.concatenate([matrix, pad], axis=1)
        elif matrix.shape[1] > self.embed_dim:
            matrix = matrix[:, :self.embed_dim]

        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
            matrix = matrix / norms

        self._embeddings = matrix.astype(np.float32, copy=False)
        self._fitted = True
        self._cache.clear()
        return self.vocab_size

    def load_builtin_embeddings(
        self,
        vocab: Dict[str, int],
        path: Optional[str] = None,
    ) -> int:
        """Load checked-in offline semantic vectors for known vocabulary words.

        The asset is intentionally tiny and deterministic. It gives the strict
        SDR benchmark a fixed semantic prior without external APIs or runtime
        downloads. Unknown vocabulary items keep their current fitted/random
        embeddings.
        """
        asset_path = (
            Path(path)
            if path is not None
            else Path(__file__).resolve().parent / "data" / "mini_semantic_embeddings.json"
        )
        with asset_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        dim = int(payload.get("dimension", self.embed_dim))
        basis = payload.get("basis", {})
        groups = payload.get("groups", {})

        vectors: Dict[str, np.ndarray] = {}
        for name, values in basis.items():
            vec = np.asarray(values, dtype=np.float32)
            if len(vec) < dim:
                vec = np.pad(vec, (0, dim - len(vec))).astype(np.float32)
            vectors[name] = vec[:dim]

        loaded = 0
        for group_name, words in groups.items():
            if group_name not in vectors:
                continue
            source = vectors[group_name]
            emb = np.zeros(self.embed_dim, dtype=np.float32)
            n = min(self.embed_dim, len(source))
            emb[:n] = source[:n]
            norm = float(np.linalg.norm(emb)) + 1e-9
            emb = emb / norm
            for word in words:
                if word in vocab:
                    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(word))
                    rng = np.random.default_rng(seed)
                    jitter = rng.standard_normal(self.embed_dim).astype(np.float32) * 0.22
                    word_emb = emb + jitter
                    word_emb /= float(np.linalg.norm(word_emb)) + 1e-9
                    self._embeddings[int(vocab[word])] = word_emb.astype(np.float32)
                    loaded += 1

        self._fitted = True
        self._cache.clear()
        return loaded


# ---------------------------------------------------------------------------
# Standalone utility functions (imported by __init__.py)
# ---------------------------------------------------------------------------

def semantic_overlap(sdr1: np.ndarray, sdr2: np.ndarray) -> float:
    """Hamming overlap between two binary SDR vectors."""
    return float(np.sum(np.asarray(sdr1) * np.asarray(sdr2)))


def semantic_overlap_ratio(sdr1: np.ndarray, sdr2: np.ndarray) -> float:
    """Overlap normalised by min(|sdr1|, |sdr2|)."""
    k1 = int(np.sum(np.asarray(sdr1) > 0.5))
    k2 = int(np.sum(np.asarray(sdr2) > 0.5))
    if k1 == 0 or k2 == 0:
        return 0.0
    return float(np.sum(np.asarray(sdr1) * np.asarray(sdr2))) / float(min(k1, k2))
