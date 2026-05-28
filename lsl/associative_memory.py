"""SparseAssociativeMemory — Phase 1: Pattern Completion (G1.5).

Mechanism (Hopfield-inspired, sparse-optimised):
  - Storage: M[active_i, active_j] += 1.0 for all co-active pairs
    (only the k×k intersection, never the full d×d matrix)
  - Retrieval: score = M @ partial → top-k winners = completed pattern
  - Consolidation: prune connections below threshold (forgetting)

Constraints enforced (from master plan):
  - Binary {0,1} outputs only
  - Update only at active bit intersections (sparse outer product)
  - No global backprop — Hebbian local rule only
  - Pattern k must be fixed (same as SemanticSDREncoder.k)

Performance:
  - Storage: O(k²) per pattern instead of O(d²)
  - Retrieval: O(n_stored × k + d) instead of O(d²)
"""
import numpy as np
from typing import Optional


class SparseAssociativeMemory:
    """Sparse Hopfield-like associative memory for SDR pattern completion.

    Stores patterns via Hebbian learning restricted to active-bit intersections.
    Retrieves via column-sum scoring then top-k selection.

    Args:
        sdr_dim: Dimension of SDR vectors (e.g. 1024)
        k: Number of active bits per pattern (e.g. 20)
        learning_rate: Strength per observation (default 1.0)
        decay_rate: Weight decay per consolidation call (default 0.99)
    """

    def __init__(
        self,
        sdr_dim: int,
        k: int = 20,
        learning_rate: float = 1.0,
        decay_rate: float = 0.99,
    ):
        self.sdr_dim = int(sdr_dim)
        self.k = int(k)
        self.lr = float(learning_rate)
        self.decay = float(decay_rate)

        # Association weight matrix — sparse in practice but stored dense
        # for simplicity. For sdr_dim=1024, this is 1024×1024 × 4B = 4MB.
        self.M = np.zeros((self.sdr_dim, self.sdr_dim), dtype=np.float32)

        # Diagonal must be zero (no self-connections — Hopfield rule)
        # Enforced in observe() after each update.
        self._pattern_count = 0

    # ------------------------------------------------------------------
    # Storage (Hebbian local rule)
    # ------------------------------------------------------------------

    def observe(self, sdr: np.ndarray, strength: float = 1.0) -> None:
        """Store a pattern via Hebbian outer product at active indices.

        Args:
            sdr: Binary array (sdr_dim,) — must have exactly k active bits
            strength: Multiplier for this pattern's learning strength
        """
        sdr = np.asarray(sdr, dtype=np.float32)
        active = np.where(sdr > 0.5)[0]

        if len(active) == 0:
            return

        # Sparse outer product: only update k×k submatrix
        # M[active_i, active_j] += lr * strength  for all (i, j) pairs
        update_strength = float(self.lr * strength)
        self.M[np.ix_(active, active)] += update_strength

        # Zero diagonal (no self-excitation)
        np.fill_diagonal(self.M, 0.0)

        self._pattern_count += 1

    # ------------------------------------------------------------------
    # Retrieval (pattern completion)
    # ------------------------------------------------------------------

    def complete(
        self,
        partial: np.ndarray,
        k: Optional[int] = None,
        max_iterations: int = 5,
    ) -> np.ndarray:
        """Complete a partial SDR pattern via iterative energy minimization.

        Args:
            partial: Partial binary array (sdr_dim,) — some bits may be zeroed
            k: Number of bits to activate in completed pattern (defaults to self.k)
            max_iterations: Convergence iterations

        Returns:
            Completed binary SDR of shape (sdr_dim,) with exactly k active bits
        """
        k = int(k) if k is not None else self.k
        s = np.asarray(partial, dtype=np.float32)

        for _ in range(max_iterations):
            active = np.where(s > 0.5)[0]

            if len(active) == 0:
                # No signal — return zeros
                return np.zeros(self.sdr_dim, dtype=np.float32)

            # Score each bit: sum of association weights from current active bits
            # Sparse: only sum columns of active bits
            scores = np.sum(self.M[:, active], axis=1)  # (sdr_dim,)

            # Select top-k bits as new state
            k_clamped = min(k, self.sdr_dim)
            top_k = np.argpartition(scores, -k_clamped)[-k_clamped:]

            s_new = np.zeros(self.sdr_dim, dtype=np.float32)
            s_new[top_k] = 1.0

            # Check convergence
            if np.array_equal(s_new, s):
                break
            s = s_new

        return s

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def consolidate(self, threshold: float = 0.5) -> int:
        """Prune weak connections (forgetting) and return pruned count.

        Args:
            threshold: Connections below this value are zeroed

        Returns:
            Number of pruned connections
        """
        weak = np.abs(self.M) < threshold
        n_pruned = int(weak.sum())
        self.M[weak] = 0.0
        np.fill_diagonal(self.M, 0.0)
        return n_pruned

    def decay_weights(self) -> None:
        """Apply exponential decay to all association weights."""
        self.M *= self.decay
        np.fill_diagonal(self.M, 0.0)

    def reset(self) -> None:
        """Reset all associations."""
        self.M.fill(0.0)
        self._pattern_count = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pattern_count(self) -> int:
        return self._pattern_count

    def mean_connection_strength(self) -> float:
        """Mean weight of non-zero connections."""
        nonzero = self.M[self.M > 0]
        return float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0

    def theoretical_capacity(self) -> int:
        """Hopfield-like capacity estimate: ~0.14 × n / log(n) patterns.

        For SDR Hopfield: capacity ≈ n / (k² / n × log(n/k))
        """
        n = self.sdr_dim
        k = self.k
        if k == 0:
            return 0
        # SDR capacity formula (Sommer & Wennekers 2001)
        capacity = int(n / max(1.0, float(k**2) / n * np.log(float(n) / k + 1)))
        return capacity

    def completion_accuracy(
        self,
        patterns: np.ndarray,
        mask_fraction: float = 0.5,
        seed: int = 0,
    ) -> float:
        """Test pattern completion accuracy by masking fraction of bits.

        Args:
            patterns: (N, sdr_dim) binary patterns previously stored
            mask_fraction: Fraction of active bits to zero out
            seed: RNG seed for reproducibility

        Returns:
            Mean fraction of active bits correctly recovered
        """
        rng = np.random.default_rng(seed)
        accuracies = []

        for pattern in patterns:
            active = np.where(pattern > 0.5)[0]
            if len(active) == 0:
                continue

            # Mask out mask_fraction of active bits
            n_mask = max(1, int(len(active) * mask_fraction))
            mask_idx = rng.choice(active, size=n_mask, replace=False)
            partial = pattern.copy()
            partial[mask_idx] = 0.0

            # Complete and compare
            completed = self.complete(partial, k=self.k)
            completed_active = set(np.where(completed > 0.5)[0])
            original_active  = set(active.tolist())

            overlap = len(completed_active & original_active)
            accuracy = overlap / max(1, len(original_active))
            accuracies.append(accuracy)

        return float(np.mean(accuracies)) if accuracies else 0.0


# ---------------------------------------------------------------------------
# Legacy class alias
# ---------------------------------------------------------------------------

class AssociativeMemory(SparseAssociativeMemory):
    """Legacy alias for SparseAssociativeMemory (backward compat).

    Old API used store()/retrieve() — mapped to observe()/complete().
    """

    def __init__(self, dim: int, capacity: int = 1000, seed: Optional[int] = None):
        # Guess k from old default sparsity ~2%
        k = max(1, int(dim * 0.02))
        super().__init__(sdr_dim=dim, k=k)

    def store(self, pattern: np.ndarray, strength: float = 1.0) -> None:
        """Legacy: alias for observe()."""
        self.observe(pattern, strength=strength)

    def retrieve(self, partial_pattern: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """Legacy: alias for complete()."""
        return self.complete(partial_pattern, max_iterations=max_iterations)

    def energy(self, pattern: np.ndarray) -> float:
        """Hopfield energy of a pattern: E = -0.5 * x^T M x"""
        x = np.asarray(pattern, dtype=np.float32)
        return float(-0.5 * x @ self.M @ x)
