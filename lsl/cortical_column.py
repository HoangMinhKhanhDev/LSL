"""Cortical Column Sequence Memory - HTM-like temporal memory.

Implements cortical column sequence learning with:
- Mini-columns: one column per token
- Cells per column: sparse active cells representing context
- Predicted cells: cells predicted by learned temporal segments
- Burst firing: unexpected input triggers burst + immediate learning
- Suppression: expected input activates only predicted cells (low cost)
- Temporal segments: local transition memory (no global backprop)

This matches cortical column dynamics:
- Correct prediction → silence/suppression
- Wrong prediction → burst firing → immediate learning
- New input → compare against all known sequences
"""
import numpy as np

from . import sparse_native
from .sparse_native import NATIVE_AVAILABLE


class CorticalColumnSequenceMemory:
    """Cortical column sequence memory for temporal pattern learning.

    Each token has a column with multiple cells. Active cells represent
    the token in context. Temporal segments connect cells across time steps.
    """
    def __init__(self, vocab_size, cells_per_column=100, sparsity=0.02, seed=42):
        """Initialize cortical column sequence memory.

        Args:
            vocab_size: Number of tokens in vocabulary
            cells_per_column: Number of cells per column (default 100, cortical column size)
            sparsity: Fraction of cells active per column (default 0.02)
            seed: Random seed for reproducibility
        """
        self.vocab_size = int(vocab_size)
        self.cells_per_column = int(cells_per_column)
        self.sparsity = float(sparsity)
        self.rng = np.random.default_rng(seed)

        # Number of active cells per column
        self.k = max(1, int(self.cells_per_column * self.sparsity))

        # Column state: for each token, which cells are active
        # Shape: (vocab_size, cells_per_column)
        self.column_active = np.zeros((vocab_size, cells_per_column), dtype=np.float32)

        # Temporal segments: transitions from (prev_token, prev_cell) to (next_token, next_cell)
        # Stored as a sparse dictionary for efficiency
        self.temporal_segments = {}  # key: (prev_token, prev_cell), value: {(next_token, next_cell): strength}

        # Prediction state
        self.predicted_cells = set()  # Set of (token, cell) tuples predicted for next step
        self.predicted_tokens = set()
        self.active_cells = set()     # Set of (token, cell) tuples currently active
        self._prev_active_cells = set()  # Track previous active cells internally
        self._context_winners = {}
        self.max_context = 4
        self.recent_tokens = []
        self.context_transitions = {}
        self.last_prediction_confidence = 0.0
        self.last_pruned_segments = 0
        self.last_pruned_context_keys = 0
        self.last_pruned_targets = 0

        # Metrics
        self.burst_count = 0
        self.suppression_count = 0
        self.total_steps = 0
        self.segment_count = 0
        self.native_topk_calls = 0
        self.native_topk_success = 0
        self._ensure_backward_compatibility()

    def _ensure_backward_compatibility(self) -> None:
        if not hasattr(self, "vocab_size"):
            self.vocab_size = 0
        if not hasattr(self, "cells_per_column"):
            self.cells_per_column = 100
        if not hasattr(self, "sparsity"):
            self.sparsity = 0.02
        if not hasattr(self, "rng"):
            self.rng = np.random.default_rng(42)
        if not hasattr(self, "k"):
            self.k = max(1, int(self.cells_per_column * self.sparsity))
        if not hasattr(self, "column_active"):
            self.column_active = np.zeros((self.vocab_size, self.cells_per_column), dtype=np.float32)
        if not hasattr(self, "temporal_segments"):
            self.temporal_segments = {}
        if not hasattr(self, "predicted_cells"):
            self.predicted_cells = set()
        if not hasattr(self, "predicted_tokens"):
            self.predicted_tokens = set()
        if not hasattr(self, "active_cells"):
            self.active_cells = set()
        if not hasattr(self, "_prev_active_cells"):
            self._prev_active_cells = set()
        if not hasattr(self, "_context_winners"):
            self._context_winners = {}
        if not hasattr(self, "max_context"):
            self.max_context = 4
        if not hasattr(self, "recent_tokens"):
            self.recent_tokens = []
        if not hasattr(self, "context_transitions"):
            self.context_transitions = {}
        if not hasattr(self, "last_prediction_confidence"):
            self.last_prediction_confidence = 0.0
        if not hasattr(self, "last_pruned_segments"):
            self.last_pruned_segments = 0
        if not hasattr(self, "last_pruned_context_keys"):
            self.last_pruned_context_keys = 0
        if not hasattr(self, "last_pruned_targets"):
            self.last_pruned_targets = 0
        if not hasattr(self, "burst_count"):
            self.burst_count = 0
        if not hasattr(self, "suppression_count"):
            self.suppression_count = 0
        if not hasattr(self, "total_steps"):
            self.total_steps = 0
        if not hasattr(self, "segment_count"):
            self.segment_count = 0
        if not hasattr(self, "native_topk_calls"):
            self.native_topk_calls = 0
        if not hasattr(self, "native_topk_success"):
            self.native_topk_success = 0

    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state) -> None:
        self.__dict__.update(dict(state or {}))
        self._ensure_backward_compatibility()

    def reset_state(self):
        """Reset active cells and predictions without losing learned segments."""
        self.active_cells.clear()
        self.predicted_cells.clear()
        self.predicted_tokens.clear()
        self._prev_active_cells.clear()
        self.recent_tokens.clear()

    def reset_all(self):
        """Reset everything including learned temporal segments."""
        self.reset_state()
        self.column_active.fill(0)
        self.temporal_segments.clear()
        self._context_winners.clear()
        self.context_transitions.clear()
        self.burst_count = 0
        self.suppression_count = 0
        self.total_steps = 0
        self.segment_count = 0

    def _context_key(self, token_id):
        return (int(token_id), tuple(sorted(self._prev_active_cells)))

    def _get_column_winner(self, token_id, burst=False, predicted_for_token=None):
        """Get active cells for a column.

        Args:
            token_id: Token ID
            burst: If True, burst (activate random cells). If False, use predicted cells.

        Returns:
            Set of cell indices for this token
        """
        if burst:
            # Burst: activate k random cells
            key = self._context_key(token_id)
            if key not in self._context_winners:
                cells = self.rng.choice(self.cells_per_column, size=self.k, replace=False)
                self._context_winners[key] = tuple(int(c) for c in cells)
                self.column_active[int(token_id), list(self._context_winners[key])] = 1.0
            return set(self._context_winners[key])
        else:
            # Use predicted cells for this token
            if predicted_for_token is None:
                predicted_for_token = {cell for (tok, cell) in self.predicted_cells if tok == token_id}
            if len(predicted_for_token) >= self.k:
                # Use top-k predicted cells
                cells = sorted(predicted_for_token)[:self.k]
                return set(cells)
            else:
                # Not enough predictions, fill with random
                key = self._context_key(token_id)
                if key in self._context_winners:
                    stored = [c for c in self._context_winners[key] if c not in predicted_for_token]
                    cells = list(predicted_for_token) + stored[:self.k - len(predicted_for_token)]
                    return set(cells)
                remaining = self.k - len(predicted_for_token)
                additional = self.rng.choice(
                    [c for c in range(self.cells_per_column) if c not in predicted_for_token],
                    size=min(remaining, self.cells_per_column - len(predicted_for_token)),
                    replace=False
                )
                cells = list(predicted_for_token) + list(additional)
                return set(cells)

    def _predict_next(self):
        """Predict next active cells based on current active cells.

        For each currently active cell, look up temporal segments
        and collect predicted cells for the next step.
        """
        predictions = {}
        for (token, cell) in self.active_cells:
            key = (token, cell)
            if key in self.temporal_segments:
                for (next_token, next_cell), strength in self.temporal_segments[key].items():
                    if next_token not in predictions:
                        predictions[next_token] = {}
                    if next_cell not in predictions[next_token]:
                        predictions[next_token][next_cell] = 0.0
                    predictions[next_token][next_cell] += strength

        # Convert to set of (token, cell) tuples
        self.predicted_cells.clear()
        for token, cells in predictions.items():
            # Keep top-k cells per token
            sorted_cells = sorted(cells.items(), key=lambda x: -x[1])[:self.k]
            for cell, _ in sorted_cells:
                self.predicted_cells.add((token, cell))

        self.predicted_tokens.clear()
        self.predicted_tokens.update(self._context_top_predictions(limit=3))
        scores = self.predict_next_token_scores()
        total = float(scores.sum())
        self.last_prediction_confidence = float(scores.max() / total) if total > 0.0 else 0.0

    def _learn_context_transition(self, token_id):
        if not self.recent_tokens:
            return
        max_len = min(self.max_context, len(self.recent_tokens))
        for length in range(1, max_len + 1):
            key = tuple(self.recent_tokens[-length:])
            if key not in self.context_transitions:
                self.context_transitions[key] = {}
            self.context_transitions[key][int(token_id)] = (
                self.context_transitions[key].get(int(token_id), 0.0) + float(length)
            )

    def _context_prediction_scores(self):
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        max_len = min(self.max_context, len(self.recent_tokens))
        for length in range(1, max_len + 1):
            key = tuple(self.recent_tokens[-length:])
            if key not in self.context_transitions:
                continue
            weight = float(length * length)
            for token, strength in self.context_transitions[key].items():
                scores[int(token)] += weight * float(strength)
        return scores

    def _context_top_predictions(self, limit=3):
        scores = {}
        max_len = min(self.max_context, len(self.recent_tokens))
        for length in range(1, max_len + 1):
            key = tuple(self.recent_tokens[-length:])
            transitions = self.context_transitions.get(key)
            if not transitions:
                continue
            weight = float(length * length)
            for token, strength in transitions.items():
                token = int(token)
                scores[token] = scores.get(token, 0.0) + weight * float(strength)
        if not scores:
            return []
        best = []
        limit = max(1, int(limit))
        for token, score in scores.items():
            if score <= 0.0:
                continue
            item = (float(score), -int(token), int(token))
            if len(best) < limit:
                best.append(item)
                continue
            worst_index = min(range(len(best)), key=lambda idx: (best[idx][0], best[idx][1]))
            if (item[0], item[1]) > (best[worst_index][0], best[worst_index][1]):
                best[worst_index] = item
        return [item[2] for item in sorted(best, key=lambda item: (-item[0], -item[1]))]

    def _learn_segment(self, prev_token, prev_cell, next_token, next_cell, strength=1.0):
        """Learn a temporal segment from previous cell to next cell.

        Args:
            prev_token: Previous token ID
            prev_cell: Previous cell index
            next_token: Next token ID
            next_cell: Next cell index
            strength: Learning strength
        """
        key = (prev_token, prev_cell)
        if key not in self.temporal_segments:
            self.temporal_segments[key] = {}
            self.segment_count += 1

        target = (next_token, next_cell)
        if target not in self.temporal_segments[key]:
            self.temporal_segments[key][target] = 0.0

        self.temporal_segments[key][target] += strength

    def prune_memory(self, max_segments=None, max_context_keys=None, max_targets_per_context=None):
        removed = 0
        removed_targets = 0
        removed_context_keys = 0
        removed_segments = 0
        if max_targets_per_context is not None:
            limit = max(1, int(max_targets_per_context))
            for key, targets in list(self.context_transitions.items()):
                if len(targets) <= limit:
                    continue
                ranked = sorted(targets.items(), key=lambda item: (-float(item[1]), int(item[0])))[:limit]
                removed_now = len(targets) - len(ranked)
                removed += removed_now
                removed_targets += removed_now
                self.context_transitions[key] = dict(ranked)
        if max_context_keys is not None:
            limit = max(1, int(max_context_keys))
            while len(self.context_transitions) > limit:
                self.context_transitions.pop(next(iter(self.context_transitions)))
                removed += 1
                removed_context_keys += 1
        if max_segments is not None:
            limit = max(1, int(max_segments))
            if len(self.temporal_segments) > limit:
                ranked_segments = sorted(
                    self.temporal_segments.items(),
                    key=lambda item: sum(float(v) for v in item[1].values()),
                    reverse=True,
                )[:limit]
                removed_now = len(self.temporal_segments) - len(ranked_segments)
                removed += removed_now
                removed_segments += removed_now
                self.temporal_segments = dict(ranked_segments)
                self.segment_count = len(self.temporal_segments)
        self.last_pruned_targets = removed_targets
        self.last_pruned_context_keys = removed_context_keys
        self.last_pruned_segments = removed_segments
        return removed

    def forward(self, token_id, learn=True):
        """Process a token and update sequence memory.

        Args:
            token_id: Current token ID
            learn: Whether to learn temporal segments

        Returns:
            Dictionary with prediction info and whether burst occurred
        """
        self.total_steps += 1

        # Check if this token was predicted
        predicted_for_token = {cell for (tok, cell) in self.predicted_cells if tok == token_id}
        was_predicted = len(predicted_for_token) > 0 or int(token_id) in self.predicted_tokens

        if was_predicted:
            # Suppression: activate only predicted cells
            self.active_cells = {
                (token_id, cell)
                for cell in self._get_column_winner(token_id, burst=False, predicted_for_token=predicted_for_token)
            }
            self.suppression_count += 1
            burst = False
        else:
            # Burst: activate random cells and learn immediately
            self.active_cells = {(token_id, cell) for cell in self._get_column_winner(token_id, burst=True)}
            self.burst_count += 1
            burst = True

        # Learn temporal segments from previous active cells to current active cells
        if learn and len(self._prev_active_cells) > 0:
            for (prev_token, prev_cell) in self._prev_active_cells:
                for (curr_token, curr_cell) in self.active_cells:
                    learning_strength = 2.0 if burst else 0.5  # Learn more strongly on burst
                    self._learn_segment(prev_token, prev_cell, curr_token, curr_cell, learning_strength)

        if learn:
            self._learn_context_transition(token_id)

        # Store current active cells for next step
        self._prev_active_cells = self.active_cells.copy()
        self.recent_tokens.append(int(token_id))
        if len(self.recent_tokens) > self.max_context:
            self.recent_tokens.pop(0)

        # Predict next step
        self._predict_next()

        return {
            "burst": burst,
            "predicted": was_predicted,
            "active_cells": len(self.active_cells),
            "predicted_cells": len(self.predicted_cells),
        }

    def predict_next_token_scores(self):
        """Get prediction scores for all tokens based on current active cells.

        Returns:
            Array of shape (vocab_size,) with prediction scores
        """
        scores = np.zeros(self.vocab_size, dtype=np.float32)

        for (token, cell) in self.active_cells:
            key = (token, cell)
            if key in self.temporal_segments:
                for (next_token, _), strength in self.temporal_segments[key].items():
                    scores[next_token] += strength

        scores += self._context_prediction_scores()

        return scores

    def topk_prediction_indices(self, scores, top_k=3, prefer_native: bool = True):
        limit = max(1, int(top_k))
        if prefer_native and NATIVE_AVAILABLE:
            self.native_topk_calls += 1
            try:
                result = sparse_native.topk_float32(scores, limit)
                self.native_topk_success += 1
                return [int(idx) for idx in result.get("indices", [])]
            except Exception:
                pass
        return [int(idx) for idx in np.argsort(scores)[-limit:][::-1]]

    def generate(self, prefix_tokens, max_steps=20, temperature=1.0, top_k=3):
        """Generate text from a prefix using sequence memory.

        Args:
            prefix_tokens: List of token IDs as context
            max_steps: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = deterministic, higher = more random)
            top_k: Number of top predictions to sample from (avoids repetition)

        Returns:
            List of generated token IDs
        """
        self.reset_state()
        self._prev_active_cells = set()

        # Process prefix
        for token_id in prefix_tokens:
            self.forward(token_id, learn=False)

        generated = list(prefix_tokens)
        recent_tokens = list(prefix_tokens[-5:])  # Track recent to avoid loops

        for _ in range(max_steps):
            # Get prediction scores
            scores = self.predict_next_token_scores()

            if scores.sum() == 0:
                # No predictions, stop
                break

            # Get top-k predictions
            top_indices = self.topk_prediction_indices(scores, top_k=top_k)
            top_scores = scores[top_indices]

            # Avoid repeating recent tokens (prevent loops)
            valid_indices = [idx for idx in top_indices if idx not in recent_tokens]
            if not valid_indices:
                valid_indices = top_indices
                valid_scores = top_scores
            else:
                valid_scores = scores[valid_indices]

            # Apply temperature and sample from valid predictions
            if temperature == 1.0:
                next_token = int(valid_indices[0])
            else:
                probs = np.exp(valid_scores / temperature - np.max(valid_scores / temperature))
                probs = probs / (probs.sum() + 1e-10)
                next_token = int(self.rng.choice(valid_indices, p=probs))

            generated.append(next_token)
            recent_tokens.append(next_token)
            if len(recent_tokens) > 10:
                recent_tokens.pop(0)

            # Stop on special tokens
            if next_token in [0, 1]:  # PAD or UNK
                break

            # Process the generated token
            self.forward(next_token, learn=False)

        return generated

    def metrics(self):
        """Get current metrics."""
        burst_rate = self.burst_count / self.total_steps if self.total_steps > 0 else 0.0
        suppression_rate = self.suppression_count / self.total_steps if self.total_steps > 0 else 0.0
        return {
            "total_steps": self.total_steps,
            "burst_count": self.burst_count,
            "suppression_count": self.suppression_count,
            "burst_rate": burst_rate,
            "suppression_rate": suppression_rate,
            "segment_count": self.segment_count,
            "active_cells": len(self.active_cells),
            "predicted_cells": len(self.predicted_cells),
            "native_topk_available": float(NATIVE_AVAILABLE),
            "native_topk_calls": float(self.native_topk_calls),
            "native_topk_success": float(self.native_topk_success),
            "native_topk_ratio": float(self.native_topk_success) / max(1.0, float(self.native_topk_calls)),
            "prediction_confidence": float(self.last_prediction_confidence),
            "last_pruned_segments": float(self.last_pruned_segments),
            "last_pruned_context_keys": float(self.last_pruned_context_keys),
            "last_pruned_targets": float(self.last_pruned_targets),
        }
