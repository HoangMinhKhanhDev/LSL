"""Long-context sparse memory built from local content-addressable stores."""
from collections import deque
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .memory import SparseKeyValueMemory


class LongContextMemory:
    """Bounded sparse memory for transitions, facts, and instructions.

    All retrieval is delegated to sparse candidate indexes. The high-level
    methods never scan stored history.
    """

    def __init__(
        self,
        capacity: int = 131072,
        vocab_size: int = 1000,
        sdr_dim: int = 4096,
        sparsity: float = 0.01,
        candidate_cap: int = 64,
        context_width: int = 4,
        store_transition_index: bool = True,
        target_cap: int = 16,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.vocab_size = int(vocab_size)
        self.context_width = max(1, int(context_width))
        self.store_transition_index = bool(store_transition_index)
        self.target_cap = max(1, int(target_cap))
        self.recent = deque(maxlen=self.context_width)
        self.seed = int(seed)

        common = dict(
            capacity=capacity,
            sdr_dim=sdr_dim,
            sparsity=sparsity,
            candidate_cap=candidate_cap,
            bucket_probe_bits=8,
            seed=seed,
        )
        self.transitions = SparseKeyValueMemory(**common)
        self.unigram_transitions = SparseKeyValueMemory(**{**common, "seed": seed + 5})
        self.facts = SparseKeyValueMemory(**{**common, "seed": seed + 11})
        self.instructions = SparseKeyValueMemory(**{**common, "seed": seed + 23})

        self.last_lookup_diag: Dict[str, float] = {}
        self.last_prediction_confidence = 0.0
        self._transition_counts: Dict[int, Dict[int, int]] = {}
        self._transition_order = deque()
        self._unigram_counts: Dict[int, Dict[int, int]] = {}
        self._unigram_order = deque()
        self._global_counts: Dict[int, int] = {}
        self._global_total = 0

    def _mix(self, tag: int, values: Iterable[int]) -> int:
        x = (int(tag) + 0x9E3779B97F4A7C15 + self.seed) & 0xFFFFFFFFFFFFFFFF
        for value in values:
            y = (int(value) + 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            x ^= y + 0x9E3779B97F4A7C15 + ((x << 6) & 0xFFFFFFFFFFFFFFFF) + (x >> 2)
            x &= 0xFFFFFFFFFFFFFFFF
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
        return int(x & 0x7FFFFFFF)

    def _transition_key(self, token_id: int) -> int:
        ctx = list(self.recent)[-(self.context_width - 1):] + [int(token_id)]
        return self._mix(101, ctx)

    def _unigram_key(self, token_id: int) -> int:
        return self._mix(103, [int(token_id)])

    def _bump_counts(self, table: Dict[int, Dict[int, int]], order, key: int, value: int) -> None:
        if key not in table:
            table[key] = {}
            order.append(key)
        bucket = table[key]
        bucket[int(value)] = bucket.get(int(value), 0) + 1
        if len(bucket) > self.target_cap:
            weakest = min(bucket.items(), key=lambda item: (item[1], item[0]))[0]
            del bucket[weakest]
        while len(table) > self.capacity and order:
            old_key = order.popleft()
            table.pop(old_key, None)

    def _best_counted_value(self, table: Dict[int, Dict[int, int]], key: int) -> Tuple[Optional[int], float, int]:
        bucket = table.get(int(key))
        if not bucket:
            return None, 0.0, 0
        value, count = max(bucket.items(), key=lambda item: item[1])
        total = sum(bucket.values())
        confidence = float(count / max(1, total))
        return int(value), confidence, int(total)

    def _probability_from_bucket(
        self,
        bucket: Optional[Dict[int, int]],
        target_id: int,
        vocab_size: int,
        alpha: float,
    ) -> Tuple[float, int]:
        if not bucket:
            return 0.0, 0
        total = int(sum(bucket.values()))
        prob = (float(bucket.get(int(target_id), 0)) + float(alpha)) / (
            float(total) + float(alpha) * max(1, int(vocab_size))
        )
        return float(prob), total

    def observe_transition(self, token_id: int, target_id: int, vocab_size: Optional[int] = None) -> None:
        vocab = int(vocab_size or self.vocab_size)
        key = self._transition_key(int(token_id))
        unigram_key = self._unigram_key(int(token_id))
        if self.store_transition_index:
            self.transitions.add(key, int(target_id), vocab_size=max(vocab, key + 1))
            self.unigram_transitions.add(unigram_key, int(target_id), vocab_size=max(vocab, unigram_key + 1))
        self._bump_counts(self._transition_counts, self._transition_order, key, int(target_id))
        self._bump_counts(self._unigram_counts, self._unigram_order, unigram_key, int(target_id))
        self._global_counts[int(target_id)] = self._global_counts.get(int(target_id), 0) + 1
        self._global_total += 1
        self.recent.append(int(token_id))

    def predict_next(
        self,
        token_id: int,
        vocab_size: Optional[int] = None,
        allow_direct_lookup: bool = True,
        return_confidence: bool = False,
        update_context: bool = False,
    ) -> Optional[int]:
        vocab = int(vocab_size or self.vocab_size)
        key = self._transition_key(int(token_id))
        value, diag = self.transitions.lookup(
            key,
            vocab_size=max(vocab, key + 1),
            return_diagnostics=True,
            allow_direct_lookup=allow_direct_lookup,
        )
        counted_value, confidence, support = self._best_counted_value(self._transition_counts, key)
        if counted_value is not None:
            value = counted_value
            diag = {**diag, "confidence": confidence, "support": float(support), "backoff": 0.0}
        elif value is not None:
            if diag.get("best_score", 0.0) < max(8, self.transitions.k // 2):
                value = None
                confidence = 0.0
            else:
                confidence = 1.0 if diag.get("best_score", 0.0) >= self.transitions.k else 0.5
                diag = {**diag, "confidence": confidence, "support": 1.0, "backoff": 0.0}

        if value is None:
            unigram_key = self._unigram_key(int(token_id))
            value, diag = self.unigram_transitions.lookup(
                unigram_key,
                vocab_size=max(vocab, unigram_key + 1),
                return_diagnostics=True,
                allow_direct_lookup=allow_direct_lookup,
            )
            counted_value, confidence, support = self._best_counted_value(self._unigram_counts, unigram_key)
            if counted_value is not None:
                value = counted_value
                diag = {**diag, "confidence": confidence, "support": float(support), "backoff": 1.0}
            elif value is not None:
                if diag.get("best_score", 0.0) < max(8, self.unigram_transitions.k // 2):
                    value = None
                    confidence = 0.0
                else:
                    confidence = 1.0 if diag.get("best_score", 0.0) >= self.unigram_transitions.k else 0.5
                    diag = {**diag, "confidence": confidence, "support": 1.0, "backoff": 1.0}

        self.last_lookup_diag = diag
        self.last_prediction_confidence = float(diag.get("confidence", 0.0))
        if update_context:
            self.recent.append(int(token_id))
        result = None if value is None else int(value)
        if return_confidence:
            return result, self.last_prediction_confidence
        return result

    def advance_context(self, token_id: int) -> None:
        self.recent.append(int(token_id))

    def target_probability(
        self,
        token_id: int,
        target_id: int,
        vocab_size: Optional[int] = None,
        alpha: float = 0.05,
        update_context: bool = False,
    ) -> float:
        """Return an interpolated local probability for target_id.

        The probability uses only local count buckets for the current sparse
        context, a single-token backoff, and a global target prior.
        """
        vocab = int(vocab_size or self.vocab_size)
        key = self._transition_key(int(token_id))
        unigram_key = self._unigram_key(int(token_id))
        p_ctx, ctx_support = self._probability_from_bucket(
            self._transition_counts.get(key),
            int(target_id),
            vocab,
            alpha,
        )
        p_uni, uni_support = self._probability_from_bucket(
            self._unigram_counts.get(unigram_key),
            int(target_id),
            vocab,
            alpha,
        )
        if self._global_total:
            p_global = (float(self._global_counts.get(int(target_id), 0)) + alpha) / (
                float(self._global_total) + alpha * max(1, vocab)
            )
        else:
            p_global = 1.0 / max(1, vocab)

        if ctx_support:
            prob = 0.82 * p_ctx + 0.13 * (p_uni if uni_support else p_global) + 0.05 * p_global
        elif uni_support:
            prob = 0.75 * p_uni + 0.25 * p_global
        else:
            prob = p_global

        self.last_lookup_diag = {
            "candidate_count": 0.0,
            "full_scan": 0.0,
            "context_support": float(ctx_support),
            "unigram_support": float(uni_support),
            "global_support": float(self._global_total),
            "probability": float(prob),
        }
        if update_context:
            self.recent.append(int(token_id))
        return float(max(prob, 1e-12))

    def top_next(self, token_id: int, vocab_size: Optional[int] = None, update_context: bool = False) -> Optional[int]:
        """Return the best next token from context counts with unigram fallback."""
        vocab = int(vocab_size or self.vocab_size)
        del vocab
        key = self._transition_key(int(token_id))
        value, confidence, support = self._best_counted_value(self._transition_counts, key)
        if value is None:
            value, confidence, support = self._best_counted_value(self._unigram_counts, self._unigram_key(int(token_id)))
        if value is None and self._global_counts:
            value = int(max(self._global_counts.items(), key=lambda item: item[1])[0])
            confidence = float(self._global_counts[value] / max(1, self._global_total))
            support = self._global_total
        self.last_prediction_confidence = float(confidence)
        self.last_lookup_diag = {
            "candidate_count": 0.0,
            "full_scan": 0.0,
            "confidence": float(confidence),
            "support": float(support),
        }
        if update_context:
            self.recent.append(int(token_id))
        return value

    def next_candidates(
        self,
        token_id: int,
        limit: int = 8,
        update_context: bool = False,
    ) -> Tuple[int, ...]:
        """Return ranked next-token candidates from local context/backoff counts."""
        scores: Dict[int, float] = {}
        key = self._transition_key(int(token_id))
        unigram_key = self._unigram_key(int(token_id))
        ctx = self._transition_counts.get(key, {})
        uni = self._unigram_counts.get(unigram_key, {})
        ctx_total = max(1, sum(ctx.values()))
        uni_total = max(1, sum(uni.values()))
        global_total = max(1, self._global_total)
        for value, count in ctx.items():
            scores[int(value)] = scores.get(int(value), 0.0) + 0.82 * float(count) / ctx_total
        for value, count in uni.items():
            scores[int(value)] = scores.get(int(value), 0.0) + 0.13 * float(count) / uni_total
        for value, count in self._global_counts.items():
            scores[int(value)] = scores.get(int(value), 0.0) + 0.05 * float(count) / global_total
        ranked = tuple(
            int(value)
            for value, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(1, int(limit))]
        )
        if update_context:
            self.recent.append(int(token_id))
        self.last_lookup_diag = {
            "candidate_count": float(len(ranked)),
            "full_scan": 0.0,
            "context_support": float(ctx_total if ctx else 0),
            "unigram_support": float(uni_total if uni else 0),
        }
        return ranked

    def observe_fact(self, subject_id: int, relation_id: int, object_id: int) -> None:
        key = self._mix(211, [int(subject_id), int(relation_id)])
        self.facts.add(key, int(object_id), vocab_size=max(self.vocab_size, key + 1))

    def query_fact(
        self,
        subject_id: int,
        relation_id: int,
        allow_direct_lookup: bool = True,
    ) -> Optional[int]:
        key = self._mix(211, [int(subject_id), int(relation_id)])
        value, diag = self.facts.lookup(
            key,
            vocab_size=max(self.vocab_size, key + 1),
            return_diagnostics=True,
            allow_direct_lookup=allow_direct_lookup,
        )
        self.last_lookup_diag = diag
        return None if value is None else int(value)

    def observe_instruction(self, command_id: int, response_id: int) -> None:
        key = self._mix(307, [int(command_id)])
        self.instructions.add(key, int(response_id), vocab_size=max(self.vocab_size, key + 1))

    def query_instruction(self, command_id: int, allow_direct_lookup: bool = True) -> Optional[int]:
        key = self._mix(307, [int(command_id)])
        value, diag = self.instructions.lookup(
            key,
            vocab_size=max(self.vocab_size, key + 1),
            return_diagnostics=True,
            allow_direct_lookup=allow_direct_lookup,
        )
        self.last_lookup_diag = diag
        return None if value is None else int(value)

    def reset_state(self) -> None:
        self.recent.clear()

    def clear(self) -> None:
        self.reset_state()
        self.transitions.clear()
        self.unigram_transitions.clear()
        self.facts.clear()
        self.instructions.clear()
        self._transition_counts.clear()
        self._transition_order.clear()
        self._unigram_counts.clear()
        self._unigram_order.clear()
        self._global_counts.clear()
        self._global_total = 0

    def diagnostics(self) -> Dict[str, float]:
        return {
            "transition_items": float(len(self.transitions)),
            "unigram_transition_items": float(len(self.unigram_transitions)),
            "fact_items": float(len(self.facts)),
            "instruction_items": float(len(self.instructions)),
            "global_transition_targets": float(len(self._global_counts)),
            "global_transition_total": float(self._global_total),
            "prediction_confidence": float(self.last_prediction_confidence),
            **{f"last_{k}": float(v) for k, v in self.last_lookup_diag.items()},
        }
