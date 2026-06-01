"""Phase 9 biological compute primitives.

These components are strict-path helpers: local state, online updates, bounded
candidate lookup, and sparse active-index computation only.
"""
import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .agent import IntegratedLSLAgent
from .cortical_column import CorticalColumnSequenceMemory
from .semantic_aliases import ALIAS_TO_GROUP, all_multilingual_terms
from . import sparse_native
from .sparse_native import NATIVE_AVAILABLE
from .text_normalization import lexical_key, looks_vietnamese, normalize_text, strip_diacritics, token_variants


def _norm(value) -> str:
    return str(value).strip().lower()


def _hash_u64(*parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little")


def _feature_bits(feature: str, dim: int, count: int, seed: int = 0) -> Tuple[int, ...]:
    bits = set()
    nonce = 0
    while len(bits) < int(count):
        bits.add(int(_hash_u64(seed, feature, nonce) % int(dim)))
        nonce += 1
    return tuple(sorted(bits))


def _tuple_overlap_sorted(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    i = 0
    j = 0
    overlap = 0
    left_len = len(left)
    right_len = len(right)
    while i < left_len and j < right_len:
        a = int(left[i])
        b = int(right[j])
        if a == b:
            overlap += 1
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return overlap


class LocalPredictiveStack:
    """Layer-local predictor with exact local pre-state -> post-state tables."""

    def __init__(self, layers: int = 3, width: int = 256, k: int = 8, theta: float = 0.05):
        self.layers = int(layers)
        self.width = int(width)
        self.k = int(k)
        self.theta = float(theta)
        self.tables: List[Dict[Tuple[int, ...], Counter]] = [defaultdict(Counter) for _ in range(self.layers)]
        self.prev_states: List[Optional[Tuple[int, ...]]] = [None for _ in range(self.layers)]
        self.error_history: List[List[float]] = [[] for _ in range(self.layers)]
        self.layer_update_counts: List[int] = [0 for _ in range(self.layers)]
        self.layer_error_sums: List[float] = [0.0 for _ in range(self.layers)]
        self.layer_suppression_sums: List[float] = [0.0 for _ in range(self.layers)]
        self.layer_step_counts: List[int] = [0 for _ in range(self.layers)]
        self.layer_confidence_sums: List[float] = [0.0 for _ in range(self.layers)]
        self.layer_anomaly_sums: List[float] = [0.0 for _ in range(self.layers)]
        self.update_count = 0
        self.step_count = 0
        self.zero_update_count = 0
        self.confidence_sum = 0.0
        self.anomaly_sum = 0.0
        self.last_confidence = 0.0
        self.last_anomaly_score = 0.0
        self._ensure_backward_compatibility()

    def _ensure_backward_compatibility(self) -> None:
        if not hasattr(self, "layers"):
            self.layers = 3
        if not hasattr(self, "width"):
            self.width = 256
        if not hasattr(self, "k"):
            self.k = 8
        if not hasattr(self, "theta"):
            self.theta = 0.05
        if not hasattr(self, "tables"):
            self.tables = [defaultdict(Counter) for _ in range(int(self.layers))]
        if not hasattr(self, "prev_states"):
            self.prev_states = [None for _ in range(int(self.layers))]
        if not hasattr(self, "error_history"):
            self.error_history = [[] for _ in range(int(self.layers))]
        if not hasattr(self, "layer_update_counts"):
            self.layer_update_counts = [0 for _ in range(int(self.layers))]
        if not hasattr(self, "layer_error_sums"):
            self.layer_error_sums = [0.0 for _ in range(int(self.layers))]
        if not hasattr(self, "layer_suppression_sums"):
            self.layer_suppression_sums = [0.0 for _ in range(int(self.layers))]
        if not hasattr(self, "layer_step_counts"):
            self.layer_step_counts = [0 for _ in range(int(self.layers))]
        if not hasattr(self, "layer_confidence_sums"):
            self.layer_confidence_sums = [0.0 for _ in range(int(self.layers))]
        if not hasattr(self, "layer_anomaly_sums"):
            self.layer_anomaly_sums = [0.0 for _ in range(int(self.layers))]
        if not hasattr(self, "update_count"):
            self.update_count = 0
        if not hasattr(self, "step_count"):
            self.step_count = 0
        if not hasattr(self, "zero_update_count"):
            self.zero_update_count = 0
        if not hasattr(self, "confidence_sum"):
            self.confidence_sum = 0.0
        if not hasattr(self, "anomaly_sum"):
            self.anomaly_sum = 0.0
        if not hasattr(self, "last_confidence"):
            self.last_confidence = 0.0
        if not hasattr(self, "last_anomaly_score"):
            self.last_anomaly_score = 0.0

    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state) -> None:
        self.__dict__.update(dict(state or {}))
        self._ensure_backward_compatibility()

    def state_for(self, token: int, layer: int = 0) -> Tuple[int, ...]:
        base = int(token)
        return _feature_bits(f"pc:{layer}:{base}", self.width, self.k, seed=layer)

    def _predict(self, layer: int, prev: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        counter = self.tables[layer].get(prev)
        if not counter:
            return None
        return tuple(max(counter.items(), key=lambda item: (item[1], item[0]))[0])

    def predict_state(self, layer: int, prev: Tuple[int, ...]) -> Tuple[Optional[Tuple[int, ...]], float]:
        counter = self.tables[layer].get(prev)
        if not counter:
            return None, 0.0
        total = max(1.0, float(sum(counter.values())))
        best_state, best_count = max(counter.items(), key=lambda item: (item[1], item[0]))
        return tuple(best_state), float(best_count) / total

    def observe(self, states: Sequence[Tuple[int, ...]], learn: bool = True) -> Dict[str, float]:
        self.step_count += 1
        updates = 0
        errors = []
        confidences = []
        suppressed_dims = 0.0
        total_dims = 0.0
        for layer, current in enumerate(states[: self.layers]):
            prev = self.prev_states[layer]
            error = 1.0
            layer_confidence = 0.0
            if prev is not None:
                predicted, confidence = self.predict_state(layer, prev)
                if predicted is not None:
                    overlap = _tuple_overlap_sorted(predicted, current)
                    error = 1.0 - overlap / max(1.0, float(self.k))
                    layer_confidence = float(confidence)
                    confidences.append(layer_confidence)
                if learn and error > self.theta:
                    self.tables[layer][prev][tuple(current)] += 1.0
                    updates += 1
            if layer_confidence <= 0.0:
                layer_confidence = max(0.0, 1.0 - error)
            self.error_history[layer].append(float(error))
            errors.append(float(error))
            layer_suppression = max(0.0, 1.0 - error)
            suppressed_dims += layer_suppression * self.width
            total_dims += self.width
            self.layer_error_sums[layer] += float(error)
            self.layer_suppression_sums[layer] += float(layer_suppression)
            self.layer_step_counts[layer] += 1
            self.layer_confidence_sums[layer] += float(layer_confidence)
            self.layer_anomaly_sums[layer] += float(max(0.0, 1.0 - layer_confidence))
            self.prev_states[layer] = tuple(current)
            if learn and error > self.theta:
                self.layer_update_counts[layer] += 1
        self.update_count += updates
        if updates == 0:
            self.zero_update_count += 1
        mean_error = sum(errors) / max(1, len(errors))
        mean_confidence = sum(confidences) / max(1, len(confidences)) if confidences else max(0.0, 1.0 - mean_error)
        mean_anomaly = max(0.0, min(1.0, 1.0 - mean_confidence if confidences else mean_error))
        self.confidence_sum += float(mean_confidence)
        self.anomaly_sum += float(mean_anomaly)
        self.last_confidence = float(mean_confidence)
        self.last_anomaly_score = float(mean_anomaly)
        return {
            "mean_error": mean_error,
            "suppression": suppressed_dims / max(1.0, total_dims),
            "updates": float(updates),
            "layer_errors": tuple(errors),
            "confidence": mean_confidence,
            "anomaly_score": mean_anomaly,
        }

    def reset_state(self) -> None:
        self.prev_states = [None for _ in range(self.layers)]

    def diagnostics(self) -> Dict[str, float]:
        layer_steps = max(1, sum(self.layer_step_counts))
        return {
            "steps": float(self.step_count),
            "updates": float(self.update_count),
            "zero_update_ratio": self.zero_update_count / max(1.0, float(self.step_count)),
            "mean_error": sum(sum(v) for v in self.error_history) / max(1, sum(len(v) for v in self.error_history)),
            "mean_confidence": self.confidence_sum / max(1.0, float(self.step_count)),
            "mean_anomaly_score": self.anomaly_sum / max(1.0, float(self.step_count)),
            "confidence_gap": abs(
                (self.confidence_sum / max(1.0, float(self.step_count)))
                - (1.0 - sum(sum(v) for v in self.error_history) / max(1, sum(len(v) for v in self.error_history)))
            ),
            "last_confidence": float(self.last_confidence),
            "last_anomaly_score": float(self.last_anomaly_score),
            "layer_update_density": sum(self.layer_update_counts) / max(1.0, float(layer_steps)),
            "mean_layer_suppression": sum(self.layer_suppression_sums) / max(1.0, float(layer_steps)),
            "mean_layer_confidence": sum(self.layer_confidence_sums) / max(1.0, float(layer_steps)),
            "mean_layer_anomaly": sum(self.layer_anomaly_sums) / max(1.0, float(layer_steps)),
            "max_layer_error": max((max(v) for v in self.error_history if v), default=0.0),
        }

    def layer_error_curve(self) -> List[float]:
        return [
            self.layer_error_sums[layer] / max(1.0, float(self.layer_step_counts[layer]))
            for layer in range(self.layers)
        ]

    def layer_confidence_curve(self) -> List[float]:
        return [
            self.layer_confidence_sums[layer] / max(1.0, float(self.layer_step_counts[layer]))
            for layer in range(self.layers)
        ]

    def layer_anomaly_curve(self) -> List[float]:
        return [
            self.layer_anomaly_sums[layer] / max(1.0, float(self.layer_step_counts[layer]))
            for layer in range(self.layers)
        ]

    def adaptive_theta(self, target_suppression: float = 0.95) -> float:
        target = max(0.0, min(1.0, float(target_suppression)))
        current = sum(self.layer_suppression_sums) / max(1.0, float(sum(self.layer_step_counts)))
        self.theta = min(0.5, max(0.001, self.theta + 0.1 * (current - target)))
        return float(self.theta)


class OnePassCausalMemory:
    """One-observation local causal and chain memory."""

    def __init__(self):
        self.direct: Dict[str, Counter] = defaultdict(Counter)

    def observe(self, cause: str, effect: str) -> None:
        self.direct[_norm(cause)][_norm(effect)] += 1.0

    def probability(self, cause: str, effect: str, vocab_size: int) -> float:
        counter = self.direct.get(_norm(cause), Counter())
        if not counter:
            return 1.0 / max(1.0, float(vocab_size))
        total = sum(counter.values())
        return float(counter.get(_norm(effect), 0.0)) / max(1.0, float(total))

    def chain(self, start: str, hops: int) -> Optional[str]:
        current = _norm(start)
        for _ in range(int(hops)):
            counter = self.direct.get(current)
            if not counter:
                return None
            current = max(counter.items(), key=lambda item: (item[1], item[0]))[0]
        return current


class VirtualSparseSDR:
    """Sparse SDR that returns active indices for very large dimensions."""

    def __init__(self, dim: int = 100000, k: int = 20, seed: int = 0):
        self.dim = int(dim)
        self.k = int(k)
        self.seed = int(seed)
        self.alias_to_group = dict(ALIAS_TO_GROUP)
        self.known_words = set(all_multilingual_terms())
        self.related: Dict[str, Counter] = defaultdict(Counter)
        self._key_cache: Dict[str, str] = {}
        self._key_cache_limit = 8192
        self.bilingual = {
            "brain": "concept:brain",
            "não": "concept:brain",
            "nao": "concept:brain",
            "memory": "concept:memory",
            "ký ức": "concept:memory",
            "ky uc": "concept:memory",
        }

    def _ensure_backward_compatibility(self) -> None:
        if not hasattr(self, "dim"):
            self.dim = 100000
        if not hasattr(self, "k"):
            self.k = 20
        if not hasattr(self, "seed"):
            self.seed = 0
        if not hasattr(self, "alias_to_group"):
            self.alias_to_group = dict(ALIAS_TO_GROUP)
        if not hasattr(self, "known_words"):
            self.known_words = set(all_multilingual_terms())
        if not hasattr(self, "related"):
            self.related = defaultdict(Counter)
        if not hasattr(self, "_key_cache"):
            self._key_cache = {}
        if not hasattr(self, "_key_cache_limit"):
            self._key_cache_limit = 8192
        if not hasattr(self, "bilingual"):
            self.bilingual = {
                "brain": "concept:brain",
                "memory": "concept:memory",
            }

    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state) -> None:
        self.__dict__.update(dict(state or {}))
        self._ensure_backward_compatibility()

    def _cached_key(self, value: str) -> str:
        raw = str(value)
        cached = self._key_cache.get(raw)
        if cached is not None:
            return cached
        cached = lexical_key(raw)
        if len(self._key_cache) >= self._key_cache_limit:
            self._key_cache.clear()
        self._key_cache[raw] = cached
        return cached

    def log2_capacity(self) -> float:
        n = float(self.dim)
        k = float(self.k)
        return (math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)) / math.log(2.0)

    def morphemes(self, word: str) -> List[str]:
        exact = normalize_text(
            word,
            normalize_unicode=True,
            compatibility_normalization=True,
            vietnamese_normalization=True,
            repair_mojibake=True,
            lowercase=True,
            strip_invisible=True,
        )
        accentless = strip_diacritics(exact)
        w = lexical_key(exact)
        parts = [f"word:{exact}", f"lex:{w}"]
        if accentless and accentless != exact:
            parts.append(f"ascii:{lexical_key(accentless)}")
        if looks_vietnamese(exact):
            parts.append("lang:vi")
        elif any(ord(ch) > 127 for ch in exact):
            parts.append("lang:unicode")
        else:
            parts.append("lang:latin")
        group = self.alias_to_group.get(w)
        if group is None:
            group = self.alias_to_group.get(lexical_key(accentless))
        if group is not None:
            parts.append(f"concept:{group}")
        for prefix in ("un", "re", "pre", "post", "anti"):
            if w.startswith(prefix) and len(w) > len(prefix) + 2:
                parts.append(f"prefix:{prefix}")
                stem = w[len(prefix):]
                if stem.endswith("ness"):
                    stem = stem[:-4]
                if stem.endswith("i"):
                    stem = stem[:-1] + "y"
                parts.append(f"stem:{stem}")
        for suffix in ("ness", "ing", "ed", "er", "ly", "s"):
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                parts.append(f"suffix:{suffix}")
                stem = w[:-len(suffix)]
                if stem.endswith("i"):
                    stem = stem[:-1] + "y"
                if stem.startswith("un") and len(stem) > 4:
                    parts.append("prefix:un")
                    parts.append(f"stem:{stem[2:]}")
                parts.append(f"stem:{stem}")
        for variant in token_variants(exact):
            variant_key = lexical_key(variant)
            if variant_key:
                parts.append(f"variant:{variant_key}")
        for other, count in self.related.get(w, Counter()).items():
            if count > 0:
                parts.append(f"related:{other}")
        return parts

    def observe_related(self, left: str, right: str) -> None:
        left_key = self._cached_key(left)
        right_key = self._cached_key(right)
        group = "pair:" + "|".join(sorted([left_key, right_key]))
        self.related[left_key][group] += 1.0
        self.related[right_key][group] += 1.0
        self.known_words.add(normalize_text(left, normalize_unicode=True, compatibility_normalization=True, vietnamese_normalization=True, repair_mojibake=True, lowercase=True, strip_invisible=True))
        self.known_words.add(normalize_text(right, normalize_unicode=True, compatibility_normalization=True, vietnamese_normalization=True, repair_mojibake=True, lowercase=True, strip_invisible=True))

    def observe_related_ids(self, left_id: int, right_id: int) -> None:
        left_key = f"tok:{int(left_id)}"
        right_key = f"tok:{int(right_id)}"
        group = "pair:" + "|".join(sorted([left_key, right_key]))
        self.related[left_key][group] += 1.0
        self.related[right_key][group] += 1.0
        self.known_words.add(left_key)
        self.known_words.add(right_key)

    def encode(self, word: str) -> Tuple[int, ...]:
        parts = self.morphemes(word)
        priority = {
            "related:": 0,
            "concept:": 1,
            "prefix:": 2,
            "stem:": 2,
            "suffix:": 3,
            "ascii:": 4,
            "variant:": 5,
            "lang:": 6,
            "lex:": 7,
            "word:": 8,
        }
        per = max(1, self.k // max(1, len(parts)))
        bits = []
        for part in sorted(parts, key=lambda value: (priority.get(value.split(":", 1)[0] + ":", 9), value)):
            if part.startswith("related:"):
                part_count = min(self.k, max(per * 4, (self.k * 3) // 4))
            elif part.startswith(("prefix:", "stem:", "concept:")):
                part_count = max(per, 3)
            elif part.startswith("lang:"):
                part_count = 1
            else:
                part_count = per
            bits.extend(_feature_bits(part, self.dim, part_count, self.seed))
        nonce = 0
        unique = []
        seen = set()
        for bit in bits:
            if bit not in seen:
                seen.add(bit)
                unique.append(bit)
        while len(unique) < self.k:
            bits.append(int(_hash_u64(self.seed, "fill", word, nonce) % self.dim))
            if bits[-1] not in seen:
                seen.add(bits[-1])
                unique.append(bits[-1])
            nonce += 1
        return tuple(sorted(unique[: self.k]))

    def overlap(self, left: str, right: str) -> int:
        return len(set(self.encode(left)) & set(self.encode(right)))

    def debug_visualize(self, word: str, width: int = 64) -> str:
        bits = self.encode(word)
        width = max(8, int(width))
        rows = int(math.ceil(self.dim / width))
        grid = [["." for _ in range(width)] for _ in range(rows)]
        for bit in bits:
            r = int(bit) // width
            c = int(bit) % width
            if 0 <= r < rows:
                grid[r][c] = "#"
        return "\n".join("".join(row) for row in grid)

    def collision_rate(self, sample_ids: Sequence[int]) -> float:
        seen = {}
        collisions = 0
        for token_id in sample_ids:
            key = tuple(self.encode(str(int(token_id))))
            if key in seen and seen[key] != int(token_id):
                collisions += 1
            else:
                seen[key] = int(token_id)
        return collisions / max(1, len(sample_ids))

    def noisy_recall(
        self,
        word: str,
        drop_rate: float = 0.4,
        trials: int = 32,
        candidate_words: Optional[Sequence[str]] = None,
    ) -> float:
        normalized_word = normalize_text(
            word,
            normalize_unicode=True,
            compatibility_normalization=True,
            vietnamese_normalization=True,
            repair_mojibake=True,
            lowercase=True,
            strip_invisible=True,
        )
        rng = np.random.default_rng(_hash_u64(self.seed, normalized_word, drop_rate, trials))
        base = self.encode(normalized_word)
        active = list(base)
        if not active:
            return 0.0
        hits = 0
        candidate_pool = [normalized_word]
        if candidate_words is None:
            candidate_pool.extend(sorted(self.known_words))
        else:
            for candidate in candidate_words:
                candidate_norm = normalize_text(
                    candidate,
                    normalize_unicode=True,
                    compatibility_normalization=True,
                    vietnamese_normalization=True,
                    repair_mojibake=True,
                    lowercase=True,
                    strip_invisible=True,
                )
                if candidate_norm not in candidate_pool:
                    candidate_pool.append(candidate_norm)
        for _ in range(int(trials)):
            keep = max(1, int(round(len(active) * (1.0 - float(drop_rate)))))
            kept = sorted(rng.choice(active, size=keep, replace=False).tolist())
            kept_set = set(kept)
            best = None
            best_score = -1
            for candidate in candidate_pool:
                score = len(kept_set & set(self.encode(candidate)))
                if score > best_score:
                    best_score = score
                    best = candidate
            hits += int(lexical_key(best or "") == lexical_key(normalized_word))
        return hits / max(1, int(trials))

    def reconstruction_accuracy(self, sample_words: Sequence[str], drop_rates: Sequence[float] = (0.2, 0.4, 0.6), candidate_words: Optional[Sequence[str]] = None) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for drop in drop_rates:
            values = [self.noisy_recall(word, drop_rate=float(drop), trials=8, candidate_words=candidate_words) for word in sample_words]
            result[f"drop_{int(float(drop) * 100)}"] = float(sum(values) / max(1, len(values)))
        return result


class HippocampalMemory:
    """Fast sparse auto-associative memory with slow consolidation."""

    def __init__(self, candidate_cap: int = 64, surprise_threshold: float = 0.5):
        self.candidate_cap = int(candidate_cap)
        self.surprise_threshold = float(surprise_threshold)
        self.fast: Dict[Tuple[str, ...], str] = {}
        self.slow: Dict[Tuple[str, ...], str] = {}
        self.value_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.feature_buckets: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
        self.encoded_count = 0
        self.seen_count = 0
        self.replay_count = 0
        self.last_candidate_count = 0
        self.last_full_scan = False
        self.last_resolution_confidence = 0.0
        self.last_compaction_removed = 0
        self.low_surprise_rejections = 0
        self.pollution_events = 0
        self.last_replay_budget = 0
        self.replay_surprise_total = 0.0
        self._ensure_backward_compatibility()

    def _ensure_backward_compatibility(self) -> None:
        if not hasattr(self, "candidate_cap"):
            self.candidate_cap = 64
        if not hasattr(self, "surprise_threshold"):
            self.surprise_threshold = 0.5
        if not hasattr(self, "fast"):
            self.fast = {}
        if not hasattr(self, "slow"):
            self.slow = {}
        if not hasattr(self, "value_counts"):
            self.value_counts = defaultdict(Counter)
        if not hasattr(self, "feature_buckets"):
            self.feature_buckets = defaultdict(list)
        if not hasattr(self, "encoded_count"):
            self.encoded_count = 0
        if not hasattr(self, "seen_count"):
            self.seen_count = 0
        if not hasattr(self, "replay_count"):
            self.replay_count = 0
        if not hasattr(self, "last_candidate_count"):
            self.last_candidate_count = 0
        if not hasattr(self, "last_full_scan"):
            self.last_full_scan = False
        if not hasattr(self, "last_resolution_confidence"):
            self.last_resolution_confidence = 0.0
        if not hasattr(self, "last_compaction_removed"):
            self.last_compaction_removed = 0
        if not hasattr(self, "low_surprise_rejections"):
            self.low_surprise_rejections = 0
        if not hasattr(self, "pollution_events"):
            self.pollution_events = 0
        if not hasattr(self, "last_replay_budget"):
            self.last_replay_budget = 0
        if not hasattr(self, "replay_surprise_total"):
            self.replay_surprise_total = 0.0

    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state) -> None:
        self.__dict__.update(dict(state or {}))
        self._ensure_backward_compatibility()

    @staticmethod
    def _transition_key(source: int) -> Tuple[str, str]:
        return ("transition", f"tok:{int(source)}")

    @staticmethod
    def _transition_key(source: int) -> Tuple[str, str]:
        return ("transition", f"tok:{int(source)}")

    def observe(self, features: Iterable[str], value: str, surprise: float = 1.0) -> bool:
        self.seen_count += 1
        key = tuple(sorted(_norm(f) for f in features if str(f).strip()))
        if not key:
            self.low_surprise_rejections += 1
            return False
        if float(surprise) <= self.surprise_threshold:
            self.low_surprise_rejections += 1
            self.pollution_events += 1
            return False
        self.value_counts[key][_norm(value)] += max(1.0, float(surprise))
        resolved = max(self.value_counts[key].items(), key=lambda item: (item[1], item[0]))[0]
        if key not in self.fast and key not in self.slow:
            for feature in key:
                self.feature_buckets[feature].append(key)
        self.fast[key] = resolved
        self.encoded_count += 1
        return True

    def observe_transition_ids(self, source: int, value: str, surprise: float = 1.0) -> bool:
        self.seen_count += 1
        key = self._transition_key(source)
        if float(surprise) <= self.surprise_threshold:
            self.low_surprise_rejections += 1
            self.pollution_events += 1
            return False
        self.value_counts[key][_norm(value)] += max(1.0, float(surprise))
        resolved = max(self.value_counts[key].items(), key=lambda item: (item[1], item[0]))[0]
        if key not in self.fast and key not in self.slow:
            for feature in key:
                self.feature_buckets[feature].append(key)
        self.fast[key] = resolved
        self.encoded_count += 1
        return True

    def observe_transition_ids(self, source: int, value: str, surprise: float = 1.0) -> bool:
        self.seen_count += 1
        key = self._transition_key(source)
        if float(surprise) <= self.surprise_threshold:
            self.low_surprise_rejections += 1
            self.pollution_events += 1
            return False
        self.value_counts[key][_norm(value)] += max(1.0, float(surprise))
        resolved = max(self.value_counts[key].items(), key=lambda item: (item[1], item[0]))[0]
        if key not in self.fast and key not in self.slow:
            for feature in key:
                self.feature_buckets[feature].append(key)
        self.fast[key] = resolved
        self.encoded_count += 1
        return True

    def consolidate(self, replay_fraction: float = 0.10) -> int:
        budget = max(1, int(len(self.fast) * float(replay_fraction))) if self.fast else 0
        ranked = sorted(
            self.fast.keys(),
            key=lambda key: (
                sum(float(v) for v in self.value_counts.get(key, {}).values()),
                len(key),
                key,
            ),
            reverse=True,
        )
        for key in ranked[:budget]:
            if key in self.value_counts:
                self.slow[key] = max(self.value_counts[key].items(), key=lambda item: (item[1], item[0]))[0]
            else:
                self.slow[key] = self.fast[key]
            self.replay_count += 1
            self.replay_surprise_total += sum(float(v) for v in self.value_counts.get(key, {}).values())
        self.last_replay_budget = budget
        return budget

    def prune(self, max_fast: Optional[int] = None, max_slow: Optional[int] = None) -> int:
        removed = 0
        if max_fast is not None:
            while len(self.fast) > int(max_fast):
                self.fast.pop(next(iter(self.fast)))
                removed += 1
        if max_slow is not None:
            while len(self.slow) > int(max_slow):
                self.slow.pop(next(iter(self.slow)))
                removed += 1
        if removed:
            self.feature_buckets.clear()
            for key in list(self.fast.keys()) + list(self.slow.keys()):
                for feature in key:
                    self.feature_buckets[feature].append(key)
            for key in list(self.value_counts.keys()):
                if key not in self.fast and key not in self.slow:
                    self.value_counts.pop(key, None)
            self.last_compaction_removed = removed
        return removed

    def resolve_conflict(self, cue_features: Iterable[str]) -> Optional[Dict[str, float]]:
        cues = tuple(sorted(_norm(f) for f in cue_features if str(f).strip()))
        if not cues:
            self.last_resolution_confidence = 0.0
            return None
        counter = self.value_counts.get(cues)
        if not counter:
            self.last_resolution_confidence = 0.0
            return None
        best_value, best_score = max(counter.items(), key=lambda item: (item[1], item[0]))
        total = max(1.0, float(sum(counter.values())))
        confidence = float(best_score) / total
        self.last_resolution_confidence = confidence
        return {"value": best_value, "confidence": confidence, "candidates": float(len(counter))}

    def compact(self, max_fast: Optional[int] = None, max_slow: Optional[int] = None) -> int:
        removed = self.prune(max_fast=max_fast, max_slow=max_slow)
        self.last_compaction_removed = removed
        return removed

    def recall(self, cue_features: Iterable[str]) -> Optional[str]:
        cues = tuple(sorted(_norm(f) for f in cue_features if str(f).strip()))
        if not cues:
            self.last_candidate_count = 0
            self.last_full_scan = False
            return None
        exact = self.slow.get(cues, self.fast.get(cues))
        if exact is not None:
            self.last_candidate_count = 1
            self.last_full_scan = False
            return exact
        candidates = []
        seen = set()
        for feature in cues:
            for key in reversed(self.feature_buckets.get(feature, [])):
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(key)
                if len(candidates) >= self.candidate_cap:
                    break
            if len(candidates) >= self.candidate_cap:
                break
        self.last_candidate_count = len(candidates)
        self.last_full_scan = False
        if not candidates:
            return None
        cue_set = set(cues)
        best = max(candidates, key=lambda key: (len(set(key) & cue_set), key))
        return self.slow.get(best, self.fast.get(best))

    def recall_transition_id(self, source: int) -> Optional[str]:
        key = self._transition_key(source)
        exact = self.slow.get(key, self.fast.get(key))
        if exact is not None:
            self.last_candidate_count = 1
            self.last_full_scan = False
            return exact
        candidates = []
        seen = set()
        for feature in key:
            for candidate in reversed(self.feature_buckets.get(feature, [])):
                if candidate in seen or candidate != key:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
                if len(candidates) >= self.candidate_cap:
                    break
            if len(candidates) >= self.candidate_cap:
                break
        self.last_candidate_count = len(candidates)
        self.last_full_scan = False
        if not candidates:
            return None
        best = candidates[0]
        return self.fast.get(best, self.slow.get(best))

    def recall_transition_id(self, source: int) -> Optional[str]:
        key = self._transition_key(source)
        exact = self.slow.get(key, self.fast.get(key))
        if exact is not None:
            self.last_candidate_count = 1
            self.last_full_scan = False
            return exact
        candidates = []
        seen = set()
        for feature in key:
            for candidate in reversed(self.feature_buckets.get(feature, [])):
                if candidate in seen or candidate != key:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
                if len(candidates) >= self.candidate_cap:
                    break
            if len(candidates) >= self.candidate_cap:
                break
        self.last_candidate_count = len(candidates)
        self.last_full_scan = False
        if not candidates:
            return None
        best = candidates[0]
        return self.fast.get(best, self.slow.get(best))

    def diagnostics(self) -> Dict[str, float]:
        return {
            "fast": float(len(self.fast)),
            "slow": float(len(self.slow)),
            "encoded": float(self.encoded_count),
            "seen": float(self.seen_count),
            "replay_budget": self.replay_count / max(1.0, float(self.seen_count)),
            "pollution_ratio": self.low_surprise_rejections / max(1.0, float(self.seen_count)),
            "pollution_events": float(self.pollution_events),
            "last_candidate_count": float(self.last_candidate_count),
            "last_full_scan": float(self.last_full_scan),
            "last_resolution_confidence": float(self.last_resolution_confidence),
            "last_compaction_removed": float(self.last_compaction_removed),
            "last_replay_budget": float(self.last_replay_budget),
            "replay_surprise_total": float(self.replay_surprise_total),
            "conflict_keys": float(sum(1 for counter in self.value_counts.values() if len(counter) > 1)),
        }


class BioNeuromodulator:
    """Dopamine/acetylcholine/serotonin style local update gates."""

    def __init__(self, novelty_window: int = 256):
        self.recent = deque(maxlen=int(novelty_window))
        self.seen = Counter()
        self.update_count = 0
        self.novel_update_count = 0
        self.reward_update_count = 0
        self.reward_total = 0.0
        self.weight_norm = 1.0
        self.sparsity = 0.02
        self.formal_count = 0
        self.casual_count = 0
        self.last_gates = {"dopamine": 0.0, "acetylcholine": 0.0, "serotonin": 0.0, "novelty": 0.0}
        self.dopamine_sum = 0.0
        self.acetylcholine_sum = 0.0
        self.serotonin_sum = 0.0
        self.gate_samples = 0
        self._ensure_backward_compatibility()

    def _ensure_backward_compatibility(self) -> None:
        if not hasattr(self, "recent"):
            self.recent = deque(maxlen=256)
        if not hasattr(self, "seen"):
            self.seen = Counter()
        if not hasattr(self, "update_count"):
            self.update_count = 0
        if not hasattr(self, "novel_update_count"):
            self.novel_update_count = 0
        if not hasattr(self, "reward_update_count"):
            self.reward_update_count = 0
        if not hasattr(self, "reward_total"):
            self.reward_total = 0.0
        if not hasattr(self, "weight_norm"):
            self.weight_norm = 1.0
        if not hasattr(self, "sparsity"):
            self.sparsity = 0.02
        if not hasattr(self, "formal_count"):
            self.formal_count = 0
        if not hasattr(self, "casual_count"):
            self.casual_count = 0
        if not hasattr(self, "last_gates"):
            self.last_gates = {"dopamine": 0.0, "acetylcholine": 0.0, "serotonin": 0.0, "novelty": 0.0}
        if not hasattr(self, "dopamine_sum"):
            self.dopamine_sum = 0.0
        if not hasattr(self, "acetylcholine_sum"):
            self.acetylcholine_sum = 0.0
        if not hasattr(self, "serotonin_sum"):
            self.serotonin_sum = 0.0
        if not hasattr(self, "gate_samples"):
            self.gate_samples = 0

    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state) -> None:
        self.__dict__.update(dict(state or {}))
        self._ensure_backward_compatibility()

    def gates(self, token: str, surprise: float, reward: float = 0.0) -> Dict[str, float]:
        token = _norm(token)
        novelty = 1.0 if self.seen[token] == 0 else 1.0 / (1.0 + self.seen[token])
        dopamine = min(1.0, max(0.0, 0.60 * float(surprise) + 0.35 * novelty + 0.25 * float(reward)))
        acetylcholine = min(1.0, max(0.0, 0.75 * novelty + 0.25 * float(surprise)))
        serotonin = min(1.0, max(0.0, 1.0 - abs(self.weight_norm - 1.0)))
        return {"dopamine": dopamine, "acetylcholine": acetylcholine, "serotonin": serotonin, "novelty": novelty}

    def observe(self, token: str, surprise: float, reward: float = 0.0) -> bool:
        token = _norm(token)
        g = self.gates(token, surprise, reward)
        self.last_gates = dict(g)
        self.dopamine_sum += float(g["dopamine"])
        self.acetylcholine_sum += float(g["acetylcholine"])
        self.serotonin_sum += float(g["serotonin"])
        self.gate_samples += 1
        should_update = g["dopamine"] * g["acetylcholine"] > 0.30
        if should_update:
            self.update_count += 1
            if self.seen[token] == 0 or surprise > 0.7:
                self.novel_update_count += 1
            if float(reward) > 0.0:
                self.reward_update_count += 1
        self.weight_norm += 0.01 * (g["dopamine"] - 0.5) - 0.06 * (self.weight_norm - 1.0)
        self.sparsity += 0.002 * (g["acetylcholine"] - 0.5) - 0.05 * (self.sparsity - 0.02)
        self.weight_norm = min(1.05, max(0.95, self.weight_norm))
        self.sparsity = min(0.022, max(0.018, self.sparsity))
        if token in {"therefore", "please", "regards", "sincerely"}:
            self.formal_count += 1
        if token in {"hey", "cool", "thanks", "yep"}:
            self.casual_count += 1
        self.seen[token] += 1
        self.reward_total += float(reward)
        self.recent.append(token)
        return should_update

    def observe_token_id(self, token_id: int, surprise: float, reward: float = 0.0) -> bool:
        token = f"tok:{int(token_id)}"
        g = self.gates(token, surprise, reward)
        self.last_gates = dict(g)
        self.dopamine_sum += float(g["dopamine"])
        self.acetylcholine_sum += float(g["acetylcholine"])
        self.serotonin_sum += float(g["serotonin"])
        self.gate_samples += 1
        should_update = g["dopamine"] * g["acetylcholine"] > 0.30
        if should_update:
            self.update_count += 1
            if self.seen[token] == 0 or surprise > 0.7:
                self.novel_update_count += 1
            if float(reward) > 0.0:
                self.reward_update_count += 1
        self.weight_norm += 0.01 * (g["dopamine"] - 0.5) - 0.06 * (self.weight_norm - 1.0)
        self.sparsity += 0.002 * (g["acetylcholine"] - 0.5) - 0.05 * (self.sparsity - 0.02)
        self.weight_norm = min(1.05, max(0.95, self.weight_norm))
        self.sparsity = min(0.022, max(0.018, self.sparsity))
        self.seen[token] += 1
        self.reward_total += float(reward)
        self.recent.append(token)
        return should_update

    def tone(self) -> str:
        return "formal" if self.formal_count >= self.casual_count else "casual"

    def curiosity_pick(self, candidates: Sequence[Tuple[str, float]]) -> str:
        return max(candidates, key=lambda item: (float(item[1]), _norm(item[0])))[0]

    def diagnostics(self) -> Dict[str, float]:
        return {
            "updates": float(self.update_count),
            "novel_update_ratio": self.novel_update_count / max(1.0, float(self.update_count)),
            "reward_update_ratio": self.reward_update_count / max(1.0, float(self.update_count)),
            "reward_mean": self.reward_total / max(1.0, float(len(self.seen))),
            "weight_norm": float(self.weight_norm),
            "sparsity": float(self.sparsity),
            "mean_dopamine": self.dopamine_sum / max(1.0, float(self.gate_samples)),
            "mean_acetylcholine": self.acetylcholine_sum / max(1.0, float(self.gate_samples)),
            "mean_serotonin": self.serotonin_sum / max(1.0, float(self.gate_samples)),
            "last_dopamine": float(self.last_gates.get("dopamine", 0.0)),
            "last_acetylcholine": float(self.last_gates.get("acetylcholine", 0.0)),
            "last_serotonin": float(self.last_gates.get("serotonin", 0.0)),
        }


@dataclass
class DendriticSegment:
    active_bits: Tuple[int, ...]
    output: int
    threshold: float
    strength: float = 1.0
    weights: Tuple[float, ...] = ()
    branch_id: int = 0
    last_activation: float = 0.0
    local_update_count: int = 0

    def __post_init__(self) -> None:
        self.active_bits = tuple(sorted({int(bit) for bit in self.active_bits}))
        if not self.weights:
            self.weights = tuple(1.0 for _ in self.active_bits)
        else:
            self.weights = tuple(float(weight) for weight in self.weights[: len(self.active_bits)])
            if len(self.weights) < len(self.active_bits):
                self.weights = self.weights + tuple(1.0 for _ in range(len(self.active_bits) - len(self.weights)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 60.0:
            return 1.0
        if value <= -60.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-float(value)))

    def flat_drive(self, bits: Iterable[int]) -> float:
        active = bits if isinstance(bits, set) else {int(bit) for bit in bits}
        return float(sum(weight for bit, weight in zip(self.active_bits, self.weights) if bit in active))

    def activation(self, bits: Iterable[int]) -> float:
        self.last_activation = self._sigmoid(self.flat_drive(bits) - float(self.threshold))
        return float(self.last_activation)

    def spike(self, bits: Iterable[int]) -> bool:
        return self.activation(bits) >= 0.5

    def hebbian_update(self, bits: Iterable[int], lr: float = 0.05, max_weight: float = 1.0) -> int:
        active = {int(bit) for bit in bits}
        branch_activation = self.activation(bits)
        if branch_activation < 0.5:
            return 0
        updates = 0
        new_weights = []
        for bit, weight in zip(self.active_bits, self.weights):
            if bit in active and weight < max_weight - 1e-9:
                weight = min(float(max_weight), float(weight) + float(lr) * branch_activation * (float(max_weight) - float(weight)))
                updates += 1
            new_weights.append(float(weight))
        if updates:
            self.weights = tuple(new_weights)
            self.local_update_count += 1
        return updates


class DendriticLayer:
    """Sparse nonlinear dendritic tree; each branch is a local mini-processor."""

    def __init__(
        self,
        input_dim: int = 1024,
        outputs: int = 2,
        segment_size: int = 3,
        branches_per_output: int = 0,
        branch_size: Optional[int] = None,
        soma_threshold: float = 0.5,
        seed: int = 0,
    ):
        self.input_dim = int(input_dim)
        self.outputs = int(outputs)
        self.segment_size = int(segment_size)
        self.branch_size = int(branch_size or segment_size)
        self.soma_threshold = float(soma_threshold)
        self.seed = int(seed)
        self.branches: List[DendriticSegment] = []
        self._branch_index = {}
        self._bit_to_branch_ids = defaultdict(list)
        self.segments = self.branches
        self.last_ops = 0
        self.last_active_branches = 0
        self.last_candidate_branches = 0
        self.last_updated_branches = 0
        self.branch_local_update_events = 0
        self.branch_growth_events = 0
        self.global_error_updates = 0
        self.native_predict_calls = 0
        self.native_predict_success = 0
        self._native_pack_dirty = True
        self._native_branch_bits = None
        self._native_branch_lengths = None
        self._native_branch_weights = None
        self._native_branch_thresholds = None
        self._native_branch_strengths = None
        self._native_branch_outputs = None
        if int(branches_per_output) > 0:
            self.initialize_tree(branches_per_output=int(branches_per_output), branch_size=self.branch_size)

    def _ensure_backward_compatibility(self) -> None:
        if not hasattr(self, "input_dim"):
            self.input_dim = 1024
        if not hasattr(self, "outputs"):
            self.outputs = 2
        if not hasattr(self, "segment_size"):
            self.segment_size = 3
        if not hasattr(self, "branch_size"):
            self.branch_size = int(getattr(self, "segment_size", 3))
        if not hasattr(self, "soma_threshold"):
            self.soma_threshold = 0.5
        if not hasattr(self, "seed"):
            self.seed = 0
        if not hasattr(self, "branches"):
            self.branches = []
        if not hasattr(self, "_branch_index"):
            self._branch_index = {}
        if not hasattr(self, "_bit_to_branch_ids"):
            self._bit_to_branch_ids = defaultdict(list)
        if not hasattr(self, "segments"):
            self.segments = self.branches
        if not hasattr(self, "last_ops"):
            self.last_ops = 0
        if not hasattr(self, "last_active_branches"):
            self.last_active_branches = 0
        if not hasattr(self, "last_candidate_branches"):
            self.last_candidate_branches = 0
        if not hasattr(self, "last_updated_branches"):
            self.last_updated_branches = 0
        if not hasattr(self, "branch_local_update_events"):
            self.branch_local_update_events = 0
        if not hasattr(self, "branch_growth_events"):
            self.branch_growth_events = 0
        if not hasattr(self, "global_error_updates"):
            self.global_error_updates = 0
        if not hasattr(self, "native_predict_calls"):
            self.native_predict_calls = 0
        if not hasattr(self, "native_predict_success"):
            self.native_predict_success = 0
        if not hasattr(self, "_native_pack_dirty"):
            self._native_pack_dirty = True
        if not hasattr(self, "_native_branch_bits"):
            self._native_branch_bits = None
        if not hasattr(self, "_native_branch_lengths"):
            self._native_branch_lengths = None
        if not hasattr(self, "_native_branch_weights"):
            self._native_branch_weights = None
        if not hasattr(self, "_native_branch_thresholds"):
            self._native_branch_thresholds = None
        if not hasattr(self, "_native_branch_strengths"):
            self._native_branch_strengths = None
        if not hasattr(self, "_native_branch_outputs"):
            self._native_branch_outputs = None

    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state) -> None:
        self.__dict__.update(dict(state or {}))
        self._ensure_backward_compatibility()

    def initialize_tree(self, branches_per_output: int = 1000, branch_size: Optional[int] = None) -> None:
        size = int(branch_size or self.branch_size)
        cursor = 0
        for output in range(max(1, self.outputs)):
            for _ in range(int(branches_per_output)):
                bits = tuple((cursor + offset) % self.input_dim for offset in range(size))
                cursor += size
                self.add_branch(bits, output=output, threshold=max(0.5, float(size) - 0.5))

    def add_branch(
        self,
        bits: Iterable[int],
        output: int = 0,
        threshold: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
        strength: float = 1.0,
    ) -> DendriticSegment:
        selected = tuple(sorted({int(bit) % self.input_dim for bit in bits}))
        if not selected:
            raise ValueError("Dendritic branch needs at least one active bit")
        key = (int(output), selected)
        if not hasattr(self, "_branch_index"):
            self._branch_index = {
                (int(branch.output), tuple(branch.active_bits)): branch
                for branch in self.branches
            }
        if key in self._branch_index:
            return self._branch_index[key]
        branch = DendriticSegment(
            selected,
            int(output),
            float(threshold if threshold is not None else max(0.5, len(selected) - 0.5)),
            strength=float(strength),
            weights=tuple(weights or ()),
            branch_id=len(self.branches),
        )
        self.branches.append(branch)
        self._branch_index[key] = branch
        self._bit_to_branch_ids.clear()
        self._native_pack_dirty = True
        return branch

    def grow_branch(
        self,
        bits: Iterable[int],
        output: int = 0,
        threshold: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
        strength: float = 1.0,
    ) -> DendriticSegment:
        selected = tuple(sorted({int(bit) % self.input_dim for bit in bits}))
        key = (int(output), selected)
        existed = key in self._branch_index if hasattr(self, "_branch_index") else False
        branch = self.add_branch(selected, output=output, threshold=threshold, weights=weights, strength=strength)
        if not existed:
            self.branch_growth_events += 1
        return branch

    def _ensure_bit_index(self) -> None:
        if not hasattr(self, "_bit_to_branch_ids"):
            self._bit_to_branch_ids = defaultdict(list)
        if self._bit_to_branch_ids and len(self._bit_to_branch_ids) > 0:
            return
        for idx, branch in enumerate(self.branches):
            branch.branch_id = idx
            for bit in branch.active_bits:
                self._bit_to_branch_ids[int(bit)].append(idx)

    def _rebuild_indexes(self) -> None:
        self._branch_index = {}
        self._bit_to_branch_ids = defaultdict(list)
        for idx, branch in enumerate(self.branches):
            branch.branch_id = idx
            key = (int(branch.output), tuple(branch.active_bits))
            self._branch_index[key] = branch
            for bit in branch.active_bits:
                self._bit_to_branch_ids[int(bit)].append(idx)
        self._native_pack_dirty = True

    def prune_branches(self, max_branches: int) -> int:
        max_branches = int(max_branches)
        if max_branches <= 0 or len(self.branches) <= max_branches:
            return 0
        ranked = sorted(
            self.branches,
            key=lambda branch: (
                int(branch.local_update_count),
                float(branch.strength),
                float(branch.last_activation),
                -int(branch.branch_id),
            ),
            reverse=True,
        )
        kept = ranked[:max_branches]
        removed = len(self.branches) - len(kept)
        self.branches[:] = kept
        self._rebuild_indexes()
        return removed

    def _ensure_native_pack(self) -> bool:
        if not NATIVE_AVAILABLE or not self.branches:
            return False
        if not self._native_pack_dirty and self._native_branch_bits is not None:
            return True
        width = max(1, max(len(branch.active_bits) for branch in self.branches))
        count = len(self.branches)
        bits = np.full((count, width), -1, dtype=np.intp)
        weights = np.zeros((count, width), dtype=np.float32)
        lengths = np.zeros(count, dtype=np.intp)
        thresholds = np.zeros(count, dtype=np.float32)
        strengths = np.zeros(count, dtype=np.float32)
        outputs = np.zeros(count, dtype=np.intp)
        for idx, branch in enumerate(self.branches):
            branch.branch_id = idx
            length = len(branch.active_bits)
            lengths[idx] = length
            thresholds[idx] = float(branch.threshold)
            strengths[idx] = float(branch.strength)
            outputs[idx] = int(branch.output)
            bits[idx, :length] = np.asarray(branch.active_bits, dtype=np.intp)
            weights[idx, :length] = np.asarray(branch.weights, dtype=np.float32)
        self._native_branch_bits = bits
        self._native_branch_lengths = lengths
        self._native_branch_weights = weights
        self._native_branch_thresholds = thresholds
        self._native_branch_strengths = strengths
        self._native_branch_outputs = outputs
        self._native_pack_dirty = False
        return True

    def _candidate_branch_ids(self, active: Iterable[int]) -> List[int]:
        self._ensure_bit_index()
        candidate_ids = set()
        for bit in active:
            candidate_ids.update(self._bit_to_branch_ids.get(int(bit), ()))
        return sorted(candidate_ids)

    def _packed_active_bits(self, bits: Iterable[int]) -> np.ndarray:
        if isinstance(bits, np.ndarray):
            packed = np.asarray(bits, dtype=np.intp).ravel()
        elif isinstance(bits, (list, tuple)):
            packed = np.fromiter((int(bit) % self.input_dim for bit in bits), dtype=np.intp, count=len(bits))
        else:
            packed = np.fromiter((int(bit) % self.input_dim for bit in bits), dtype=np.intp)
        if packed.size <= 1:
            return packed
        if np.all(packed[1:] > packed[:-1]):
            return packed
        return np.unique(packed)

    def observe(self, bits: Iterable[int], output: int) -> None:
        if isinstance(bits, tuple):
            selected = bits[: self.segment_size]
        else:
            selected = tuple(sorted(int(b) % self.input_dim for b in bits))[: self.segment_size]
        if len(selected) < self.segment_size:
            return
        self.add_branch(selected, output=int(output), threshold=max(0.5, float(len(selected)) - 0.5))

    def observe_or(self, bits: Iterable[int], output: int) -> None:
        selected = tuple(sorted(int(b) % self.input_dim for b in bits))[: self.segment_size]
        if selected:
            self.add_branch(selected, output=int(output), threshold=0.5)

    def branch_activations(self, bits: Iterable[int], output: Optional[int] = None) -> List[float]:
        activations = []
        self.last_ops = 0
        self.last_active_branches = 0
        active_ids = self._packed_active_bits(bits)
        active = set(int(bit) for bit in active_ids.tolist())
        candidate_ids = self._candidate_branch_ids(active_ids)
        self.last_candidate_branches = len(candidate_ids)
        if not candidate_ids:
            candidate_ids = list(range(len(self.branches)))
        for branch_id in candidate_ids:
            branch = self.branches[branch_id]
            if output is not None and int(branch.output) != int(output):
                continue
            self.last_ops += len(branch.active_bits)
            activation = branch.activation(active)
            activations.append(activation)
            if activation >= 0.5:
                self.last_active_branches += 1
        return activations

    def predict(self, bits: Iterable[int], prefer_native: bool = True) -> Optional[int]:
        active_ids = self._packed_active_bits(bits)
        candidate_ids = self._candidate_branch_ids(active_ids)
        self.last_candidate_branches = len(candidate_ids)
        if prefer_native and self._ensure_native_pack():
            self.native_predict_calls += 1
            try:
                if candidate_ids and hasattr(sparse_native, "dendrite_predict_candidates"):
                    stats = sparse_native.dendrite_predict_candidates(
                        self._native_branch_bits,
                        self._native_branch_lengths,
                        self._native_branch_weights,
                        self._native_branch_thresholds,
                        self._native_branch_strengths,
                        self._native_branch_outputs,
                        active_ids,
                        np.asarray(candidate_ids, dtype=np.intp),
                    )
                else:
                    stats = sparse_native.dendrite_predict(
                        self._native_branch_bits,
                        self._native_branch_lengths,
                        self._native_branch_weights,
                        self._native_branch_thresholds,
                        self._native_branch_strengths,
                        self._native_branch_outputs,
                        active_ids,
                    )
                self.last_ops = int(stats.get("ops", 0))
                self.last_active_branches = int(stats.get("active_branches", 0))
                best = int(stats.get("best_output", -1))
                self.native_predict_success += 1
                return best if best >= 0 else None
            except Exception:
                pass
        active = set(int(bit) for bit in active_ids.tolist())
        votes = Counter()
        self.last_ops = 0
        self.last_active_branches = 0
        candidate_ids = self._candidate_branch_ids(active)
        self.last_candidate_branches = len(candidate_ids)
        if not candidate_ids:
            return None
        for branch_id in candidate_ids:
            if branch_id < 0 or branch_id >= len(self.branches):
                continue
            branch = self.branches[branch_id]
            self.last_ops += len(branch.active_bits)
            activation = branch.activation(active)
            if activation >= 0.5:
                votes[branch.output] += branch.strength * activation
                self.last_active_branches += 1
        if not votes:
            return None
        return int(max(votes.items(), key=lambda item: (item[1], -item[0]))[0])

    def soma_activation(self, bits: Iterable[int], output: Optional[int] = None) -> float:
        activations = self.branch_activations(bits, output=output)
        suprathreshold = [activation for activation in activations if activation >= 0.5]
        return DendriticSegment._sigmoid(sum(suprathreshold) - self.soma_threshold)

    def soma_spike(self, bits: Iterable[int], output: Optional[int] = None) -> bool:
        return self.soma_activation(bits, output=output) >= 0.5

    def learn_branch_local(self, bits: Iterable[int], output: Optional[int] = None, lr: float = 0.05) -> int:
        self.last_ops = 0
        self.last_active_branches = 0
        self.last_updated_branches = 0
        update_ops = 0
        for branch in self.branches:
            if output is not None and int(branch.output) != int(output):
                continue
            self.last_ops += len(branch.active_bits)
            activation = branch.activation(bits)
            if activation >= 0.5:
                self.last_active_branches += 1
                updates = branch.hebbian_update(bits, lr=lr)
                if updates:
                    self.last_updated_branches += 1
                    self.branch_local_update_events += 1
                    self._native_pack_dirty = True
                    update_ops += updates
        return update_ops

    def branch_activity_overlap(self, samples: Sequence[Iterable[int]]) -> float:
        active_sets = []
        for branch in self.branches:
            active = {
                idx
                for idx, sample in enumerate(samples)
                if branch.activation(sample) >= 0.5
            }
            if active:
                active_sets.append(active)
        if len(active_sets) < 2:
            return 0.0
        overlaps = []
        for i in range(len(active_sets)):
            for j in range(i + 1, len(active_sets)):
                union = active_sets[i] | active_sets[j]
                if union:
                    overlaps.append(len(active_sets[i] & active_sets[j]) / len(union))
        return sum(overlaps) / max(1, len(overlaps))

    def dense_ops_proxy(self) -> int:
        return max(1, self.input_dim * max(1, len(self.branches)) * max(1, self.outputs))

    def diagnostics(self) -> Dict[str, float]:
        branches = max(1, len(self.branches))
        mean_branch_size = sum(len(branch.active_bits) for branch in self.branches) / float(branches)
        return {
            "segments": float(len(self.branches)),
            "branches": float(len(self.branches)),
            "branches_per_neuron": float(len(self.branches)) / max(1.0, float(self.outputs)),
            "mean_branch_size": float(mean_branch_size),
            "last_active_branches": float(self.last_active_branches),
            "last_active_branch_ratio": float(self.last_active_branches) / float(branches),
            "last_candidate_branches": float(self.last_candidate_branches),
            "last_candidate_branch_ratio": float(self.last_candidate_branches) / float(branches),
            "last_updated_branches": float(self.last_updated_branches),
            "last_zero_update_branch_ratio": 1.0 - float(self.last_updated_branches) / float(branches),
            "branch_local_update_events": float(self.branch_local_update_events),
            "branch_growth_events": float(self.branch_growth_events),
            "global_error_updates": float(self.global_error_updates),
            "native_predict_available": float(NATIVE_AVAILABLE),
            "native_predict_calls": float(self.native_predict_calls),
            "native_predict_success": float(self.native_predict_success),
            "native_predict_ratio": float(self.native_predict_success) / max(1.0, float(self.native_predict_calls)),
            "last_ops": float(self.last_ops),
            "dense_ops_proxy": float(self.dense_ops_proxy()),
            "compute_density_gain": float(self.dense_ops_proxy()) / max(1.0, float(self.last_ops)),
        }


class BioComputeAgent(IntegratedLSLAgent):
    """Integrated Phase 9 agent with optional biological compute modules."""

    def __init__(
        self,
        vocab_size: int = 4000,
        tokenizer: str = "subword",
        candidate_cap: int = 128,
        seed: int = 0,
        use_pc_v2: bool = True,
        use_sdr_v2: bool = True,
        use_columns: bool = True,
        use_hippocampus: bool = True,
        use_dendrites: bool = True,
        use_neuromodulation: bool = True,
    ):
        super().__init__(vocab_size=vocab_size, tokenizer=tokenizer, candidate_cap=candidate_cap, seed=seed)
        self.use_pc_v2 = bool(use_pc_v2)
        self.use_sdr_v2 = bool(use_sdr_v2)
        self.use_columns = bool(use_columns)
        self.use_hippocampus = bool(use_hippocampus)
        self.use_dendrites = bool(use_dendrites)
        self.use_neuromodulation = bool(use_neuromodulation)
        self.pc_v2 = LocalPredictiveStack(layers=3, width=256, k=8, theta=0.05)
        self.sdr_v2 = VirtualSparseSDR(dim=100000, k=20, seed=seed)
        self.columns = CorticalColumnSequenceMemory(
            vocab_size=max(128, vocab_size),
            cells_per_column=100,
            sparsity=0.02,
            seed=seed,
        )
        self.hippocampus = HippocampalMemory(candidate_cap=candidate_cap)
        self.bio_modulator = BioNeuromodulator()
        self.dendrites = DendriticLayer(input_dim=2048, outputs=8, segment_size=3)
        self.sdr_observations = 0
        self.sdr_active_total = 0

    def build_tokenizer(self, text: str) -> None:
        super().build_tokenizer(text)
        if self.use_columns and self.columns.vocab_size < self.vocab_size:
            self.columns = CorticalColumnSequenceMemory(
                vocab_size=max(128, self.vocab_size),
                cells_per_column=100,
                sparsity=0.02,
                seed=self.seed,
            )

    def observe_text(self, text: str, source: str = "text", learn_transitions: bool = True) -> None:
        super().observe_text(text, source=source, learn_transitions=learn_transitions)
        tokens = self.tokenizer.encode(text) if learn_transitions else []
        for token in tokens:
            if self.use_sdr_v2:
                bits = self.sdr_v2.encode(str(token))
                self.sdr_observations += 1
                self.sdr_active_total += len(bits)
            if self.use_columns and int(token) < self.columns.vocab_size:
                self.columns.forward(int(token), learn=True)
            if self.use_pc_v2:
                states = [self.pc_v2.state_for(token + layer * 997, layer) for layer in range(self.pc_v2.layers)]
                self.pc_v2.observe(states, learn=True)
            if self.use_neuromodulation:
                self.bio_modulator.observe(str(token), surprise=0.8 if self.bio_modulator.seen[str(token)] == 0 else 0.05)

    def predict_next_token_id(self, prefix: Sequence[int]) -> Optional[int]:
        tokens = [int(token) for token in prefix]
        if not tokens:
            return None
        votes = Counter()
        if self.use_columns:
            self.columns.reset_state()
            for token in tokens:
                if int(token) < self.columns.vocab_size:
                    self.columns.forward(int(token), learn=False)
            scores = self.columns.predict_next_token_scores()
            if float(scores.sum()) > 0.0:
                top = int(scores.argmax())
                votes[top] += float(scores[top])
        self.long_context.reset_state()
        for token in tokens[:-1]:
            self.long_context.advance_context(int(token))
        remembered, confidence = self.long_context.predict_next(
            int(tokens[-1]),
            vocab_size=self.vocab_size,
            return_confidence=True,
            update_context=False,
        )
        if remembered is not None:
            votes[int(remembered)] += max(0.01, float(confidence))
        if not votes:
            return None
        return int(max(votes.items(), key=lambda item: (item[1], -item[0]))[0])

    def observe_fact(self, key: str, value: str, surprise: float = 1.0) -> None:
        self.observe_text(f"The value for {key} is {value}.", source="bio_fact", learn_transitions=False)
        if self.use_hippocampus:
            self.hippocampus.observe(["fact", key], value, surprise=surprise)

    def recall_fact(self, key: str) -> Optional[str]:
        if self.use_hippocampus:
            value = self.hippocampus.recall(["fact", key])
            if value is not None:
                return value
        return self.answer(f"What is the value for {key}?")

    def observe_context_pattern(self, bits: Iterable[int], output: int) -> None:
        if self.use_dendrites:
            self.dendrites.observe(bits, output)

    def predict_context_pattern(self, bits: Iterable[int]) -> Optional[int]:
        if not self.use_dendrites:
            return None
        return self.dendrites.predict(bits)

    def consolidate(self, replay_fraction: float = 0.10) -> int:
        if not self.use_hippocampus:
            return 0
        return self.hippocampus.consolidate(replay_fraction=replay_fraction)

    def diagnostics(self) -> Dict[str, float]:
        column_metrics = self.columns.metrics()
        return {
            **super().diagnostics(),
            **{f"pc_v2_{k}": v for k, v in self.pc_v2.diagnostics().items()},
            "sdr_v2_observations": float(self.sdr_observations),
            "sdr_v2_mean_active": self.sdr_active_total / max(1.0, float(self.sdr_observations)),
            **{f"column_{k}": float(v) for k, v in column_metrics.items()},
            **{f"hippocampus_{k}": v for k, v in self.hippocampus.diagnostics().items()},
            **{f"bio_mod_{k}": v for k, v in self.bio_modulator.diagnostics().items()},
            **{f"dendrite_{k}": v for k, v in self.dendrites.diagnostics().items()},
        }
