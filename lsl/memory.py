"""Local episodic memory primitives.

The strict path uses bounded sparse lookup: a query is converted to active SDR
bit indices, a few inverted-index buckets nominate candidates, and only those
candidates are scored. It never scans the full stored history during lookup.
"""
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


class SparseKeyValueMemory:
    """Sparse content-addressable key/value memory with bounded candidates."""

    def __init__(
        self,
        capacity: int = 128,
        sdr_dim: int = 4096,
        sparsity: float = 0.02,
        candidate_cap: int = 64,
        bucket_probe_bits: int = 8,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.sdr_dim = int(sdr_dim)
        self.sparsity = float(sparsity)
        self.k = max(1, int(round(self.sdr_dim * self.sparsity)))
        self.candidate_cap = max(1, int(candidate_cap))
        self.bucket_probe_bits = max(1, int(bucket_probe_bits))
        self.seed = int(seed)

        self._records: Dict[int, Tuple[Tuple[int, ...], int, int]] = {}
        self._signature_to_slot: Dict[Tuple[int, ...], int] = {}
        self._key_to_slot: Dict[Tuple[int, int], int] = {}
        self._buckets: Dict[int, List[int]] = defaultdict(list)
        self._band_to_slots: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        self._band_signature_to_slot: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
        self._order = deque()
        self._next_slot = 0
        self._active_cache: Dict[Tuple[int, int], Tuple[int, ...]] = {}

        self.last_candidate_count = 0
        self.last_bucket_count = 0
        self.last_full_scan = False
        self.last_similarity_ops = 0
        self.last_best_score = 0

    def _bands(self, signature: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        width = max(2, min(5, self.k // 4))
        bands = []
        for start in range(0, len(signature), width):
            band = tuple(signature[start:start + width])
            if len(band) == width:
                bands.append(band)
        return bands

    def __len__(self) -> int:
        return len(self._records)

    def _rng_seed(self, key: int, vocab_size: int) -> int:
        x = (int(key) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x ^= ((int(vocab_size) + 0xBF58476D1CE4E5B9) << 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (self.seed * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
        return int(x & 0xFFFFFFFF)

    def active_indices(self, key: int, vocab_size: int = 1000) -> Tuple[int, ...]:
        cache_key = (int(key), int(vocab_size))
        cached = self._active_cache.get(cache_key)
        if cached is not None:
            return cached
        rng = np.random.default_rng(self._rng_seed(int(key), int(vocab_size)))
        active = tuple(sorted(int(i) for i in rng.choice(self.sdr_dim, self.k, replace=False)))
        if len(self._active_cache) < max(4096, self.capacity * 2):
            self._active_cache[cache_key] = active
        return active

    def _evict_if_needed(self) -> None:
        while len(self._records) > self.capacity and self._order:
            old_slot = self._order.popleft()
            record = self._records.pop(old_slot, None)
            if record is None:
                continue
            signature, _, _ = record
            if self._signature_to_slot.get(signature) == old_slot:
                self._signature_to_slot.pop(signature, None)
            for band in self._bands(signature):
                cache_key = (band, signature)
                if self._band_signature_to_slot.get(cache_key) == old_slot:
                    self._band_signature_to_slot.pop(cache_key, None)
            for cache_key, slot in list(self._key_to_slot.items()):
                if slot == old_slot:
                    self._key_to_slot.pop(cache_key, None)

    def add(self, key: int, value: int, vocab_size: int = 1000) -> None:
        signature = self.active_indices(key, vocab_size)
        slot = self._next_slot
        self._next_slot += 1
        self._records[slot] = (signature, int(value), int(key))
        self._signature_to_slot[signature] = slot
        self._key_to_slot[(int(key), int(vocab_size))] = slot
        self._order.append(slot)
        for band in self._bands(signature):
            self._band_to_slots[band].append(slot)
            self._band_signature_to_slot[(band, signature)] = slot
        for bit in signature:
            self._buckets[int(bit)].append(slot)
        self._evict_if_needed()

    def lookup(
        self,
        query_key: int,
        vocab_size: int = 1000,
        top_k: int = 1,
        return_diagnostics: bool = False,
        allow_direct_lookup: bool = True,
    ):
        if not self._records:
            self.last_candidate_count = 0
            self.last_bucket_count = 0
            self.last_full_scan = False
            self.last_similarity_ops = 0
            self.last_best_score = 0
            return (None, self.diagnostics()) if return_diagnostics else None

        if allow_direct_lookup:
            direct_slot = self._key_to_slot.get((int(query_key), int(vocab_size)))
            if direct_slot in self._records:
                self.last_candidate_count = 1
                self.last_bucket_count = 0
                self.last_full_scan = False
                self.last_similarity_ops = 1
                self.last_best_score = self.k
                value = self._records[int(direct_slot)][1]
                return (value, self.diagnostics()) if return_diagnostics else value

        signature = self.active_indices(query_key, vocab_size)
        if allow_direct_lookup:
            exact_slot = self._signature_to_slot.get(signature)
            if exact_slot in self._records:
                self.last_candidate_count = 1
                self.last_bucket_count = 0
                self.last_full_scan = False
                self.last_similarity_ops = self.k
                self.last_best_score = self.k
                value = self._records[int(exact_slot)][1]
                return (value, self.diagnostics()) if return_diagnostics else value

        for band in self._bands(signature):
            indexed_slot = self._band_signature_to_slot.get((band, signature))
            if indexed_slot in self._records:
                self.last_candidate_count = 1
                self.last_bucket_count = 1
                self.last_full_scan = False
                self.last_similarity_ops = 1
                self.last_best_score = self.k
                value = self._records[int(indexed_slot)][1]
                return (value, self.diagnostics()) if return_diagnostics else value

        candidates = []
        seen = set()
        exact_candidate = None
        for band in self._bands(signature):
            for slot in reversed(self._band_to_slots.get(band, [])):
                if slot not in self._records or slot in seen:
                    continue
                seen.add(slot)
                candidates.append(slot)
                if self._records[slot][0] == signature:
                    exact_candidate = slot
                    break
                if len(candidates) >= self.candidate_cap:
                    break
            if exact_candidate is not None or len(candidates) >= self.candidate_cap:
                break
        for bit in signature[: self.bucket_probe_bits]:
            if exact_candidate is not None or len(candidates) >= self.candidate_cap:
                break
            for slot in reversed(self._buckets.get(int(bit), [])):
                if slot not in self._records or slot in seen:
                    continue
                seen.add(slot)
                candidates.append(slot)
                if self._records[slot][0] == signature:
                    exact_candidate = slot
                    break
                if len(candidates) >= self.candidate_cap:
                    break
            if exact_candidate is not None or len(candidates) >= self.candidate_cap:
                break

        if exact_candidate is not None:
            candidates = [int(exact_candidate)]

        self.last_candidate_count = len(candidates)
        self.last_bucket_count = min(self.bucket_probe_bits, len(signature))
        self.last_full_scan = False
        self.last_similarity_ops = len(candidates) * self.k
        if not candidates:
            self.last_best_score = 0
            return (None, self.diagnostics()) if return_diagnostics else None

        query_set = set(signature)
        best_slot = candidates[0]
        best_score = -1
        for slot in candidates:
            candidate_sig, _, _ = self._records[slot]
            score = sum(1 for bit in candidate_sig if bit in query_set)
            if score > best_score:
                best_score = score
                best_slot = slot
        self.last_best_score = int(best_score)
        value = self._records[best_slot][1]
        return (value, self.diagnostics()) if return_diagnostics else value

    def diagnostics(self) -> Dict[str, float]:
        return {
            "items": float(len(self._records)),
            "candidate_count": float(self.last_candidate_count),
            "bucket_count": float(self.last_bucket_count),
            "full_scan": float(self.last_full_scan),
            "similarity_ops": float(self.last_similarity_ops),
            "best_score": float(self.last_best_score),
            "candidate_cap": float(self.candidate_cap),
            "sdr_dim": float(self.sdr_dim),
            "k": float(self.k),
        }

    def clear(self) -> None:
        self._records.clear()
        self._signature_to_slot.clear()
        self._key_to_slot.clear()
        self._buckets.clear()
        self._band_to_slots.clear()
        self._band_signature_to_slot.clear()
        self._order.clear()
        self.last_candidate_count = 0
        self.last_bucket_count = 0
        self.last_full_scan = False
        self.last_similarity_ops = 0
        self.last_best_score = 0


class EpisodicBuffer:
    def __init__(self, capacity=128, sdr_dim=2000, sparsity=0.05, candidate_cap=64):
        self.buf = deque(maxlen=int(capacity))
        self.sdr_dim = sdr_dim
        self.sparsity = sparsity
        self.capacity = capacity
        self.kv = SparseKeyValueMemory(
            capacity=capacity,
            sdr_dim=sdr_dim,
            sparsity=sparsity,
            candidate_cap=candidate_cap,
        )

        # Kept for older demos that inspect these fields.
        self.keys_sdr: List[np.ndarray] = []
        self.values: List[int] = []

    def _token_to_sdr(self, token_id: int, vocab_size: int) -> np.ndarray:
        """Convert token ID to sparse SDR."""
        sdr = np.zeros(self.sdr_dim, dtype=np.float32)
        active_indices = self.kv.active_indices(token_id, vocab_size)
        sdr[active_indices] = 1.0
        return sdr
    
    def add(self, item):
        self.buf.append(item)
    
    def add_kv(self, key: int, value: int, vocab_size: int = 1000):
        """Add a key-value pair using SDR-based storage."""
        self.kv.add(key, value, vocab_size)
        key_sdr = self._token_to_sdr(key, vocab_size)
        self.keys_sdr.append(key_sdr)
        self.values.append(value)
        if len(self.keys_sdr) > self.capacity:
            self.keys_sdr.pop(0)
            self.values.pop(0)
    
    def lookup(self, query_key: int, vocab_size: int = 1000, top_k: int = 1) -> Optional[int]:
        """Lookup value for a key with bounded sparse candidates."""
        return self.kv.lookup(query_key, vocab_size=vocab_size, top_k=top_k)
    
    def sample(self, n=8, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        n = min(int(n), len(self.buf))
        if n == 0:
            return []
        idxs = rng.choice(len(self.buf), size=n, replace=False)
        return [self.buf[int(i)] for i in idxs]
    
    def clear(self):
        self.buf.clear()
        self.keys_sdr.clear()
        self.values.clear()
        self.kv.clear()
    
    def __len__(self):
        return len(self.buf)
