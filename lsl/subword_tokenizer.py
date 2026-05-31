"""Deterministic Unicode-aware subword tokenizer.

This tokenizer keeps the lightweight local BPE spirit of the original
implementation, but adds production-oriented pieces needed by the Phase 2
foundation work:

- Unicode/Vietnamese normalization without stripping diacritics.
- UTF-8 byte fallback so unseen characters do not become `<UNK>`.
- Persistent encode/decode cache keyed by tokenizer fingerprint.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .text_normalization import normalize_text


class SimpleSubwordTokenizer:
    def __init__(
        self,
        vocab_size: int = 8000,
        max_merges: Optional[int] = None,
        min_pair_count: int = 2,
        normalize_unicode: bool = True,
        vietnamese_normalization: bool = False,
        normalization_form: str = "NFC",
        byte_fallback: bool = True,
        cache_dir: Optional[str] = None,
        cache_capacity: int = 50000,
    ):
        self.requested_vocab_size = int(vocab_size)
        self.vocab_size = int(vocab_size)
        self.max_merges = max_merges
        self.min_pair_count = int(min_pair_count)
        self.normalize_unicode = bool(normalize_unicode)
        self.vietnamese_normalization = bool(vietnamese_normalization)
        self.normalization_form = str(normalization_form or "NFC")
        self.byte_fallback = bool(byte_fallback)
        self.cache_dir = cache_dir
        self.cache_capacity = int(cache_capacity)
        self.special_tokens = ["<PAD>", "<UNK>"]
        self.required_tokens = [" ", "</w>"]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_order: List[Tuple[str, str]] = []
        self._word_cache: Dict[str, Tuple[int, ...]] = {}
        self._encode_text_cache: Dict[str, Tuple[int, ...]] = {}
        self._decode_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _normalize_text(self, text: str, *, lowercase: bool = False) -> str:
        return normalize_text(
            text,
            normalize_unicode=self.normalize_unicode,
            normalization_form=self.normalization_form,
            vietnamese_normalization=self.vietnamese_normalization,
            repair_mojibake=True,
            lowercase=lowercase,
        )

    def _words(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", self._normalize_text(text, lowercase=True), re.UNICODE)

    def _initial_units(self, word: str) -> Tuple[str, ...]:
        if re.match(r"^\w+$", word, re.UNICODE):
            return tuple(list(" " + word) + ["</w>"])
        return (word,)

    def _pair_counts(self, corpus: Dict[Tuple[str, ...], int]) -> Counter:
        counts = Counter()
        for units, freq in corpus.items():
            for pair in zip(units, units[1:]):
                counts[pair] += freq
        return counts

    def _merge_units(self, units: Tuple[str, ...], pair: Tuple[str, str], merged: str) -> Tuple[str, ...]:
        out = []
        i = 0
        while i < len(units):
            if i < len(units) - 1 and units[i] == pair[0] and units[i + 1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(units[i])
                i += 1
        return tuple(out)

    @staticmethod
    def byte_token(value: int) -> str:
        return f"<0x{int(value) & 255:02X}>"

    @staticmethod
    def is_byte_token(token: str) -> bool:
        return bool(re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", str(token)))

    @staticmethod
    def byte_value(token: str) -> int:
        return int(str(token)[3:5], 16)

    def build_vocab(self, text: str) -> None:
        word_counts = Counter(self._words(text))
        corpus = defaultdict(int)
        for word, count in word_counts.items():
            corpus[self._initial_units(word)] += int(count)

        base_units = set(self.required_tokens)
        for units in corpus:
            base_units.update(units)

        self.merges.clear()
        self.merge_order.clear()
        self._word_cache.clear()
        self._encode_text_cache.clear()
        self._decode_cache.clear()
        max_merges = self.max_merges
        reserved = len(self.special_tokens) + len(self.required_tokens) + (256 if self.byte_fallback else 0)
        if max_merges is None:
            max_merges = max(0, self.requested_vocab_size - reserved - len(base_units))

        for _ in range(max_merges):
            pairs = self._pair_counts(corpus)
            if not pairs:
                break
            pair, count = pairs.most_common(1)[0]
            if count < self.min_pair_count:
                break
            merged = pair[0] + pair[1]
            if merged in base_units:
                break
            next_corpus = defaultdict(int)
            for units, freq in corpus.items():
                next_corpus[self._merge_units(units, pair, merged)] += freq
            corpus = next_corpus
            self.merges[pair] = merged
            self.merge_order.append(pair)
            base_units.add(merged)
            if len(base_units) + reserved >= self.requested_vocab_size:
                break

        token_counts = Counter()
        for units, freq in corpus.items():
            for unit in units:
                token_counts[unit] += freq

        self.token_to_id = {}
        for token in self.special_tokens:
            self.token_to_id[token] = len(self.token_to_id)
        if self.byte_fallback:
            for byte in range(256):
                self.token_to_id[self.byte_token(byte)] = len(self.token_to_id)
        for token in self.required_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

        target_size = max(self.requested_vocab_size, len(self.token_to_id))
        for token, _ in token_counts.most_common(max(0, target_size - len(self.token_to_id))):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        if self.cache_dir:
            self.load_cache()

    def _apply_merges(self, units: Tuple[str, ...]) -> Tuple[str, ...]:
        result = units
        for pair in self.merge_order:
            result = self._merge_units(result, pair, self.merges[pair])
        return result

    def _encode_unknown_unit(self, unit: str, unk: int) -> Tuple[int, ...]:
        if not self.byte_fallback:
            return (int(unk),)
        raw = str(unit).replace("</w>", "")
        ids = [self.token_to_id[self.byte_token(byte)] for byte in raw.encode("utf-8")]
        if str(unit).endswith("</w>") and "</w>" in self.token_to_id:
            ids.append(self.token_to_id["</w>"])
        return tuple(ids) if ids else (int(unk),)

    def _encode_word(self, word: str, unk: int) -> Tuple[int, ...]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        ids: List[int] = []
        for unit in self._apply_merges(self._initial_units(word)):
            token_id = self.token_to_id.get(unit)
            if token_id is None:
                ids.extend(self._encode_unknown_unit(unit, unk))
            else:
                ids.append(int(token_id))
        encoded = tuple(ids)
        if len(self._word_cache) < self.cache_capacity:
            self._word_cache[word] = encoded
        return encoded

    def _encode_cache_key(self, normalized: str, max_tokens: Optional[int]) -> str:
        h = hashlib.sha256()
        h.update(str(max_tokens).encode("ascii"))
        h.update(b"\0")
        h.update(normalized.encode("utf-8"))
        return h.hexdigest()

    def encode(self, text: str, max_tokens: Optional[int] = None) -> List[int]:
        normalized = self._normalize_text(text, lowercase=True)
        key = self._encode_cache_key(normalized, max_tokens)
        cached = self._encode_text_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return list(cached)
        self._cache_misses += 1
        unk = self.token_to_id.get("<UNK>", 1)
        ids: List[int] = []
        words = re.finditer(r"\w+|[^\w\s]", normalized, re.UNICODE)
        for match in words:
            ids.extend(self._encode_word(match.group(0), unk))
            if max_tokens is not None and len(ids) >= int(max_tokens):
                ids = ids[: int(max_tokens)]
                break
        if len(self._encode_text_cache) < self.cache_capacity:
            self._encode_text_cache[key] = tuple(ids)
        return ids

    def _decode_cache_key(self, token_ids: Iterable[int]) -> str:
        h = hashlib.sha256()
        for idx in token_ids:
            h.update(int(idx).to_bytes(8, "little", signed=True))
        return h.hexdigest()

    def decode(self, token_ids: Iterable[int]) -> str:
        ids = tuple(int(idx) for idx in token_ids)
        key = self._decode_cache_key(ids)
        cached = self._decode_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        parts: List[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                parts.append(bytes(pending_bytes).decode("utf-8", errors="replace"))
                pending_bytes.clear()

        for idx in ids:
            token = self.id_to_token.get(int(idx), "<UNK>")
            if self.is_byte_token(token):
                pending_bytes.append(self.byte_value(token))
                continue
            flush_bytes()
            if token == "<PAD>":
                continue
            if token in self.special_tokens:
                parts.append(token)
            else:
                parts.append(str(token).replace("</w>", ""))
        flush_bytes()
        text = "".join(parts)
        text = text.replace(" </w>", " ").replace("</w>", "")
        text = re.sub(r"\s+([.,!?;:)\]])", r"\1", text)
        text = re.sub(r"([(])\s+", r"\1", text)
        out = text.strip()
        if len(self._decode_cache) < self.cache_capacity:
            self._decode_cache[key] = out
        return out

    def unk_rate(self, token_ids: Iterable[int]) -> float:
        ids = list(token_ids)
        if not ids:
            return 0.0
        unk = self.token_to_id.get("<UNK>", 1)
        return sum(1 for idx in ids if int(idx) == unk) / len(ids)

    def train_vietnamese(self, text: str) -> None:
        self.vietnamese_normalization = True
        self.normalization_form = "NFC"
        self.build_vocab(text)

    def cache_stats(self) -> Dict[str, float]:
        hits = float(self._cache_hits)
        misses = float(self._cache_misses)
        return {
            "encode_entries": float(len(self._encode_text_cache)),
            "word_entries": float(len(self._word_cache)),
            "decode_entries": float(len(self._decode_cache)),
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / max(1.0, hits + misses),
        }

    def fingerprint(self) -> str:
        payload = {
            "version": 2,
            "vocab": self.token_to_id,
            "merges": self.merge_order,
            "normalize_unicode": self.normalize_unicode,
            "vietnamese_normalization": self.vietnamese_normalization,
            "normalization_form": self.normalization_form,
            "byte_fallback": self.byte_fallback,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def enable_cache(self, cache_dir: str, load: bool = True) -> None:
        self.cache_dir = str(cache_dir)
        if load:
            self.load_cache()

    def _cache_path(self) -> Optional[Path]:
        if not self.cache_dir or not self.token_to_id:
            return None
        return Path(self.cache_dir) / f"subword_cache_{self.fingerprint()}.json.gz"

    def save_cache(self) -> Optional[str]:
        path = self._cache_path()
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "LSLSubwordTokenizerCache",
            "version": 1,
            "fingerprint": self.fingerprint(),
            "encode": {key: list(value) for key, value in list(self._encode_text_cache.items())[: self.cache_capacity]},
            "decode": dict(list(self._decode_cache.items())[: self.cache_capacity]),
        }
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return str(path)

    def load_cache(self) -> bool:
        path = self._cache_path()
        if path is None or not path.exists():
            return False
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("format") != "LSLSubwordTokenizerCache" or payload.get("fingerprint") != self.fingerprint():
            return False
        self._encode_text_cache = {
            str(key): tuple(int(x) for x in value)
            for key, value in dict(payload.get("encode", {})).items()
        }
        self._decode_cache = {str(key): str(value) for key, value in dict(payload.get("decode", {})).items()}
        return True

    def to_dict(self) -> Dict[str, object]:
        return {
            "format": "LSLSimpleSubwordTokenizer",
            "version": 2,
            "requested_vocab_size": int(self.requested_vocab_size),
            "vocab_size": int(self.vocab_size),
            "max_merges": self.max_merges,
            "min_pair_count": int(self.min_pair_count),
            "normalize_unicode": bool(self.normalize_unicode),
            "vietnamese_normalization": bool(self.vietnamese_normalization),
            "normalization_form": self.normalization_form,
            "byte_fallback": bool(self.byte_fallback),
            "special_tokens": list(self.special_tokens),
            "required_tokens": list(self.required_tokens),
            "token_to_id": dict(self.token_to_id),
            "merge_order": [list(pair) for pair in self.merge_order],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SimpleSubwordTokenizer":
        if payload.get("format") != "LSLSimpleSubwordTokenizer":
            raise ValueError("Unsupported tokenizer payload")
        tok = cls(
            vocab_size=int(payload.get("requested_vocab_size", payload.get("vocab_size", 8000))),
            max_merges=payload.get("max_merges"),
            min_pair_count=int(payload.get("min_pair_count", 2)),
            normalize_unicode=bool(payload.get("normalize_unicode", True)),
            vietnamese_normalization=bool(payload.get("vietnamese_normalization", False)),
            normalization_form=str(payload.get("normalization_form", "NFC")),
            byte_fallback=bool(payload.get("byte_fallback", True)),
        )
        tok.special_tokens = [str(x) for x in payload.get("special_tokens", ["<PAD>", "<UNK>"])]
        tok.required_tokens = [str(x) for x in payload.get("required_tokens", [" ", "</w>"])]
        tok.token_to_id = {str(key): int(value) for key, value in dict(payload.get("token_to_id", {})).items()}
        tok.id_to_token = {idx: token for token, idx in tok.token_to_id.items()}
        tok.merge_order = [tuple(str(x) for x in pair) for pair in payload.get("merge_order", [])]
        tok.merges = {tuple(pair): "".join(pair) for pair in tok.merge_order}
        tok.vocab_size = int(payload.get("vocab_size", len(tok.token_to_id)))
        return tok

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleSubwordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
