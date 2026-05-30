"""Small deterministic subword tokenizer.

This is a lightweight BPE-style tokenizer for benchmark use. It trains merges
from corpus text, falls back to character units, and therefore sharply reduces
unknown tokens compared with a fixed word vocabulary.
"""
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


class SimpleSubwordTokenizer:
    def __init__(self, vocab_size: int = 8000, max_merges: int = None, min_pair_count: int = 2):
        self.vocab_size = int(vocab_size)
        self.max_merges = max_merges
        self.min_pair_count = int(min_pair_count)
        self.special_tokens = ["<PAD>", "<UNK>"]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_order: List[Tuple[str, str]] = []

    def _words(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

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

    def build_vocab(self, text: str) -> None:
        word_counts = Counter(self._words(text))
        corpus = defaultdict(int)
        for word, count in word_counts.items():
            corpus[self._initial_units(word)] += int(count)

        base_units = set()
        for units in corpus:
            base_units.update(units)

        self.merges.clear()
        self.merge_order.clear()
        max_merges = self.max_merges
        if max_merges is None:
            max_merges = max(0, self.vocab_size - len(self.special_tokens) - len(base_units))

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
            if len(base_units) + len(self.special_tokens) >= self.vocab_size:
                break

        token_counts = Counter()
        for units, freq in corpus.items():
            for unit in units:
                token_counts[unit] += freq

        self.token_to_id = {}
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx

        for token, _ in token_counts.most_common(max(0, self.vocab_size - len(self.special_tokens))):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    def _apply_merges(self, units: Tuple[str, ...]) -> Tuple[str, ...]:
        result = units
        for pair in self.merge_order:
            result = self._merge_units(result, pair, self.merges[pair])
        return result

    def encode(self, text: str) -> List[int]:
        unk = self.token_to_id.get("<UNK>", 1)
        ids: List[int] = []
        for word in self._words(text):
            units = self._apply_merges(self._initial_units(word))
            ids.extend(self.token_to_id.get(unit, unk) for unit in units)
        return ids

    def decode(self, token_ids: Iterable[int]) -> str:
        text = ""
        for idx in token_ids:
            token = self.id_to_token.get(int(idx), "<UNK>")
            if token in self.special_tokens:
                piece = token
            else:
                piece = token.replace("</w>", "")
            text += piece
        text = text.replace(" </w>", " ").replace("</w>", "")
        return re.sub(r"\s+([.,!?;:)\]])", r"\1", text).strip()

    def unk_rate(self, token_ids: Iterable[int]) -> float:
        ids = list(token_ids)
        if not ids:
            return 0.0
        unk = self.token_to_id.get("<UNK>", 1)
        return sum(1 for idx in ids if int(idx) == unk) / len(ids)
