"""Optional offline semantic prior quantized into SDRs."""
import json
import os
import hashlib
from typing import Dict, Iterable, Tuple

import numpy as np


class OfflinePriorSDR:
    """Loads checked-in embeddings and converts them to sparse binary codes."""

    def __init__(self, dim: int = 1024, k: int = 20, seed: int = 0):
        self.dim = int(dim)
        self.k = int(k)
        self.seed = int(seed)
        self.embeddings: Dict[str, np.ndarray] = {}

    def load_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if "groups" in raw and "basis" in raw:
            vectors = {}
            for group, tokens in raw["groups"].items():
                basis = np.asarray(raw["basis"][group], dtype=np.float32)
                for token in tokens:
                    vectors[str(token)] = basis
            self.embeddings = vectors
            return
        self.embeddings = {str(key): np.asarray(value, dtype=np.float32) for key, value in raw.items()}

    def load_builtin(self) -> None:
        root = os.path.dirname(__file__)
        self.load_json(os.path.join(root, "data", "mini_semantic_embeddings.json"))

    def encode(self, token: str) -> Tuple[int, ...]:
        token = str(token).lower()
        vec = self.embeddings.get(token)
        if vec is None:
            digest = hashlib.blake2b(f"{token}:{self.seed}".encode("utf-8"), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little") & 0xFFFFFFFF)
            return tuple(sorted(int(i) for i in rng.choice(self.dim, self.k, replace=False)))
        rng = np.random.default_rng(self.seed)
        projection = rng.standard_normal((len(vec), self.dim)).astype(np.float32)
        scores = vec @ projection
        active = np.argpartition(scores, -self.k)[-self.k:]
        return tuple(sorted(int(i) for i in active))

    @staticmethod
    def overlap(a: Iterable[int], b: Iterable[int]) -> int:
        return len(set(int(x) for x in a) & set(int(x) for x in b))
