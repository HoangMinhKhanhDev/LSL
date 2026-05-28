"""EpisodicBuffer - short-lived store of recent experiences.

Used for replay-based consolidation, analogous to hippocampal replay during
quiet wakefulness or sleep. Replay drives selective transfer of plastic
weights into long-term storage.
"""
import numpy as np
from collections import deque


class EpisodicBuffer:
    def __init__(self, capacity=128):
        self.buf = deque(maxlen=int(capacity))

    def add(self, item):
        self.buf.append(item)

    def sample(self, n=8, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        n = min(int(n), len(self.buf))
        if n == 0:
            return []
        idxs = rng.choice(len(self.buf), size=n, replace=False)
        return [self.buf[int(i)] for i in idxs]

    def clear(self):
        self.buf.clear()

    def __len__(self):
        return len(self.buf)
