"""Event-driven sparse state memory for replacing dense recurrent bottlenecks."""
from typing import Dict, Iterable, Tuple

import numpy as np


class EventDrivenSSM:
    """Sparse state update that touches only active event dimensions."""

    def __init__(self, dim: int, decay: float = 0.92, active_cap: int = 128):
        self.dim = int(dim)
        self.decay = float(decay)
        self.active_cap = int(active_cap)
        self.state: Dict[int, float] = {}
        self.last_ops = 0

    def forward(self, active_indices: Iterable[int], values=None) -> Tuple[np.ndarray, Dict[str, float]]:
        idxs = [int(i) % self.dim for i in active_indices]
        if values is None:
            vals = [1.0] * len(idxs)
        else:
            vals = [float(v) for v in values]
        touched = set(idxs) | set(self.state.keys())
        for idx in list(touched):
            value = self.state.get(idx, 0.0) * self.decay
            self.state[idx] = value
            if abs(value) < 1e-6:
                self.state.pop(idx, None)
        for idx, value in zip(idxs, vals):
            self.state[idx] = self.state.get(idx, 0.0) + float(value)
        if len(self.state) > self.active_cap:
            keep = sorted(self.state.items(), key=lambda item: abs(item[1]), reverse=True)[: self.active_cap]
            self.state = {int(k): float(v) for k, v in keep}
        out = np.zeros(self.dim, dtype=np.float32)
        for idx, value in self.state.items():
            out[int(idx)] = float(value)
        self.last_ops = len(touched) + len(idxs) + len(self.state)
        return out, self.diagnostics()

    def diagnostics(self) -> Dict[str, float]:
        return {
            "active_state": float(len(self.state)),
            "ops": float(self.last_ops),
            "dense_ops_equivalent": float(self.dim * 2),
            "ops_fraction": float(self.last_ops / max(1, self.dim * 2)),
        }

    def reset_state(self) -> None:
        self.state.clear()
        self.last_ops = 0
