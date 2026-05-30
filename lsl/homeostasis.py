"""Local self-tuning controller for sparse online learning."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class HomeostaticState:
    sparsity: float = 0.02
    fatigue_rate: float = 0.18
    decay_rate: float = 0.995
    suppression_threshold: float = 0.02
    local_lr: float = 0.05


class HomeostaticController:
    """Keeps sparse activity and update strength in stable local ranges."""

    def __init__(
        self,
        target_sparsity: float = 0.02,
        target_error: float = 0.10,
        adapt_rate: float = 0.01,
        min_lr: float = 0.005,
        max_lr: float = 0.12,
    ):
        self.target_sparsity = float(target_sparsity)
        self.target_error = float(target_error)
        self.adapt_rate = float(adapt_rate)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.state = HomeostaticState(sparsity=target_sparsity)
        self.steps = 0
        self.error_ema = target_error
        self.sparsity_ema = target_sparsity

    def observe(self, active_count: int, total_count: int, local_error: float) -> HomeostaticState:
        observed_sparsity = float(active_count) / max(1.0, float(total_count))
        err = max(0.0, float(local_error))
        self.steps += 1
        ema = 0.10
        self.sparsity_ema = (1.0 - ema) * self.sparsity_ema + ema * observed_sparsity
        self.error_ema = (1.0 - ema) * self.error_ema + ema * err

        sparse_delta = self.sparsity_ema - self.target_sparsity
        error_delta = self.error_ema - self.target_error
        rate = self.adapt_rate

        self.state.suppression_threshold *= 1.0 + 0.05 * rate * sparse_delta / max(self.target_sparsity, 1e-6)
        self.state.suppression_threshold = min(0.20, max(0.001, self.state.suppression_threshold))

        self.state.fatigue_rate *= 1.0 + 0.15 * rate * sparse_delta / max(self.target_sparsity, 1e-6)
        self.state.fatigue_rate = min(0.60, max(0.02, self.state.fatigue_rate))

        self.state.local_lr *= 1.0 + 0.25 * rate * error_delta / max(self.target_error, 1e-6)
        self.state.local_lr = min(self.max_lr, max(self.min_lr, self.state.local_lr))

        if self.error_ema < self.target_error * 0.8:
            self.state.decay_rate = min(0.9995, self.state.decay_rate + 0.0002)
        elif self.error_ema > self.target_error * 1.2:
            self.state.decay_rate = max(0.9700, self.state.decay_rate - 0.0003)

        self.state.sparsity = self.target_sparsity
        return self.state

    def diagnostics(self) -> Dict[str, float]:
        return {
            "steps": float(self.steps),
            "target_sparsity": float(self.target_sparsity),
            "sparsity_ema": float(self.sparsity_ema),
            "target_error": float(self.target_error),
            "error_ema": float(self.error_ema),
            "fatigue_rate": float(self.state.fatigue_rate),
            "decay_rate": float(self.state.decay_rate),
            "suppression_threshold": float(self.state.suppression_threshold),
            "local_lr": float(self.state.local_lr),
        }
