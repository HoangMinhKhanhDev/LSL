"""Neuromodulator - global gating signal for plasticity.

Mimics how dopamine / noradrenaline gate learning in the brain:

- High prediction error (surprise) increases plasticity.
- Excessively high surprise (chaos) is suppressed - chaos guard.
- Reward and novelty both bias the modulator upward.
- A baseline tracks the running average of surprise so the system reacts to
  *relative* surprise, not absolute scale.
"""
import numpy as np
from collections import deque


class Neuromodulator:
    def __init__(self, recent_capacity=64):
        self.surprise_baseline = 1.0
        self.surprise_var = 1.0
        self.recent_inputs = deque(maxlen=int(recent_capacity))

    def update_baseline(self, surprise):
        s = float(surprise)
        self.surprise_baseline = 0.95 * self.surprise_baseline + 0.05 * s
        diff = s - self.surprise_baseline
        self.surprise_var = 0.95 * self.surprise_var + 0.05 * (diff * diff)

    def novelty(self, token_id):
        n_recent = sum(1 for t in self.recent_inputs if t == int(token_id))
        nov = 1.0 / (1.0 + n_recent)
        self.recent_inputs.append(int(token_id))
        return float(nov)

    def compute(self, surprise, novelty=0.5, reward=0.0):
        std = max(float(np.sqrt(self.surprise_var)), 1e-4)
        z = (float(surprise) - self.surprise_baseline) / std
        self.update_baseline(surprise)
        if z > 4.0:
            base = 0.1
        elif z >= 0.0:
            base = float(np.tanh(0.5 * z))
        else:
            base = 0.1 * float(np.tanh(z))
        return float(np.clip(base + 0.3 * float(novelty) + 0.5 * float(reward),
                             -1.0, 2.0))

    def reset(self):
        self.surprise_baseline = 1.0
        self.surprise_var = 1.0
        self.recent_inputs.clear()
