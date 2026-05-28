"""DynamicCircuitRouter - real-time circuit routing.

At each forward step only the top-k most relevant neurons are gated through.
Routing depends on the current input AND the global state, so the *path* the
signal takes through the network changes step-by-step.

Biological analogue: lateral inhibition + cortical attention - only some
neuron populations participate in any given thought, the rest are suppressed.
"""
import numpy as np


class DynamicCircuitRouter:
    def __init__(self, n_neurons, k_ratio=0.3, usage_penalty=0.1, history_decay=0.95):
        self.n_neurons = int(n_neurons)
        self.k = max(1, int(self.n_neurons * float(k_ratio)))
        self.usage_penalty = float(usage_penalty)
        self.history_decay = float(history_decay)
        self.usage_history = np.zeros(self.n_neurons, dtype=np.float32)
        self._last_mask = None

    def gate(self, activations, state):
        a = np.abs(np.asarray(activations, dtype=np.float32))
        s = np.abs(np.asarray(state, dtype=np.float32))
        score = a + 0.5 * s - self.usage_penalty * self.usage_history
        if self.k >= self.n_neurons:
            mask = np.ones_like(score)
        else:
            thresh = np.partition(score, -self.k)[-self.k]
            mask = (score >= thresh).astype(np.float32)
        self.usage_history = (self.history_decay * self.usage_history
                              + (1.0 - self.history_decay) * mask)
        self._last_mask = mask.copy()
        return mask

    def last_mask(self):
        """Return the routing mask from the most recent gate() call."""
        return self._last_mask

    def active_indices(self):
        """Return indices of active neurons from the last gate() call."""
        if self._last_mask is None:
            return []
        return [i for i, v in enumerate(self._last_mask) if v > 0.5]

    def reset(self):
        self.usage_history[:] = 0.0
