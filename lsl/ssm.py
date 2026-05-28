"""LivingSSM - state space model layer with living weights and synaptic fatigue.

It serves as a bio-plausible recurrent layer that models long-range dependencies
using an exponential-decay compressed memory state, replacing attention.
"""
import numpy as np
from .synapse import LivingSynapseLayer


class LivingSSM:
    def __init__(self, dim, slow_init=0.1, seed=None):
        self.dim = int(dim)
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 10000, size=3)

        # Automatically scale initialization to prevent signal decay in small dimensions
        ssm_slow_init = float(0.3 * (8.0 / dim) ** 0.5)
        self.B_proj = LivingSynapseLayer(dim, dim, slow_init=ssm_slow_init, seed=int(seeds[0]))
        self.C_proj = LivingSynapseLayer(dim, dim, slow_init=ssm_slow_init, seed=int(seeds[1]))

        # Learnable decay gate alpha
        self.alpha_slow = rng.uniform(0.8, 0.95, size=dim).astype(np.float32)
        self.alpha_live = np.zeros(dim, dtype=np.float32)

        # Internal state s_t
        self.s = np.zeros(dim, dtype=np.float32)
        self._step_count = 0

        # Caches for local updates
        self._last_x = None
        self._last_s_prev = None
        self._last_tanh_Bx = None
        self._last_s = None
        self._last_y = None

    def forward(self, x):
        """Forward pass for a single token vector x of shape (dim,)."""
        x = np.asarray(x, dtype=np.float32)
        self._last_x = x.copy()
        self._last_s_prev = self.s.copy()
        self._step_count += 1

        # Compute B_proj(x)
        Bx = self.B_proj.forward(x)
        tanh_Bx = np.tanh(Bx)
        self._last_tanh_Bx = tanh_Bx.copy()

        # State update
        alpha_eff = np.clip(self.alpha_slow + self.alpha_live, 0.0, 0.99)
        self.s = alpha_eff * self._last_s_prev + (1.0 - alpha_eff) * tanh_Bx
        self._last_s = self.s.copy()

        # Compute C_proj(s)
        y = self.C_proj.forward(self.s)
        self._last_y = y.copy()
        return y

    def hebbian_update(self, modulator, lr=0.05, decay=0.001, max_norm=12.0, k_frac=0.05, use_sparse=True):
        """Hebbian update for both B and C layers."""
        if self._last_x is None:
            return
        if use_sparse:
            self.B_proj.top_k_hebbian_update(modulator, lr=lr, k_frac=k_frac, decay=decay, max_norm=max_norm)
            self.C_proj.top_k_hebbian_update(modulator, lr=lr, k_frac=k_frac, decay=decay, max_norm=max_norm)
        else:
            self.B_proj.hebbian_update(modulator, lr=lr, decay=decay, max_norm=max_norm)
            self.C_proj.hebbian_update(modulator, lr=lr, decay=decay, max_norm=max_norm)

    def inference_plasticity(self, lr=0.003, max_norm=12.0):
        """Unsupervised activity-dependent plasticity during inference."""
        self.B_proj.inference_plasticity(lr=lr, max_norm=max_norm)
        self.C_proj.inference_plasticity(lr=lr, max_norm=max_norm)

    def pc_update(self, e_ssm, modulator, lr=0.05, decay=0.001, max_norm=12.0, k_frac=0.05, use_sparse=True):
        """Update SSM weights using local prediction error e_ssm.

        Args:
            e_ssm: local prediction error at the output of SSM (h_ssm - h_ssm_pred), shape (dim,)
            modulator: global neuromodulatory gain
            lr: learning rate
            decay: weight decay
            max_norm: clipping threshold
            k_frac: fraction for sparse updates
            use_sparse: whether to use top-k updates
        """
        if self._last_x is None:
            return

        gain = float(modulator)

        # Update C projection layer: output is y_t, input is s_t. Error is e_ssm.
        if use_sparse:
            self.C_proj.top_k_supervised_update(e_ssm * gain, lr=lr, k_frac=k_frac, decay=decay, max_norm=max_norm)
        else:
            self.C_proj.supervised_local_update(e_ssm * gain, lr=lr, decay=decay, max_norm=max_norm)

        # Local error projection: project e_ssm back to state s_t using W_eff.T
        W_C_eff = self.C_proj.effective_weight()
        err_s = W_C_eff.T @ e_ssm

        # Calculate error for B projection layer:
        # s_t = alpha * s_prev + (1 - alpha) * tanh(B(x))
        # d_s_t / d_B(x) = (1 - alpha) * (1 - tanh(B(x))^2)
        alpha_eff = np.clip(self.alpha_slow + self.alpha_live, 0.0, 0.99)
        err_B = err_s * (1.0 - alpha_eff) * (1.0 - self._last_tanh_Bx ** 2)

        if use_sparse:
            self.B_proj.top_k_supervised_update(err_B * gain, lr=lr, k_frac=k_frac, decay=decay, max_norm=max_norm)
        else:
            self.B_proj.supervised_local_update(err_B * gain, lr=lr, decay=decay, max_norm=max_norm)

        # Update alpha:
        # d_s_t / d_alpha = s_prev - tanh(B(x))
        d_alpha = self._last_s_prev - self._last_tanh_Bx
        # alpha_live update
        self.alpha_live += lr * gain * err_s * d_alpha

        # Simple norm clipping for alpha_live
        alpha_norm = float(np.linalg.norm(self.alpha_live))
        if alpha_norm > max_norm and alpha_norm > 0:
            self.alpha_live *= (max_norm / alpha_norm)

    def consolidate(self, threshold=0.005, fraction=0.3):
        """Transfer W_live/alpha_live -> W_slow/alpha_slow."""
        n = 0
        n += self.B_proj.consolidate(threshold, fraction)
        n += self.C_proj.consolidate(threshold, fraction)

        # Consolidate alpha_live to alpha_slow
        mask = np.abs(self.alpha_live) > float(threshold)
        if mask.any():
            self.alpha_slow[mask] += float(fraction) * self.alpha_live[mask]
            self.alpha_live[mask] *= (1.0 - float(fraction))
            self.alpha_slow = np.clip(self.alpha_slow, 0.0, 0.99)
            n += int(mask.sum())
        return n

    def decay_live(self, rate=0.999):
        self.B_proj.decay_live(rate)
        self.C_proj.decay_live(rate)
        self.alpha_live *= float(rate)

    def recover_fatigue(self, rate=0.98):
        self.B_proj.recover_fatigue(rate)
        self.C_proj.recover_fatigue(rate)

    def reset_state(self):
        self.s[:] = 0.0
        self._step_count = 0
        self._last_x = None
        self._last_s_prev = None
        self._last_tanh_Bx = None
        self._last_s = None
        self._last_y = None

    def reset_live(self):
        self.B_proj.W_live[:] = 0.0
        self.B_proj.fatigue[:] = 0.0
        self.C_proj.W_live[:] = 0.0
        self.C_proj.fatigue[:] = 0.0
        self.alpha_live[:] = 0.0
        self.reset_state()

    def live_norm(self):
        return float(np.linalg.norm(self.B_proj.W_live) +
                     np.linalg.norm(self.C_proj.W_live) +
                     np.linalg.norm(self.alpha_live))

    def slow_norm(self):
        return float(np.linalg.norm(self.B_proj.W_slow) +
                     np.linalg.norm(self.C_proj.W_slow) +
                     np.linalg.norm(self.alpha_slow))

    def effective_weight_norms(self):
        return {
            "b": self.B_proj.effective_weight_norm(),
            "c": self.C_proj.effective_weight_norm(),
        }
