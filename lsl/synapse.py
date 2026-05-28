"""LivingSynapseLayer - local online synapse primitive.

The layer has stable weights (W_slow), plastic weights (W_live), and synaptic
fatigue. Forward passes never mutate W_live; learning happens only through
explicit local update methods.
"""
import numpy as np


class LivingSynapseLayer:
    def __init__(self, in_dim, out_dim, slow_init=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.W_slow = np.asfortranarray(
            rng.standard_normal((out_dim, in_dim)).astype(np.float32)
            * float(slow_init)
        )
        self.W_live = np.zeros((out_dim, in_dim), dtype=np.float32, order="F")
        self.fatigue = np.zeros((out_dim, in_dim), dtype=np.float32, order="F")
        self._last_pre = None
        self._last_post = None
        self.last_forward_ops = {
            "mode": "none",
            "ops": 0,
            "active_inputs": 0,
            "fatigue_touched": 0,
        }

    def effective_weight(self):
        return (self.W_slow + self.W_live) * (1.0 - self.fatigue)

    def effective_weight_norm(self):
        return float(np.linalg.norm(self.effective_weight()))

    def forward(self, x, use_sparse=False, return_stats=False):
        """Compute post-synaptic activity and fatigue.

        Sparse mode touches only active input columns. This is the strict SDR
        path: no full effective matrix, no dense fatigue outer product.
        """
        x = np.asarray(x, dtype=np.float32)
        active = np.where(np.abs(x) > 1e-5)[0]

        if (use_sparse or len(active) == 1) and 0 < len(active) < 0.5 * len(x):
            active = active.astype(np.intp, copy=False)
            x_active = x[active]
            fatigue_cols = self.fatigue[:, active]
            W_cols = (
                self.W_slow[:, active] + self.W_live[:, active]
            ) * (1.0 - fatigue_cols)
            post = W_cols @ x_active

            max_s = float(np.max(np.abs(post))) + 1e-8
            self.fatigue[:, active] = (
                0.98 * fatigue_cols
                + 0.02 * np.abs(post[:, None] * x_active[None, :]) / max_s
            )
            self.last_forward_ops = {
                "mode": "sparse",
                "ops": int(self.out_dim * len(active)),
                "active_inputs": int(len(active)),
                "fatigue_touched": int(self.out_dim * len(active)),
            }
        else:
            W_eff = self.effective_weight()
            post = W_eff @ x
            sig = np.abs(np.outer(post, x))
            max_s = float(sig.max()) + 1e-8
            self.fatigue = 0.98 * self.fatigue + 0.02 * (sig / max_s)
            self.last_forward_ops = {
                "mode": "dense",
                "ops": int(self.out_dim * self.in_dim),
                "active_inputs": int(len(active)),
                "fatigue_touched": int(self.out_dim * self.in_dim),
            }

        np.clip(self.fatigue, 0.0, 0.9, out=self.fatigue)
        self._last_pre = x.copy()
        self._last_post = post.copy()
        if return_stats:
            return post, dict(self.last_forward_ops)
        return post

    def _active_pre(self):
        if self._last_pre is None:
            return np.array([], dtype=np.intp)
        return np.where(np.abs(self._last_pre) > 1e-5)[0].astype(np.intp)

    def hebbian_update(self, modulator, lr=0.05, decay=0.001, max_norm=12.0):
        if self._last_pre is None:
            return
        pre, post = self._last_pre, self._last_post
        active = self._active_pre()
        if 0 < len(active) < 0.1 * len(pre):
            self.W_live[:, active] *= (1.0 - lr * decay)
            for c in active:
                self.W_live[:, c] += lr * float(modulator) * post * pre[c]
        else:
            self.W_live *= (1.0 - lr * decay)
            self.W_live += lr * float(modulator) * np.outer(post, pre)
        self._clip_norm(max_norm)

    def supervised_local_update(self, error_signal, lr=0.05, decay=0.001,
                                max_norm=12.0):
        if self._last_pre is None:
            return
        pre = self._last_pre
        err = np.asarray(error_signal, dtype=np.float32)
        active = self._active_pre()
        if 0 < len(active) < 0.1 * len(pre):
            self.W_live[:, active] *= (1.0 - lr * decay)
            for c in active:
                self.W_live[:, c] += lr * err * pre[c]
        else:
            self.W_live *= (1.0 - lr * decay)
            self.W_live += lr * np.outer(err, pre)
        self._clip_norm(max_norm)

    def top_k_hebbian_update(self, modulator, lr=0.05, k_frac=0.05,
                             decay=0.001, max_norm=12.0):
        if self._last_pre is None:
            return
        if self.in_dim <= 32 or self.out_dim <= 32:
            self.hebbian_update(modulator, lr=lr, decay=decay,
                                max_norm=max_norm)
            return

        pre, post = self._last_pre, self._last_post
        active = self._active_pre()
        if len(active) == 0:
            return

        k_pre = max(1, int(k_frac * len(active)))
        k_post = max(1, int(k_frac * self.out_dim))
        top_pre = active[np.argpartition(np.abs(pre[active]), -k_pre)[-k_pre:]]
        top_post = np.argpartition(np.abs(post), -k_post)[-k_post:]

        self.W_live[np.ix_(top_post, top_pre)] *= (1.0 - lr * decay)
        mod = float(modulator)
        for c in top_pre:
            self.W_live[top_post, c] += lr * mod * post[top_post] * pre[c]
        self._clip_norm(max_norm)

    def top_k_supervised_update(self, error_signal, lr=0.05, k_frac=0.05,
                                decay=0.001, max_norm=12.0):
        if self._last_pre is None:
            return
        if self.in_dim <= 32 or self.out_dim <= 32:
            self.supervised_local_update(error_signal, lr=lr, decay=decay,
                                         max_norm=max_norm)
            return

        pre = self._last_pre
        err = np.asarray(error_signal, dtype=np.float32)
        active = self._active_pre()
        if len(active) == 0:
            return

        k_pre = max(1, int(k_frac * len(active)))
        k_err = max(1, int(k_frac * self.out_dim))
        top_pre = active[np.argpartition(np.abs(pre[active]), -k_pre)[-k_pre:]]
        top_err = np.argpartition(np.abs(err), -k_err)[-k_err:]

        self.W_live[np.ix_(top_err, top_pre)] *= (1.0 - lr * decay)
        for c in top_pre:
            self.W_live[top_err, c] += lr * err[top_err] * pre[c]
        self._clip_norm(max_norm)

    def inference_plasticity(self, lr=0.003, max_norm=12.0):
        if self._last_pre is None:
            return
        pre, post = self._last_pre, self._last_post
        active = self._active_pre()
        if 0 < len(active) < 0.1 * len(pre):
            for c in active:
                self.W_live[:, c] += lr * post * pre[c]
        else:
            self.W_live += lr * np.outer(post, pre)
        self._clip_norm(max_norm)

    def decay_live(self, rate=0.999):
        self.W_live *= float(rate)

    def recover_fatigue(self, rate=0.98):
        self.fatigue *= float(rate)

    def consolidate(self, threshold=0.005, fraction=0.3):
        mask = np.abs(self.W_live) > float(threshold)
        if not mask.any():
            return 0
        self.W_slow[mask] += float(fraction) * self.W_live[mask]
        self.W_live[mask] *= (1.0 - float(fraction))
        return int(mask.sum())

    def _clip_norm(self, max_norm):
        n = float(np.linalg.norm(self.W_live))
        if n > max_norm and n > 0:
            self.W_live *= max_norm / n
