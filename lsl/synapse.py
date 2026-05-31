"""LivingSynapseLayer - local online synapse primitive.

The layer has stable weights (W_slow), plastic weights (W_live), and synaptic
fatigue. Forward passes never mutate W_live; learning happens only through
explicit local update methods.
"""
import numpy as np

from . import sparse_native


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
        self._last_active = None
        self._last_active_values = None
        self.last_forward_ops = {
            "mode": "none",
            "ops": 0,
            "active_inputs": 0,
            "fatigue_touched": 0,
        }
        self.last_update_ops = {
            "mode": "none",
            "ops": 0,
            "active_inputs": 0,
            "weights_touched": 0,
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

        if (use_sparse or len(active) == 1) and 0 < len(active) < 0.8 * len(x):
            active = active.astype(np.intp, copy=False)
            x_active = x[active]
            fatigue_cols = self.fatigue[:, active]
            W_cols = (
                self.W_slow[:, active] + self.W_live[:, active]
            ) * (1.0 - fatigue_cols)
            post = W_cols @ x_active

            max_s = float(np.max(np.abs(post))) + 1e-8
            updated_fatigue = (
                0.98 * fatigue_cols
                + 0.02 * np.abs(post[:, None] * x_active[None, :]) / max_s
            )
            self.fatigue[:, active] = np.clip(updated_fatigue, 0.0, 0.9)
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
        self._last_active = active.astype(np.intp, copy=False)
        self._last_active_values = x[self._last_active].astype(np.float32, copy=True)
        self._last_post = post.copy()
        if return_stats:
            return post, dict(self.last_forward_ops)
        return post

    def forward_active(self, active_indices, active_values=None, return_stats=False):
        active = np.asarray(active_indices, dtype=np.intp)
        if active_values is None:
            x_active = np.ones(len(active), dtype=np.float32)
        else:
            x_active = np.asarray(active_values, dtype=np.float32)
        if len(active) == 0:
            post = np.zeros(self.out_dim, dtype=np.float32)
            self.last_forward_ops = {
                "mode": "sparse_active",
                "ops": 0,
                "active_inputs": 0,
                "fatigue_touched": 0,
            }
        elif sparse_native.NATIVE_AVAILABLE:
            try:
                post, stats = sparse_native.forward_active(
                    self.W_slow,
                    self.W_live,
                    self.fatigue,
                    active,
                    x_active,
                )
                self.last_forward_ops = {
                    "mode": stats.get("mode", "native_sparse_active"),
                    "ops": int(stats.get("ops", self.out_dim * len(active))),
                    "active_inputs": int(stats.get("active_inputs", len(active))),
                    "fatigue_touched": int(stats.get("touched", self.out_dim * len(active))),
                }
            except Exception:
                fatigue_cols = self.fatigue[:, active]
                W_cols = (
                    self.W_slow[:, active] + self.W_live[:, active]
                ) * (1.0 - fatigue_cols)
                post = W_cols @ x_active
                max_s = float(np.max(np.abs(post))) + 1e-8
                updated_fatigue = (
                    0.98 * fatigue_cols
                    + 0.02 * np.abs(post[:, None] * x_active[None, :]) / max_s
                )
                self.fatigue[:, active] = np.clip(updated_fatigue, 0.0, 0.9)
                self.last_forward_ops = {
                    "mode": "sparse_active_fallback",
                    "ops": int(self.out_dim * len(active)),
                    "active_inputs": int(len(active)),
                    "fatigue_touched": int(self.out_dim * len(active)),
                }
        else:
            fatigue_cols = self.fatigue[:, active]
            W_cols = (
                self.W_slow[:, active] + self.W_live[:, active]
            ) * (1.0 - fatigue_cols)
            post = W_cols @ x_active
            max_s = float(np.max(np.abs(post))) + 1e-8
            updated_fatigue = (
                0.98 * fatigue_cols
                + 0.02 * np.abs(post[:, None] * x_active[None, :]) / max_s
            )
            self.fatigue[:, active] = np.clip(updated_fatigue, 0.0, 0.9)
            self.last_forward_ops = {
                "mode": "sparse_active",
                "ops": int(self.out_dim * len(active)),
                "active_inputs": int(len(active)),
                "fatigue_touched": int(self.out_dim * len(active)),
            }
        self._last_pre = None
        self._last_active = active
        self._last_active_values = x_active
        self._last_post = post.copy()
        if return_stats:
            return post, dict(self.last_forward_ops)
        return post

    def _active_pre(self):
        if self._last_active is not None:
            return self._last_active.astype(np.intp, copy=False)
        if self._last_pre is None:
            return np.array([], dtype=np.intp)
        return np.where(np.abs(self._last_pre) > 1e-5)[0].astype(np.intp)

    def _active_values(self):
        if self._last_active_values is not None:
            return self._last_active_values.astype(np.float32, copy=False)
        if self._last_pre is None:
            return np.array([], dtype=np.float32)
        active = self._active_pre()
        return self._last_pre[active].astype(np.float32, copy=False)

    def _clip_active_columns(self, active, max_norm):
        if len(active) == 0:
            return
        norms = np.linalg.norm(self.W_live[:, active], axis=0)
        scale = np.ones_like(norms, dtype=np.float32)
        mask = norms > float(max_norm)
        scale[mask] = float(max_norm) / (norms[mask] + 1e-12)
        self.W_live[:, active] *= scale

    def hebbian_update_active(self, modulator, lr=0.05, decay=0.001, max_norm=12.0):
        if self._last_active is None or self._last_post is None:
            return
        active = self._last_active.astype(np.intp, copy=False)
        values = self._active_values()
        post = self._last_post
        if sparse_native.NATIVE_AVAILABLE and len(active) > 0:
            try:
                stats = sparse_native.hebbian_update_active(
                    self.W_live,
                    active,
                    values,
                    post,
                    float(modulator),
                    float(lr),
                    float(decay),
                    float(max_norm),
                )
                self.last_update_ops = {
                    "mode": stats.get("mode", "native_sparse_active_hebbian"),
                    "ops": int(stats.get("ops", self.out_dim * len(active))),
                    "active_inputs": int(stats.get("active_inputs", len(active))),
                    "weights_touched": int(stats.get("touched", self.out_dim * len(active))),
                }
                return
            except Exception:
                pass
        self.W_live[:, active] *= (1.0 - lr * decay)
        for i, c in enumerate(active):
            self.W_live[:, c] += lr * float(modulator) * post * values[i]
        self._clip_active_columns(active, max_norm)
        self.last_update_ops = {
            "mode": "sparse_active_hebbian",
            "ops": int(self.out_dim * len(active)),
            "active_inputs": int(len(active)),
            "weights_touched": int(self.out_dim * len(active)),
        }

    def supervised_local_update_active(self, error_signal, lr=0.05, decay=0.001,
                                       max_norm=12.0):
        if self._last_active is None:
            return
        active = self._last_active.astype(np.intp, copy=False)
        values = self._active_values()
        err = np.asarray(error_signal, dtype=np.float32)
        if sparse_native.NATIVE_AVAILABLE and len(active) > 0:
            try:
                stats = sparse_native.supervised_update_active(
                    self.W_live,
                    active,
                    values,
                    err,
                    float(lr),
                    float(decay),
                    float(max_norm),
                )
                self.last_update_ops = {
                    "mode": stats.get("mode", "native_sparse_active_supervised"),
                    "ops": int(stats.get("ops", self.out_dim * len(active))),
                    "active_inputs": int(stats.get("active_inputs", len(active))),
                    "weights_touched": int(stats.get("touched", self.out_dim * len(active))),
                }
                return
            except Exception:
                pass
        self.W_live[:, active] *= (1.0 - lr * decay)
        for i, c in enumerate(active):
            self.W_live[:, c] += lr * err * values[i]
        self._clip_active_columns(active, max_norm)
        self.last_update_ops = {
            "mode": "sparse_active_supervised",
            "ops": int(self.out_dim * len(active)),
            "active_inputs": int(len(active)),
            "weights_touched": int(self.out_dim * len(active)),
        }

    def target_update_from_active(self, active_indices, target_index, active_values=None,
                                  lr=0.05, decay=0.001, max_abs=12.0):
        active = np.asarray(active_indices, dtype=np.intp)
        if active_values is None:
            values = np.ones(len(active), dtype=np.float32)
        else:
            values = np.asarray(active_values, dtype=np.float32)
        target = int(target_index)
        if sparse_native.NATIVE_AVAILABLE and len(active) > 0:
            try:
                stats = sparse_native.target_update_active(
                    self.W_live,
                    active,
                    values,
                    target,
                    float(lr),
                    float(decay),
                    float(max_abs),
                )
                self.last_update_ops = {
                    "mode": stats.get("mode", "native_sparse_active_target"),
                    "ops": int(stats.get("ops", len(active))),
                    "active_inputs": int(stats.get("active_inputs", len(active))),
                    "weights_touched": int(stats.get("touched", len(active))),
                }
                return
            except Exception:
                pass
        scale = 1.0 - float(lr) * float(decay)
        self.W_live[target, active] *= scale
        self.W_live[target, active] += float(lr) * values
        np.clip(self.W_live[target, active], -float(max_abs), float(max_abs), out=self.W_live[target, active])
        self.last_update_ops = {
            "mode": "sparse_active_target",
            "ops": int(len(active)),
            "active_inputs": int(len(active)),
            "weights_touched": int(len(active)),
        }

    def hebbian_update(self, modulator, lr=0.05, decay=0.001, max_norm=12.0):
        if self._last_pre is None:
            return
        pre, post = self._last_pre, self._last_post
        active = self._active_pre()
        if 0 < len(active) < 0.1 * len(pre):
            self.W_live[:, active] *= (1.0 - lr * decay)
            for c in active:
                self.W_live[:, c] += lr * float(modulator) * post * pre[c]
            self.last_update_ops = {
                "mode": "sparse_hebbian",
                "ops": int(self.out_dim * len(active)),
                "active_inputs": int(len(active)),
                "weights_touched": int(self.out_dim * len(active)),
            }
        else:
            self.W_live *= (1.0 - lr * decay)
            self.W_live += lr * float(modulator) * np.outer(post, pre)
            self.last_update_ops = {
                "mode": "dense_hebbian",
                "ops": int(self.out_dim * self.in_dim),
                "active_inputs": int(len(active)),
                "weights_touched": int(self.out_dim * self.in_dim),
            }
        self._clip_norm(max_norm)

    def hebbian_update_dense(self, modulator, lr=0.05, decay=0.001, max_norm=12.0):
        if self._last_pre is None:
            return
        pre, post = self._last_pre, self._last_post
        active = self._active_pre()
        self.W_live *= (1.0 - lr * decay)
        self.W_live += lr * float(modulator) * np.outer(post, pre)
        self._clip_norm(max_norm)
        self.last_update_ops = {
            "mode": "dense_hebbian_forced",
            "ops": int(self.out_dim * self.in_dim),
            "active_inputs": int(len(active)),
            "weights_touched": int(self.out_dim * self.in_dim),
        }

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
