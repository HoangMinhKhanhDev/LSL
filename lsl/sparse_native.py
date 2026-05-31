"""Optional native sparse kernels for strict active-index synapse paths."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from . import _sparse_native
except ImportError:  # pragma: no cover - exercised when extension is not built
    _sparse_native = None


NATIVE_AVAILABLE = _sparse_native is not None


def require_native() -> None:
    if _sparse_native is None:
        raise RuntimeError(
            "lsl._sparse_native is not available. Build it with "
            "`python setup.py build_ext --inplace` or install the package."
        )


def forward_active(
    w_slow: np.ndarray,
    w_live: np.ndarray,
    fatigue: np.ndarray,
    active_indices: np.ndarray,
    active_values: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    require_native()
    return _sparse_native.forward_active(
        w_slow,
        w_live,
        fatigue,
        np.asarray(active_indices, dtype=np.intp),
        np.asarray(active_values, dtype=np.float32),
    )


def hebbian_update_active(
    w_live: np.ndarray,
    active_indices: np.ndarray,
    active_values: np.ndarray,
    post: np.ndarray,
    modulator: float,
    lr: float,
    decay: float,
    max_norm: float,
) -> Dict[str, int]:
    require_native()
    return _sparse_native.hebbian_update_active(
        w_live,
        np.asarray(active_indices, dtype=np.intp),
        np.asarray(active_values, dtype=np.float32),
        np.asarray(post, dtype=np.float32),
        float(modulator),
        float(lr),
        float(decay),
        float(max_norm),
    )


def supervised_update_active(
    w_live: np.ndarray,
    active_indices: np.ndarray,
    active_values: np.ndarray,
    error: np.ndarray,
    lr: float,
    decay: float,
    max_norm: float,
) -> Dict[str, int]:
    require_native()
    if not hasattr(_sparse_native, "supervised_update_active"):
        raise RuntimeError("lsl._sparse_native was built without supervised_update_active")
    return _sparse_native.supervised_update_active(
        w_live,
        np.asarray(active_indices, dtype=np.intp),
        np.asarray(active_values, dtype=np.float32),
        np.asarray(error, dtype=np.float32),
        float(lr),
        float(decay),
        float(max_norm),
    )


def target_update_active(
    w_live: np.ndarray,
    active_indices: np.ndarray,
    active_values: np.ndarray,
    target_index: int,
    lr: float,
    decay: float,
    max_abs: float,
) -> Dict[str, int]:
    require_native()
    if not hasattr(_sparse_native, "target_update_active"):
        raise RuntimeError("lsl._sparse_native was built without target_update_active")
    return _sparse_native.target_update_active(
        w_live,
        np.asarray(active_indices, dtype=np.intp),
        np.asarray(active_values, dtype=np.float32),
        int(target_index),
        float(lr),
        float(decay),
        float(max_abs),
    )


def score_active(
    w_slow: np.ndarray,
    w_live: np.ndarray,
    fatigue: np.ndarray,
    active_indices: np.ndarray,
    active_values: np.ndarray,
    target_index: int = -1,
) -> Dict[str, float]:
    require_native()
    if not hasattr(_sparse_native, "score_active"):
        raise RuntimeError("lsl._sparse_native was built without score_active")
    return _sparse_native.score_active(
        w_slow,
        w_live,
        fatigue,
        np.asarray(active_indices, dtype=np.intp),
        np.asarray(active_values, dtype=np.float32),
        int(target_index),
    )


def dendrite_predict(
    branch_bits: np.ndarray,
    branch_lengths: np.ndarray,
    branch_weights: np.ndarray,
    branch_thresholds: np.ndarray,
    branch_strengths: np.ndarray,
    branch_outputs: np.ndarray,
    active_bits: np.ndarray,
) -> Dict[str, float]:
    require_native()
    if not hasattr(_sparse_native, "dendrite_predict"):
        raise RuntimeError("lsl._sparse_native was built without dendrite_predict")
    return _sparse_native.dendrite_predict(
        np.asarray(branch_bits, dtype=np.intp),
        np.asarray(branch_lengths, dtype=np.intp),
        np.asarray(branch_weights, dtype=np.float32),
        np.asarray(branch_thresholds, dtype=np.float32),
        np.asarray(branch_strengths, dtype=np.float32),
        np.asarray(branch_outputs, dtype=np.intp),
        np.asarray(active_bits, dtype=np.intp),
    )


def topk_float32(scores: np.ndarray, k: int) -> Dict[str, object]:
    require_native()
    if not hasattr(_sparse_native, "topk_float32"):
        raise RuntimeError("lsl._sparse_native was built without topk_float32")
    return _sparse_native.topk_float32(np.asarray(scores, dtype=np.float32), int(k))
