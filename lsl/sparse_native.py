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

