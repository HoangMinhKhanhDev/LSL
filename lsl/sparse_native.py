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


def simple_tokenize(text: str, max_tokens: int | None = None):
    if _sparse_native is None or not hasattr(_sparse_native, "simple_tokenize"):
        import re

        lowered = str(text).lower()
        tokens = [match.group(0) for match in re.finditer(r"\w+|[^\w\s]", lowered, re.UNICODE)]
        return tokens if max_tokens is None else tokens[: int(max_tokens)]
    if max_tokens is None:
        return _sparse_native.simple_tokenize(str(text))
    return _sparse_native.simple_tokenize(str(text), int(max_tokens))


def best_signature_match(
    query_active: np.ndarray,
    candidate_signatures: np.ndarray,
    candidate_lengths: np.ndarray,
    candidate_values: np.ndarray,
) -> Dict[str, float]:
    if _sparse_native is None or not hasattr(_sparse_native, "best_signature_match"):
        query = {int(x) for x in np.asarray(query_active, dtype=np.intp).tolist()}
        signatures = np.asarray(candidate_signatures, dtype=np.intp)
        lengths = np.asarray(candidate_lengths, dtype=np.intp)
        values = np.asarray(candidate_values, dtype=np.intp)
        best_position = -1
        best_value = -1
        best_score = -1
        for i in range(signatures.shape[0]):
            length = max(0, min(int(lengths[i]), signatures.shape[1]))
            score = 0
            for j in range(length):
                bit = int(signatures[i, j])
                if bit >= 0 and bit in query:
                    score += 1
            if score > best_score:
                best_score = score
                best_position = i
                best_value = int(values[i])
        return {
            "mode": "python_best_signature_match",
            "best_position": float(best_position),
            "best_value": float(best_value),
            "best_score": float(best_score),
            "candidate_count": float(signatures.shape[0]),
            "ops": float(signatures.shape[0] * signatures.shape[1]),
        }
    return _sparse_native.best_signature_match(
        np.asarray(query_active, dtype=np.intp),
        np.asarray(candidate_signatures, dtype=np.intp),
        np.asarray(candidate_lengths, dtype=np.intp),
        np.asarray(candidate_values, dtype=np.intp),
    )


def forward_active_batch(
    w_slow: np.ndarray,
    w_live: np.ndarray,
    fatigue: np.ndarray,
    active_indices: np.ndarray,
    active_values: np.ndarray,
    lengths: np.ndarray,
):
    if _sparse_native is None or not hasattr(_sparse_native, "forward_active_batch"):
        slow = np.asarray(w_slow, dtype=np.float32)
        live = np.asarray(w_live, dtype=np.float32)
        fat = np.asarray(fatigue, dtype=np.float32)
        active_indices = np.asarray(active_indices, dtype=np.intp)
        active_values = np.asarray(active_values, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.intp)
        batch = int(active_indices.shape[0])
        out_dim = int(slow.shape[0])
        posts = np.zeros((batch, out_dim), dtype=np.float32)
        total_ops = 0
        for row in range(batch):
            count = max(0, min(int(lengths[row]), int(active_indices.shape[1])))
            max_abs = 1e-8
            for out in range(out_dim):
                acc = 0.0
                for j in range(count):
                    col = int(active_indices[row, j])
                    if col < 0 or col >= slow.shape[1]:
                        raise IndexError("active index out of bounds")
                    value = float(active_values[row, j])
                    acc += float((slow[out, col] + live[out, col]) * (1.0 - fat[out, col]) * value)
                    total_ops += 1
                posts[row, out] = acc
                max_abs = max(max_abs, abs(acc))
            for out in range(out_dim):
                p = float(posts[row, out])
                for j in range(count):
                    col = int(active_indices[row, j])
                    value = float(active_values[row, j])
                    next_value = 0.98 * float(fat[out, col]) + 0.02 * (abs(p * value) / max_abs)
                    fat[out, col] = min(0.9, max(0.0, next_value))
        return posts, {"mode": "python_sparse_active_batch", "ops": total_ops, "batch": batch, "touched": total_ops}
    return _sparse_native.forward_active_batch(
        np.asarray(w_slow, dtype=np.float32),
        np.asarray(w_live, dtype=np.float32),
        np.asarray(fatigue, dtype=np.float32),
        np.asarray(active_indices, dtype=np.intp),
        np.asarray(active_values, dtype=np.float32),
        np.asarray(lengths, dtype=np.intp),
    )


def simple_tokenize_cached(text: str, max_tokens: int | None = None):
    return simple_tokenize(text, max_tokens=max_tokens)


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
