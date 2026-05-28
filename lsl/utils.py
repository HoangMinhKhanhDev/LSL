"""Small numerical helpers used across the LSL package."""
import numpy as np


def softmax(x, axis=-1, temp=1.0):
    x = np.asarray(x, dtype=np.float32) / float(temp)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def one_hot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    v[int(idx)] = 1.0
    return v


def safe_log(p, eps=1e-10):
    return float(np.log(max(float(p), eps)))
