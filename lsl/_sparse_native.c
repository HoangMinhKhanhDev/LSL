#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <numpy/arrayobject.h>


static PyObject *stats_dict(const char *mode, npy_intp out_dim, npy_intp active_count) {
    npy_intp touched = out_dim * active_count;
    return Py_BuildValue(
        "{s:s,s:n,s:n,s:n}",
        "mode", mode,
        "ops", touched,
        "active_inputs", active_count,
        "touched", touched
    );
}


static PyObject *forward_active(PyObject *self, PyObject *args) {
    PyObject *slow_obj;
    PyObject *live_obj;
    PyObject *fatigue_obj;
    PyObject *active_obj;
    PyObject *values_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &slow_obj, &live_obj, &fatigue_obj, &active_obj, &values_obj)) {
        return NULL;
    }

    PyArrayObject *slow = (PyArrayObject *)PyArray_FROM_OTF(slow_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    PyArrayObject *live = (PyArrayObject *)PyArray_FROM_OTF(live_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    PyArrayObject *fatigue = (PyArrayObject *)PyArray_FROM_OTF(fatigue_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE);
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(active_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!slow || !live || !fatigue || !active || !values) {
        Py_XDECREF(slow);
        Py_XDECREF(live);
        Py_XDECREF(fatigue);
        Py_XDECREF(active);
        Py_XDECREF(values);
        return NULL;
    }
    if (PyArray_NDIM(slow) != 2 || PyArray_NDIM(live) != 2 || PyArray_NDIM(fatigue) != 2) {
        PyErr_SetString(PyExc_ValueError, "weight and fatigue arrays must be 2D");
        goto fail;
    }
    if (PyArray_DIM(slow, 0) != PyArray_DIM(live, 0) || PyArray_DIM(slow, 1) != PyArray_DIM(live, 1) ||
        PyArray_DIM(slow, 0) != PyArray_DIM(fatigue, 0) || PyArray_DIM(slow, 1) != PyArray_DIM(fatigue, 1)) {
        PyErr_SetString(PyExc_ValueError, "weight and fatigue arrays must have matching shape");
        goto fail;
    }
    if (PyArray_NDIM(active) != 1 || PyArray_NDIM(values) != 1 || PyArray_DIM(active, 0) != PyArray_DIM(values, 0)) {
        PyErr_SetString(PyExc_ValueError, "active indices and values must be matching 1D arrays");
        goto fail;
    }

    npy_intp out_dim = PyArray_DIM(slow, 0);
    npy_intp in_dim = PyArray_DIM(slow, 1);
    npy_intp active_count = PyArray_DIM(active, 0);
    npy_intp out_shape[1] = {out_dim};
    PyArrayObject *post = (PyArrayObject *)PyArray_SimpleNew(1, out_shape, NPY_FLOAT32);
    if (!post) {
        goto fail;
    }

    float max_abs = 1.0e-8f;
    for (npy_intp row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (npy_intp j = 0; j < active_count; j++) {
            npy_intp col = *(npy_intp *)PyArray_GETPTR1(active, j);
            if (col < 0 || col >= in_dim) {
                Py_DECREF(post);
                PyErr_SetString(PyExc_IndexError, "active index out of bounds");
                goto fail;
            }
            float value = *(float *)PyArray_GETPTR1(values, j);
            float ws = *(float *)PyArray_GETPTR2(slow, row, col);
            float wl = *(float *)PyArray_GETPTR2(live, row, col);
            float f = *(float *)PyArray_GETPTR2(fatigue, row, col);
            acc += (ws + wl) * (1.0f - f) * value;
        }
        *(float *)PyArray_GETPTR1(post, row) = acc;
        float a = fabsf(acc);
        if (a > max_abs) {
            max_abs = a;
        }
    }

    for (npy_intp row = 0; row < out_dim; row++) {
        float p = *(float *)PyArray_GETPTR1(post, row);
        for (npy_intp j = 0; j < active_count; j++) {
            npy_intp col = *(npy_intp *)PyArray_GETPTR1(active, j);
            float value = *(float *)PyArray_GETPTR1(values, j);
            float old = *(float *)PyArray_GETPTR2(fatigue, row, col);
            float sig = fabsf(p * value) / max_abs;
            float next = 0.98f * old + 0.02f * sig;
            if (next < 0.0f) {
                next = 0.0f;
            } else if (next > 0.9f) {
                next = 0.9f;
            }
            *(float *)PyArray_GETPTR2(fatigue, row, col) = next;
        }
    }

    PyObject *stats = stats_dict("native_sparse_active", out_dim, active_count);
    Py_DECREF(slow);
    Py_DECREF(live);
    Py_DECREF(fatigue);
    Py_DECREF(active);
    Py_DECREF(values);
    return Py_BuildValue("NN", (PyObject *)post, stats);

fail:
    Py_XDECREF(slow);
    Py_XDECREF(live);
    Py_XDECREF(fatigue);
    Py_XDECREF(active);
    Py_XDECREF(values);
    return NULL;
}


static PyObject *hebbian_update_active(PyObject *self, PyObject *args) {
    PyObject *live_obj;
    PyObject *active_obj;
    PyObject *values_obj;
    PyObject *post_obj;
    double modulator;
    double lr;
    double decay;
    double max_norm;
    if (!PyArg_ParseTuple(args, "OOOOdddd", &live_obj, &active_obj, &values_obj, &post_obj, &modulator, &lr, &decay, &max_norm)) {
        return NULL;
    }

    PyArrayObject *live = (PyArrayObject *)PyArray_FROM_OTF(live_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE);
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(active_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *post = (PyArrayObject *)PyArray_FROM_OTF(post_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!live || !active || !values || !post) {
        Py_XDECREF(live);
        Py_XDECREF(active);
        Py_XDECREF(values);
        Py_XDECREF(post);
        return NULL;
    }
    if (PyArray_NDIM(live) != 2 || PyArray_NDIM(active) != 1 || PyArray_NDIM(values) != 1 || PyArray_NDIM(post) != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid array dimensions");
        goto fail;
    }

    npy_intp out_dim = PyArray_DIM(live, 0);
    npy_intp in_dim = PyArray_DIM(live, 1);
    npy_intp active_count = PyArray_DIM(active, 0);
    if (PyArray_DIM(values, 0) != active_count || PyArray_DIM(post, 0) != out_dim) {
        PyErr_SetString(PyExc_ValueError, "array shapes do not match");
        goto fail;
    }

    float scale = (float)(1.0 - lr * decay);
    float step = (float)(lr * modulator);
    for (npy_intp j = 0; j < active_count; j++) {
        npy_intp col = *(npy_intp *)PyArray_GETPTR1(active, j);
        if (col < 0 || col >= in_dim) {
            PyErr_SetString(PyExc_IndexError, "active index out of bounds");
            goto fail;
        }
        float value = *(float *)PyArray_GETPTR1(values, j);
        for (npy_intp row = 0; row < out_dim; row++) {
            float *w = (float *)PyArray_GETPTR2(live, row, col);
            float p = *(float *)PyArray_GETPTR1(post, row);
            *w = (*w) * scale + step * p * value;
        }
    }

    for (npy_intp j = 0; j < active_count; j++) {
        npy_intp col = *(npy_intp *)PyArray_GETPTR1(active, j);
        double norm2 = 0.0;
        for (npy_intp row = 0; row < out_dim; row++) {
            float w = *(float *)PyArray_GETPTR2(live, row, col);
            norm2 += (double)w * (double)w;
        }
        double norm = sqrt(norm2);
        if (norm > max_norm && norm > 0.0) {
            float clip = (float)(max_norm / norm);
            for (npy_intp row = 0; row < out_dim; row++) {
                float *w = (float *)PyArray_GETPTR2(live, row, col);
                *w *= clip;
            }
        }
    }

    PyObject *stats = stats_dict("native_sparse_active_hebbian", out_dim, active_count);
    Py_DECREF(live);
    Py_DECREF(active);
    Py_DECREF(values);
    Py_DECREF(post);
    return stats;

fail:
    Py_XDECREF(live);
    Py_XDECREF(active);
    Py_XDECREF(values);
    Py_XDECREF(post);
    return NULL;
}


static PyObject *supervised_update_active(PyObject *self, PyObject *args) {
    PyObject *live_obj;
    PyObject *active_obj;
    PyObject *values_obj;
    PyObject *error_obj;
    double lr;
    double decay;
    double max_norm;
    if (!PyArg_ParseTuple(args, "OOOOddd", &live_obj, &active_obj, &values_obj, &error_obj, &lr, &decay, &max_norm)) {
        return NULL;
    }

    PyArrayObject *live = (PyArrayObject *)PyArray_FROM_OTF(live_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE);
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(active_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *error = (PyArrayObject *)PyArray_FROM_OTF(error_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!live || !active || !values || !error) {
        Py_XDECREF(live);
        Py_XDECREF(active);
        Py_XDECREF(values);
        Py_XDECREF(error);
        return NULL;
    }
    if (PyArray_NDIM(live) != 2 || PyArray_NDIM(active) != 1 || PyArray_NDIM(values) != 1 || PyArray_NDIM(error) != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid array dimensions");
        goto fail;
    }

    npy_intp out_dim = PyArray_DIM(live, 0);
    npy_intp in_dim = PyArray_DIM(live, 1);
    npy_intp active_count = PyArray_DIM(active, 0);
    if (PyArray_DIM(values, 0) != active_count || PyArray_DIM(error, 0) != out_dim) {
        PyErr_SetString(PyExc_ValueError, "array shapes do not match");
        goto fail;
    }

    float scale = (float)(1.0 - lr * decay);
    float step = (float)lr;
    for (npy_intp j = 0; j < active_count; j++) {
        npy_intp col = *(npy_intp *)PyArray_GETPTR1(active, j);
        if (col < 0 || col >= in_dim) {
            PyErr_SetString(PyExc_IndexError, "active index out of bounds");
            goto fail;
        }
        float value = *(float *)PyArray_GETPTR1(values, j);
        for (npy_intp row = 0; row < out_dim; row++) {
            float *w = (float *)PyArray_GETPTR2(live, row, col);
            float e = *(float *)PyArray_GETPTR1(error, row);
            *w = (*w) * scale + step * e * value;
        }
    }

    for (npy_intp j = 0; j < active_count; j++) {
        npy_intp col = *(npy_intp *)PyArray_GETPTR1(active, j);
        double norm2 = 0.0;
        for (npy_intp row = 0; row < out_dim; row++) {
            float w = *(float *)PyArray_GETPTR2(live, row, col);
            norm2 += (double)w * (double)w;
        }
        double norm = sqrt(norm2);
        if (norm > max_norm && norm > 0.0) {
            float clip = (float)(max_norm / norm);
            for (npy_intp row = 0; row < out_dim; row++) {
                float *w = (float *)PyArray_GETPTR2(live, row, col);
                *w *= clip;
            }
        }
    }

    PyObject *stats = stats_dict("native_sparse_active_supervised", out_dim, active_count);
    Py_DECREF(live);
    Py_DECREF(active);
    Py_DECREF(values);
    Py_DECREF(error);
    return stats;

fail:
    Py_XDECREF(live);
    Py_XDECREF(active);
    Py_XDECREF(values);
    Py_XDECREF(error);
    return NULL;
}


static PyObject *target_update_active(PyObject *self, PyObject *args) {
    PyObject *live_obj;
    PyObject *active_obj;
    PyObject *values_obj;
    Py_ssize_t target;
    double lr;
    double decay;
    double max_abs;
    if (!PyArg_ParseTuple(args, "OOOnddd", &live_obj, &active_obj, &values_obj, &target, &lr, &decay, &max_abs)) {
        return NULL;
    }

    PyArrayObject *live = (PyArrayObject *)PyArray_FROM_OTF(live_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE);
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(active_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!live || !active || !values) {
        Py_XDECREF(live);
        Py_XDECREF(active);
        Py_XDECREF(values);
        return NULL;
    }
    if (PyArray_NDIM(live) != 2 || PyArray_NDIM(active) != 1 || PyArray_NDIM(values) != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid array dimensions");
        goto fail;
    }

    npy_intp out_dim = PyArray_DIM(live, 0);
    npy_intp in_dim = PyArray_DIM(live, 1);
    npy_intp active_count = PyArray_DIM(active, 0);
    if (PyArray_DIM(values, 0) != active_count) {
        PyErr_SetString(PyExc_ValueError, "active indices and values must match");
        goto fail;
    }
    if (target < 0 || target >= out_dim) {
        PyErr_SetString(PyExc_IndexError, "target index out of bounds");
        goto fail;
    }

    float scale = (float)(1.0 - lr * decay);
    float step = (float)lr;
    float cap = (float)max_abs;
    for (npy_intp j = 0; j < active_count; j++) {
        npy_intp col = *(npy_intp *)PyArray_GETPTR1(active, j);
        if (col < 0 || col >= in_dim) {
            PyErr_SetString(PyExc_IndexError, "active index out of bounds");
            goto fail;
        }
        float value = *(float *)PyArray_GETPTR1(values, j);
        float *w = (float *)PyArray_GETPTR2(live, (npy_intp)target, col);
        float next = (*w) * scale + step * value;
        if (next > cap) {
            next = cap;
        } else if (next < -cap) {
            next = -cap;
        }
        *w = next;
    }

    PyObject *stats = stats_dict("native_sparse_active_target", 1, active_count);
    Py_DECREF(live);
    Py_DECREF(active);
    Py_DECREF(values);
    return stats;

fail:
    Py_XDECREF(live);
    Py_XDECREF(active);
    Py_XDECREF(values);
    return NULL;
}


static PyObject *score_active(PyObject *self, PyObject *args) {
    PyObject *slow_obj;
    PyObject *live_obj;
    PyObject *fatigue_obj;
    PyObject *active_obj;
    PyObject *values_obj;
    Py_ssize_t target;
    if (!PyArg_ParseTuple(args, "OOOOOn", &slow_obj, &live_obj, &fatigue_obj, &active_obj, &values_obj, &target)) {
        return NULL;
    }

    PyArrayObject *slow = (PyArrayObject *)PyArray_FROM_OTF(slow_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    PyArrayObject *live = (PyArrayObject *)PyArray_FROM_OTF(live_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    PyArrayObject *fatigue = (PyArrayObject *)PyArray_FROM_OTF(fatigue_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(active_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!slow || !live || !fatigue || !active || !values) {
        Py_XDECREF(slow);
        Py_XDECREF(live);
        Py_XDECREF(fatigue);
        Py_XDECREF(active);
        Py_XDECREF(values);
        return NULL;
    }
    if (PyArray_NDIM(slow) != 2 || PyArray_NDIM(live) != 2 || PyArray_NDIM(fatigue) != 2 ||
        PyArray_NDIM(active) != 1 || PyArray_NDIM(values) != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid array dimensions");
        goto fail;
    }
    if (PyArray_DIM(slow, 0) != PyArray_DIM(live, 0) || PyArray_DIM(slow, 1) != PyArray_DIM(live, 1) ||
        PyArray_DIM(slow, 0) != PyArray_DIM(fatigue, 0) || PyArray_DIM(slow, 1) != PyArray_DIM(fatigue, 1)) {
        PyErr_SetString(PyExc_ValueError, "weight and fatigue arrays must have matching shape");
        goto fail;
    }

    npy_intp out_dim = PyArray_DIM(slow, 0);
    npy_intp in_dim = PyArray_DIM(slow, 1);
    npy_intp active_count = PyArray_DIM(active, 0);
    if (PyArray_DIM(values, 0) != active_count) {
        PyErr_SetString(PyExc_ValueError, "active indices and values must match");
        goto fail;
    }

    npy_intp best_index = 0;
    float best_score = -3.402823466e+38F;
    double positive_sum = 0.0;
    float target_score = 0.0f;
    for (npy_intp row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (npy_intp j = 0; j < active_count; j++) {
            npy_intp col = *(npy_intp *)PyArray_GETPTR1(active, j);
            if (col < 0 || col >= in_dim) {
                PyErr_SetString(PyExc_IndexError, "active index out of bounds");
                goto fail;
            }
            float value = *(float *)PyArray_GETPTR1(values, j);
            float ws = *(float *)PyArray_GETPTR2(slow, row, col);
            float wl = *(float *)PyArray_GETPTR2(live, row, col);
            float f = *(float *)PyArray_GETPTR2(fatigue, row, col);
            acc += (ws + wl) * (1.0f - f) * value;
        }
        if (acc > best_score) {
            best_score = acc;
            best_index = row;
        }
        if (acc > 0.0f) {
            positive_sum += (double)acc;
        }
        if (row == (npy_intp)target) {
            target_score = acc > 0.0f ? acc : 0.0f;
        }
    }

    PyObject *stats = Py_BuildValue(
        "{s:s,s:n,s:n,s:n,s:n,s:f,s:f,s:d}",
        "mode", "native_sparse_active_score",
        "ops", out_dim * active_count,
        "active_inputs", active_count,
        "touched", out_dim * active_count,
        "best_index", best_index,
        "best_score", (double)best_score,
        "target_score", (double)target_score,
        "positive_sum", positive_sum
    );
    Py_DECREF(slow);
    Py_DECREF(live);
    Py_DECREF(fatigue);
    Py_DECREF(active);
    Py_DECREF(values);
    return stats;

fail:
    Py_XDECREF(slow);
    Py_XDECREF(live);
    Py_XDECREF(fatigue);
    Py_XDECREF(active);
    Py_XDECREF(values);
    return NULL;
}


static int contains_active(PyArrayObject *active, npy_intp active_count, npy_intp bit) {
    for (npy_intp i = 0; i < active_count; i++) {
        if (*(npy_intp *)PyArray_GETPTR1(active, i) == bit) {
            return 1;
        }
    }
    return 0;
}


static inline int active_contains_sorted(const npy_intp *active, npy_intp active_count, npy_intp bit) {
    npy_intp lo = 0;
    npy_intp hi = active_count - 1;
    while (lo <= hi) {
        npy_intp mid = lo + ((hi - lo) >> 1);
        npy_intp value = active[mid];
        if (value == bit) {
            return 1;
        }
        if (value < bit) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return 0;
}


static inline int active_contains_lookup(
    const unsigned char *mask,
    npy_intp mask_size,
    const npy_intp *sorted,
    npy_intp sorted_count,
    npy_intp bit
) {
    if (mask != NULL) {
        return bit >= 0 && bit < mask_size && mask[bit] != 0;
    }
    return active_contains_sorted(sorted, sorted_count, bit);
}


typedef struct {
    npy_intp index;
    float value;
} TopKItem;


static int compare_npy_intp_ascending(const void *left, const void *right) {
    const npy_intp a = *(const npy_intp *)left;
    const npy_intp b = *(const npy_intp *)right;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}


static int compare_topk_descending(const void *left, const void *right) {
    const TopKItem *a = (const TopKItem *)left;
    const TopKItem *b = (const TopKItem *)right;
    if (a->value > b->value) {
        return -1;
    }
    if (a->value < b->value) {
        return 1;
    }
    if (a->index > b->index) {
        return -1;
    }
    if (a->index < b->index) {
        return 1;
    }
    return 0;
}


static inline int topk_is_better(const TopKItem *candidate, const TopKItem *baseline) {
    if (candidate->value > baseline->value) {
        return 1;
    }
    if (candidate->value < baseline->value) {
        return 0;
    }
    return candidate->index > baseline->index;
}


static void topk_swap(TopKItem *left, TopKItem *right) {
    TopKItem tmp = *left;
    *left = *right;
    *right = tmp;
}


static void topk_heap_sift_up(TopKItem *heap, npy_intp index) {
    while (index > 0) {
        npy_intp parent = (index - 1) / 2;
        if (!topk_is_better(&heap[parent], &heap[index])) {
            break;
        }
        topk_swap(&heap[parent], &heap[index]);
        index = parent;
    }
}


static void topk_heap_sift_down(TopKItem *heap, npy_intp size) {
    npy_intp index = 0;
    while (1) {
        npy_intp left = index * 2 + 1;
        npy_intp right = left + 1;
        npy_intp worst = index;
        if (left < size && !topk_is_better(&heap[left], &heap[worst])) {
            worst = left;
        }
        if (right < size && !topk_is_better(&heap[right], &heap[worst])) {
            worst = right;
        }
        if (worst == index) {
            break;
        }
        topk_swap(&heap[index], &heap[worst]);
        index = worst;
    }
}


static int is_word_char(Py_UCS4 ch) {
    return Py_UNICODE_ISALNUM(ch) || ch == '_';
}


static PyObject *simple_tokenize(PyObject *self, PyObject *args) {
    PyObject *text_obj;
    Py_ssize_t max_tokens = -1;
    if (!PyArg_ParseTuple(args, "O|n", &text_obj, &max_tokens)) {
        return NULL;
    }

    PyObject *text = PyObject_Str(text_obj);
    if (!text) {
        return NULL;
    }
    PyObject *lower = PyObject_CallMethod(text, "lower", NULL);
    Py_DECREF(text);
    if (!lower) {
        return NULL;
    }
    if (!PyUnicode_Check(lower)) {
        Py_DECREF(lower);
        PyErr_SetString(PyExc_TypeError, "simple_tokenize expects a unicode string");
        return NULL;
    }

    Py_ssize_t length = PyUnicode_GetLength(lower);
    int kind = PyUnicode_KIND(lower);
    const void *data = PyUnicode_DATA(lower);
    PyObject *out = PyList_New(0);
    if (!out) {
        Py_DECREF(lower);
        return NULL;
    }

    Py_ssize_t i = 0;
    while (i < length) {
        if (max_tokens >= 0 && PyList_GET_SIZE(out) >= max_tokens) {
            break;
        }
        Py_UCS4 ch = PyUnicode_READ(kind, data, i);
        if (Py_UNICODE_ISSPACE(ch)) {
            i++;
            continue;
        }
        Py_ssize_t start = i;
        if (is_word_char(ch)) {
            i++;
            while (i < length) {
                ch = PyUnicode_READ(kind, data, i);
                if (!is_word_char(ch)) {
                    break;
                }
                i++;
            }
        } else {
            i++;
        }
        PyObject *token = PyUnicode_Substring(lower, start, i);
        if (!token) {
            Py_DECREF(out);
            Py_DECREF(lower);
            return NULL;
        }
        if (PyList_Append(out, token) < 0) {
            Py_DECREF(token);
            Py_DECREF(out);
            Py_DECREF(lower);
            return NULL;
        }
        Py_DECREF(token);
    }
    Py_DECREF(lower);
    return out;
}


static PyObject *best_signature_match(PyObject *self, PyObject *args) {
    PyObject *query_obj;
    PyObject *signatures_obj;
    PyObject *lengths_obj;
    PyObject *values_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &query_obj, &signatures_obj, &lengths_obj, &values_obj)) {
        return NULL;
    }

    PyArrayObject *query = (PyArrayObject *)PyArray_FROM_OTF(query_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *signatures = (PyArrayObject *)PyArray_FROM_OTF(signatures_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *lengths = (PyArrayObject *)PyArray_FROM_OTF(lengths_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!query || !signatures || !lengths || !values) {
        Py_XDECREF(query);
        Py_XDECREF(signatures);
        Py_XDECREF(lengths);
        Py_XDECREF(values);
        return NULL;
    }
    if (PyArray_NDIM(query) != 1 || PyArray_NDIM(signatures) != 2 || PyArray_NDIM(lengths) != 1 || PyArray_NDIM(values) != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid signature arrays");
        goto match_fail;
    }

    npy_intp candidate_count = PyArray_DIM(signatures, 0);
    npy_intp width = PyArray_DIM(signatures, 1);
    npy_intp active_count = PyArray_DIM(query, 0);
    if (PyArray_DIM(lengths, 0) != candidate_count || PyArray_DIM(values, 0) != candidate_count) {
        PyErr_SetString(PyExc_ValueError, "candidate arrays must have the same length");
        goto match_fail;
    }

    npy_intp query_max = -1;
    const npy_intp *query_data = (const npy_intp *)PyArray_DATA(query);
    for (npy_intp i = 0; i < active_count; i++) {
        if (query_data[i] > query_max) {
            query_max = query_data[i];
        }
    }
    unsigned char *query_mask = NULL;
    npy_intp query_mask_size = 0;
    npy_intp *query_sorted = NULL;
    if (active_count > 0) {
        if (query_max >= 0 && query_max <= 1048576) {
            query_mask_size = query_max + 1;
            query_mask = (unsigned char *)PyMem_Calloc((size_t)query_mask_size, sizeof(unsigned char));
            if (!query_mask) {
                PyErr_NoMemory();
                goto match_fail;
            }
            for (npy_intp i = 0; i < active_count; i++) {
                npy_intp bit = query_data[i];
                if (bit >= 0 && bit < query_mask_size) {
                    query_mask[bit] = 1;
                }
            }
        } else {
            query_sorted = (npy_intp *)PyMem_Malloc((size_t)active_count * sizeof(npy_intp));
            if (!query_sorted) {
                PyErr_NoMemory();
                goto match_fail;
            }
            memcpy(query_sorted, query_data, (size_t)active_count * sizeof(npy_intp));
            qsort(query_sorted, (size_t)active_count, sizeof(npy_intp), compare_npy_intp_ascending);
        }
    }

    npy_intp best_position = -1;
    npy_intp best_value = -1;
    npy_intp best_score = -1;
    npy_intp ops = 0;
    for (npy_intp i = 0; i < candidate_count; i++) {
        npy_intp length = *(npy_intp *)PyArray_GETPTR1(lengths, i);
        if (length < 0) {
            length = 0;
        }
        if (length > width) {
            length = width;
        }
        npy_intp score = 0;
        for (npy_intp j = 0; j < length; j++) {
            npy_intp bit = *(npy_intp *)PyArray_GETPTR2(signatures, i, j);
            if (bit < 0) {
                continue;
            }
            ops += 1;
            if (active_contains_lookup(query_mask, query_mask_size, query_sorted, active_count, bit)) {
                score += 1;
            }
        }
        if (score > best_score) {
            best_score = score;
            best_position = i;
            best_value = *(npy_intp *)PyArray_GETPTR1(values, i);
        }
    }

    PyObject *stats = Py_BuildValue(
        "{s:s,s:n,s:n,s:n,s:n,s:n}",
        "mode", "native_best_signature_match",
        "best_position", best_position,
        "best_value", best_value,
        "best_score", best_score,
        "candidate_count", candidate_count,
        "ops", ops
    );
    PyMem_Free(query_mask);
    PyMem_Free(query_sorted);
    Py_DECREF(query);
    Py_DECREF(signatures);
    Py_DECREF(lengths);
    Py_DECREF(values);
    return stats;

match_fail:
    PyMem_Free(query_mask);
    PyMem_Free(query_sorted);
    Py_XDECREF(query);
    Py_XDECREF(signatures);
    Py_XDECREF(lengths);
    Py_XDECREF(values);
    return NULL;
}


static PyObject *forward_active_batch(PyObject *self, PyObject *args) {
    PyObject *slow_obj;
    PyObject *live_obj;
    PyObject *fatigue_obj;
    PyObject *active_obj;
    PyObject *values_obj;
    PyObject *lengths_obj;
    if (!PyArg_ParseTuple(args, "OOOOOO", &slow_obj, &live_obj, &fatigue_obj, &active_obj, &values_obj, &lengths_obj)) {
        return NULL;
    }

    PyArrayObject *slow = (PyArrayObject *)PyArray_FROM_OTF(slow_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    PyArrayObject *live = (PyArrayObject *)PyArray_FROM_OTF(live_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
    PyArrayObject *fatigue = (PyArrayObject *)PyArray_FROM_OTF(fatigue_obj, NPY_FLOAT32, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE);
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(active_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *values = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *lengths = (PyArrayObject *)PyArray_FROM_OTF(lengths_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!slow || !live || !fatigue || !active || !values || !lengths) {
        Py_XDECREF(slow);
        Py_XDECREF(live);
        Py_XDECREF(fatigue);
        Py_XDECREF(active);
        Py_XDECREF(values);
        Py_XDECREF(lengths);
        return NULL;
    }
    if (PyArray_NDIM(slow) != 2 || PyArray_NDIM(live) != 2 || PyArray_NDIM(fatigue) != 2 ||
        PyArray_NDIM(active) != 2 || PyArray_NDIM(values) != 2 || PyArray_NDIM(lengths) != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid batch array dimensions");
        goto batch_fail;
    }

    npy_intp batch = PyArray_DIM(active, 0);
    npy_intp width = PyArray_DIM(active, 1);
    npy_intp out_dim = PyArray_DIM(slow, 0);
    npy_intp in_dim = PyArray_DIM(slow, 1);
    if (PyArray_DIM(live, 0) != out_dim || PyArray_DIM(live, 1) != in_dim ||
        PyArray_DIM(fatigue, 0) != out_dim || PyArray_DIM(fatigue, 1) != in_dim ||
        PyArray_DIM(values, 0) != batch || PyArray_DIM(values, 1) != width ||
        PyArray_DIM(lengths, 0) != batch) {
        PyErr_SetString(PyExc_ValueError, "batch array shapes do not match");
        goto batch_fail;
    }

    npy_intp out_shape[2] = {batch, out_dim};
    PyArrayObject *post = (PyArrayObject *)PyArray_SimpleNew(2, out_shape, NPY_FLOAT32);
    if (!post) {
        goto batch_fail;
    }

    npy_intp ops = 0;
    npy_intp touched = 0;
    const char *slow_base = PyArray_BYTES(slow);
    const char *live_base = PyArray_BYTES(live);
    char *fatigue_base = PyArray_BYTES(fatigue);
    float *post_data = (float *)PyArray_DATA(post);
    const npy_intp slow_row_stride = PyArray_STRIDES(slow)[0];
    const npy_intp slow_col_stride = PyArray_STRIDES(slow)[1];
    const npy_intp live_row_stride = PyArray_STRIDES(live)[0];
    const npy_intp live_col_stride = PyArray_STRIDES(live)[1];
    const npy_intp fatigue_row_stride = PyArray_STRIDES(fatigue)[0];
    const npy_intp fatigue_col_stride = PyArray_STRIDES(fatigue)[1];
    const npy_intp *active_data = (const npy_intp *)PyArray_DATA(active);
    const float *values_data = (const float *)PyArray_DATA(values);
    const npy_intp *lengths_data = (const npy_intp *)PyArray_DATA(lengths);
    for (npy_intp row = 0; row < batch; row++) {
        npy_intp active_count = lengths_data[row];
        if (active_count < 0) {
            active_count = 0;
        }
        if (active_count > width) {
            active_count = width;
        }
        const npy_intp *active_row = active_data + row * width;
        const float *values_row = values_data + row * width;
        float max_abs = 1.0e-8f;
        for (npy_intp out = 0; out < out_dim; out++) {
            const char *slow_row = slow_base + out * slow_row_stride;
            const char *live_row = live_base + out * live_row_stride;
            char *fatigue_row = fatigue_base + out * fatigue_row_stride;
            float acc = 0.0f;
            for (npy_intp j = 0; j < active_count; j++) {
                npy_intp col = active_row[j];
                if (col < 0 || col >= in_dim) {
                    Py_DECREF(post);
                    PyErr_SetString(PyExc_IndexError, "active index out of bounds");
                    goto batch_fail;
                }
                float value = values_row[j];
                float ws = *(float *)(slow_row + col * slow_col_stride);
                float wl = *(float *)(live_row + col * live_col_stride);
                float f = *(float *)(fatigue_row + col * fatigue_col_stride);
                acc += (ws + wl) * (1.0f - f) * value;
                ops += 1;
                touched += 1;
            }
            post_data[row * out_dim + out] = acc;
            float a = fabsf(acc);
            if (a > max_abs) {
                max_abs = a;
            }
        }
        for (npy_intp out = 0; out < out_dim; out++) {
            const char *slow_row = slow_base + out * slow_row_stride;
            const char *live_row = live_base + out * live_row_stride;
            char *fatigue_row = fatigue_base + out * fatigue_row_stride;
            float p = post_data[row * out_dim + out];
            for (npy_intp j = 0; j < active_count; j++) {
                npy_intp col = active_row[j];
                float value = values_row[j];
                float old = *(float *)(fatigue_row + col * fatigue_col_stride);
                float sig = fabsf(p * value) / max_abs;
                float next = 0.98f * old + 0.02f * sig;
                if (next < 0.0f) {
                    next = 0.0f;
                } else if (next > 0.9f) {
                    next = 0.9f;
                }
                *(float *)(fatigue_row + col * fatigue_col_stride) = next;
            }
        }
    }

    PyObject *stats = Py_BuildValue(
        "{s:s,s:n,s:n,s:n}",
        "mode", "native_sparse_active_batch",
        "ops", ops,
        "batch", batch,
        "touched", touched
    );
    Py_DECREF(slow);
    Py_DECREF(live);
    Py_DECREF(fatigue);
    Py_DECREF(active);
    Py_DECREF(values);
    Py_DECREF(lengths);
    return Py_BuildValue("NN", (PyObject *)post, stats);

batch_fail:
    Py_XDECREF(slow);
    Py_XDECREF(live);
    Py_XDECREF(fatigue);
    Py_XDECREF(active);
    Py_XDECREF(values);
    Py_XDECREF(lengths);
    return NULL;
}


static PyObject *dendrite_predict(PyObject *self, PyObject *args) {
    PyObject *bits_obj;
    PyObject *lengths_obj;
    PyObject *weights_obj;
    PyObject *thresholds_obj;
    PyObject *strengths_obj;
    PyObject *outputs_obj;
    PyObject *active_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &bits_obj, &lengths_obj, &weights_obj, &thresholds_obj, &strengths_obj, &outputs_obj, &active_obj)) {
        return NULL;
    }

    PyArrayObject *bits = (PyArrayObject *)PyArray_FROM_OTF(bits_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *lengths = (PyArrayObject *)PyArray_FROM_OTF(lengths_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *weights = (PyArrayObject *)PyArray_FROM_OTF(weights_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *thresholds = (PyArrayObject *)PyArray_FROM_OTF(thresholds_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *strengths = (PyArrayObject *)PyArray_FROM_OTF(strengths_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *outputs = (PyArrayObject *)PyArray_FROM_OTF(outputs_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(active_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!bits || !lengths || !weights || !thresholds || !strengths || !outputs || !active) {
        Py_XDECREF(bits);
        Py_XDECREF(lengths);
        Py_XDECREF(weights);
        Py_XDECREF(thresholds);
        Py_XDECREF(strengths);
        Py_XDECREF(outputs);
        Py_XDECREF(active);
        return NULL;
    }
    if (PyArray_NDIM(bits) != 2 || PyArray_NDIM(weights) != 2 || PyArray_NDIM(lengths) != 1 ||
        PyArray_NDIM(thresholds) != 1 || PyArray_NDIM(strengths) != 1 || PyArray_NDIM(outputs) != 1 ||
        PyArray_NDIM(active) != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid dendrite array dimensions");
        goto dendrite_fail;
    }

    npy_intp branch_count = PyArray_DIM(bits, 0);
    npy_intp branch_width = PyArray_DIM(bits, 1);
    npy_intp active_count = PyArray_DIM(active, 0);
    if (PyArray_DIM(weights, 0) != branch_count || PyArray_DIM(weights, 1) != branch_width ||
        PyArray_DIM(lengths, 0) != branch_count || PyArray_DIM(thresholds, 0) != branch_count ||
        PyArray_DIM(strengths, 0) != branch_count || PyArray_DIM(outputs, 0) != branch_count) {
        PyErr_SetString(PyExc_ValueError, "dendrite array shapes do not match");
        goto dendrite_fail;
    }

    const npy_intp *active_data = (const npy_intp *)PyArray_DATA(active);
    npy_intp active_max = -1;
    for (npy_intp i = 0; i < active_count; i++) {
        if (active_data[i] > active_max) {
            active_max = active_data[i];
        }
    }
    unsigned char *active_mask = NULL;
    npy_intp active_mask_size = 0;
    npy_intp *active_sorted = NULL;
    if (active_count > 0) {
        if (active_max >= 0 && active_max <= 1048576) {
            active_mask_size = active_max + 1;
            active_mask = (unsigned char *)PyMem_Calloc((size_t)active_mask_size, sizeof(unsigned char));
            if (!active_mask) {
                PyErr_NoMemory();
                goto dendrite_fail;
            }
            for (npy_intp i = 0; i < active_count; i++) {
                npy_intp bit = active_data[i];
                if (bit >= 0 && bit < active_mask_size) {
                    active_mask[bit] = 1;
                }
            }
        } else {
            active_sorted = (npy_intp *)PyMem_Malloc((size_t)active_count * sizeof(npy_intp));
            if (!active_sorted) {
                PyErr_NoMemory();
                goto dendrite_fail;
            }
            memcpy(active_sorted, active_data, (size_t)active_count * sizeof(npy_intp));
            qsort(active_sorted, (size_t)active_count, sizeof(npy_intp), compare_npy_intp_ascending);
        }
    }

    npy_intp max_output = -1;
    const npy_intp *output_data = (const npy_intp *)PyArray_DATA(outputs);
    for (npy_intp i = 0; i < branch_count; i++) {
        if (output_data[i] > max_output) {
            max_output = output_data[i];
        }
    }
    if (max_output < 0) {
        max_output = 0;
    }
    double *vote_scores = (double *)PyMem_Calloc((size_t)(max_output + 1), sizeof(double));
    if (!vote_scores) {
        PyErr_NoMemory();
        goto dendrite_fail;
    }

    npy_intp active_branches = 0;
    npy_intp ops = 0;
    const npy_intp *lengths_data = (const npy_intp *)PyArray_DATA(lengths);
    const float *thresholds_data = (const float *)PyArray_DATA(thresholds);
    const float *strengths_data = (const float *)PyArray_DATA(strengths);
    const npy_intp *bits_base = (const npy_intp *)PyArray_DATA(bits);
    const char *weights_base = (const char *)PyArray_DATA(weights);
    const npy_intp bits_row_stride = PyArray_STRIDES(bits)[0] / (npy_intp)sizeof(npy_intp);
    const npy_intp bits_col_stride = PyArray_STRIDES(bits)[1] / (npy_intp)sizeof(npy_intp);
    const npy_intp weights_row_stride = PyArray_STRIDES(weights)[0];
    const npy_intp weights_col_stride = PyArray_STRIDES(weights)[1];
    for (npy_intp b = 0; b < branch_count; b++) {
        npy_intp length = lengths_data[b];
        if (length < 0) {
            length = 0;
        }
        if (length > branch_width) {
            length = branch_width;
        }
        const npy_intp *bits_row = bits_base + b * bits_row_stride;
        const char *weights_row = weights_base + b * weights_row_stride;
        float drive = 0.0f;
        for (npy_intp j = 0; j < length; j++) {
            ops += active_count;
            npy_intp bit = bits_row[j * bits_col_stride];
            if (active_contains_lookup(active_mask, active_mask_size, active_sorted, active_count, bit)) {
                drive += *(float *)(weights_row + j * weights_col_stride);
            }
        }
        float threshold = thresholds_data[b];
        float activation = 1.0f / (1.0f + expf(-(drive - threshold)));
        if (activation < 0.5f) {
            continue;
        }
        active_branches += 1;
        npy_intp output = output_data[b];
        if (output < 0 || output > max_output) {
            continue;
        }
        double vote = (double)strengths_data[b] * (double)activation;
        vote_scores[output] += vote;
    }

    npy_intp best_output = -1;
    double best_score = 0.0;
    for (npy_intp i = 0; i <= max_output; i++) {
        if (vote_scores[i] > best_score ||
            (vote_scores[i] == best_score && best_output >= 0 && i < best_output)) {
            best_output = i;
            best_score = vote_scores[i];
        }
    }

    PyMem_Free(active_mask);
    PyMem_Free(active_sorted);
    PyMem_Free(vote_scores);
    Py_DECREF(bits);
    Py_DECREF(lengths);
    Py_DECREF(weights);
    Py_DECREF(thresholds);
    Py_DECREF(strengths);
    Py_DECREF(outputs);
    Py_DECREF(active);
    return Py_BuildValue(
        "{s:s,s:n,s:n,s:n,s:n,s:d}",
        "mode", "native_dendrite_predict",
        "ops", ops,
        "branches", branch_count,
        "active_branches", active_branches,
        "best_output", best_output,
        "best_score", best_score
    );

dendrite_fail:
    PyMem_Free(active_mask);
    PyMem_Free(active_sorted);
    Py_XDECREF(bits);
    Py_XDECREF(lengths);
    Py_XDECREF(weights);
    Py_XDECREF(thresholds);
    Py_XDECREF(strengths);
    Py_XDECREF(outputs);
    Py_XDECREF(active);
    return NULL;
}


static PyObject *dendrite_predict_candidates(PyObject *self, PyObject *args) {
    PyObject *bits_obj;
    PyObject *lengths_obj;
    PyObject *weights_obj;
    PyObject *thresholds_obj;
    PyObject *strengths_obj;
    PyObject *outputs_obj;
    PyObject *active_obj;
    PyObject *candidate_ids_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &bits_obj, &lengths_obj, &weights_obj, &thresholds_obj, &strengths_obj, &outputs_obj, &active_obj, &candidate_ids_obj)) {
        return NULL;
    }

    PyArrayObject *bits = (PyArrayObject *)PyArray_FROM_OTF(bits_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *lengths = (PyArrayObject *)PyArray_FROM_OTF(lengths_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *weights = (PyArrayObject *)PyArray_FROM_OTF(weights_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *thresholds = (PyArrayObject *)PyArray_FROM_OTF(thresholds_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *strengths = (PyArrayObject *)PyArray_FROM_OTF(strengths_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *outputs = (PyArrayObject *)PyArray_FROM_OTF(outputs_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(active_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *candidate_ids = (PyArrayObject *)PyArray_FROM_OTF(candidate_ids_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!bits || !lengths || !weights || !thresholds || !strengths || !outputs || !active || !candidate_ids) {
        Py_XDECREF(bits);
        Py_XDECREF(lengths);
        Py_XDECREF(weights);
        Py_XDECREF(thresholds);
        Py_XDECREF(strengths);
        Py_XDECREF(outputs);
        Py_XDECREF(active);
        Py_XDECREF(candidate_ids);
        return NULL;
    }
    if (PyArray_NDIM(bits) != 2 || PyArray_NDIM(weights) != 2 || PyArray_NDIM(lengths) != 1 ||
        PyArray_NDIM(thresholds) != 1 || PyArray_NDIM(strengths) != 1 || PyArray_NDIM(outputs) != 1 ||
        PyArray_NDIM(active) != 1 || PyArray_NDIM(candidate_ids) != 1) {
        PyErr_SetString(PyExc_ValueError, "invalid dendrite array dimensions");
        goto dendrite_candidates_fail;
    }

    npy_intp branch_count = PyArray_DIM(bits, 0);
    npy_intp branch_width = PyArray_DIM(bits, 1);
    npy_intp active_count = PyArray_DIM(active, 0);
    npy_intp candidate_count = PyArray_DIM(candidate_ids, 0);
    if (PyArray_DIM(weights, 0) != branch_count || PyArray_DIM(weights, 1) != branch_width ||
        PyArray_DIM(lengths, 0) != branch_count || PyArray_DIM(thresholds, 0) != branch_count ||
        PyArray_DIM(strengths, 0) != branch_count || PyArray_DIM(outputs, 0) != branch_count) {
        PyErr_SetString(PyExc_ValueError, "dendrite array shapes do not match");
        goto dendrite_candidates_fail;
    }

    const npy_intp *active_data = (const npy_intp *)PyArray_DATA(active);
    npy_intp active_max = -1;
    for (npy_intp i = 0; i < active_count; i++) {
        if (active_data[i] > active_max) {
            active_max = active_data[i];
        }
    }
    unsigned char *active_mask = NULL;
    npy_intp active_mask_size = 0;
    npy_intp *active_sorted = NULL;
    if (active_count > 0) {
        if (active_max >= 0 && active_max <= 1048576) {
            active_mask_size = active_max + 1;
            active_mask = (unsigned char *)PyMem_Calloc((size_t)active_mask_size, sizeof(unsigned char));
            if (!active_mask) {
                PyErr_NoMemory();
                goto dendrite_candidates_fail;
            }
            for (npy_intp i = 0; i < active_count; i++) {
                npy_intp bit = active_data[i];
                if (bit >= 0 && bit < active_mask_size) {
                    active_mask[bit] = 1;
                }
            }
        } else {
            active_sorted = (npy_intp *)PyMem_Malloc((size_t)active_count * sizeof(npy_intp));
            if (!active_sorted) {
                PyErr_NoMemory();
                goto dendrite_candidates_fail;
            }
            memcpy(active_sorted, active_data, (size_t)active_count * sizeof(npy_intp));
            qsort(active_sorted, (size_t)active_count, sizeof(npy_intp), compare_npy_intp_ascending);
        }
    }

    npy_intp max_output = -1;
    const npy_intp *output_data = (const npy_intp *)PyArray_DATA(outputs);
    for (npy_intp i = 0; i < branch_count; i++) {
        if (output_data[i] > max_output) {
            max_output = output_data[i];
        }
    }
    if (max_output < 0) {
        max_output = 0;
    }
    double *vote_scores = (double *)PyMem_Calloc((size_t)(max_output + 1), sizeof(double));
    if (!vote_scores) {
        PyErr_NoMemory();
        goto dendrite_candidates_fail;
    }

    npy_intp active_branches = 0;
    npy_intp ops = 0;
    const npy_intp *candidate_data = (const npy_intp *)PyArray_DATA(candidate_ids);
    const npy_intp *lengths_data = (const npy_intp *)PyArray_DATA(lengths);
    const float *thresholds_data = (const float *)PyArray_DATA(thresholds);
    const float *strengths_data = (const float *)PyArray_DATA(strengths);
    const npy_intp *bits_base = (const npy_intp *)PyArray_DATA(bits);
    const char *weights_base = (const char *)PyArray_DATA(weights);
    const npy_intp bits_row_stride = PyArray_STRIDES(bits)[0] / (npy_intp)sizeof(npy_intp);
    const npy_intp bits_col_stride = PyArray_STRIDES(bits)[1] / (npy_intp)sizeof(npy_intp);
    const npy_intp weights_row_stride = PyArray_STRIDES(weights)[0];
    const npy_intp weights_col_stride = PyArray_STRIDES(weights)[1];
    for (npy_intp idx = 0; idx < candidate_count; idx++) {
        npy_intp b = candidate_data[idx];
        if (b < 0 || b >= branch_count) {
            continue;
        }
        npy_intp length = lengths_data[b];
        if (length < 0) {
            length = 0;
        }
        if (length > branch_width) {
            length = branch_width;
        }
        const npy_intp *bits_row = bits_base + b * bits_row_stride;
        const char *weights_row = weights_base + b * weights_row_stride;
        float drive = 0.0f;
        for (npy_intp j = 0; j < length; j++) {
            ops += active_count;
            npy_intp bit = bits_row[j * bits_col_stride];
            if (active_contains_lookup(active_mask, active_mask_size, active_sorted, active_count, bit)) {
                drive += *(float *)(weights_row + j * weights_col_stride);
            }
        }
        float threshold = thresholds_data[b];
        float activation = 1.0f / (1.0f + expf(-(drive - threshold)));
        if (activation < 0.5f) {
            continue;
        }
        active_branches += 1;
        npy_intp output = output_data[b];
        if (output < 0 || output > max_output) {
            continue;
        }
        double vote = (double)strengths_data[b] * (double)activation;
        vote_scores[output] += vote;
    }

    npy_intp best_output = -1;
    double best_score = 0.0;
    for (npy_intp i = 0; i <= max_output; i++) {
        if (vote_scores[i] > best_score ||
            (vote_scores[i] == best_score && best_output >= 0 && i < best_output)) {
            best_output = i;
            best_score = vote_scores[i];
        }
    }

    PyMem_Free(active_mask);
    PyMem_Free(active_sorted);
    PyMem_Free(vote_scores);
    Py_DECREF(bits);
    Py_DECREF(lengths);
    Py_DECREF(weights);
    Py_DECREF(thresholds);
    Py_DECREF(strengths);
    Py_DECREF(outputs);
    Py_DECREF(active);
    Py_DECREF(candidate_ids);
    return Py_BuildValue(
        "{s:s,s:n,s:n,s:n,s:n,s:d}",
        "mode", "native_dendrite_predict_candidates",
        "ops", ops,
        "branches", branch_count,
        "active_branches", active_branches,
        "best_output", best_output,
        "best_score", best_score
    );

dendrite_candidates_fail:
    PyMem_Free(active_mask);
    PyMem_Free(active_sorted);
    Py_XDECREF(bits);
    Py_XDECREF(lengths);
    Py_XDECREF(weights);
    Py_XDECREF(thresholds);
    Py_XDECREF(strengths);
    Py_XDECREF(outputs);
    Py_XDECREF(active);
    Py_XDECREF(candidate_ids);
    return NULL;
}


static PyObject *topk_float32(PyObject *self, PyObject *args) {
    PyObject *scores_obj;
    Py_ssize_t k_raw;
    if (!PyArg_ParseTuple(args, "On", &scores_obj, &k_raw)) {
        return NULL;
    }
    PyArrayObject *scores = (PyArrayObject *)PyArray_FROM_OTF(scores_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!scores) {
        return NULL;
    }
    if (PyArray_NDIM(scores) != 1) {
        Py_DECREF(scores);
        PyErr_SetString(PyExc_ValueError, "scores must be a 1D float32 array");
        return NULL;
    }
    npy_intp n = PyArray_DIM(scores, 0);
    npy_intp k = (npy_intp)k_raw;
    if (k < 0) {
        k = 0;
    }
    if (k > n) {
        k = n;
    }
    PyObject *indices = PyList_New(k);
    PyObject *values = PyList_New(k);
    if (!indices || !values) {
        Py_XDECREF(indices);
        Py_XDECREF(values);
        Py_DECREF(scores);
        return NULL;
    }
    TopKItem *heap = (TopKItem *)PyMem_Malloc((size_t)k * sizeof(TopKItem));
    if (!heap && k > 0) {
        Py_DECREF(indices);
        Py_DECREF(values);
        Py_DECREF(scores);
        PyErr_NoMemory();
        return NULL;
    }
    npy_intp heap_size = 0;
    const float *score_data = (const float *)PyArray_DATA(scores);
    for (npy_intp i = 0; i < n; i++) {
        TopKItem item = {i, score_data[i]};
        if (heap_size < k) {
            heap[heap_size] = item;
            topk_heap_sift_up(heap, heap_size);
            heap_size += 1;
            continue;
        }
        if (k > 0 && topk_is_better(&item, &heap[0])) {
            heap[0] = item;
            topk_heap_sift_down(heap, heap_size);
        }
    }
    qsort(heap, (size_t)heap_size, sizeof(TopKItem), compare_topk_descending);
    for (npy_intp rank = 0; rank < heap_size; rank++) {
        PyList_SET_ITEM(indices, rank, PyLong_FromSsize_t(heap[rank].index));
        PyList_SET_ITEM(values, rank, PyFloat_FromDouble((double)heap[rank].value));
    }
    PyMem_Free(heap);
    Py_DECREF(scores);
    PyObject *result = Py_BuildValue("{s:s,s:O,s:O,s:n}", "mode", "native_topk_float32", "indices", indices, "scores", values, "ops", n * k);
    Py_DECREF(indices);
    Py_DECREF(values);
    return result;
}


static PyMethodDef SparseMethods[] = {
    {"forward_active", forward_active, METH_VARARGS, "Run active-index sparse forward and fatigue update."},
    {"forward_active_batch", forward_active_batch, METH_VARARGS, "Run sparse forward on a batch of active-index inputs."},
    {"hebbian_update_active", hebbian_update_active, METH_VARARGS, "Run active-index local Hebbian update."},
    {"supervised_update_active", supervised_update_active, METH_VARARGS, "Run active-index local supervised update."},
    {"target_update_active", target_update_active, METH_VARARGS, "Run active-index single-target local update."},
    {"score_active", score_active, METH_VARARGS, "Score active-index input without allocating a full output vector."},
    {"best_signature_match", best_signature_match, METH_VARARGS, "Score sparse candidate signatures against an active query."},
    {"simple_tokenize", simple_tokenize, METH_VARARGS, "Tokenize text into lowercase word and punctuation tokens."},
    {"dendrite_predict", dendrite_predict, METH_VARARGS, "Score dendritic branches and return the winning output."},
    {"dendrite_predict_candidates", dendrite_predict_candidates, METH_VARARGS, "Score only candidate dendritic branches and return the winning output."},
    {"topk_float32", topk_float32, METH_VARARGS, "Return top-k float32 indices without allocating argsort output."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef sparsemodule = {
    PyModuleDef_HEAD_INIT,
    "_sparse_native",
    "Native sparse active-index kernels.",
    -1,
    SparseMethods
};


PyMODINIT_FUNC PyInit__sparse_native(void) {
    import_array();
    return PyModule_Create(&sparsemodule);
}
