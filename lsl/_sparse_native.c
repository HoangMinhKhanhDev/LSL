#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
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

    npy_intp *vote_outputs = (npy_intp *)PyMem_Calloc((size_t)branch_count, sizeof(npy_intp));
    double *vote_scores = (double *)PyMem_Calloc((size_t)branch_count, sizeof(double));
    if (!vote_outputs || !vote_scores) {
        PyMem_Free(vote_outputs);
        PyMem_Free(vote_scores);
        PyErr_NoMemory();
        goto dendrite_fail;
    }

    npy_intp unique = 0;
    npy_intp active_branches = 0;
    npy_intp ops = 0;
    for (npy_intp b = 0; b < branch_count; b++) {
        npy_intp length = *(npy_intp *)PyArray_GETPTR1(lengths, b);
        if (length < 0) {
            length = 0;
        }
        if (length > branch_width) {
            length = branch_width;
        }
        float drive = 0.0f;
        for (npy_intp j = 0; j < length; j++) {
            ops += active_count;
            npy_intp bit = *(npy_intp *)PyArray_GETPTR2(bits, b, j);
            if (contains_active(active, active_count, bit)) {
                drive += *(float *)PyArray_GETPTR2(weights, b, j);
            }
        }
        float threshold = *(float *)PyArray_GETPTR1(thresholds, b);
        float activation = 1.0f / (1.0f + expf(-(drive - threshold)));
        if (activation < 0.5f) {
            continue;
        }
        active_branches += 1;
        npy_intp output = *(npy_intp *)PyArray_GETPTR1(outputs, b);
        double vote = (double)(*(float *)PyArray_GETPTR1(strengths, b)) * (double)activation;
        npy_intp found = -1;
        for (npy_intp i = 0; i < unique; i++) {
            if (vote_outputs[i] == output) {
                found = i;
                break;
            }
        }
        if (found < 0) {
            found = unique;
            vote_outputs[unique] = output;
            vote_scores[unique] = 0.0;
            unique += 1;
        }
        vote_scores[found] += vote;
    }

    npy_intp best_output = -1;
    double best_score = 0.0;
    for (npy_intp i = 0; i < unique; i++) {
        if (best_output < 0 || vote_scores[i] > best_score ||
            (vote_scores[i] == best_score && vote_outputs[i] < best_output)) {
            best_output = vote_outputs[i];
            best_score = vote_scores[i];
        }
    }

    PyMem_Free(vote_outputs);
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
    Py_XDECREF(bits);
    Py_XDECREF(lengths);
    Py_XDECREF(weights);
    Py_XDECREF(thresholds);
    Py_XDECREF(strengths);
    Py_XDECREF(outputs);
    Py_XDECREF(active);
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
    char *used = (char *)PyMem_Calloc((size_t)n, sizeof(char));
    if (!used && n > 0) {
        Py_DECREF(indices);
        Py_DECREF(values);
        Py_DECREF(scores);
        PyErr_NoMemory();
        return NULL;
    }
    for (npy_intp rank = 0; rank < k; rank++) {
        npy_intp best = -1;
        float best_value = -3.402823466e+38F;
        for (npy_intp i = 0; i < n; i++) {
            if (used[i]) {
                continue;
            }
            float value = *(float *)PyArray_GETPTR1(scores, i);
            if (best < 0 || value > best_value || (value == best_value && i > best)) {
                best = i;
                best_value = value;
            }
        }
        if (best < 0) {
            best = 0;
            best_value = 0.0f;
        }
        used[best] = 1;
        PyList_SET_ITEM(indices, rank, PyLong_FromSsize_t(best));
        PyList_SET_ITEM(values, rank, PyFloat_FromDouble((double)best_value));
    }
    PyMem_Free(used);
    Py_DECREF(scores);
    PyObject *result = Py_BuildValue("{s:s,s:O,s:O,s:n}", "mode", "native_topk_float32", "indices", indices, "scores", values, "ops", n * k);
    Py_DECREF(indices);
    Py_DECREF(values);
    return result;
}


static PyMethodDef SparseMethods[] = {
    {"forward_active", forward_active, METH_VARARGS, "Run active-index sparse forward and fatigue update."},
    {"hebbian_update_active", hebbian_update_active, METH_VARARGS, "Run active-index local Hebbian update."},
    {"supervised_update_active", supervised_update_active, METH_VARARGS, "Run active-index local supervised update."},
    {"target_update_active", target_update_active, METH_VARARGS, "Run active-index single-target local update."},
    {"score_active", score_active, METH_VARARGS, "Score active-index input without allocating a full output vector."},
    {"dendrite_predict", dendrite_predict, METH_VARARGS, "Score dendritic branches and return the winning output."},
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
