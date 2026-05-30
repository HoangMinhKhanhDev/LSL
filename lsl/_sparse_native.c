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


static PyMethodDef SparseMethods[] = {
    {"forward_active", forward_active, METH_VARARGS, "Run active-index sparse forward and fatigue update."},
    {"hebbian_update_active", hebbian_update_active, METH_VARARGS, "Run active-index local Hebbian update."},
    {"supervised_update_active", supervised_update_active, METH_VARARGS, "Run active-index local supervised update."},
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
