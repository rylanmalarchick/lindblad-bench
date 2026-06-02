/**
 * lindblad_ext.c — CPython C extension: NumPy interface to the lindblad-bench
 * C library.
 *
 * Exposes:
 *   propagate_step(P, rho_in) -> rho_out
 *   evolve(H, cops, rho0, t_end, dt) -> rho_final
 *
 * Inputs and outputs are NumPy complex128 (C99 double complex) arrays in
 * row-major order. NumPy complex128 is layout-compatible with double complex,
 * so the marshalling is a memcpy into the library's aligned buffers; the
 * arrays are copied in, not borrowed.
 *
 * Build with: pip install -e python/
 * Requires: C library built at ../build/liblindblad_bench.a
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <complex.h>
#include <math.h>

#include "lindblad_bench.h"

/* Coerce a Python object to a C-contiguous complex128 2-D square array.
 * On success returns a new reference and writes the side length to *dim.
 * On failure sets a Python exception and returns NULL. */
static PyArrayObject *as_square_c128(PyObject *obj, const char *name, size_t *dim)
{
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (!arr) {
        return NULL; /* PyArray_FROM_OTF already set the error */
    }
    if (PyArray_NDIM(arr) != 2) {
        PyErr_Format(PyExc_ValueError, "%s must be a 2-D array", name);
        Py_DECREF(arr);
        return NULL;
    }
    npy_intp rows = PyArray_DIM(arr, 0);
    npy_intp cols = PyArray_DIM(arr, 1);
    if (rows != cols) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be square, got (%ld, %ld)", name,
                     (long)rows, (long)cols);
        Py_DECREF(arr);
        return NULL;
    }
    *dim = (size_t)rows;
    return arr;
}

/* Allocate a fresh complex128 dim×dim array and copy src into it. */
static PyObject *c128_from_matrix(const lb_matrix_t *src)
{
    npy_intp dims[2] = {(npy_intp)src->dim, (npy_intp)src->dim};
    PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!out) {
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *)out), src->data,
           src->dim * src->dim * sizeof(double complex));
    return out;
}

/* --------------------------------------------------------------------------
 * propagate_step(P, rho_in) -> rho_out
 * -------------------------------------------------------------------------- */
static PyObject *
lb_ext_propagate_step(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *P_obj = NULL, *rho_obj = NULL;
    if (!PyArg_ParseTuple(args, "OO", &P_obj, &rho_obj)) {
        return NULL;
    }

    PyArrayObject *P_arr = NULL, *rho_arr = NULL;
    PyObject *result = NULL;
    lb_propagator_t prop = {0};
    lb_matrix_t rho_in = {0}, rho_out = {0};

    size_t d2 = 0, d = 0;
    P_arr = as_square_c128(P_obj, "P", &d2);
    if (!P_arr) {
        goto done;
    }
    rho_arr = as_square_c128(rho_obj, "rho_in", &d);
    if (!rho_arr) {
        goto done;
    }
    if (d * d != d2) {
        PyErr_Format(PyExc_ValueError,
                     "P (%zu x %zu) and rho_in (%zu x %zu) are inconsistent: "
                     "P must be d^2 x d^2 where rho_in is d x d",
                     d2, d2, d, d);
        goto done;
    }

    if (lb_matrix_alloc(&prop.P, d2) != 0 ||
        lb_matrix_alloc(&rho_in, d) != 0 ||
        lb_matrix_alloc(&rho_out, d) != 0) {
        PyErr_NoMemory();
        goto done;
    }
    prop.d = d;
    prop.dt = 0.0;
    memcpy(prop.P.data, PyArray_DATA(P_arr), d2 * d2 * sizeof(double complex));
    memcpy(rho_in.data, PyArray_DATA(rho_arr), d * d * sizeof(double complex));

    if (lb_propagate_step(&prop, &rho_in, &rho_out) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "lb_propagate_step failed");
        goto done;
    }

    result = c128_from_matrix(&rho_out);

done:
    lb_propagator_free(&prop);
    lb_matrix_free(&rho_in);
    lb_matrix_free(&rho_out);
    Py_XDECREF(P_arr);
    Py_XDECREF(rho_arr);
    return result;
}

/* --------------------------------------------------------------------------
 * evolve(H, cops, rho0, t_end, dt) -> rho_final
 * -------------------------------------------------------------------------- */
static PyObject *
lb_ext_evolve(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *H_obj = NULL, *cops_obj = NULL, *rho0_obj = NULL;
    double t_end = 0.0, dt = 0.0;
    if (!PyArg_ParseTuple(args, "OOOdd", &H_obj, &cops_obj, &rho0_obj,
                          &t_end, &dt)) {
        return NULL;
    }
    if (dt <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dt must be positive");
        return NULL;
    }

    PyArrayObject *H_arr = NULL, *rho0_arr = NULL;
    PyObject *cops_seq = NULL;
    PyObject *result = NULL;
    lb_system_t sys;
    int sys_inited = 0;
    lb_matrix_t rho0 = {0}, rho_out = {0};

    size_t d = 0, d_rho = 0;
    H_arr = as_square_c128(H_obj, "H", &d);
    if (!H_arr) {
        goto done;
    }
    rho0_arr = as_square_c128(rho0_obj, "rho0", &d_rho);
    if (!rho0_arr) {
        goto done;
    }
    if (d_rho != d) {
        PyErr_Format(PyExc_ValueError,
                     "rho0 dimension %zu does not match H dimension %zu",
                     d_rho, d);
        goto done;
    }

    lb_system_init(&sys, d);
    sys_inited = 1;
    if (lb_matrix_alloc(&sys.H, d) != 0) {
        PyErr_NoMemory();
        goto done;
    }
    memcpy(sys.H.data, PyArray_DATA(H_arr), d * d * sizeof(double complex));

    cops_seq = PySequence_Fast(cops_obj, "cops must be a sequence of arrays");
    if (!cops_seq) {
        goto done;
    }
    Py_ssize_t n_cops = PySequence_Fast_GET_SIZE(cops_seq);
    if (n_cops > LB_MAX_COPS) {
        PyErr_Format(PyExc_ValueError,
                     "too many collapse operators: %zd (max %d)",
                     n_cops, LB_MAX_COPS);
        goto done;
    }
    for (Py_ssize_t k = 0; k < n_cops; k++) {
        PyObject *item = PySequence_Fast_GET_ITEM(cops_seq, k); /* borrowed */
        size_t dk = 0;
        PyArrayObject *Lk_arr = as_square_c128(item, "collapse operator", &dk);
        if (!Lk_arr) {
            goto done;
        }
        if (dk != d) {
            PyErr_Format(PyExc_ValueError,
                         "collapse operator %zd has dimension %zu, expected %zu",
                         k, dk, d);
            Py_DECREF(Lk_arr);
            goto done;
        }
        lb_matrix_t Lk = {0};
        if (lb_matrix_alloc(&Lk, d) != 0) {
            PyErr_NoMemory();
            Py_DECREF(Lk_arr);
            goto done;
        }
        memcpy(Lk.data, PyArray_DATA(Lk_arr), d * d * sizeof(double complex));
        int added = lb_system_add_cop(&sys, &Lk);
        lb_matrix_free(&Lk);
        Py_DECREF(Lk_arr);
        if (added != 0) {
            PyErr_SetString(PyExc_RuntimeError, "lb_system_add_cop failed");
            goto done;
        }
    }

    if (lb_matrix_alloc(&rho0, d) != 0 || lb_matrix_alloc(&rho_out, d) != 0) {
        PyErr_NoMemory();
        goto done;
    }
    memcpy(rho0.data, PyArray_DATA(rho0_arr), d * d * sizeof(double complex));

    size_t n_steps = 0;
    if (lb_evolve(&sys, &rho0, t_end, dt, &rho_out, NULL, &n_steps) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "lb_evolve failed");
        goto done;
    }

    result = c128_from_matrix(&rho_out);

done:
    lb_matrix_free(&rho0);
    lb_matrix_free(&rho_out);
    if (sys_inited) {
        lb_system_free(&sys);
    }
    Py_XDECREF(cops_seq);
    Py_XDECREF(H_arr);
    Py_XDECREF(rho0_arr);
    return result;
}

static PyMethodDef lb_methods[] = {
    {"propagate_step", lb_ext_propagate_step, METH_VARARGS,
     "propagate_step(P, rho_in) -> rho_out: apply one Lindblad step "
     "rho_out = reshape(P @ vec(rho_in))."},
    {"evolve",         lb_ext_evolve,         METH_VARARGS,
     "evolve(H, cops, rho0, t_end, dt) -> rho_final: evolve rho0 from t=0 to "
     "t_end with fixed timestep dt."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef lb_module = {
    PyModuleDef_HEAD_INIT, "_lindblad_ext", NULL, -1, lb_methods
};

PyMODINIT_FUNC
PyInit__lindblad_ext(void)
{
    import_array();
    return PyModule_Create(&lb_module);
}
