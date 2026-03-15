/**
 * lindblad_ext.c — CPython C extension: zero-copy NumPy interface to
 * the lindblad-bench C library.
 *
 * Exposes:
 *   propagate_step(P, rho_in) -> rho_out
 *   evolve(H, cops, rho0, t_end, dt) -> rho_final
 *
 * Build with: pip install -e python/
 * Requires: C library built at ../build/liblindblad_bench.a
 *
 * TODO: implement after core C library is validated.
 *       Stub returns NotImplementedError so Python package imports cleanly.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *
lb_ext_propagate_step(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "lindblad_ext C extension not yet implemented. "
                    "Use reference/reference.py for now.");
    return NULL;
}

static PyObject *
lb_ext_evolve(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "lindblad_ext C extension not yet implemented. "
                    "Use reference/reference.py for now.");
    return NULL;
}

static PyMethodDef lb_methods[] = {
    {"propagate_step", lb_ext_propagate_step, METH_VARARGS,
     "Apply one Lindblad propagation step (C fast path)."},
    {"evolve",         lb_ext_evolve,         METH_VARARGS,
     "Evolve density matrix from t=0 to t=t_end (C fast path)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef lb_module = {
    PyModuleDef_HEAD_INIT, "_lindblad_ext", NULL, -1, lb_methods
};

PyMODINIT_FUNC
PyInit__lindblad_ext(void)
{
    return PyModule_Create(&lb_module);
}
