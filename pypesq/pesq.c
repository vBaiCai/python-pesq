#include <Python.h>
#include "arrayobject.h"
#include <stdio.h>
#include <math.h>
#include "pesq.h"
#include "dsp.h"

static char module_docstring[] = 
    "This module provides an interface for calculate PESQ.";
static char pesq_docstring[] = 
    "Compute PESQ.";
static PyObject *_pesq(PyObject *self, PyObject *arg);

#if PY_MAJOR_VERSION >= 3
static PyMethodDef module_methods[] = {
    {"_pesq", _pesq, METH_VARARGS, pesq_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pesqmodule = {
    PyModuleDef_HEAD_INIT,
    "pesq_core",
    module_docstring,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_pesq_core(void){
    import_array();     //essential for numpy
    return PyModule_Create(&pesqmodule);
};

#else 
static PyMethodDef PesqCoreMethods[] = {
    {"_pesq", _pesq, METH_VARARGS, module_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initpesq_core(void){
    PyObject *m;
    import_array();     //essential for numpy

    m = Py_InitModule("pesq_core", PesqCoreMethods);
    if (m == NULL)
        return;

}
#endif

static PyObject *_pesq(PyObject *self, PyObject *args){

    PyArrayObject *ref, *deg;
    long fs;

    if(!PyArg_ParseTuple(args, "O!O!l", &PyArray_Type, &ref,
        &PyArray_Type, &deg, &fs)){
        return NULL;
    }

    float pesq = compute_pesq(ref->data, deg->data, ref->dimensions[0], deg->dimensions[0], fs);

    return Py_BuildValue("f", pesq);
}