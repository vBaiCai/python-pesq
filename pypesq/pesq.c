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

static PyObject *_pesq(PyObject *self, PyObject *args){

    PyArrayObject *ref, *deg;
    long fs;

    if(!PyArg_ParseTuple(args, "O!O!l", &PyArray_Type, &ref,
        &PyArray_Type, &deg, &fs)){
        return NULL;
    }

    // change continuous arrays into C *array
    // int i;
    // float *cref = (float*)ref->data;
    // for(i=0; i<5; i++){
    //     //printf("%.9g\n", *(cref++));
    // }
    // //printf("%i\n", ref->strides[0]);
    // //printf("%i\n", ref->nd);

    //printf("%ld,%ld,%ld\n", ref->dimensions[0], deg->dimensions[0], fs);
    float pesq = compute_pesq(ref->data, deg->data, ref->dimensions[0], deg->dimensions[0], fs);

    // if(NULL==ref || NULL==deg){
    //     return NULL;
    // }
    return Py_BuildValue("f", pesq);
    return PyFloat_FromDouble(pesq);
}

