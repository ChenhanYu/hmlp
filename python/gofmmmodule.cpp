#include <stdio.h>
#include <stdlib.h>

/** Python API */
#include <Python.h>

/** PyArrayObjest */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <hmlp.h>

static PyObject *Evaluate( PyObject *self, PyObject *args )
{
  PyObject *ret = Py_BuildValue( "i", 0 );
  return ret;
}; /** end Evaluate() */


static PyObject *Compress( PyObject *self, PyObject *args )
{
  double stol = 0.0, budget = 0.0;
  PyObject *Kobject;
  //int Kshape[ 3 ] = { 0, 0, 0 };

  printf( "before parsing\n" ); fflush( stdout );

  /** parse 3 inputs */
  //if ( !PyArg_ParseTuple( args, "Odd", &Kobject, &stol, &budget ) ) 
  //  return NULL;
  if ( !PyArg_ParseTuple( args, "O", &Kobject ) ) 
    return NULL;

  printf( "after parsing\n" ); fflush( stdout );

  /** interpret the objects as numpy arrays. */
  //PyObject *Karray = PyArray_FROM_OTF( Kobject, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
  PyObject *Karray = PyArray_FROM_O( Kobject );
  //PyObject *Karray = PyArray_FromAny( Kobject, PyArray_DescrFromType(NPY_FLOAT64), 0, 0, NPY_ARRAY_C_CONTIGUOUS, NULL );

  printf( "after PyArray_FROM_OTF\n" ); fflush( stdout );

  ///* If that didn't work, throw an exception. */
  //if ( Karray == NULL ) 
  //{
  //  Py_XDECREF( Karray );
  //  return NULL;
  //}

  ///** detect the number of modes of K */
  //size_t Kmodes = (size_t)PyArray_NDIM( Karray );
  //size_t m = (size_t)PyArray_DIM( Karray, 0 );
  //size_t n = (size_t)PyArray_DIM( Karray, 1 );

  //printf( "modes %lu m %lu n %lu stol %3.1E budge %.2lf\n", Kmodes, m, n, stol, budget );


  PyObject *ret = Py_BuildValue( "i", 0 );
  return ret;
}; /** end Compress() */



static PyMethodDef gofmmMethods[] = {
  { "Evaluate", Evaluate, METH_VARARGS, "Evaluate." },
  { "Compress", Compress, METH_VARARGS, "Compress." },
  { NULL, NULL, 0, NULL }        /* Sentinel */
}; 

static PyObject *gofmmError;

PyMODINIT_FUNC initgofmm( void )
{
  printf( "initgofmm\n" ); fflush( stdout );
  PyObject *m = Py_InitModule( "gofmm", gofmmMethods );
  if ( m == NULL ) return;
  gofmmError = PyErr_NewException( "gofmm.error", NULL, NULL);
  Py_INCREF( gofmmError );
  PyModule_AddObject( m, "error", gofmmError );
}; /** end initgofmm() */



///**
// *  @brief Python wrapper for Compress()
// */ 
//static PyObject * Compress_wrapper( PyObject *self, PyObject *args )
//{
//  hmlp::Data<double> *K;
//  double stol = 0.0, budget = 0.0;
//
//  /** parse arguments */
//  if ( !PyArg_ParseTuple( args, "Odd", K, &stol, &budget ) ) 
//  {
//    return NULL;
//  }
//
//  /** run the actual function */
//  //auto tree = Compress<double>( *K, stol, budget );
//  //return Py_BuildValue( "O", &tree );
//
// 
//  return Py_BuildValue( "O", K );
//
//}; /** end Compress_wrapper() */
//
//
//static PyMethodDef CompressMethods[] = {
//  {"Compress",  Compress_wrapper, METH_VARARGS, "Execute Compress."},
//  {NULL, NULL, 0, NULL}        /* Sentinel */
//};
//
//PyMODINIT_FUNC initCompress(void)
//{
//  Py_InitModule("Compress", CompressMethods);
//};




