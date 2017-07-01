#include <stdio.h>
#include <stdlib.h>
#include <Python.h>

#include <hmlp.h>

static PyObject *hmlp_init( PyObject *self, PyObject *args )
{
  printf( "here hmlp_init()\n" ); fflush( stdout );
  hmlp_init();
  printf( "finish hmlp_init()\n" ); fflush( stdout );
  PyObject *ret = Py_BuildValue( "i", 0 );
  return ret;
}; /** end hmlp_init() */

static PyObject *hmlp_set_num_workers( PyObject *self, PyObject *args )
{
  int nworkers = 1;
  if ( !PyArg_ParseTuple( args, "i", &nworkers ) ) return NULL;
  hmlp_set_num_workers( nworkers );
  PyObject *ret = Py_BuildValue( "i", 0 );
  return ret;
}; /** end hmlp_set_num_workers() */

static PyObject *hmlp_run( PyObject *self, PyObject *args )
{
  hmlp_run();
  PyObject *ret = Py_BuildValue( "i", 0 );
  return ret;
}; /** end hmlp_run) */

static PyObject *hmlp_finalize( PyObject *self, PyObject *args )
{
  hmlp_finalize();
  PyObject *ret = Py_BuildValue( "i", 0 );
  return ret;
}; /** end hmlp_finalize() */




static PyMethodDef hmlpMethods[] = {
  { "init",            hmlp_init,            METH_VARARGS, "Initialize HMLP." },
  { "set_num_workers", hmlp_set_num_workers, METH_VARARGS, "change number of workers." },
  { "run",             hmlp_run,             METH_VARARGS, "Execute all tasks." },
  { "finalize",        hmlp_finalize,        METH_VARARGS, "Finalize HMLP." },
  { NULL, NULL, 0, NULL }        /* Sentinel */
}; 

static PyObject *HMLPError;

PyMODINIT_FUNC inithmlp( void )
{
  printf( "inithmlp\n" ); fflush( stdout );
  PyObject *m = Py_InitModule( "hmlp", hmlpMethods );
  if ( m == NULL ) return;
  HMLPError = PyErr_NewException( "hmlp.error", NULL, NULL);
  Py_INCREF( HMLPError );
  PyModule_AddObject( m, "error", HMLPError );
}; /** end inithmlp() */



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




