## cython stl support
from libcpp.string cimport string
from libcpp.vector cimport vector

## numpy support
import numpy as np

## cython numpy support
cimport numpy as np
from cython.operator cimport dereference as deref


## import hmlp::Data<T> 
cdef extern from "${PROJECT_SOURCE_DIR}/frame/containers/data.hpp" namespace "hmlp":

	cdef cppclass Data[T]:

		Data() except +

		void resize( size_t m, size_t n )

		size_t row()

		size_t col()

		T getvalue( size_t i )

		void setvalue( size_t i, T v )

		void read( size_t m, size_t n, string &filename )

	## end cppclass Data[T]

## end extern #


## create a Python wrapper for hmlp::Data<T>
cdef class PyData:

	## pointer to hmlp::Data<T>
	cdef Data[double] *_this

	def __cinit__( self ):
		self._this = new Data[double]()
		#print "execute cinit"

	#def __dealloc__( self ):
		#print "dealloc"

	def __getitem__( self, i ):
		return self._this.getvalue( i )

	def __setitem__( self, i, v ):
		self._this.setvalue( i, v )

	cpdef resize( self, size_t m, size_t n ):
		self._this.resize( m, n )

	cpdef row( self ):
		return self._this.row()

	cpdef col( self ):
		return self._this.col()

	cpdef read( self, m, n, filename ):
		self._this.read( m, n, filename )

## end class PyData



## import dTree_t from hmlp::gofmm #
cdef extern from "${PROJECT_SOURCE_DIR}/frame/gofmm/gofmm.hpp" namespace "hmlp::gofmm":

	## hmlp::gofmm::dTree_t #
	cdef cppclass dTree_t:

	  ## constructor #
		dTree_t() except +

    ## number of points #
		size_t n

    ## leaf node size #
		size_t m

    ## tree depth #
		size_t depth

	## end cppclass dTree_t #


  ## hmlp::gofmm::dSPDMatrix_t #
	cdef cppclass dSPDMatrix_t:

	  ## constructor #
		dSPDMatrix_t() except +

		void resize( size_t m, size_t n )

		size_t row()

		size_t col()

		double getvalue( size_t i )

		void setvalue( size_t i, double v )

		void randspd()

	## end cppclass dSPDMatrix_t:

	## hmlp::gofmm::Compress() #
	dTree_t *Compress( dSPDMatrix_t *K, double stol, double budget )

  ## hmlp::gofmm::Evaluate() #
	Data[double] Evaluate( dTree_t *tree, Data[double] *weights )

  ## hmlp::gofmm::ComputeError() #
	double ComputeError( dTree_t *tree, size_t gid, Data[double] *potentials )

## end extern from #


## dTree_t python class wrapper #
cdef class PyTree:

	cdef dTree_t *_this

	def __cinit__( self ):
		self._this = new dTree_t()
		#print "execute cinit dTree_t()"

	def __dealloc__( self ):
		if self._this != NULL:
			del self._this
		#print "dealloc"

	cdef reallocate( self, dTree_t *tree_ptr ):
		if self._this != NULL:
			del self._this
		self._this = tree_ptr

	cpdef num_points( self ):
		return self._this.n

	cpdef leaf_node_size( self ):
		return self._this.m

	cpdef depth( self ):
		return self._this.depth

## end class PyTree #


## create a Python wrapper for SPDMatrix
cdef class PySPDMatrix:

	cdef dSPDMatrix_t *_this

	def __cinit__( self ):
		self._this = new dSPDMatrix_t()
		#print "execute cinit"

	#def __dealloc__( self ):
		#print "dealloc"

	def __getitem__( self, i ):
		return self._this.getvalue( i )

	def __setitem__( self, i, v ):
		self._this.setvalue( i, v )

	cpdef resize( self, size_t m, size_t n ):
		self._this.resize( m, n )

	cpdef row( self ):
		return self._this.row()

	cpdef col( self ):
		return self._this.col()

	cpdef randspd( self ):
		self._this.randspd()

## end class PySPDMatrix #



def PyEvaluate( PyTree tree, rhs ):
	## currently, we only support double precision #
	assert rhs.dtype == np.double
	
	n = rhs.shape[ 0 ]
	nrhs = rhs.shape[ 1 ]

	weights = PyData()
	weights.resize( nrhs, n )

	for i in range( n ):
		for j in range( nrhs ):
			weights[ i * nrhs + j ] = rhs[ i, j ]

	cdef Data[double] potentials = Evaluate( tree._this, weights._this );

  ## new an numpy ndarray for output
	ret = rhs

	for i in range( n ):
		for j in range( nrhs ):
			ret[ i ][ j ] = potentials.getvalue( i * nrhs + j )

	return ret

## end PyEvaluate() #


## TODO: not sure if python is passed by reference
def PyCompress( PySPDMatrix K, double stol, double budget ):

	#print "before Compress()"
	cdef dTree_t *tree = Compress( K._this, stol, budget )
	#print "after Compress()"

	H = PyTree()
	H.reallocate( tree )
	#print "inside PyCompress()"
	
	return H

## end Compress() 

def PyComputeError( PyTree tree, size_t gid, u ):

	n = u.shape[ 0 ]
	nrhs = u.shape[ 1 ]

	potentials = PyData();
	potentials.resize( n, nrhs )

	for j in range( nrhs ):
		potentials[ j ] = u[ 0 ][ j ]

	cdef double ret = ComputeError( tree._this, gid, potentials._this )

	return ret

## end PyComputeError() 
