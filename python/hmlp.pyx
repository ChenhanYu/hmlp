
#cdef extern from "${PROJECT_SOURCE_DIR}/include/pyhmlp.h":
cdef extern from "${PROJECT_SOURCE_DIR}/include/hmlp.h":

	cdef void hmlp_init()
	cdef void hmlp_set_num_workers( int n_worker )
	cdef void hmlp_run()
	cdef void hmlp_finalize()

	cdef enum ks_type: 
		KS_GAUSSIAN,
		KS_POLYNOMIAL,
		KS_LAPLACE,
		KS_GAUSSIAN_VAR_BANDWIDTH,
		KS_TANH,
		KS_QUARTIC,
		KS_MULTIQUADRATIC,
		KS_EPANECHNIKOV

	cdef cppclass kernel_s[T]:
		ks_type type
		T powe
		T scal
		T cons
		T *hi
		T *hj
		T *h


	

  ## prototype of kernel summation in double precision
	cdef void dgsks( kernel_s[double] *kernel, 
			int m, int n, int k, 
			double *u,             int *umap, 
			double *A, double *A2, int *amap, 
			double *B, double *B2, int *bmap, 
			double *w,             int *wmap )


## end extern from #


cdef class Runtime:

	def __cinit__( self ):
		hmlp_init()
		print "execute hmlp_init()"

	def __dealloc__( self ):
		hmlp_finalize()

	cpdef init( self ):
		hmlp_init()

	cpdef set_num_workers( self, int nworkers ):
		hmlp_set_num_workers( nworkers )

	cpdef run( self ):
		hmlp_run()

	cpdef finalize( self ):
		hmlp_finalize()

#end class Runtime
