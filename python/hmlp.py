##  
##  HMLP (High-Performance Machine Learning Primitives)
##  
##  Copyright (C) 2014-2018, The University of Texas at Austin
##  
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##  
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
##  GNU General Public License for more details.
##  
##  You should have received a copy of the GNU General Public License
##  along with this program. If not, see the LICENSE file.
##  


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


cdef class PyRuntime:

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
