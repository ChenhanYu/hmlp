/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  


#include <tuple>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>

#include <external/hmlp_blas_lapack.h>

#include <Data.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace hmlp;



int main( int argc, char *argv[] )
{
	using T = float;

	size_t m = 18; 
  size_t n = 5000000;
 
	std::string inputfile( "/work/02794/ych/data/SUSY.csv" );
	std::string outputfile( "/work/02794/ych/data/SUSY5M18D.bin" );

	Data<T> X( m, n );

	X.readmtx<1, true>( m, n, inputfile );
	X.write( outputfile );



  return 0;
};
