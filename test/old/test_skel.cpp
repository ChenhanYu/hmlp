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
#include <base/hmlp_blas_lapack.h>
#include <base/Data.hpp>
#include <primitives/lowrank.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace hmlp::lowrank;


template<typename T>
void test_skel( int m, int n, int s )
{
  double beg, pmid_t, id_t;
  std::vector<T> A( m * n );
  std::vector<T> P;
  std::vector<int> jpjv;

  std::default_random_engine generator;
  std::normal_distribution<T> gaussian( 0.0, 1.0 );

  for ( int i = 0; i < m * n; i ++ )
  {
    A[ i ] = ( (T) std::rand() ) / RAND_MAX;
    A[ i ] /= 10.0;

    //A[ i ] = gaussian( generator );
  }

  beg = omp_get_wtime();
  //pmid<T>( m, n, s, A, jpjv, P );
  pmid_t = omp_get_wtime() - beg;

  beg = omp_get_wtime();
  //id<T>( m, n, s, A, jpjv, P );
  id_t = omp_get_wtime() - beg;

  printf( "%d, %d %d, %5.2fs, %5.2fs\n", m, n, s, pmid_t, id_t );
};

int main( int argc, char *argv[] )
{
  int m, n, s;

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &s );
  
  test_skel<double>( m, n, s );

  hmlp::Data<double> X;

  X.resize( 3, 10 );

  std::cout << X.size() << std::endl;
 
  std::size_t arg1, arg2;
  std::tie( arg1, arg2 ) = X.shape();

  std::cout << arg1 << std::endl;
  std::cout << arg2 << std::endl;
  

  X.rand();
  X.Print();
  X.randn( 0.0, 1.0 );
  X.Print();



  return 0;
};
