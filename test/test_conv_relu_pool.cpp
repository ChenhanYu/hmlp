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
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <hmlp_blas_lapack.h>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

template<typename T>
void compute_error
(
  int m, int n,
  T *test, int ldtest,
  T *goal, int ldgoal
)
{
  std::tuple<int, int, T> max_err( -1, -1, (T)0.0 );
  T abs_err = 0.0, rel_err = 0.0, nrm2 = 0.0;

  for ( auto j = 0; j < n; j ++ ) 
  {
    for ( auto i = 0; i < m; i ++ ) 
    {
      auto tmp_goal = goal[ j * ldgoal + i ];
      auto tmp_test = test[ j * ldtest + i ];
      auto err = fabs( tmp_test - tmp_goal );
      if ( err > std::get<2>( max_err ) ) 
      {
        max_err = std::make_tuple( i, j, err );
      }
      rel_err += err * err;
      nrm2    += tmp_goal * tmp_goal;
    }
  }

  abs_err = sqrt( rel_err );
  rel_err /= nrm2;
  rel_err = sqrt( rel_err );

  if ( rel_err > TOLERANCE ) 
  {
	  printf( "rel error %E, abs error %E, max error %E at (%d %d)\n", 
		  rel_err, abs_err, std::get<2>( max_err ), 
      std::get<0>( max_err ), std::get<1>( max_err ) );
  }
}


/* 
 * --------------------------------------------------------------------------
 * @brief  This is the test routine to exam the correctness of GSKS. XA and
 *         XB are d leading coordinate tables, and u, w have to be rhs
 *         leading. In this case, dgsks() doesn't need to know the size of
 *         nxa and nxb as long as those index map--amap, bmap, umap and wmap
 *         --are within the legal range.
 *
 * @param  *kernel gsks data structure
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * --------------------------------------------------------------------------
 */
template<typename T>
void test_conv_relu_pool( int m, int n, int k ) 
{
  T *A, *B, *C, *C_ref;
  double ref_beg, ref_time, gkmx_beg, gkmx_time;
  double flops = ( (double)( m * n ) / GFLOPS ) * ( 2.0 * k + 0.0 );

  int n_iter = 1;

  // ------------------------------------------------------------------------
  // Memory allocation for all common buffers
  // ------------------------------------------------------------------------
#ifdef HMLP_MIC_AVX512
  A     = (T*)hbw_malloc( sizeof(T) * m * k );
  B     = (T*)hbw_malloc( sizeof(T) * k * n );
  C     = (T*)hbw_malloc( sizeof(T) * m * n );
  C_ref = (T*)hbw_malloc( sizeof(T) * m * n );
#else
  A     = (T*)malloc( sizeof(T) * m * k );
  B     = (T*)malloc( sizeof(T) * k * n );
  //C     = (T*)malloc( sizeof(T) * m * n );
  //C_ref = (T*)malloc( sizeof(T) * m * n );
  posix_memalign( (void**)&C,     32, sizeof(T) * m * n );
  posix_memalign( (void**)&C_ref, 32, sizeof(T) * m * n );
#endif
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  for ( auto i = 0; i < m; i ++ ) 
  {
    for ( auto p = 0; p < k; p ++ ) 
    {
      A[ i * k + p ] = (T)( rand() % 100 ) / 1000.0;	
    }
  }

  for ( auto j = 0; j < n; j ++ ) 
  {
    for ( auto p = 0; p < k; p ++ ) 
    {
      B[ j * k + p ] = (T)( rand() % 100 ) / 1000.0;	
    }
  }
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Call my implementation (NN)
  // ------------------------------------------------------------------------
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) gkmx_beg = omp_get_wtime();
    gkmx_dconv_relu_pool
    (
      HMLP_OP_N, HMLP_OP_N,
      m, n, k,
      A, m,
      B, k,
      C, m 
    );
  }
  gkmx_time = omp_get_wtime() - gkmx_beg;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call the reference function (NN)
  // ------------------------------------------------------------------------
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) ref_beg = omp_get_wtime();
    hmlp::xgemm
    ( 
      "N", "N", 
      m, n, k, 
      1.0, A,     m, 
           B,     k, 
      0.0, C_ref, m 
    );

    for ( auto j = 0; j < n; j ++ )
    {
      #pragma omp parallel for
      for ( auto i = 0; i < m; i ++ )
      {
         C_ref[ ( j / 4 ) * m + i ] = std::max( C_ref[ ( j / 4 ) * m + i  ], C_ref[ j * m + i ] );
      }
    }
  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time  /= n_iter;
  gkmx_time /= n_iter;

  compute_error( m, n, C, m, C_ref, m );

  printf( "NN %5d, %5d, %5d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / gkmx_time, flops / ref_time );



#ifdef HMLP_MIC_AVX512
  hbw_free( A );
  hbw_free( B );
  hbw_free( C );
  hbw_free( C_ref );
#else
  free( A );
  free( B );
  free( C );
  free( C_ref );
#endif

};


int main( int argc, char *argv[] )
{
  int m, n, k;

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );

  test_conv_relu_pool<double>( m, n, k );

  return 0;
};
