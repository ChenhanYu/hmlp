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






#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <hmlp_util.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define NUM_POINTS 10240
#define TOLERANCE 1E-13
#define GFLOPS 1073741824

using namespace hmlp;

template<typename T>
void compute_error(
    int    r,
    int    n,
    T *D,
    int    *I,
    T *D_gold,
    int    *I_gold
    )
{

  int    i, j, p;
  T *D1, *D2;
  int    *I1, *I2, *Set1, *Set2;

  #ifdef HMLP_MIC_AVX512
    D1 = (T*)hbw_malloc( sizeof(T) * r * n );
    D2 = (T*)hbw_malloc( sizeof(T) * r * n );
    I1 = (int*)hbw_malloc( sizeof(int) * r * n );
    I2 = (int*)hbw_malloc( sizeof(int) * r * n );
  #else
    D1 = (T*)malloc( sizeof(T) * r * n );
    D2 = (T*)malloc( sizeof(T) * r * n );
    I1 = (int*)malloc( sizeof(int) * r * n );
    I2 = (int*)malloc( sizeof(int) * r * n );
  #endif

  // Check error using bubbleSort.
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      D1[ j * r + i ] = D[ j * r + i ];
      I1[ j * r + i ] = I[ j * r + i ];
      D2[ j * r + i ] = D_gold[ j * r + i ];
      I2[ j * r + i ] = I_gold[ j * r + i ];
    }
    bubble_sort<T>( r, &D1[ j * r ], &I1[ j * r ] );
    bubble_sort<T>( r, &D2[ j * r ], &I2[ j * r ] );
  }

  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      if ( I1[ j * r + i ] != I2[ j * r + i ] ) {
        if ( fabs( D1[ j * r + i ] - D2[ j * r + i ] ) > TOLERANCE ) {
          printf( "D[ %d ][ %d ] != D_gold, %E, %E\n", i, j, D1[ j * r + i ], D2[ j * r + i ] );
          printf( "I[ %d ][ %d ] != I_gold, %d, %d\n", i, j, I1[ j * r + i ], I2[ j * r + i ] );
          break;
        }
      }
    }
  }

  free(D1);
  free(D2);
  free(I1);
  free(I2);
}


/*
 * --------------------------------------------------------------------------
 * @brief  This is the test routine to exam the correctness of GSKNN. XA and
 *         XB are d leading coordinate tables, and u, w have to be rhs
 *         leading. In this case, dgsknn() doesn't need to know the size of
 *         nxa and nxb as long as those index map--amap, bmap, umap and wmap
 *         --are within the legal range.
 *
 * @param  *kernel gsknn data structure
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * --------------------------------------------------------------------------
 */
template<typename T>
void test_gsknn( int m, int n, int k, int r )
{
  int    i, j, p, nx, iter, n_iter;
  int    *amap, *bmap, *I, *I_mkl;
  T *XA, *XB, *XA2, *XB2, *D, *D_mkl;
  double tmp, error, flops;
  double ref_beg, ref_time, dgsknn_beg, dgsknn_time;

  nx     = NUM_POINTS;
  n_iter = 1;


  // ------------------------------------------------------------------------
  // Memory allocation for all common buffers
  // ------------------------------------------------------------------------
#ifdef HMLP_MIC_AVX512
  amap  = (int*)hbw_malloc( sizeof(int) * m );
  bmap  = (int*)hbw_malloc( sizeof(int) * n );
  I     = (int*)hbw_malloc( sizeof(int) * r * n );
  I_mkl = (int*)hbw_malloc( sizeof(int) * r * n );
  XA    = (T*)hbw_malloc( sizeof(T) * k * nx );   // k   leading
  XA2   = (T*)hbw_malloc( sizeof(T) * nx );
  D     = (T*)hbw_malloc( sizeof(T) * r * n );
  D_mkl = (T*)hbw_malloc( sizeof(T) * r * n );
#else
  amap = (int*)malloc( sizeof(int) * m );
  bmap = (int*)malloc( sizeof(int) * n );
  I     = (int*)malloc( sizeof(int) * r * n );
  I_mkl = (int*)malloc( sizeof(int) * r * n );
  XA   = (T*)malloc( sizeof(T) * k * nx );   // k   leading
  XA2  = (T*)malloc( sizeof(T) * nx );
  D     = (T*)malloc( sizeof(T) * r * n );
  D_mkl = (T*)malloc( sizeof(T) * r * n );
#endif
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------

  for ( i = 0; i < m; i ++ )
  {
    amap[ i ] = i;
  }

  for ( j = 0; j < n; j ++ )
  {
    bmap[ j ] = j;
  }

  // random[ 0, 0.1 ]
  for ( i = 0; i < nx; i ++ )
  {
    for ( p = 0; p < k; p ++ )
    {
      XA[ i * k + p ] = (T)( rand() % 100 ) / 1000.0;
    }
  }
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Compute XA2
  // ------------------------------------------------------------------------
  for ( i = 0; i < nx; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Use the same coordinate table
  // ------------------------------------------------------------------------
  XB  = XA;
  XB2 = XA2;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Initialize D ( distance ) to the maximum double and I ( index ) to -1.
  // ------------------------------------------------------------------------

  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      D[ j * r + i ]     = 1.79E+308;
      I[ j * r + i ]     = -1;
      D_mkl[ j * r + i ] = 1.79E+308;
      I_mkl[ j * r + i ] = -1;
    }
  }

  // ------------------------------------------------------------------------
  // Call my implementation
  // ------------------------------------------------------------------------
  for ( iter = -1; iter < n_iter; iter ++ )
  {
    if ( iter == 0 ) dgsknn_beg = omp_get_wtime();
    dgsknn(
        n, m, k, r,
        XB, XB2, bmap,
        XA, XA2, amap,
        D,       I
    );
  }
  dgsknn_time = omp_get_wtime() - dgsknn_beg;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call the reference function
  // ------------------------------------------------------------------------
  for ( iter = -1; iter < n_iter; iter ++ )
  {
    if ( iter == 0 ) ref_beg = omp_get_wtime();

    dgsknn_ref(
        m, n, k, r,
        XA, XA2, amap,
        XB, XB2, bmap,
        D_mkl,   I_mkl
        );

  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time   /= n_iter;
  dgsknn_time /= n_iter;

  compute_error<T>(
      r,
      n,
      D,
      I,
      D_mkl,
      I_mkl
      );

  flops = ( (double)( m * n ) / GFLOPS ) * ( 2.0 * k + 37.0 );

#ifdef MATLAB_OUTPUT
  printf( "%d, %d, %d, %d, %5.2lf, %5.2lf;\n",
      m, n, k, r, flops / dgsknn_time, flops / ref_time );
#else
  printf( "%d, %5.2lf, %5.2lf\n",
      k, flops / dgsknn_time, flops / ref_time );
#endif

  free( amap );
  free( bmap );
  free( XA );
  free( XA2 );
  free( D );
  free( I );
  free( D_mkl );
  free( I_mkl );
}



int main( int argc, char *argv[] )
{
  int    m, n, k, r;

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );
  sscanf( argv[ 4 ], "%d", &r );

  test_gsknn<double>( m, n, k, r );

  return 0;
}
