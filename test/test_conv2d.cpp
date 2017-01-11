/*
 * test_conv_relu_pool.c
 *
 * Chenhan D. Yu
 *
 * Department of Computer Science, University of Texas at Austin
 *
 * Purpose: 
 * this is the main function to exam the correctness between dgemm_tn()
 * and dgemm() from the BLAS library.
 *
 * Todo:
 *
 * Modification:
 *
 * */

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
void test_conv2d
( 
  int w0, int h0, int d0, int s, int p, int batchSize,
  int w1, int h1, int d1
) 
{
  T *A, *B, *C, *C_ref;
  double ref_beg, ref_time, gkmx_beg, gkmx_time;

  int n_iter = 3;
  int m = d1;
  int nx = ( w0 - w1 + 2 * p ) / s + 1;
  int ny = ( h0 - h1 + 2 * p ) / s + 1;
  int n = nx * ny;
  int k = w1 * h1 * d0;

  //printf( "m %4d n %4d k %4d\n", m, n, k );

  double flops = ( (double)( m * n ) / GFLOPS ) * ( 2.0 * k + 0.0 ) * batchSize;

  // ------------------------------------------------------------------------
  // Memory allocation for all common buffers
  // ------------------------------------------------------------------------
#ifdef HMLP_MIC_AVX512
  A     = (T*)hbw_malloc( sizeof(T) * m * k );
  B     = (T*)hbw_malloc( sizeof(T) * w0 * h0 * d0 * batchSize );
  C     = (T*)hbw_malloc( sizeof(T) * m * n * batchSize );
  C_ref = (T*)hbw_malloc( sizeof(T) * m * n * batchSize );
#else
  A     = (T*)malloc( sizeof(T) * m * k );
  B     = (T*)malloc( sizeof(T) * w0 * h0 * d0 * batchSize );
  posix_memalign( (void**)&C,     32, sizeof(T) * m * n * batchSize );
  posix_memalign( (void**)&C_ref, 32, sizeof(T) * m * n * batchSize );
#endif
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  for ( auto i = 0; i < m; i ++ ) 
  {
    for ( auto p = 0; p < k; p ++ ) 
    {
      //A[ i * k + p ] = (T)( rand() % 100 ) / 1000.0;	
      A[ i * k + p ] = (T)1.0;	
    }
  }

  for ( auto j = 0; j < w0 * h0 * batchSize; j ++ ) 
  {
    for ( auto p = 0; p < d0; p ++ ) 
    {
      //B[ j * d0 + p ] = (T)( rand() % 100 ) / 1000.0;	
      B[ j * d0 + p ] = p + 1.0;	
    }
  }
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call my implementation (NN)
  // ------------------------------------------------------------------------
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) gkmx_beg = omp_get_wtime();
    //sconv2d
    dconv2d
    (
      w0, h0, d0, s, p, batchSize,
      B,
      w1, h1, d1,
      A,
      C
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
    //sconv2d_ref
    dconv2d_ref
    (
      w0, h0, d0, s, p, batchSize,
      B,
      w1, h1, d1,
      A,
      C_ref
    );
  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time  /= n_iter;
  gkmx_time /= n_iter;

  compute_error( m, n, C, m, C_ref, m );

#ifdef MATLAB_OUTPUT
  printf( "NN %5d, %5d, %5d, %5.2lf (%5.2lfms), %5.2lf (%5.2lfms);\n", 
      m, n, k, flops / gkmx_time, gkmx_time, flops / ref_time, ref_time );
#else
  printf( "%5d, %5.2lf, %5.2lf\n", 
      k, flops / gkmx_time, flops / ref_time );
#endif



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
  int w0, h0, d0, s, p, batchSize;
  int w1, h1, d1;

  sscanf( argv[ 1 ], "%d", &w0 );
  sscanf( argv[ 2 ], "%d", &h0 );
  sscanf( argv[ 3 ], "%d", &d0 );
  sscanf( argv[ 4 ], "%d", &s  );
  sscanf( argv[ 5 ], "%d", &p  );
  sscanf( argv[ 6 ], "%d", &batchSize );
  sscanf( argv[ 7 ], "%d", &w1 );
  sscanf( argv[ 8 ], "%d", &h1 );
  sscanf( argv[ 9 ], "%d", &d1 );

  test_conv2d<double>
  //test_conv2d<float>
  ( 
    w0, h0, d0, s, p, batchSize,
    w1, h1, d1
  );

  return 0;
};
