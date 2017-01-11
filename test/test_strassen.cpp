/*
 * test_strassen.c
 *
 * Jianyu Huang
 *
 * Department of Computer Science, University of Texas at Austin
 *
 * Purpose: 
 * this is the main function to exam the correctness between dstrassen()
 * and dgemm() from the BLAS library.
 *
 * Todo:
 *
 * Modification:
 *
 * */

#include <tuple>
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
void hmlp_printmatrix(
    T *A,
    int    lda,
    int    m,
    int    n
    )
{
  int    i, j;
  for ( i = 0; i < m; i ++ ) {
    for ( j = 0; j < n; j ++ ) {
      printf("%lf\t", A[j * lda + i]);  //Assume T is double...
    }
    printf("\n");
  }
}

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
void test_strassen( int m, int n, int k ) 
{
  T *A, *B, *C, *C_ref;
  double ref_beg, ref_time, strassen_beg, strassen_time;
  double flops = ( (double)( m * n ) / GFLOPS ) * ( 2.0 * k + 0.0 );

  int lda, ldb, ldc, ldc_ref;
  int MR = 8;

  ldc     = ( ( m - 1 ) / MR + 1 ) * MR;
  ldc_ref = ( ( m - 1 ) / MR + 1 ) * MR;

  int n_iter = 3;

  // ------------------------------------------------------------------------
  // Memory allocation for all common buffers
  // ------------------------------------------------------------------------
#ifdef HMLP_MIC_AVX512
  A     = (T*)hbw_malloc( sizeof(T) * m * k );
  B     = (T*)hbw_malloc( sizeof(T) * k * n );
  //C     = (T*)hbw_malloc( sizeof(T) * m * n );
  //C_ref = (T*)hbw_malloc( sizeof(T) * m * n );
#else
  A     = (T*)malloc( sizeof(T) * m * k );
  B     = (T*)malloc( sizeof(T) * k * n );
  //C     = (T*)malloc( sizeof(T) * m * n );
  //C_ref = (T*)malloc( sizeof(T) * m * n );
#endif
  //C     = hmlp_malloc( m, n, sizeof(T) );
  //C_ref = hmlp_malloc( m, n, sizeof(T) );
  int ALIGN_SIZE = 32;
  posix_memalign( (void**)&C,     (size_t)ALIGN_SIZE, sizeof(T) * ldc     * ( n + 4 ) );
  posix_memalign( (void**)&C_ref, (size_t)ALIGN_SIZE, sizeof(T) * ldc_ref * ( n + 4 ) );

  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  for ( auto i = 0; i < m; i ++ ) 
  {
    for ( auto p = 0; p < k; p ++ ) 
    {
      A[ i * k + p ] = (T)( rand() % 100 ) / 1000.0;	
      //A[ i * k + p ] = (T)( 1.0 );
    }
  }

  for ( auto j = 0; j < n; j ++ ) 
  {
    for ( auto p = 0; p < k; p ++ ) 
    {
      B[ j * k + p ] = (T)( rand() % 100 ) / 1000.0;	
      //B[ j * k + p ] = (T)( 1.0 );
    }
  }

  for ( auto i = 0; i < m; i ++ ) 
  {
    for ( auto j = 0; j < n; j ++ ) 
    {
      C[ j * ldc + i ] = (T)( 1.0 );
      C_ref[ j * ldc_ref + i ] = (T)( 1.0 );
    }
  }
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Call my implementation (NN)
  // ------------------------------------------------------------------------
  lda = m;
  ldb = k;
  for ( auto iter = 0; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) strassen_beg = omp_get_wtime();
    dstrassen
    (
      HMLP_OP_N, HMLP_OP_N,
      m, n, k,
      A, lda,
      B, ldb,
      C, ldc
    );
  }
  strassen_time = omp_get_wtime() - strassen_beg;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call the reference function (NN)
  // ------------------------------------------------------------------------
  for ( auto iter = 0; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) ref_beg = omp_get_wtime();
    hmlp::xgemm
    ( 
      "N", "N", 
      m, n, k, 
      1.0, A,     lda, 
           B,     ldb, 
      1.0, C_ref, ldc_ref 
    );
  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time  /= n_iter;
  strassen_time /= n_iter;


  //printf( "C:" );
  //hmlp_printmatrix( C, ldc, m, n );
  //printf( "C_ref:" );
  //hmlp_printmatrix( C_ref, ldc_ref, m, n );


  compute_error( m, n, C, m, C_ref, m );

  printf( "NN %5d, %5d, %5d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / strassen_time, flops / ref_time );


  // ------------------------------------------------------------------------
  // Call my implementation (TN)
  // ------------------------------------------------------------------------
  lda = k;
  ldb = k;
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) strassen_beg = omp_get_wtime();
    dstrassen
    (
      HMLP_OP_T, HMLP_OP_N,
      m, n, k,
      A, lda,
      B, ldb,
      C, ldc 
    );
  }
  strassen_time = omp_get_wtime() - strassen_beg;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call the reference function (TN)
  // ------------------------------------------------------------------------
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) ref_beg = omp_get_wtime();
    hmlp::xgemm
    ( 
      "T", "N", 
      m, n, k, 
      1.0, A,     lda, 
           B,     ldb, 
      1.0, C_ref, ldc_ref 
    );
  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time  /= n_iter;
  strassen_time /= n_iter;


  compute_error( m, n, C, m, C_ref, m );

  printf( "TN %5d, %5d, %5d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / strassen_time, flops / ref_time );



  // ------------------------------------------------------------------------
  // Call my implementation (TT)
  // ------------------------------------------------------------------------
  lda = k;
  ldb = n;
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) strassen_beg = omp_get_wtime();
    dstrassen
    (
      HMLP_OP_T, HMLP_OP_T,
      m, n, k,
      A, lda,
      B, ldb,
      C, ldc
    );
  }
  strassen_time = omp_get_wtime() - strassen_beg;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call the reference function (TT)
  // ------------------------------------------------------------------------
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) ref_beg = omp_get_wtime();
    hmlp::xgemm
    ( 
      "T", "T", 
      m, n, k, 
      1.0, A,     lda, 
           B,     ldb, 
      1.0, C_ref, ldc_ref 
    );
  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time  /= n_iter;
  strassen_time /= n_iter;

  //printf( "C:" );
  //hmlp_printmatrix( C, ldc, m, n );
  //printf( "C_ref:" );
  //hmlp_printmatrix( C_ref, ldc_ref, m, n );


  compute_error( m, n, C, m, C_ref, m );

  printf( "TT %5d, %5d, %5d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / strassen_time, flops / ref_time );



  // ------------------------------------------------------------------------
  // Call my implementation (NT)
  // ------------------------------------------------------------------------
  lda = m;
  ldb = n;
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) strassen_beg = omp_get_wtime();
    dstrassen
    (
      HMLP_OP_N, HMLP_OP_T,
      m, n, k,
      A, lda,
      B, ldb,
      C, ldc 
    );
  }
  strassen_time = omp_get_wtime() - strassen_beg;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call the reference function (NT)
  // ------------------------------------------------------------------------
  for ( auto iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) ref_beg = omp_get_wtime();
    hmlp::xgemm
    ( 
      "N", "T", 
      m, n, k, 
      1.0, A,     lda, 
           B,     ldb, 
      1.0, C_ref, ldc_ref 
    );
  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time  /= n_iter;
  strassen_time /= n_iter;


  //printf( "C:" );
  //hmlp_printmatrix( C, ldc, m, n );
  //printf( "C_ref:" );
  //hmlp_printmatrix( C_ref, ldc_ref, m, n );


  compute_error( m, n, C, m, C_ref, m );

#ifdef MATLAB_OUTPUT
  printf( "NT %5d, %5d, %5d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / strassen_time, flops / ref_time );
#else
  printf( "%5d, %5.2lf, %5.2lf;\n", 
      k, flops / strassen_time, flops / ref_time );
#endif

}


int main( int argc, char *argv[] )
{
  int m, n, k;

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );

  test_strassen<double>( m, n, k );

  return 0;
}
