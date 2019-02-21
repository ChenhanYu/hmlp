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

#include <blas_lapack.hpp>

// #define DEBUG_XGEMM 1

#ifndef USE_BLAS
#warning BLAS/LAPACK routines are not compiled (-HMLP_USE_BLAS=false)
#endif

extern "C"
{
#include <external/blas_lapack_prototypes.h>
}; /** end extern "C" */






namespace hmlp
{

/** 
 *  BLAS level-1 wrappers: DOT, NRM2 
 */


/**
 *  @brief DDOT wrapper
 */ 
double xdot( int n, const double *dx, int incx, const double *dy, int incy )
{
  double ret_val;
#ifdef USE_BLAS
  ret_val = ddot_( &n, dx, &incx, dy, &incy );
#else
  printf( "xdot must enables USE_BLAS.\n" );
  exit( 1 );
#endif
  return ret_val;
}; /** end xdot() */



/**
 *  @brief SDOT wrapper
 */ 
float xdot( int n, const float *dx, int incx, const float *dy, int incy )
{
  float ret_val;
#ifdef USE_BLAS
  ret_val = sdot_( &n, dx, &incx, dy, &incy );
#else
  printf( "xdot must enables USE_BLAS.\n" );
  exit( 1 );
#endif
  return ret_val;
}; /** end xdot() */


/**
 *  @brief DNRM2 wrapper
 */ 
double xnrm2( int n, double *x, int incx )
{
  double ret_val;
#ifdef USE_BLAS
  ret_val = dnrm2_( &n, x, &incx );
#else
  printf( "xnrm2 must enables USE_BLAS.\n" );
  exit( 1 );
#endif
  return ret_val;
}; /** end xnrm2() */


/**
 *  @brief SNRM2 wrapper
 */ 
float  xnrm2( int n,  float *x, int incx )
{
  float ret_val;
#ifdef USE_BLAS
  ret_val = snrm2_( &n, x, &incx );
#else
  printf( "xnrm2 must enables USE_BLAS.\n" );
  exit( 1 );
#endif
  return ret_val;
}; /** end xnrm2() */






void xaxpy( const int n, const float* alpha, const float* x, const int incx, float* y, const int incy )
{
#ifdef USE_BLAS
  saxpy_( &n, alpha, x, &incx, y, &incy );
#else
  printf( "xaxpy must enables USE_BLAS.\n" );
  exit( 1 );
#endif
};

void xaxpy( const int n, const double* alpha, const double* x, const int incx, double* y, const int incy )
{
#ifdef USE_BLAS
  daxpy_( &n, alpha, x, &incx, y, &incy );
#else
  printf( "xaxpy must enables USE_BLAS.\n" );
  exit( 1 );
#endif
};


/** 
 *  BLAS level-3 wrappers: GEMM, TRSM 
 */


/**
 *  @brief DGEMM wrapper
 */ 
void xgemm
(
  const char *transA, const char *transB,
  int m, int n, int k, 
  double alpha, const double *A, int lda,
                const double *B, int ldb, 
  double beta,        double *C, int ldc
)
{
  double beg, xgemm_time = 0.0;
  double gflops = (double)( m ) * n * ( 2 * k ) / 1E+9;
  beg = omp_get_wtime();
#ifdef USE_BLAS
  dgemm_
  (
    transA, transB, 
    &m, &n, &k, 
    &alpha, A, &lda, 
            B, &ldb, 
    &beta,  C, &ldc 
  );
#else
  //printf( "xgemm: configure HMLP_USE_BLAS=true to enable BLAS support.\n" );
  for ( int p = 0; p < k; p ++ )
  {
    for ( int j = 0; j < n; j ++ )
    {
      for ( int i = 0; i < m; i ++ )
      {
        double a, b;
        if ( *transA == 'T' ) a = A[ i * lda + p ];
        else                  a = A[ p * lda + i ];
        if ( *transB == 'T' ) b = B[ p * ldb + j ];
        else                  b = B[ j * ldb + p ];
        if ( p == 0 )
        {
          C[ j * ldc + i ] = beta * C[ j * ldc + i ] + alpha * a * b;
        }
        else 
        {
          C[ j * ldc + i ] += alpha * a * b;
        }
      }
    }
  }
#endif
  xgemm_time = omp_get_wtime() - beg;
#ifdef DEBUG_XGEMM
  printf( "dgemm %s%s m %d n %d k %d, %5.2lf GFLOPS %5.2lf s\n", 
      transA, transB, m, n, k, gflops / xgemm_time, xgemm_time );
#endif

#ifdef DEBUG_XGEMM
  printf( "hmlp::xgemm debug\n" );
  for ( int i = 0; i < m; i ++ )
  {
    for ( int j = 0; j < n; j ++ )
    {
      printf( "%E ", C[ j * ldc + i ] / alpha );
    }
    printf( "\n" );
  }
#endif
};                

/**
 *  @brief SGEMM wrapper
 */ 
void xgemm
(
  const char *transA, const char *transB,
  int m, int n, int k, 
  float alpha, const float *A, int lda,
               const float *B, int ldb, 
  float beta,        float *C, int ldc
)
{
  double beg, xgemm_time = 0.0;
  double gflops = (double)( m ) * n * ( 2 * k ) / 1E+9;
  beg = omp_get_wtime();
#ifdef USE_BLAS
  sgemm_
  (
   transA, transB, 
   &m, &n, &k, 
   &alpha, A, &lda, 
           B, &ldb, 
   &beta,  C, &ldc 
  );
#else
  //printf( "xgemm: configure HMLP_USE_BLAS=true to enable BLAS support.\n" );
  for ( int p = 0; p < k; p ++ )
  {
    for ( int j = 0; j < n; j ++ )
    {
      for ( int i = 0; i < m; i ++ )
      {
        double a, b;
        if ( *transA == 'T' ) a = A[ i * lda + p ];
        else                  a = A[ p * lda + i ];
        if ( *transB == 'T' ) b = B[ p * ldb + j ];
        else                  b = B[ j * ldb + p ];
        if ( p == 0 )
        {
          C[ j * ldc + i ] = beta * C[ j * ldc + i ] + alpha * a * b;
        }
        else 
        {
          C[ j * ldc + i ] += alpha * a * b;
        }
      }
    }
  }
#endif
  xgemm_time = omp_get_wtime() - beg;
#ifdef DEBUG_XGEMM
  printf( "sgemm %s%s m %d n %d k %d, %5.2lf GFLOPS %5.2lf s\n", 
      transA, transB, m, n, k, gflops / xgemm_time, xgemm_time );
#endif
#ifdef DEBUG_XGEMM
  printf( "hmlp::xgemm debug\n" );
  for ( int i = 0; i < m; i ++ )
  {
    for ( int j = 0; j < n; j ++ )
    {
      printf( "%E ", C[ j * ldc + i ] );
    }
    printf( "\n" );
  }
#endif
};


void xsyrk
(
  const char *uplo, const char *trans,
  int n, int k, 
  double alpha, double *A, int lda,
  double beta,  double *C, int ldc
)
{
#ifdef USE_BLAS
  dsyrk_
  (
   uplo, trans, 
   &n, &k, 
   &alpha, A, &lda, 
   &beta,  C, &ldc 
  );
#else
  printf( "xsyrk must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xsyrk() */


void xsyrk
(
  const char *uplo, const char *trans,
  int n, int k, 
  float alpha, float *A, int lda,
  float beta,  float *C, int ldc
)
{
#ifdef USE_BLAS
  ssyrk_
  (
   uplo, trans, 
   &n, &k, 
   &alpha, A, &lda, 
   &beta,  C, &ldc 
  );
#else
  printf( "xsyrk must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xsyrk() */





/**
 *  @brief DTRSM wrapper
 */ 
void xtrsm
( 
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  double alpha,
  double *A, int lda,
  double *B, int ldb 
)
{
  double beg, xtrsm_time = 0.0;
  double gflops = (double)( m ) * ( m - 1 ) * n / 1E+9;
  beg = omp_get_wtime();

#ifdef USE_BLAS
  dtrsm_
  (
    side, uplo,
	  transA, diag,
	  &m, &n,
	  &alpha,
	  A, &lda,
	  B, &ldb
  );
#else
  printf( "xtrsm must enables USE_BLAS.\n" );
  exit( 1 );
#endif
  
  xtrsm_time = omp_get_wtime() - beg;
#ifdef DEBUG_XTRSM
  printf( "dtrsm m %d n %d, %5.2lf GFLOPS, %5.2lf s\n", 
      m, n, gflops / xtrsm_time, xtrsm_time );
#endif
}; /** end xtrsm() */


/**
 *  @brief STRSM wrapper
 */ 
void xtrsm
( 
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  float alpha,
  float *A, int lda,
  float *B, int ldb 
)
{
#ifdef USE_BLAS
  strsm_
  (
    side, uplo,
	  transA, diag,
	  &m, &n,
	  &alpha,
	  A, &lda,
	  B, &ldb
  );
#else
  printf( "xtrsm must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xtrsm() */


/**
 *  @brief DTRMM wrapper
 */ 
void xtrmm
( 
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  double alpha,
  double *A, int lda,
  double *B, int ldb 
)
{
#ifdef USE_BLAS
  dtrmm_
  (
    side, uplo,
	  transA, diag,
	  &m, &n,
	  &alpha,
	  A, &lda,
	  B, &ldb
  );
#else
  printf( "xtrmm must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xtrmm() */


/**
 *  @brief DTRMM wrapper
 */ 
void xtrmm
( 
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  float alpha,
  float *A, int lda,
  float *B, int ldb 
)
{
#ifdef USE_BLAS
  strmm_
  (
    side, uplo,
	  transA, diag,
	  &m, &n,
	  &alpha,
	  A, &lda,
	  B, &ldb
  );
#else
  printf( "xtrmm must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xtrmm() */


/**
 *  LAPACK routine wrappers: POTR(F,S), GETR(F,S), GECON, GEQRF, 
 *  ORGQR, ORMQR, GEQP3, GELS
 */


/**
 *  @brief DLASWP wrapper
 */ 
void xlaswp( int n, double *A, int lda, 
    int k1, int k2, int *ipiv, int incx )
{
#ifdef USE_BLAS
  dlaswp_( &n, A, &lda, &k1, &k2, ipiv, &incx );
#else
  printf( "xlaswp must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xlaswp() */


/**
 *  @brief SLASWP wrapper
 */ 
void xlaswp( int n, float *A, int lda, 
    int k1, int k2, int *ipiv, int incx )
{
#ifdef USE_BLAS
  slaswp_( &n, A, &lda, &k1, &k2, ipiv, &incx );
#else
  printf( "xlaswp must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xlaswp() */


/**
 *  @brief DPOTRF wrapper
 */ 
void xpotrf( const char *uplo, int n, double *A, int lda )
{
#ifdef USE_BLAS
  int info;
  dpotrf_( uplo, &n, A, &lda, &info );
  if ( info ) printf( "xpotrf error code %d\n", info );
#else
  printf( "xpotrf must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xpotrf() */


/**
 *  @brief SPOTRF wrapper
 */ 
void xpotrf( const char *uplo, int n, float *A, int lda )
{
#ifdef USE_BLAS
  int info;
  spotrf_( uplo, &n, A, &lda, &info );
  if ( info ) printf( "xpotrf error code %d\n", info );
#else
  printf( "xpotrf must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xpotrf() */


/**
 *  @brief DPOTRS wrapper
 */ 
void xpotrs( const char *uplo, 
  int n, int nrhs, double *A, int lda, double *B, int ldb )
{
#ifdef USE_BLAS
  int info;
  dpotrs_( uplo, &n, &nrhs, A, &lda, B, &ldb, &info );
  if ( info ) printf( "xpotrs error code %d\n", info );
#else
  printf( "xpotrs must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xpotrs() */


/**
 *  @brief SPOTRS wrapper
 */ 
void xpotrs( const char *uplo, 
  int n, int nrhs, float *A, int lda, float *B, int ldb )
{
#ifdef USE_BLAS
  int info;
  spotrs_( uplo, &n, &nrhs, A, &lda, B, &ldb, &info );
  if ( info ) printf( "xpotrs error code %d\n", info );
#else
  printf( "xpotrs must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xpotrs() */


/**
 *  @brief DGETRF wrapper
 */ 
void xgetrf( int m, int n, double *A, int lda, int *ipiv )
{
#ifdef USE_BLAS
  int info;
  dgetrf_( &m, &n, A, &lda, ipiv, &info );
#else
  printf( "xgetrf must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgetrf() */


/**
 *  @brief SGETRF wrapper
 */ 
void xgetrf( int m, int n, float *A, int lda, int *ipiv )
{
#ifdef USE_BLAS
  int info;
  sgetrf_( &m, &n, A, &lda, ipiv, &info );
#else
  printf( "xgetrf must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgetrf() */


/**
 *  @brief DGETRS wrapper
 */ 
void xgetrs
(
  const char *trans,
  int m, int nrhs, 
  double *A, int lda, int *ipiv,
  double *B, int ldb
)
{
#ifdef USE_BLAS
  int info;
  dgetrs_
  (
    trans,
    &m, &nrhs, 
    A, &lda, ipiv, 
    B, &ldb, &info
  );
#else
  printf( "xgetrs must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgetrs() */


/**
 *  @brief SGETRS wrapper
 */ 
void xgetrs
(
  const char *trans,
  int m, int nrhs, 
  float *A, int lda, int *ipiv,
  float *B, int ldb
)
{
#ifdef USE_BLAS
  int info;
  sgetrs_
  (
    trans,
    &m, &nrhs, 
    A, &lda, ipiv, 
    B, &ldb, &info
  );
#else
  printf( "xgetrs must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgetrs() */


/**
 *  @brief DGECON wrapper
 */ 
void xgecon
(
  const char *norm,
  int n,
  double *A, int lda, 
  double anorm, 
  double *rcond, 
  double *work, int *iwork 
)
{
#ifdef USE_BLAS
  int info;
  dgecon_
  (
    norm,
    &n,
    A, &lda,
    &anorm,
    rcond,
    work, iwork, &info
  );
#else
  printf( "xgecon must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgecon() */



/**
 *  @brief SGECON wrapper
 */ 
void xgecon
(
  const char *norm,
  int n,
  float *A, int lda, 
  float anorm, 
  float *rcond, 
  float *work, int *iwork 
)
{
#ifdef USE_BLAS
  int info;
  sgecon_
  (
    norm,
    &n,
    A, &lda,
    &anorm,
    rcond,
    work, iwork, &info
  );
#else
  printf( "xgecon must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgecon() */


/**
 *  @brief DGEQRF wrapper
 */ 
void xgeqrf
(
  int m, int n, 
  double *A, int lda, 
  double *tau, 
  double *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  dgeqrf_
  (
    &m, &n,
    A, &lda, 
    tau,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgeqrf has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgeqrf must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgeqrf() */



/**
 *  @brief SGEQRF wrapper
 */ 
void xgeqrf
(
  int m, int n, 
  float *A, int lda, 
  float *tau, 
  float *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  sgeqrf_
  (
    &m, &n,
    A, &lda, 
    tau,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgeqrf has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgeqrf must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgeqrf() */


/**
 *  @brief SORGQR wrapper
 */ 
void xorgqr
(
  int m, int n, int k,
  double *A, int lda, 
  double *tau,
  double *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  dorgqr_
  (
    &m, &n, &k,
    A, &lda,
    tau,
    work, &lwork, &info
  );
#else
  printf( "xorgqr must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xorgqr() */


/**
 *  @brief SORGQR wrapper
 */ 
void xorgqr
(
  int m, int n, int k,
  float *A, int lda, 
  float *tau,
  float *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  sorgqr_
  (
    &m, &n, &k,
    A, &lda,
    tau,
    work, &lwork, &info
  );
#else
  printf( "xorgqr must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xorgqr() */


/**
 *  @brief DORMQR wrapper
 */ 
void xormqr
(
  const char *side, const char *trans,
  int m, int n, int k, 
  double *A, int lda, 
  double *tau,
  double *C, int ldc, 
  double *work, int lwork
)
{
#ifdef USE_BLAS
  int info;
  dormqr_
  (
    side, trans,
    &m, &n, &k,
    A, &lda,
    tau,
    C, &ldc,
    work, &lwork, &info
  );
#else
  printf( "xormqr must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xormqr() */


/**
 *  @brief SORMQR wrapper
 */ 
void xormqr
(
  const char *side, const char *trans,
  int m, int n, int k, 
  float *A, int lda, 
  float *tau,
  float *C, int ldc, 
  float *work, int lwork
)
{
#ifdef USE_BLAS
  int info;
  sormqr_
  (
    side, trans,
    &m, &n, &k,
    A, &lda,
    tau,
    C, &ldc,
    work, &lwork, &info
  );
#else
  printf( "xormqr must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xormqr() */


/**
 *  @brief DGEQP3 wrapper
 */ 
void xgeqp3
(
  int m, int n,
  double *A, int lda, int *jpvt,
  double *tau,
  double *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  dgeqp3_
  (
    &m, &n, 
    A, &lda, jpvt,
    tau,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgeqp3 has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgeqp3 must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end geqp3() */


/**
 *  @brief SGEQP3 wrapper
 */ 
void xgeqp3
(
  int m, int n,
  float *A, int lda, int *jpvt,
  float *tau,
  float *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  sgeqp3_
  (
    &m, &n, 
    A, &lda, jpvt,
    tau,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgeqp3 has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgeqp3 must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end geqp3() */


/**
 *  @brief DGEQP4 wrapper
 */ 
void xgeqp4
(
  int m, int n,
  double *A, int lda, int *jpvt,
  double *tau,
  double *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  dgeqp4
  (
    &m, &n, 
    A, &lda, jpvt,
    tau,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgeqp4 has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgeqp4 must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end geqp4() */


/**
 *  @brief SGEQP4 wrapper
 */ 
void xgeqp4
(
  int m, int n,
  float *A, int lda, int *jpvt,
  float *tau,
  float *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  sgeqp4
  (
    &m, &n, 
    A, &lda, jpvt,
    tau,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgeqp4 has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgeqp4 must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end geqp4() */


/**
 *  @brief DGELS wrapper
 */ 
void xgels
(
  const char *trans,
  int m, int n, int nrhs,
  double *A, int lda,
  double *B, int ldb,
  double *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  dgels_
  (
    trans,
    &m, &n, &nrhs,
    A, &lda,
    B, &ldb,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgels has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgels must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end gels() */


/**
 *  @brief SGELS wrapper
 */ 
void xgels
(
  const char *trans,
  int m, int n, int nrhs,
  float *A, int lda,
  float *B, int ldb,
  float *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  sgels_
  (
    trans,
    &m, &n, &nrhs,
    A, &lda,
    B, &ldb,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgels has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgels must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end gels() */



/**
 *  @brief DGESDD wrapper
 */ 
void xgesdd
( 
  const char *jobz, 
  int m, int n, 
  double *A, int lda, 
  double *S, 
  double *U, int ldu, 
  double *VT, int ldvt, 
  double *work, int lwork, int *iwork
)
{
#ifdef USE_BLAS
  int info;
  dgesdd_
  (
    jobz,
    &m, &n,
    A, &lda,
    S,
    U, &ldu,
    VT, &ldvt,
    work, &lwork, iwork, &info
  );
#else
  printf( "xgesdd must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgesdd() */

/**
 *  @brief SGESDD wrapper
 */ 
void xgesdd
( 
  const char *jobz, 
  int m, int n, 
  float *A, int lda, 
  float *S, 
  float *U, int ldu, 
  float *VT, int ldvt, 
  float *work, int lwork, int *iwork
)
{
#ifdef USE_BLAS
  int info;
  sgesdd_
  (
    jobz,
    &m, &n,
    A, &lda,
    S,
    U, &ldu,
    VT, &ldvt,
    work, &lwork, iwork, &info
  );
#else
  printf( "xgesdd must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xgesdd() */



void xstev
(
  const char *jobz,
  int n, 
  double *D, 
  double *E, 
  double *Z, int ldz, 
  double *work  
)
{
#ifdef USE_BLAS
  int info;
  dstev_( jobz, &n, D, E, Z, &ldz, work, &info );
#else
  printf( "xstev must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xstev() */


void xstev
(
  const char *jobz,
  int n, 
  float *D, 
  float *E, 
  float *Z, int ldz, 
  float *work  
)
{
#ifdef USE_BLAS
  int info;
  sstev_( jobz, &n, D, E, Z, &ldz, work, &info );
#else
  printf( "xstev must enables USE_BLAS.\n" );
  exit( 1 );
#endif
}; /** end xstev() */


}; /** end namespace hmlp */
