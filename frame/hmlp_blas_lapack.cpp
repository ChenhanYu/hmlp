#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <hmlp_blas_lapack.h>

// #define DEBUG_XGEMM 1



extern "C"
{
  void dgemm_( const char *transA, const char *transB, 
      int *m, int *n, int *k, 
      double *alpha,
      double *A, int *lda, 
      double *B, int *ldb, double *beta, 
      double *C, int *ldc );
  void sgemm_( const char *transA, const char *transB, 
      int *m, int *n, int *k, 
      float *alpha,
      float *A, int *lda, 
      float *B, int *ldb, float *beta, 
      float *C, int *ldc );
  void dgeqp3_(
      int *m, int *n,
      double *A, int *lda, int *jpvt,
      double *work, int *lwork, int *info ); 
  void sgeqp3_(
      int *m, int *n,
      float *A, int *lda, int *jpvt,
      float *work, int *lwork, int *info ); 
};


namespace hmlp
{

/**
 *  @brief DGEMM wrapper
 */ 
void xgemm
(
  const char *transA, const char *transB,
  int m, int n, int k, 
  double alpha,
  double *A, int lda,
  double *B, int ldb, double beta,
  double *C, int ldc
)
{
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
  float alpha,
  float *A, int lda,
  float *B, int ldb, float beta,
  float *C, int ldc
)
{
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



/**
 *  @brief DGEQP3 wrapper
 */ 
void xgeqp3
(
  int m, int n,
  double *A, int lda, int *jpvt,
  double *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  dgeqp3_
  (
    &m, &n, 
    A, &lda, jpvt,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgeqp3 has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgeqp3 must enables USE_BLAS.\n" );
#endif
};


/**
 *  @brief SGEQP3 wrapper
 */ 
void xgeqp3
(
  int m, int n,
  float *A, int lda, int *jpvt,
  float *work, int lwork 
)
{
#ifdef USE_BLAS
  int info;
  sgeqp3_
  (
    &m, &n, 
    A, &lda, jpvt,
    work, &lwork, &info
  );
  if ( info ) 
  {
    printf( "xgeqp3 has illegal values at parameter %d\n", info );
  }
#else
  printf( "xgeqp3 must enables USE_BLAS.\n" );
#endif
};



}; // end namespace hmlp
