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
  void dgeref_(
      int *m, int *n, 
      double *A, int *lda, int *ipiv, int *info );
  void sgeref_(
      int *m, int *n, 
      float *A, int *lda, int *ipiv, int *info );
  void dgeqrf_(
      int *m, int *n, 
      double *A, int *lda, 
      double *tau, 
      double *work, int *lwork, int *info );
  void sgeqrf_(
      int *m, int *n, 
      float *A, int *lda, 
      float *tau, 
      float *work, int *lwork, int *info );
  void dormqr_( 
      const char *side, const char *trans,
      int *m, int *n, int *k, 
      double *A, int *lda, 
      double *tau,
      double *C, int *ldc, 
      double *work, int *lwork, int *info );
  void sormqr_( 
      const char *side, const char *trans,
      int *m, int *n, int *k, 
      float *A, int *lda, 
      float *tau,
      float *C, int *ldc, 
      float *work, int *lwork, int *info );
  void dgeqp3_(
      int *m, int *n,
      double *A, int *lda, int *jpvt,
      double *tau,
      double *work, int *lwork, int *info ); 
  void sgeqp3_(
      int *m, int *n,
      float *A, int *lda, int *jpvt,
      float *tau,
      float *work, int *lwork, int *info );
  void dgels_(
      const char *trans,
      int *m, int *n, int *nrhs,
      double *A, int *lda,
      double *B, int *ldb,
      double *work, int *lwork, int *info );
  void sgels_(
      const char *trans,
      int *m, int *n, int *nrhs,
      float *A, int *lda,
      float *B, int *ldb,
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
#endif
};



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
#endif
};

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
#endif
};


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
#endif
};






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
#endif
};


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
#endif
};


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
#endif
};


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
#endif
};


}; // end namespace hmlp
