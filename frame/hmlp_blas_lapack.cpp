#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// #define DEBUG_XGEMM 1


extern "C"
{
  void dgemm_( char *transA, char *transB, 
      int *m, int *n, int *k, 
      double *alpha,
      double *A, int *lda, 
      double *B, int *ldb, double *beta, 
      double *C, int *ldc );
  void sgemm_( char *transA, char *transB, 
      int *m, int *n, int *k, 
      float *alpha,
      float *A, int *lda, 
      float *B, int *ldb, float *beta, 
      float *C, int *ldc );
};


namespace hmlp
{

void xgemm(
    char *transA, char *transB,
    int m, int n, int k, 
    double alpha,
    double *A, int lda,
    double *B, int ldb, double beta,
    double *C, int ldc
    )
{
  dgemm_( transA, transB, 
      &m, &n, &k, 
      &alpha,
      A, &lda, 
      B, &ldb, &beta, 
      C, &ldc );
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
}

void xgemm(
    char *transA, char *transB,
    int m, int n, int k, 
    float alpha,
    float *A, int lda,
    float *B, int ldb, float beta,
    float *C, int ldc
    )
{
  sgemm_( transA, transB, 
      &m, &n, &k, 
      &alpha,
      A, &lda, 
      B, &ldb, &beta, 
      C, &ldc );
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
}


}; // end namespace hmlp
