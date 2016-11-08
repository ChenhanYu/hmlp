#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <hmlp_blas_lapack.h>


namespace hmlp
{

  
void xgemm_batched
(
  cublasHandle_t handle,
  cublasOperation_t transA, cublasOperation_t transB,
  int m, int n, int k, 
  double alpha,
  double *Aarray[], int lda,
  double *Barray[], int ldb, double beta,
  double *Carray[], int ldc,
  int batchSize
)
{
  cublasDgemmBatched
  (
    handle,
    transA, transB,
    m, n, k,
    &alpha,
    (const double**)Aarray, lda,
    (const double**)Barray, ldb, &beta,
                    Carray, ldc,
    batchSize
  );
};


void xgemm_batched
(
  cublasHandle_t handle,
  cublasOperation_t transA, cublasOperation_t transB,
  int m, int n, int k, 
  float alpha,
  float *Aarray[], int lda,
  float *Barray[], int ldb, float beta,
  float *Carray[], int ldc,
  int batchSize
)
{
  cublasSgemmBatched
  (
    handle,
    transA, transB,
    m, n, k,
    &alpha,
    (const float**)Aarray, lda,
    (const float**)Barray, ldb, &beta,
                   Carray, ldc,
    batchSize
  );
};



}; // end namespace hmlp
