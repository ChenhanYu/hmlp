#ifndef HMLP_BLAS_LAPACK_H
#define HMLP_BLAS_LAPACK_H


#ifdef HMLP_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#endif

namespace hmlp
{

void xgemm
(
  char *transA, char *transB,
  int m, int n, int k, 
  float alpha,
  float *A, int lda,
  float *B, int ldb, float beta,
  float *C, int ldc
);

void xgemm
(
  char *transA, char *transB,
  int m, int n, int k, 
  double alpha,
  double *A, int lda,
  double *B, int ldb, double beta,
  double *C, int ldc
);

#ifdef HMLP_USE_CUDA
void xgemm_batched
(
  cublasHandle_t handle,
  cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, 
  float alpha,
  float *Aarray[], int lda,
  float *Barray[], int ldb, float beta,
  float *Carray[], int ldc,
  int batchSize
);

void xgemm_batched
(
  cublasHandle_t handle,
  cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, 
  double alpha,
  double *Aarray[], int lda,
  double *Barray[], int ldb, double beta,
  double *Carray[], int ldc,
  int batchSize
);
#endif


}; // end namespace hmlp

#endif // define HMLP_BLAS_LAPACK_H
