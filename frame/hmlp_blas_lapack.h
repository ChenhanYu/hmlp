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
  const char *transA, const char *transB,
  int m, int n, int k, 
  float alpha,
  float *A, int lda,
  float *B, int ldb, float beta,
  float *C, int ldc
);

void xgemm
(
  const char *transA, const char *transB,
  int m, int n, int k, 
  double alpha,
  double *A, int lda,
  double *B, int ldb, double beta,
  double *C, int ldc
);

void xgeqrf
(
  int m, int n, 
  double *A, int lda, 
  double *tau, 
  double *work, int lwork 
);

void xgeqrf
(
  int m, int n, 
  float *A, int lda, 
  float *tau, 
  float *work, int lwork 
);

void xormqr
(
  const char *side, const char *trans,
  int m, int n, int k, 
  float *A, int lda, 
  float *tau,
  float *C, int ldc, 
  float *work, int lwork
);

void xormqr
(
  const char *side, const char *trans,
  int m, int n, int k, 
  double *A, int lda, 
  double *tau,
  double *C, int ldc, 
  double *work, int lwork
);

void xgeqp3
(
  int m, int n,
  float *A, int lda, int *jpvt, 
  float *tau,
  float *work, int lwork 
);

void xgeqp3
(
  int m, int n,
  double *A, int lda, int *jpvt,
  double *tau,
  double *work, int lwork 
);

void xgels
(
  const char *trans,
  int m, int n, int nrhs,
  float *A, int lda,
  float *B, int ldb,
  float *work, int lwork 
);

void xgels
(
  const char *trans,
  int m, int n, int nrhs,
  double *A, int lda,
  double *B, int ldb,
  double *work, int lwork 
);

#ifdef HMLP_USE_CUDA
// cublasSgemm wrapper
void xgemm
(
  cublasHandle_t handle,
  cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, 
  float alpha,
  float *A, int lda,
  float *B, int ldb, float beta,
  float *C, int ldc
);

// cublasDgemm wrapper
void xgemm
(
  cublasHandle_t handle,
  cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, 
  double alpha,
  double *A, int lda,
  double *B, int ldb, double beta,
  double *C, int ldc
);

// cublasSgemmBatched wrapper
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

// cublasDgemmBatched wrapper
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
