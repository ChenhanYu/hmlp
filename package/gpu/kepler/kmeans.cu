#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include<hmlp.h>


template<typename T>
void kmeans
(
  cudaStream_t stream, 
  //hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  T *Aarray[], T *A2array[], int lda,
  T *Barray[], T *B2array[], int ldb,
  thrust::pair<T,int>  *Carray[], int ldc, 
  int batchSize
)
{
  cublasHandle_t handle;

  // create handle;

  thrust::device_vector<T> Varray( m * n * batchSize, 0.0 );

  xgemm_batched
  (
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    1.0,
    Aarray, lda,
    Barray, ldb, 0.0,
    Varray, ldc,
    batchSize
  );

  // Compute the 2-norm here and reduce
  

};

void dkmeans
(
  cudaStream_t stream, 
  //hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  double *Aarray[], double *A2array[], int lda,
  double *Barray[], double *B2array[], int ldb,
  thrust::pair<double,int>  *Carray[], int ldc, 
  int batchSize
)
{
  keams<double>
  (
    stream,
    m, n, k,
    Aarray, A2array, lda,
    Barray, B2array, ldb,
    C,               ldc,
    batchSize
  );
}

void skmeans
(
  cudaStream_t stream, 
  //hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  float *Aarray[], float *A2array[], int lda,
  float *Barray[], float *B2array[], int ldb,
  thrust::pair<float,int>  *Carray[], int ldc, 
  int batchSize
)
{
  keams<float>
  (
    stream,
    m, n, k,
    Aarray, A2array, lda,
    Barray, B2array, ldb,
    C,               ldc,
    batchSize
  );
}
