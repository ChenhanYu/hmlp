#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include<hmlp.h>
#include<hmlp_blas_lapack.h>
#include<gkmx_gpu.hpp>

using namespace hmlp;


template <typename TV, typename TC>
struct kmeans 
{
  __host__ __device__ __forceinline__ TC operator()
  ( 
    const TV& x, int i, int j, int b 
  ) const 
  { 
    return thrust::make_pair( x , j ); 
  }
  TV** A2;
  TV** B2;
};

template <typename TC>
struct argmin 
{
  __host__ __device__ __forceinline__ TC operator()
  ( 
    const TC& lhs, const TC& rhs, int i, int j, int b 
  ) const 
  {
    return lhs.first < rhs.first ? lhs : rhs;
  }
};




template<typename T>
void kmeans_ref
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
  using TC = thrust::pair<T, int>;

  cublasHandle_t handle;
  cublasCreate( &handle );

  printf( "m %d ldc %d n %d k %d batchSize %d\n", m, ldc, n, k, batchSize );

  thrust::device_vector<T>  Varray( ldc * n * batchSize, 0.0 );
  thrust::device_vector<T*> Varrayp( batchSize );


  printf( "after allocate\n" );

  kmeans<T, TC> opkernel;
  argmin<TC> opreduce;

  // Declare <TC> initial value.
  TC initC( 999999.99, -1 );

  opkernel.A2 = A2array;
  opkernel.B2 = B2array;


  printf( "after thrust::device\n" );

  for ( int i = 0; i < batchSize; i ++ ) 
  {
    Varrayp[ i ] = Varray.data().get() + i * ldc * n;
  }

  xgemm_batched
  (
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    m, n, k,
    1.0,
    Aarray, lda,
    Barray, ldb, 0.0,
    Varrayp.data().get(), ldc,
    batchSize
  );


  printf( "after xgemm_batched\n" );

  // Compute the 2-norm here and reduce
  gkmx::transform
  <T, TC, false, true, kmeans<T, TC> >
  (
    0,
    m, n, 
    Varrayp.data().get(), (T*)NULL, 
    Carray, (TC*)NULL, ldc, ldc * n,
    batchSize, 
    opkernel
  )
  ;
  printf( "after transform\n" );


  gkmx::reduce
  <TC, false, argmin<TC> >
  (
    0,
    m, n,
    Carray, Carray[ 0 ], ldc, ldc * n,
    batchSize,
    opreduce, initC
  );
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
  kmeans_ref<double>
  (
    stream,
    m, n, k,
    Aarray, A2array, lda,
    Barray, B2array, ldb,
    Carray,          ldc,
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
  kmeans_ref<float>
  (
    stream,
    m, n, k,
    Aarray, A2array, lda,
    Barray, B2array, ldb,
    Carray,          ldc,
    batchSize
  );
}
