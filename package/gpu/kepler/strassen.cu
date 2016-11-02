/**
 *  -- GKMX (version 1.1.0) --
 *
 *  NVIDIA Corp, Santa Clara
 *
 *  @date June 2016
 *  @author Chenhan D. Yu
 *
 */  

#include <utility>
#include <thrust/functional.h>

// This is ugly but ... we will see later.
#define PRECISION_d 1

// Tunned parameters 
#include <dgemm_param_nn.h>
#include <dgemm_param_nt.h>
#include <dgemm_param_tn.h>
#include <dgemm_param_tt.h>
#include <sgemm_param_nn.h>
#include <sgemm_param_nt.h>
#include <sgemm_param_tn.h>
#include <sgemm_param_tt.h>

// GKMX template
#include <strassen_gpu.hpp>

using namespace hmlp::strassen;


template <typename T>
struct identity 
{
  __host__ __device__ __forceinline__ T operator()( const T& x, int i, int j, int b ) const 
  {
    return x; 
  }
  T** A2;
  T** B2;
};



/**
 *  @brief This gkmm instance shows how to implement Gaussian kernel matrix
 *         using GKMX. op1 and op2 compose the FMA instruction. The initial
 *         value of FMA is zero. sq2nrm is set
 *         to be true so that a^2 -2ab + b^2 expansion is used. We use gkmx
 *         structure to pass in the double pointer arrays of A2 and B2.
 *         The rbf kernel performs a bandwidth scaling then takes the
 *         exponential function before stored back.
 */ 
void dstrassen
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  const double *Aarray[], int lda,
  const double *Barray[], int ldb,
        double *Carray[], int ldc,
  int batchSize
)
{
  // Declare semi-rings.
  thrust::plus<double> op1;
  thrust::multiplies<double> op2; 

  // Declare kernel operation
  identity<double> opkernel;

  // Declare <TV> initial value.
  double initV = 0.0;

  // Declare SQ2NRM
  const bool sq2nrm = false;

  strassen
  <sq2nrm, identity<double>, thrust::plus<double>, thrust::multiplies<double>,
  double, double, double, double>
  (
    stream, 
    transA, transB,
    m, n, k,
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    batchSize,
    opkernel, op1, op2, initV
  );
};

//void gkmm_dfma
//(
//  cudaStream_t stream, 
//  hmlpOperation_t transA, hmlpOperation_t transB, 
//  int m, int n, int k,
//  const double *Aarray, int lda, int loa, 
//  const double *Barray, int ldb, int lob,
//        double *Carray, int ldc, int loc,
//  int batchSize
//)
//{
//  // Declare semi-rings.
//  thrust::plus<double> op1;
//  thrust::multiplies<double> op2; 
//
//  // Declare kernel operation
//  identity<double> opkernel;
//
//  // Declare <TV> initial value.
//  double initV = 0.0;
//
//  // Declare SQ2NRM
//  const bool sq2nrm = false;
//
//  gkmm
//  <sq2nrm, identity<double>, thrust::plus<double>, thrust::multiplies<double>,
//  double, double, double, double>
//  (
//    stream, 
//    transA, transB,
//    m, n, k,
//    Aarray, lda, loa,
//    Barray, ldb, lob,
//    Carray, ldc, loc,
//    batchSize,
//    opkernel, op1, op2, initV
//  );
//};










///** 
// *  @brief This is the same gkmm instance that takes strided matrices.
// *
// */ 
//void dgkmm_batched_strided_instance(
//    cudaStream_t stream, cublasOperation_t transA, cublasOperation_t transB, 
//    int m, int n, int k,
//    const double *Aarray, int lda, int loa,
//    const double *Barray, int ldb, int lob,
//    double *Carray, int ldc, int loc,
//    int batchSize, struct gkmx_s<double> *gkmx )
//{
//  // Declare semi-rings.
//  add<double> op1;
//  mul<double> op2; 
//
//  // Declare kernel operation
//  gaussian<double,double> opkernel;
//
//  // Declare <TV> initial value.
//  double init1 = 0.0;
//
//  // Declare SQ2NRM
//  const bool sq2nrm = true;
//
//  opkernel.A2 = gkmx->A2;
//  opkernel.B2 = gkmx->B2;
//  opkernel.h  = gkmx->h;
//
//
//  gkmm_template_strided_batched
//    <double, double, double, double, 
//    sq2nrm, gaussian<double,double>, add<double>, mul<double> >
//      ( stream, transA, transB,
//        m, n, k,
//        Aarray, lda, loa, 
//        Barray, ldb, lob, 
//        Carray, ldc, loc,
//        batchSize,
//        opkernel,
//        op1, op2, init1 );
//}
//
///**
// *  @brief This gkmm instance shows how to implement one kmeans iteration
// *         using GKRX. op1 and op2 compose the FMA instruction. The initial
// *         value of FMA is zero. sq2nrm is set
// *         to be true so that a^2 -2ab + b^2 expansion is used. We use gkmx
// *         structure to pass in the double pointer arrays of A2 and B2.
// *         The rbf kernel performs identify transfromation. Only the first
// *         column of C is meaningful. The pair contains the centroid assignment
// *         and the distance.
// *         
// */ 
//void dgkrm_batched_instance(
//    cudaStream_t stream, cublasOperation_t transA, cublasOperation_t transB, 
//    int m, int n, int k,
//    const double *Aarray[], int lda,
//    const double *Barray[], int ldb,
//    thrust::pair<double,int> *Carray[], int ldc, 
//    int batchSize, struct gkmx_s<double> *gkmx )
//{
//  // Declare semi-rings.
//  add<double> op1;
//  mul<double> op2; 
//
//  // Declare kernel operation
//  kmeans<double, thrust::pair<double,int> > opkernel;
//
//  // Declare reduce operation
//  argmin<thrust::pair<double,int> > opreduce; 
//
//  // Declare <TV> initial value.
//  double init1 = 0.0;
//
//  // Declare <TC> initial value.
//  thrust::pair<double,int> init2;
//  init2.first  = 999999.99;
//  init2.second = -1;
//
//  // Declare SQ2NRM
//  const bool sq2nrm = true;
//
//  opkernel.A2 = gkmx->A2;
//  opkernel.B2 = gkmx->B2;
//
//  gkrm_template_batched<
//  double, double, thrust::pair<double,int>, double, 
//  sq2nrm, kmeans<double, thrust::pair<double,int> >, add<double>, mul<double>, argmin<thrust::pair<double,int> > >(
//      stream, transA, transB,
//      m, n, k,
//      Aarray, lda,
//      Barray, ldb,
//      Carray, ldc,
//      batchSize,
//      opkernel, op1, op2, init1, opreduce, init2 );
//}
//
///*
// *
// */ 
//void sgkrm_batched_strided_instance(
//    cudaStream_t stream, cublasOperation_t transA, cublasOperation_t transB, 
//    int m, int n, int k,
//    const float *Aarray, int lda, int loa,
//    const float *Barray, int ldb, int lob,
//    thrust::pair<float,int> *Carray, int ldc, int loc, 
//    int batchSize, struct gkmx_s<float> *gkmx )
//{
//  // Declare semi-rings.
//  add<float> op1;
//  mul<float> op2; 
//
//  // Declare kernel operation
//  kmeans<float, thrust::pair<float,int> > opkernel;
//
//  // Declare reduce operation
//  argmin<thrust::pair<float,int> > opreduce; 
//
//  // Declare <TV> initial value.
//  float init1 = 0.0;
//
//  // Declare <TC> initial value.
//  thrust::pair<float,int> init2;
//  init2.first  = 999999.99;
//  init2.second = -1;
//
//  // Declare SQ2NRM
//  const bool sq2nrm = true;
//
//  // Use precomputed squared 2-norm of A and B.
//  opkernel.A2 = gkmx->A2;
//  opkernel.B2 = gkmx->B2;
//
//  gkrm_template_strided_batched<
//  float, float, thrust::pair<float,int>, float, 
//  sq2nrm, kmeans<float, thrust::pair<float,int> >, add<float>, mul<float>, argmin<thrust::pair<float,int> > >(
//      stream, transA, transB,
//      m, n, k,
//      Aarray, lda, loa, 
//      Barray, ldb, lob, 
//      Carray, ldc, loc,
//      batchSize,
//      opkernel, op1, op2, init1, opreduce, init2 );
//}
//
//
//void linkage_disequilibria(
//    cudaStream_t stream, cublasOperation_t transA, cublasOperation_t transB, 
//    int m, int n, int k,
//    const unsigned long *Aarray, int lda, int loa,
//    const unsigned long *Barray, int ldb, int lob,
//          unsigned long *Carray, int ldc, int loc, 
//    int batchSize )
//{
//  add<unsigned long> op1;
//  linkage<unsigned long> op2;
//  identity<unsigned long> opkernel;
//  unsigned long init1 = 0.0;
//  const bool sq2nrm = false;
//
//  gkmm_template_strided_batched<
//    unsigned long, unsigned long, unsigned long, unsigned long, 
//    sq2nrm, identity<unsigned long>, add<unsigned long>, linkage<unsigned long> >(
//      stream, transA, transB,
//      m, n, k,
//      Aarray, lda, loa,
//      Barray, ldb, lob,
//      Carray, ldc, loc,
//      batchSize,
//      opkernel,
//      op1, op2, init1 );
//}
//
//
///*
// *
// */ 
//void dgkrm_batched_strided_instance(
//    cudaStream_t stream, cublasOperation_t transA, cublasOperation_t transB, 
//    int m, int n, int k,
//    const double *Aarray, int lda, int loa,
//    const double *Barray, int ldb, int lob,
//    thrust::pair<double,int> *Carray, int ldc, int loc, 
//    int batchSize, struct gkmx_s<double> *gkmx )
//{
//  // Declare semi-rings.
//  add<double> op1;
//  mul<double> op2; 
//
//  // Declare kernel operation
//  kmeans<double, thrust::pair<double,int> > opkernel;
//
//  // Declare reduce operation
//  argmin<thrust::pair<double,int> > opreduce; 
//
//  // Declare <TV> initial value.
//  double init1 = 0.0;
//
//  // Declare <TC> initial value.
//  thrust::pair<double,int> init2;
//  init2.first  = 999999.99;
//  init2.second = -1;
//
//  // Declare SQ2NRM
//  const bool sq2nrm = true;
//
//  opkernel.A2 = gkmx->A2;
//  opkernel.B2 = gkmx->B2;
//
//  gkrm_template_strided_batched<
//  double, double, thrust::pair<double,int>, double, 
//  sq2nrm, kmeans<double, thrust::pair<double,int> >, add<double>, mul<double>, argmin<thrust::pair<double,int> > >(
//      stream, transA, transB,
//      m, n, k,
//      Aarray, lda, loa, 
//      Barray, ldb, lob, 
//      Carray, ldc, loc,
//      batchSize,
//      opkernel, op1, op2, init1, opreduce, init2 );
//}
//
///*
// *
// */ 
//void dgkmmv_batched_instance(
//    cudaStream_t stream, cublasOperation_t transA, cublasOperation_t transB, 
//    int m, int n, int k,
//    const double *Aarray[], int lda,
//    const double *Barray[], int ldb,
//          double *Carray[], int ldc,
//    int batchSize, struct gkmx_s<double> *gkmx )
//{
//  // Declare semi-rings.
//  add<double> op1;
//  mul<double> op2; 
//
//  // Declare kernel operation
//  gaussian<double,double> opkernel;
//
//  // Declare reduce operation
//  ks<double> opreduce; 
//
//  // Declare <TV> initial value.
//  double init1 = 0.0;
//
//  // Declare <TC> initial value.
//  double init2 = 0.0;
//
//  // Declare SQ2NRM
//  const bool sq2nrm = true;
//
//  opkernel.A2 = gkmx->A2;
//  opkernel.B2 = gkmx->B2;
//  opkernel.h  = gkmx->h;
//
//  // Setup 
//  opkernel.w  = gkmx->w;
//
//  gkmmv_template_batched<
//  double, double, double, double, 
//  sq2nrm, gaussian<double,double>, add<double>, mul<double>, ks<double> >(
//      stream, transA, transB,
//      m, n, k,
//      Aarray, lda,
//      Barray, ldb,
//      Carray, ldc,
//      batchSize,
//      opkernel, op1, op2, init1, opreduce, init2 );
//}
//
///*
// *
// */ 
//void dgkmmv_batched_strided_instance(
//    cudaStream_t stream, cublasOperation_t transA, cublasOperation_t transB, 
//    int m, int n, int k,
//    const double *Aarray, int lda, int loa,
//    const double *Barray, int ldb, int lob,
//          double *Carray, int ldc, int loc, 
//    int batchSize, struct gkmx_s<double> *gkmx )
//{
//  // Declare semi-rings.
//  add<double> op1;
//  mul<double> op2; 
//
//  // Declare kernel operation
//  gaussian<double,double> opkernel;
//
//  // Declare reduce operation
//  ks<double> opreduce; 
//
//  // Declare <TV> initial value.
//  double init1 = 0.0;
//
//  // Declare <TC> initial value.
//  double init2 = 0.0;
//
//  // Declare SQ2NRM
//  const bool sq2nrm = true;
//
//  opkernel.A2 = gkmx->A2;
//  opkernel.B2 = gkmx->B2;
//  opkernel.h  = gkmx->h;
//
//  // Setup 
//  opkernel.w  = gkmx->w;
//
//  gkmmv_template_strided_batched<
//    double, double, double, double, 
//    sq2nrm, gaussian<double,double>, add<double>, mul<double>, ks<double> >(
//      stream, transA, transB,
//      m, n, k,
//      Aarray, lda, loa, 
//      Barray, ldb, lob, 
//      Carray, ldc, loc,
//      batchSize,
//      opkernel,
//      op1, op2, init1,
//      opreduce, init2 );
//}
//
//
///**
// *  @brief Compute the square 2-norm of X (d leading).
// */ 
//void dsq2nrm_batched_n(
//    int d, int n, double* X2array[], const double* Xarray[], double* X, int ldx, int batchSize )
//{
//  Square2nrmBatched<double, false, false>( d, n, X2array, Xarray, NULL, ldx, 0, batchSize );
//}
//
//
///**
// *  @brief Compute the square 2-norm of X (n leading).
// */ 
//void dsq2nrm_batched_t(
//    int d, int n, double* X2array[], const double* Xarray[], double* X, int ldx, int batchSize )
//{
//  Square2nrmBatched<double, false, true>( d, n, X2array, Xarray, NULL, ldx, 0, batchSize );
//}
//
