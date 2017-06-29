#ifndef HMLP_BLAS_LAPACK_H
#define HMLP_BLAS_LAPACK_H

#include <containers/view.hpp>

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

void xtrsm
( 
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  float alpha,
  float *A, int lda,
  float *B, int ldb 
);

void xtrsm
( 
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  double alpha,
  double *A, int lda,
  double *B, int ldb 
);

void xtrmm
(
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  float alpha,
  float *A, int lda,
  float *B, int ldb 
);

void xtrmm
(
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  double alpha,
  double *A, int lda,
  double *B, int ldb 
);

void xpotrf
(
  const char *uplo, 
  int n, double *A, int lda
);

void xpotrf
(
  const char *uplo, 
  int n, float *A, int lda
);

void xgetrf
(
  int m, int n, 
  double *A, int lda, int *ipiv
);

void xgetrf
(
  int m, int n, 
  float *A, int lda, int *ipiv
);

void xgetrs
(
  const char *trans,
  int m, int nrhs, 
  double *A, int lda, int *ipiv,
  double *B, int ldb
);

void xgetrs
(
  const char *trans,
  int m, int nrhs, 
  float *A, int lda, int *ipiv,
  float *B, int ldb
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

void xorgqr
(
  int m, int n, int k,
  double *A, int lda, 
  double *tau,
  double *work, int lwork 
);

void xorgqr
(
  int m, int n, int k,
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

void xgecon
(
  const char *norm,
  int n,
  float *A, int lda, 
  float anorm, 
  float *rcond, 
  float *work, int *iwork 
);

void xgecon
(
  const char *norm,
  int n,
  double *A, int lda, 
  double anorm, 
  double *rcond, 
  double *work, int *iwork 
);

double xdot
(
  int n,
  double *dx, int incx,
  double *dy, int incy
);

float xdot
(
  int n,
  float *dx, int incx,
  float *dy, int incy
);


#ifdef HMLP_USE_CUDA
// cublasSgemm wrapper
void xgemm
(
  cublasHandle_t &handle,
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
  cublasHandle_t &handle,
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
  cublasHandle_t &handle,
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
  cublasHandle_t &handle,
  cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, 
  double alpha,
  double *Aarray[], int lda,
  double *Barray[], int ldb, double beta,
  double *Carray[], int ldc,
  int batchSize
);


void xgeqp3
(
  cublasHandle_t &handle,
  int m, int n,
  float *A, int lda,
  int *jpvt,
  float *tau,
  float *work, int lwork
);


void xgeqp3
(
  cublasHandle_t &handle,
  int m, int n,
  double *A, int lda,
  int *jpvt,
  double *tau,
  double *work, int lwork
);

#endif


/**
 *  @brief
 */ 
template<size_t NB = 512, typename T>
void xgemm_var1( 
    T alpha, hmlp::View<T> &A, 
             hmlp::View<T> &B, 
    T beta,  hmlp::View<T> &C )
{
  /** all subviews */
  hmlp::View<T> AL, AR, 
                A0, A1, A2;
  hmlp::View<T> BT, BB, 
                B0, B1, B2;
  
  
  A.partition1x2( AL, AR, 0, TOP);
  B.partition2x1( BT,
                  BB, 0, TOP ); 

  while ( AL.col() < A.col() )
  {
    size_t b = std::min( AR.col(), NB );

    /** repartition A */
    Repartition1x2To1x3( AL,      AR,
                         /** **** */
                         A0,  A1, A2, b, RIGHT );

    /** repartition B */
    Repartition2x1To3x1( BT, /**/ B0
                             /**/ B1,
                         BB, /**/ B2, b, BOTTOM );


    /** --------------------------------------------------- */
    xgemmTask( alpha, A, B, beta, C );
    /** --------------------------------------------------- */

    /** merge B */
    ContinueWith1x3To1x2( AL,      AR,
                          /** **** */
                          A0,  A1, A2, LEFT );


    /** merge B */
    ContinueWith3x1To2x1( BT, /**/ B0,
                              /**/ B1,
                          BB, /**/ B2, TOP );

  } /** end while */

}; /** end xgemm_var1() */


/**
 *  @brief [ A * BL + CL, A * BR + CR ] 
 */ 
template<size_t NB = 512, typename T>
void xgemm_var2( 
    T alpha, hmlp::View<T> &A, 
             hmlp::View<T> &B, 
    T beta,  hmlp::View<T> &C )
{

}; /** end xgemm_var2() */


/**
 *  @brief [ AT * B + CT; AB * B + CB ] 
 */ 
template<size_t NB = 512, typename T>
void xgemm_var3( 
    T alpha, hmlp::View<T> &A, 
             hmlp::View<T> &B, 
    T beta,  hmlp::View<T> &C )
{
  /** all subviews */
  hmlp::View<T> AT, A0, CT, C0, 
                AB, A1, CB, C1,
                    A2,     C2;



}; /** end xgemm_var3() */




}; // end namespace hmlp

#endif // define HMLP_BLAS_LAPACK_H
