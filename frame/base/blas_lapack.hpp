/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  


#ifndef HMLP_BLAS_LAPACK_H
#define HMLP_BLAS_LAPACK_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifdef HMLP_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#endif

namespace hmlp
{

/*
 *  Level-1
 */ 
void xaxpy( const int n, const  float* alpha, const  float* x, const int incx,  float* y, const int incy );
void xaxpy( const int n, const double* alpha, const double* x, const int incx, double* y, const int incy );



/*
 *  Level-3
 */ 
void xgemm
(
  const char *transA, const char *transB,
  int m, int n, int k, 
  float alpha, const float *A, int lda,
               const float *B, int ldb, 
  float beta,        float *C, int ldc
);

void xgemm
(
  const char *transA, const char *transB,
  int m, int n, int k, 
  double alpha, const double *A, int lda,
                const double *B, int ldb, 
  double beta,        double *C, int ldc
);



void xsyrk
(
  const char *uplo, const char *trans,
  int n, int k, 
  double alpha, double *A, int lda,
  double beta,  double *C, int ldc
);

void xsyrk
(
  const char *uplo, const char *trans,
  int n, int k, 
  float alpha, float *A, int lda,
  float beta,  float *C, int ldc
);


void xtrsm
( 
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  float alpha, float *A, int lda,
               float *B, int ldb 
);

void xtrsm
( 
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  double alpha, double *A, int lda,
                double *B, int ldb 
);

void xtrmm
(
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  float alpha, float *A, int lda,
               float *B, int ldb 
);

void xtrmm
(
  const char *side, const char *uplo,
  const char *transA, const char *diag,
  int m, int n,
  double alpha, double *A, int lda,
                double *B, int ldb 
);

void xlaswp
( 
  int n, double *A, int lda, 
  int k1, int k2, int *ipiv, int incx 
);

void xlaswp
( 
  int n, float *A, int lda, 
  int k1, int k2, int *ipiv, int incx 
);

/** Cholesky family */
void xpotrf( const char *uplo, int n, double *A, int lda );
void xpotrf( const char *uplo, int n, float  *A, int lda );
void xpotrs( const char *uplo, int n, int nrhs, double *A, int lda, double *B, int ldb );
void xpotrs( const char *uplo, int n, int nrhs, float  *A, int lda, float  *B, int ldb );
void xposv( const char *uplo, int n, int nrhs, double *A, int lda, double *B, int ldb );
void xposv( const char *uplo, int n, int nrhs, float  *A, int lda, float  *B, int ldb );



/** LU family */
void xgetrf( int m, int n, double *A, int lda, int *ipiv );
void xgetrf( int m, int n, float  *A, int lda, int *ipiv );
void xgetrs( const char *trans, int m, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb );
void xgetrs( const char *trans, int m, int nrhs, float  *A, int lda, int *ipiv, float  *B, int ldb );

/** QR family */
void xgeqrf( int m, int n, double *A, int lda, double *tau, double *work, int lwork );
void xgeqrf( int m, int n, float  *A, int lda, float  *tau, float  *work, int lwork );
void xorgqr( int m, int n, int k, double *A, int lda, double *tau, double *work, int lwork );
void xorgqr( int m, int n, int k, float  *A, int lda, float  *tau, float  *work, int lwork );
void xormqr( const char *side, const char *trans,
  int m, int n, int k, float  *A, int lda, float  *tau, float  *C, int ldc, float  *work, int lwork );
void xormqr( const char *side, const char *trans,
  int m, int n, int k, double *A, int lda, double *tau, double *C, int ldc, double *work, int lwork );
void xgeqp3( int m, int n, float  *A, int lda, int *jpvt, float  *tau, float  *work, int lwork );
void xgeqp3( int m, int n, double *A, int lda, int *jpvt, double *tau, double *work, int lwork );

void xgeqp4( int m, int n, float  *A, int lda, int *jpvt, float  *tau, float  *work, int lwork );
void xgeqp4( int m, int n, double *A, int lda, int *jpvt, double *tau, double *work, int lwork );
void xgels( const char *trans, int m, int n, int nrhs, float  *A, int lda, float  *B, int ldb, float  *work, int lwork );
void xgels( const char *trans, int m, int n, int nrhs, double *A, int lda, double *B, int ldb, double *work, int lwork );


void xgecon( const char *norm, int n, float  *A, int lda, float  anorm, float  *rcond, float  *work, int *iwork );
void xgecon( const char *norm, int n, double *A, int lda, double anorm, double *rcond, double *work, int *iwork );

void xstev( const char *jobz, int n, double *D, double *E, double *Z, int ldz, double *work );
void xstev( const char *jobz, int n, float  *D, float  *E, float  *Z, int ldz, float  *work );


double xdot( int n, const double *dx, int incx, const double *dy, int incy );
float  xdot( int n, const  float *dx, int incx, const  float *dy, int incy );

double xnrm2( int n, double *x, int incx );
float  xnrm2( int n,  float *x, int incx );



#ifdef HMLP_USE_CUDA

/** cublasSgemm wrapper */
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

/** cublasDgemm wrapper */
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

/** cublasSgemmBatched wrapper */
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

/** cublasDgemmBatched wrapper */
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

#endif /** end ifdef HMLP_USE_CUDA */

}; /** end namespace hmlp */

#endif /** define HMLP_BLAS_LAPACK_H */
