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


#ifndef HMLP_H
#define HMLP_H


/** HMLP runtime API */
void hmlp_init();
void hmlp_set_num_workers( int n_worker );
void hmlp_run();
void hmlp_finalize();


typedef enum
{
  HMLP_OP_N,
  HMLP_OP_T
} hmlpOperation_t;


void gkmx_sfma
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  float *A, int lda,
  float *B, int ldb,
  float *C, int ldc
);


void gkmx_dfma
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
);


void gkmx_dfma_simple
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
);


void gkmx_dconv_relu_pool
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
);

void dstrassen(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
);

void sconv2d
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  float *B,
  int w1, int h1, int d1,
	float *A,
	float *C
);

void dconv2d
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  double *B,
  int w1, int h1, int d1,
	double *A,
	double *C
);

void sconv2d_ref
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  float *B,
  int w1, int h1, int d1,
	float *A,
	float *C
);

void dconv2d_ref
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  double *B,
  int w1, int h1, int d1,
	double *A,
	double *C
);






typedef enum
{
  KS_GAUSSIAN,
  KS_POLYNOMIAL,
  KS_LAPLACE,
  KS_GAUSSIAN_VAR_BANDWIDTH,
  KS_TANH,
  KS_QUARTIC,
  KS_MULTIQUADRATIC,
  KS_EPANECHNIKOV
} ks_type;

template<typename T>
struct kernel_s
{
  ks_type type;
  T powe;
  T scal;
  T cons;
  T *hi;
  T *hj;
  T *h;
};

//typedef struct kernel_s ks_t;

void gsks
(
  kernel_s<double> *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
);

void gsks
(
  kernel_s<float> *kernel,
  int m, int n, int k,
  float *u,            int *umap,
  float *A, float *A2, int *amap,
  float *B, float *B2, int *bmap,
  float *w,            int *wmap
);

void dgsks
(
  kernel_s<double> *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
);

void sgsks
(
  kernel_s<float> *kernel,
  int m, int n, int k,
  float *u,            int *umap,
  float *A, float *A2, int *amap,
  float *B, float *B2, int *bmap,
  float *w,            int *wmap
);

void dgsks_ref
(
  kernel_s<double> *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
);

void dgsknn
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
);

void gsknn
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
);

void dgsknn_ref
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
);

#ifdef HMLP_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/pair.h>


void dsq2nrm
(
  hmlpOperation_t transX, 
  int d, int n, 
  double* X2array[], const double* Xarray[], double* X, int ldx, 
  int batchSize 
);

void gkmm_dfma
(
  cudaStream_t stream,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  const double *Aarray[], int lda,
  const double *Barray[], int ldb,
        double *Carray[], int ldc,
  int batchSize
);

void gkmm_dfma
(
  cudaStream_t stream,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  const double *Aarray, int lda, int loa,
  const double *Barray, int ldb, int lob,
        double *Carray, int ldc, int loc,
  int batchSize
);

void gkmm_mixfma
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  const double *Aarray, int lda, int loa, 
  const double *Barray, int ldb, int lob,
        float  *Carray, int ldc, int loc,
  int batchSize
);

void gkrm_dkmeans
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  double *Aarray[], double *A2array[], int lda,
  double *Barray[], double *B2array[], int ldb,
  thrust::pair<double,int>  *Carray[], int ldc, 
  int batchSize
);

void dkmeans
(
  cudaStream_t stream, 
  int m, int n, int k,
  double *Aarray[], double *A2array[], int lda,
  double *Barray[], double *B2array[], int ldb,
  thrust::pair<double,int>  *Carray[], int ldc, 
  int batchSize
);

void dstrassen
(
  cudaStream_t stream,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  const double *Aarray[], int lda,
  const double *Barray[], int ldb,
        double *Carray[], int ldc,
  int batchSize
);

void dstrassen
(
  cudaStream_t stream,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  const double *Aarray, int lda, int loa,
  const double *Barray, int ldb, int lob,
        double *Carray, int ldc, int loc,
  int batchSize
);
#endif // end ifdef HMLP_USE_CUDA

#endif // define HMLP_H
