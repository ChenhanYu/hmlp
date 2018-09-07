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




/** GSKS templates */
#include <primitives/gsks.hpp>

/** reference kernels */
#include <gsks_ref_mrxnr.hpp>

/** Haswell kernels */
#include <rank_k_d8x6.hpp>
#include <gsks_d8x6.hpp>


using namespace hmlp;


void gsks
(
  kernel_s<float> *kernel,
  int m, int n, int k,
  float *u,            int *umap,
  float *A, float *A2, int *amap,
  float *B, float *B2, int *bmap,
  float *w,            int *wmap
)
{
  printf( "gsks() in single precision is not implemented yet\n" );
  exit( 1 );
};



void gsks
(
  kernel_s<double> *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
)
{
  switch ( kernel->type )
  {
    case KS_GAUSSIAN:
    {
      /** kernel instances */
      rank_k_asm_d8x6 semiringkernel;
      gsks_gaussian_int_d8x6 microkernel;

      gsks::gsks<
        /** MC, NC, KC, MR, NR */
        72, 960, 256, 8, 6, 
        /** PACK_MC, PACK_NC, PACK_MR, PACK_NR, SIMD_SIZE */
        72, 960,      8, 6, 32,
        /** USE_L2NORM, USE_VAR_BANDWIDTH, USE_STRASSEN */
        true, false, false,
        rank_k_asm_d8x6, gsks_gaussian_int_d8x6,
        double, double, double, double>
      ( 
        kernel,
        m, n, k,
        u,     umap,
        A, A2, amap,
        B, B2, bmap,
        w,     wmap,
        semiringkernel, microkernel 
      );
      break;
    }
    case KS_GAUSSIAN_VAR_BANDWIDTH:
    {
      break;
    }
    case KS_POLYNOMIAL:
    {
      rank_k_asm_d8x6 semiringkernel;
      gsks_polynomial_int_d8x6 microkernel;

      gsks::gsks<
        72, 960, 256, 8, 6, 
        72, 960,      8, 6, 32,
        true, false, false,
        rank_k_asm_d8x6, gsks_polynomial_int_d8x6,
        double, double, double, double>
      ( 
        kernel,
        m, n, k,
        u,     umap,
        A, A2, amap,
        B, B2, bmap,
        w,     wmap,
        semiringkernel, microkernel 
      );
      break;
    }
    case KS_LAPLACE:
      break;
    case KS_TANH:
      break;
    case KS_QUARTIC:
      break;
    case KS_MULTIQUADRATIC:
      break;
    case KS_EPANECHNIKOV:
      break;
    default:
      exit( 1 );
  }
};







void dgsks_ref
(
  kernel_s<double> *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
)
{
  gsks::gsks_ref<double>
  (
    kernel,
    m, n, k,
    u,     umap,
    A, A2, amap,
    B, B2, bmap,
    w,     wmap
  );
}
