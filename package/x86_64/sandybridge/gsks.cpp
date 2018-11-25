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

/** sandy-bridge kernels */
#include <rank_k_d8x4.hpp>
#include <gsks_d8x4.hpp>
#include <variable_bandwidth_gaussian_d8x4.hpp>

// Knights Landing
// #include <rank_k_int_d24x8.hpp>
// #include <gaussian_int_d24x8.hpp>

using namespace hmlp::gsks;



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
    case GAUSSIAN:
    {
      rank_k_asm_d8x4 semiringkernel;
      gsks_gaussian_int_d8x4 fusedkernel;

      gsks<
        104, 
        4096, 
        256, 
        8, 
        4, 
        104, 
        4096, 
        8, 
        4, 
        32,
        true,  /** USE_L2NORM */
        false, /** USE_VAR_BANDWIDTH */
        false, /** USE_STRASSEN */
        rank_k_asm_d8x4,
        gsks_gaussian_int_d8x4,
        double, double, double, double>
          ( 
           kernel,
           m, n, k,
           u,     umap,
           A, A2, amap,
           B, B2, bmap,
           w,     wmap,
           semiringkernel, fusedkernel 
          );

      // Knights Landing
      // rank_k_int_d24x8 semiringkernel;
      // gaussian_int_d24x8 microkernel;
      // gsks<
      //   120, 14400, 336, 24, 8, 120, 14400, 24, 8, 64,
      //   true, false, false,
      //   rank_k_int_d24x8, gaussian_int_d24x8,
      //   double, double, double, double>
      // ( 
      //   kernel,
      //   m, n, k,
      //   u,     umap,
      //   A, A2, amap,
      //   B, B2, bmap,
      //   w,     wmap,
      //   semiringkernel, microkernel 
      // );

      break;
    }
    case GAUSSIAN_VAR_BANDWIDTH:
      {
        rank_k_asm_d8x4 semiringkernel;
        variable_bandwidth_gaussian_int_d8x4 fusedkernel;

        gsks<
          104, 
          4096, 
          256, 
          8, 
          4, 
          104, 
          4096, 
          8, 
          4, 
          32,
          true,  // USE_L2NORM
          true,  // USE_VAR_BANDWIDTH
          false, // USE_STRASSEN
          rank_k_asm_d8x4, variable_bandwidth_gaussian_int_d8x4,
          double, double, double, double>
        ( 
          kernel,
          m, n, k,
          u,     umap,
          A, A2, amap,
          B, B2, bmap,
          w,     wmap,
          semiringkernel, fusedkernel 
        );

        break;
      }
    case POLYNOMIAL:
      {
        rank_k_asm_d8x4 semiringkernel;


        break;
      }
    case LAPLACE:
      break;
    case TANH:
      break;
    case QUARTIC:
      break;
    case MULTIQUADRATIC:
      break;
    case EPANECHNIKOV:
      break;
    default:
      exit( 1 );
  }
};


void sgsks
(
  kernel_s<float> *kernel,
  int m, int n, int k,
  float *u,            int *umap,
  float *A, float *A2, int *amap,
  float *B, float *B2, int *bmap,
  float *w,            int *wmap
)
{
  gsks( kernel, m, n, k,
      u,     umap,
      A, A2, amap,
      B, B2, bmap,
      w,     wmap );
};

void dgsks
(
  kernel_s<double> *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
)
{
  gsks( kernel, m, n, k,
      u,     umap,
      A, A2, amap,
      B, B2, bmap,
      w,     wmap );
};










void dgsks_ref
(
  //ks_t *kernel,
  kernel_s<double> *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
)
{
  gsks_ref<double>
  (
    kernel,
    m, n, k,
    u,     umap,
    A, A2, amap,
    B, B2, bmap,
    w,     wmap
  );
}
