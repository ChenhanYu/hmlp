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

/** Skylake kernels */
//#include <rank_k_d6x32.hpp>
//#include <gsks_d6x32.hpp>
#include <rank_k_d12x16.hpp>
#include <gsks_d12x16.hpp>


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
    case KS_GAUSSIAN:
    {
      //rank_k_opt_d6x32 semiringkernel;
      //gsks_gaussian_int_d6x32 fusedkernel;
      rank_k_opt_d12x16 semiringkernel;
      gsks_gaussian_int_d12x16 fusedkernel;

      //const size_t MR = rank_k_opt_d6x32::mr; 
      //const size_t NR = rank_k_opt_d6x32::nr; 
      //const size_t PACK_MR = rank_k_opt_d6x32::pack_mr; 
      //const size_t PACK_NR = rank_k_opt_d6x32::pack_nr; 
      const size_t MR = rank_k_opt_d12x16::mr; 
      const size_t NR = rank_k_opt_d12x16::nr; 
      const size_t PACK_MR = rank_k_opt_d12x16::pack_mr; 
      const size_t PACK_NR = rank_k_opt_d12x16::pack_nr; 
      const size_t MC = 480;
      const size_t NC = 3072;
      const size_t KC = 384;
      //const size_t ALIGN_SIZE = rank_k_opt_d6x32::align_size;
      const size_t ALIGN_SIZE = rank_k_opt_d12x16::align_size;

      gsks<MC, NC, KC, MR, NR, MC, NC, PACK_MR, PACK_NR, ALIGN_SIZE,
        true,  /** USE_L2NORM */
        false, /** USE_VAR_BANDWIDTH */
        false, /** USE_STRASSEN */
        //rank_k_opt_d6x32,
        //gsks_gaussian_int_d6x32,
        rank_k_opt_d12x16,
        gsks_gaussian_int_d12x16,
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
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      break;
    case KS_POLYNOMIAL:
      break;
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
