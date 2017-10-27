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


/** GSKNN templates */
#include <primitives/gsknn.hpp>

/** Sandy-bridge micro-kernels */
#include <rank_k_d8x4.hpp>
#include <knn_d8x4.hpp>

using namespace hmlp::gsknn;

void gsknn
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
)
{
  const bool USE_STRASSEN = false;

  rank_k_asm_d8x4 semiringkernel;
  knn_int_d8x4    fusedkernel;
  gsknn<
    104, 2048, 256, 8, 4, 
    104, 2048,      8, 4, 32,
    USE_STRASSEN,
    rank_k_asm_d8x4,
    knn_int_d8x4,
    double, double, double, double>
  (
    m, n, k, r,
    A, A2, amap,
    B, B2, bmap,
    D,     I,
    semiringkernel, fusedkernel
  );
};

void gsknn
(
  int m, int n, int k, int r,
  float *A, float *A2, int *amap,
  float *B, float *B2, int *bmap,
  float *D,            int *I
)
{
  printf( "not implemented yet\n" );
  exit( 1 );
};

void dgsknn
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
)
{
  gsknn( m, n, k, r, A, A2, amap, B, B2, bmap, D, I );
};

void sgsknn
(
  int m, int n, int k, int r,
  float *A, float *A2, int *amap,
  float *B, float *B2, int *bmap,
  float *D,            int *I
)
{
  gsknn( m, n, k, r, A, A2, amap, B, B2, bmap, D, I );
};



void dgsknn_ref
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
)
{
  gsknn_ref<double>
  (
    m, n, k, r,
    A, A2, amap,
    B, B2, bmap,
    D,     I
  );
}
