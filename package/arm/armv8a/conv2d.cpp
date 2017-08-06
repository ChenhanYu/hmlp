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




// GKMX templates
#include <conv2d.hpp>

// Armv8a
#include <rank_k_d6x8.hpp>
#include <conv_relu_pool2x2_6x8.hpp>

using namespace hmlp::cnn;

void sconv2d
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  float *B,
  int w1, int h1, int d1,
	float *A,
	float *C
)
{
  rank_k_asm_s8x12 semiringkernel;
  rank_k_asm_s8x12 microkernel;

  conv2d
  //<120, 3072, 640, 8, 12, 120, 3072, 8, 12, 16,
  <120, 768, 640, 8, 12, 120, 768, 8, 12, 16,
  false,
  rank_k_asm_s8x12, rank_k_asm_s8x12,
  float, float, float, float>
	(
    w0, h0, d0, s, p, batchSize,
    B,
    w1, h1, d1,
    A,
    C,
	  semiringkernel,
	  microkernel
	);
};

void dconv2d
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  double *B,
  int w1, int h1, int d1,
	double *A,
	double *C
)
{
  rank_k_asm_d6x8 semiringkernel;
  rank_k_asm_d6x8 microkernel;
  //conv_relu_pool2x2_asm_d6x8 microkernel;

  conv2d
  //<120, 3072, 240, 6, 8, 120, 3072, 6, 8, 16,
  <120, 768, 240, 6, 8, 120, 768, 6, 8, 16,
  false,
  rank_k_asm_d6x8, rank_k_asm_d6x8,
  //rank_k_asm_d6x8, conv_relu_pool2x2_asm_d6x8,
  double, double, double, double>
	(
    w0, h0, d0, s, p, batchSize, 
    B,
    w1, h1, d1,
    A,
    C,
	  semiringkernel,
	  microkernel
	);
};

void sconv2d_ref
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  float *B,
  int w1, int h1, int d1,
	float *A,
	float *C
)
{
  conv2d_ref
  (
    w0, h0, d0, s, p, batchSize, 
    B,
    w1, h1, d1,
    A,
    C
 );
};

void dconv2d_ref
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  double *B,
  int w1, int h1, int d1,
	double *A,
	double *C
)
{
  conv2d_ref
  (
    w0, h0, d0, s, p, batchSize, 
    B,
    w1, h1, d1,
    A,
    C
 );
};
