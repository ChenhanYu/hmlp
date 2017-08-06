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



/** CONV2D templates */
#include <primitives/conv2d.hpp>

/** Sandy-bridge */
#include <rank_k_d8x4.hpp>

using namespace hmlp::cnn;

void dconv2d
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  double *B,
  int w1, int h1, int d1,
	double *A,
	double *C
)
{
  // Sandy-bridge
  rank_k_asm_d8x4 semiringkernel;
  rank_k_asm_d8x4 microkernel;

  conv2d
  <104, 1024, 256, 8, 4, 104, 1024, 8, 4, 32,
  false,
	rank_k_asm_d8x4, rank_k_asm_d8x4,
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
