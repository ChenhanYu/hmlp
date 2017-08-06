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




/** CNN templates */
#include <primitives/cnn.hpp>

/** Sandy-bridge micro-kernel */
#include <rank_k_d8x4.hpp>

using namespace hmlp::cnn;


void dcnn(
  hmlpOperation_t transA, hmlpOperation_t transB,
	int m, int n, int k,
	double *A, int lda,
	double *B, int ldb,
	double *C, int ldc
	)
{
  // Sandy-bridge
	rank_k_int_d8x4 semiringkernel;
  rank_k_int_d8x4 microkernel;
  cnn
  <104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
  false,
	rank_k_int_d8x4, rank_k_int_d8x4,
	double, double, double, double>
	(
    transA, transB,
	  m, n, k,
	  A, lda,
	  B, ldb,
	  C, ldc,
	  semiringkernel,
	  microkernel
	);
}
