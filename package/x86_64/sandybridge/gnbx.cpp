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


/** GNBX templates */
#include <primitives/gnbx.hpp>

/** Sandy-bridge micro-kernels */
#include <rank_k_d8x4.hpp>

using namespace hmlp::gnbx;

void gnbx
(
	int m, int n, int k,
	float  *A, int lda,
  float  *B, int ldb,
  double *C, int ldc
)
{
  /** packA kernel */
  hmlp::pack2D_pbxib<8, float, double> packakernel;
  packakernel.trans = false;
  packakernel.ldx = lda;

  /** packB kernel */
  hmlp::pack2D_pbxib<4, float, double> packbkernel;
  packbkernel.trans = true;
  packbkernel.ldx = ldb;


  /** microkernel */
  rank_k_asm_d8x4 semiringkernel;
  rank_k_asm_d8x4 microkernel;

  gnbx<
    104, 4096, 256, 8, 4, 
    104, 4096,      8, 4, 32,
    false, true,
    hmlp::pack2D_pbxib<8, float, double>,
    hmlp::pack2D_pbxib<4, float, double>,
    rank_k_asm_d8x4, 
    rank_k_asm_d8x4,
    float, double, float, double, double, double>
  (
    m, n, k,
    A, packakernel,
    B, packbkernel,
    C, ldc,
    0, // batchId
    semiringkernel,
    microkernel
  );
};
