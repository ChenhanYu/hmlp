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

/** Haswell micro-kernels */
#include <rank_k_d8x6.hpp>
#include <rank_k_d6x8.hpp>

using namespace hmlp;

void gnbx
(
	int m, int n, int k,
	float  *A, int lda,
  float  *B, int ldb,
  double *C, int ldc
)
{
  /** microkernel */
  //rank_k_asm_d8x6 semiringkernel;
  //rank_k_asm_d8x6 microkernel;
  rank_k_asm_d6x8 semiringkernel;
  rank_k_asm_d6x8 microkernel;

  //const size_t PACK_MR = rank_k_asm_d8x6::pack_mr; 
  //const size_t PACK_NR = rank_k_asm_d8x6::pack_nr; 
  const size_t PACK_MR = rank_k_asm_d6x8::pack_mr; 
  const size_t PACK_NR = rank_k_asm_d6x8::pack_nr; 

  const size_t MC = 72;
  const size_t NC = 2040;
  const size_t KC = 256;


  /** packA kernel */
  pack2D_pbxib<PACK_MR, float, double> packakernel;
  packakernel.trans = false;
  packakernel.ldx = lda;

  /** packB kernel */
  pack2D_pbxib<PACK_NR, float, double> packbkernel;
  packbkernel.trans = true;
  packbkernel.ldx = ldb;

  /** UnpackC kernel */
  unpack2D_ibxjb<PACK_MR, double, double> unpackckernel;
  unpackckernel.rs_c = 1;
  unpackckernel.cs_c = ldc;



  gnbx::gnbx<
    MC, NC, KC,
    false, true,
    pack2D_pbxib<PACK_MR, float, double>,
    pack2D_pbxib<PACK_NR, float, double>,
    unpack2D_ibxjb<PACK_MR, double, double>,
    //rank_k_asm_d8x6, 
    //rank_k_asm_d8x6,
    rank_k_asm_d6x8, 
    rank_k_asm_d6x8,
    float, double, float, double, double, double>
  (
    m, n, k,
    A, packakernel,
    B, packbkernel,
    C, unpackckernel,
    0, // batchId
    semiringkernel,
    microkernel
  );
};
