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
#include <primitives/nbody.hpp>

/** Skylake micro-kernels */
#include <rank_k_d6x32.hpp>

using namespace hmlp;

template<typename T>
struct identity 
{
  inline T operator()( const T& x, int i, int j, int b ) const 
  {
    return x; 
  }
  T** A2;
  T** B2;
};

void nbody
(
	int m, int n, int k,
	float  *A, int lda,
  float  *B, int ldb,
  float  *C, int ldc
)
{
  using TINPUT = float;
  using TINNER = double;

  /** microkernel */
  rank_k_opt_d6x32 semiringkernel;
  rank_k_opt_d6x32 microkernel;

  const size_t PACK_MR = rank_k_opt_d6x32::pack_mr; 
  const size_t PACK_NR = rank_k_opt_d6x32::pack_nr; 
  const size_t MC = 192;
  const size_t NC = 5760;
  const size_t KC = 336;

  /** ObjA */
  MatrixLike<PACK_MR, TINPUT, TINNER> ObjA;
  ObjA.Set( A, m, k, 1, lda, false );

  /** ObjB */
  MatrixLike<PACK_NR, TINPUT, TINNER> ObjB;
  ObjB.Set( B, k, n, 1, ldb, true );

  /** ObjC */
  //MatrixLike<PACK_MR, double, double> ObjC;
  MatrixLike<PACK_MR, TINPUT, TINNER> ObjC;
  ObjC.Set( C, m, n, 1, ldc, false );

  /** General N-body operator (these 6 types are essential) */
  nbody::nbody<MC, NC, KC, TINNER, TINNER, TINNER>
  (
    0, m, n, k,
    ObjA, 
    ObjB, 
    ObjC,
    semiringkernel,
    microkernel
  );

}; /** end gnbx() */

  
void nbody
(
	int m, int n, int k,
	double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
)
{
  ///** microkernel */
  //rank_k_opt_s12x32 semiringkernel;
  //rank_k_opt_s12x32 microkernel;

  //const size_t PACK_MR = rank_k_opt_s12x32::pack_mr; 
  //const size_t PACK_NR = rank_k_opt_s12x32::pack_nr; 
  //const size_t MC = 480;
  //const size_t NC = 3072;
  //const size_t KC = 384;


  using TINPUT = double;
  using TINNER = double;

  /** microkernel */
  rank_k_opt_d6x32 semiringkernel;
  rank_k_opt_d6x32 microkernel;

  const size_t PACK_MR = rank_k_opt_d6x32::pack_mr; 
  const size_t PACK_NR = rank_k_opt_d6x32::pack_nr; 
  const size_t MC = 128;
  const size_t NC = 2880;
  const size_t KC = 336;

  /** ObjA */
  MatrixLike<PACK_MR, TINPUT, TINNER> ObjA;
  ObjA.Set( A, m, k, 1, lda, false );

  /** ObjB */
  MatrixLike<PACK_NR, TINPUT, TINNER> ObjB;
  ObjB.Set( B, k, n, 1, ldb, true );

  /** ObjC */
  //MatrixLike<PACK_MR, double, double> ObjC;
  MatrixLike<PACK_MR, TINPUT, TINNER> ObjC;
  ObjC.Set( C, m, n, 1, ldc, false );

  /** General N-body operator (these 6 types are essential) */
  nbody::nbody<MC, NC, KC, TINNER, TINNER, TINNER>
  (
    0, m, n, k,
    ObjA, 
    ObjB, 
    ObjC,
    semiringkernel,
    microkernel
  );

}; /** end nbody() */

//void gnbx_simple
//(
//	int m, int n, int k,
//	double *A, int lda,
//  double *B, int ldb,
//  double *C, int ldc
//)
//{
//  std::plus<float> op1;
//  std::multiplies<float> op2;
//  identity<float> opkernel;
//  float initV = 0.0;
//
//  const size_t MR = 8; 
//  const size_t NR = 4; 
//  const size_t MC = 128;
//  const size_t NC = 4096;
//  const size_t KC = 384;
//
//  /** ObjA, stored in double, computed in float */
//  MatrixLike<MR, double, float> ObjA;
//  ObjA.Set( A, m, k, 1, lda, false );
//
//  /** ObjB, stored in double, computed in float */
//  MatrixLike<NR, double, float> ObjB;
//  ObjB.Set( B, k, n, 1, ldb, true );
//
//  /** ObjC, stored in double, computed in float */
//  MatrixLike<MR, double, float> ObjC;
//  ObjC.Set( C, m, n, 1, ldc, false );
//
//  /** General N-body operator (these 6 types are essential) */
//  gnbx::gnbx<MR, NR, MC, NC, KC, float, float, float, float>
//  (
//    0, m, n, k,
//    ObjA, 
//    ObjB, 
//    ObjC,
//    opkernel, op1, op2, initV
//  );
//
//}; /** end gnbx() */
