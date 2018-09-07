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



#ifndef RANK_K_HPP
#define RANK_K_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>

#include <hmlp.h>
#include <hmlp_internal.hpp>
#include <hmlp_util.hpp>

/** Use Thread Control Interface (TCI). */
#include <hmlp_tci.hpp>
/** for USE_STRASSEN */
//#include <primitives/strassen.hpp>

/** reference microkernels */
#include <packing.hpp>
#include <semiring_mrxnr.hpp>
#include <fused_mrxnr.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{
/**
 *  @brief Macro kernel contains the 3rd and 2nd loops. Depending on the
 *         configuration of the communicator, the 3rd loop may be parallelized.
 *         b_next is the prefetch pointer.
 */ 
template<int KC, typename SEMIRINGKERNEL, typename TA, typename TB, typename TV>
void rank_k_macro_kernel
(
  tci::Comm &Comm3rd,
  int ic, int jc, int pc,
  int  m, int  n, int  k,
  TA *packA,
  TB *packB,
  TV *V, int rs_v, int cs_v,
  SEMIRINGKERNEL semiringkernel
)
{
  /** Get all block sizes */
  const static int MR         = SEMIRINGKERNEL::mr;
  const static int NR         = SEMIRINGKERNEL::nr;
  const static int PACK_MR    = SEMIRINGKERNEL::pack_mr;
  const static int PACK_NR    = SEMIRINGKERNEL::pack_nr;
  /** Create subcommunicators for each loop. */
  auto Comm2nd = Comm3rd.Split( hmlp_read_nway_from_env( "KS_JR_NT" ) );
  /** Compute loop ranges for each thread */
  auto Loop3rd = Comm3rd.DistributeOver1DGangs(        0, n,      NR );
  auto Pack3rd = Comm3rd.DistributeOver1DGangs(        0, n, PACK_NR );
  auto Loop2nd = Comm2nd.DistributeOver1DThreads(      0, m,      MR );
  auto Pack2nd = Comm2nd.DistributeOver1DThreads(      0, m, PACK_MR );
  /** Distribute range [0,n) over Comm3rd (jr loop). */
  for ( int j  = Loop3rd.beg(), jp  = Pack3rd.beg();
            j  < Loop3rd.end();
            j += Loop3rd.inc(), jp += Pack3rd.inc() )
  {
    struct aux_s<TA, TB, TV, TV> aux;
    aux.pc       = pc;
    aux.b_next   = packB;
    aux.do_packC = 0;
    aux.jb       = std::min( n - j, NR );
    /** Distribute range [0,m) over Comm2nd (ir loop). */
    for ( int i  = Loop2nd.beg(), ip  = Pack2nd.beg();
              i  < Loop2nd.end();
              i += Loop2nd.inc(), ip += Pack2nd.inc() )
    {
      aux.ib = std::min( m - i, MR );
      /** Increase the b_next pointer. */
      if ( i + MR >= m ) aux.b_next += Pack3rd.inc() * k;
           
      if ( aux.jb == NR && aux.ib == MR )                 
      {
        semiringkernel( k, &packA[ ip * k ], &packB[ jp * k ],
          &V[ i * rs_v + j * cs_v ], rs_v, cs_v, &aux );
      }
      else
      {
        TV vtmp[ MR * NR ];

        if ( pc ) // initilize ctmp
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
            for ( auto ii = 0; ii < aux.ib; ii ++ )
              vtmp[ jj * MR + ii ] = 
                V[ ( j + jj ) * cs_v + ( i + ii ) * rs_v ];
        }

        semiringkernel( k, &packA[ ip * k ], &packB[ jp * k ],
          vtmp, 1, MR, &aux );

        for ( auto jj = 0; jj < aux.jb; jj ++ )
          for ( auto ii = 0; ii < aux.ib; ii ++ )
            V[ ( j + jj ) * cs_v + ( i + ii ) * rs_v ] 
              = vtmp[ jj * MR + ii ];
      }
    } /** end 2nd loop */
  } /** end 3rd loop */
}; /** end rank_k_macro_kernel() */



/**
 *  @breif This function contains the loop body of the 6th to 4th loops,
 *         including all packing and unpacking routines. Notice that this
 *         function is executed by all threads in the root communicator.
 *         To access each thread in different level of communicators, use
 *         their ids.
 */ 
template<
  int MC, int NC, int KC,
  typename TPACKA, typename TPACKB, typename TV,
  typename     TA, typename     TB, typename TC,
  typename SEMIRINGKERNEL>
void rank_k_internal
(
  tci::Comm &Comm6th,
  int batchId, int m, int n, int k, int k_stra,
  TA& A, 
  TB& B, 
  TV* V, int rs_v, int cs_v,
  SEMIRINGKERNEL semiringkernel
)
{
  /** Get all block sizes. */
  const static int MR         = SEMIRINGKERNEL::mr;
  const static int NR         = SEMIRINGKERNEL::nr;
  const static int PACK_MR    = SEMIRINGKERNEL::pack_mr;
  const static int PACK_NR    = SEMIRINGKERNEL::pack_nr;
  const static int ALIGN_SIZE = SEMIRINGKERNEL::align_size;
  const static int PACK_MC    = ( MC / MR ) * PACK_MR;
  const static int PACK_NC    = ( NC / NR ) * PACK_NR;
  /** Create subcommunicators for each loop. */
  auto Comm5th = Comm6th.Split( hmlp_read_nway_from_env( "KS_JC_NT" ) );
  auto Comm4th = Comm5th.Split( 1 );
  auto Comm3th = Comm4th.Split( hmlp_read_nway_from_env( "KS_IC_NT" ) );
  /** Adjuest nc and pack_nc if the 6th loop is parallelized. */
  int nc = Comm6th.BalanceOver1DGangs( n, NC, NR );
  int pack_nc = ( nc / NR ) * PACK_NR;
  /** Allocate packB (shared over Comm4th, private for each Comm5th gang). */
  auto *packB = Comm4th.AllocateSharedMemory<ALIGN_SIZE, TPACKB>( KC * ( pack_nc + 1 ) );
  /** Allocate packA (shared over Comm3th, private for each Comm4th gang). */
  auto *packA = Comm3th.AllocateSharedMemory<ALIGN_SIZE, TPACKA>( KC * ( PACK_MC + 1 ) );
  /** Distribute range [0,n) over Comm6th. */
  auto Loop6th = Comm6th.DistributeOver1DGangs(      0, n, nc );
  /** Distribute range [k_stra,k) over Comm5th. */
  auto Loop5th = Comm5th.DistributeOver1DGangs( k_stra, k, KC );
  /** Distribute range [0,m) over Comm4th. */
  auto Loop4th = Comm4th.DistributeOver1DGangs(      0, m, MC );
  /** Distribute range [0,n) over Comm6th. */
  for ( int jc  = Loop6th.beg(); 
            jc  < Loop6th.end(); 
            jc += Loop6th.inc() )
  {
    auto jb = std::min( n - jc, nc );
    /** Distribute range [k_stra,k) over Comm5th. */
    for ( int pc  = Loop5th.beg(); 
              pc  < Loop5th.end(); 
              pc += Loop5th.inc() )
    {
      auto pb = std::min( k - pc, KC );
      /** Distribute range [0,jb) over Comm4th. */
      auto LooppkB = Comm4th.DistributeOver1DThreads( 0, jb,      NR );
      auto PackpkB = Comm4th.DistributeOver1DThreads( 0, jb, PACK_NR );
      /** PackB and typecast from TB to TPACKB.  */
      for ( int j  = LooppkB.beg(), jp  = PackpkB.beg();
                j  < LooppkB.end();
                j += LooppkB.inc(), jp += PackpkB.inc() )
      {
        B.Pack( k, pc, pb, n, jc + j, std::min( jb - j, NR ), 
            &packB[ jp * pb ] );
      }
      /** Synchronize all threads in Comm4th. */
      Comm4th.Barrier();
      /** Distribute range [0,m) over Comm4th. */
      for ( int ic  = Loop4th.beg();
                ic  < Loop4th.end();
                ic += Loop4th.inc() )
      {
        auto ib = std::min( m - ic, MC );
        /** Distribute range [0,ib) over Comm3th. */
        auto LooppkA = Comm3th.DistributeOver1DThreads( 0, ib, MR );
        auto PackpkA = Comm3th.DistributeOver1DThreads( 0, ib, PACK_MR );
        /** packA and typecast from TA to TPACKA. */
        for ( int i  = LooppkA.beg(), ip  = PackpkA.beg();
                  i  < LooppkA.end();
                  i += LooppkA.inc(), ip += PackpkA.inc() )
        {
          A.Pack( m, ic + i, std::min( ib - i, MR ), 
              k, pc, pb, &packA[ ip * pb ] );
        }
        /** Synchronize all threads in Comm3th. */
        Comm3th.Barrier();
        /** Otherwise, invoke the semiubg rank-k kernel. */
        rank_k_macro_kernel<KC>( Comm3th, 
          ic, jc, pc, ib, jb, pb, packA, packB,
          V + ic * rs_v + jc * cs_v, rs_v, cs_v,
          semiringkernel );
        /** Synchronize all threads in Comm3th. */
        Comm3th.Barrier();
      } /** end 4th loop */
      Comm4th.Barrier();
    } /** end 5th loop */
    Comm5th.Barrier();
  } /** end 6th loop */
  Comm6th.Barrier();
  /** Free packing buffer. */
  Comm3th.FreeSharedMemory( packA );
  Comm4th.FreeSharedMemory( packB );
}; /** end nbody_internal() */





/**
 *  @breif This is the main routine of gkmx. All packing buffers are
 *         managed here. The communicator and the parallel section 
 *         start here.
 *
 */ 
template<
  int MC, int NC, int KC,
  typename TPACKA, typename TPACKB, typename TV,
  typename     TA, typename     TB, typename TC,
  typename SEMIRINGKERNEL>
void rank_k
(
  int batchId, int m, int n, int k,
  TA& A, 
  TB& B, 
  TC& C,
  SEMIRINGKERNEL semiringkernel
)
{
  const static int MR         = SEMIRINGKERNEL::mr;
  const static int NR         = SEMIRINGKERNEL::nr;
  const static int PACK_MR    = SEMIRINGKERNEL::pack_mr;
  const static int PACK_NR    = SEMIRINGKERNEL::pack_nr;
  const static int ALIGN_SIZE = SEMIRINGKERNEL::align_size;
  const static int PACK_MC    = ( MC / MR ) * PACK_MR;
  const static int PACK_NC    = ( NC / NR ) * PACK_NR;
  const static bool USE_STRASSEN = false; 

  /** Early return if possible. */
  if ( m == 0 || n == 0 || k == 0 ) return;
  /** Type C must be MatrixLike.  */
  if ( !is_same<TC, MatrixLike<PACK_MR, TV, TV>>::value )
  {
    exit( 1 );
  }
  /** Now get the pointer, row and column stride. */
  auto *V = reinterpret_cast<TV*>( C.X );
  auto rs_v = C.rs;
  auto cs_v = C.cs;


  int k_stra = 0; 
  if ( USE_STRASSEN )
  {
    assert( typeid(TPACKA) == typeid(TPACKB) );
    assert( typeid(TC) == typeid(TV) );
    k_stra = k - k % KC;

    if ( k_stra == k ) k_stra -= KC;
  }

  tci::Parallelize( NULL, rank_k_internal<MC, NC, KC, TPACKA, TPACKB, TV, 
      TA, TB, TC, SEMIRINGKERNEL>, 
      batchId, m, n, k, k_stra, A, B, C, V, rs_v, cs_v, 
      semiringkernel );
}; /** end rank_k() */

}; /** end namespace hmlp */

#endif /** define RANK_K_HPP */
