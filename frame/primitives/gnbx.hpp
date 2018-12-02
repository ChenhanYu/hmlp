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



#ifndef GNBX_HPP
#define GNBX_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>

#include <hmlp.h>
#include <hmlp_internal.hpp>
#include <hmlp_base.hpp>

/** for USE_STRASSEN */
//#include <primitives/strassen.hpp>

/** reference microkernels */
#include <packing.hpp>
#include <semiring_mrxnr.hpp>
#include <fused_mrxnr.hpp>

using namespace std;


namespace hmlp
{
namespace gnbx
{

  /**
 *  @brief Macro kernel contains the 3rd and 2nd loops. Depending on the
 *         configuration of the communicator, the 3rd loop may be parallelized.
 *         b_next is the prefetch pointer.
 */ 
template<int KC, typename SEMIRINGKERNEL, typename TA, typename TB, typename TV>
void rank_k_macro_kernel
(
  Worker &Comm4th,
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

  /** Get ic loop communicator */
  thread_communicator &ic_comm = *Comm4th.comm;

  /** Compute loop ranges for each thread */
  auto Loop3rd = Comm4th.DistributeOver1DGangs(        0, n,      NR );
  auto Pack3rd = Comm4th.DistributeOver1DGangs(        0, n, PACK_NR );
  auto Loop2nd = Comm4th.DistributeOver1DThreads(      0, m,      MR );
  auto Pack2nd = Comm4th.DistributeOver1DThreads(      0, m, PACK_MR );

  /** Loop 3rd (jr loop) */
  for ( int j  = get<0>( Loop3rd ), jp  = get<0>( Pack3rd );
            j  < get<1>( Loop3rd );
            j += get<2>( Loop3rd ), jp += get<2>( Pack3rd ) )
  {
    struct aux_s<TA, TB, TV, TV> aux;
    aux.pc       = pc;
    aux.b_next   = packB;
    aux.do_packC = 0;
    aux.jb       = std::min( n - j, NR );

    /** Loop 2nd (ir loop) */
    for ( int i  = get<0>( Loop2nd ), ip  = get<0>( Pack2nd );
              i  < get<1>( Loop2nd );
              i += get<2>( Loop2nd ), ip += get<2>( Pack2nd ) )
    {
      aux.ib = std::min( m - i, MR );
      if ( i + MR >= m ) 
      {
        aux.b_next += ic_comm.GetNumThreads() * PACK_NR * k;
      }
      
      if ( aux.jb == NR && aux.ib == MR )                 
      {
        semiringkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          &V[ i * rs_v + j * cs_v ], rs_v, cs_v,
          &aux
        );
      }
      else                                                 // corner case
      {
        TV vtmp[ MR * NR ];

        if ( pc ) // initilize ctmp
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
            for ( auto ii = 0; ii < aux.ib; ii ++ )
              vtmp[ jj * MR + ii ] = 
                V[ ( j + jj ) * cs_v + ( i + ii ) * rs_v ];
        }

        semiringkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          vtmp, 1, MR,
          &aux
        );

        for ( auto jj = 0; jj < aux.jb; jj ++ )
          for ( auto ii = 0; ii < aux.ib; ii ++ )
            V[ ( j + jj ) * cs_v + ( i + ii ) * rs_v ] = vtmp[ jj * MR + ii ];
      }
    }                                                      // end 2nd loop
  }                                                        // end 3rd loop
};                                                         // end rank_k_macro_kernel





/**
 *  @brief fused_macro_kernel contains the 3rd, 2nd loops and the fused micro
 *         kernel. Notice that here C has type TC, which is differnet from the
 *         one in rank_k_macro_kernel. ctmp used in the conner case is also
 *         type TC. 
 */ 
template<int KC, typename FUSEDKERNEL, typename TA, typename TB, typename TC, typename TV>
void fused_macro_kernel
(
  Worker &Comm4th,
  int m, int n,
  int ic, int jc, int pc,
  int mc, int nc, int kc,
  TA *packA,
  TB *packB,
  TC *C,
  TV *V, int rs_v, int cs_v,
  int batchId,
  FUSEDKERNEL fusedkernel
)
{
  /** Get all block sizes */
  const static int MR         = FUSEDKERNEL::mr;
  const static int NR         = FUSEDKERNEL::nr;
  const static int PACK_MR    = FUSEDKERNEL::pack_mr;
  const static int PACK_NR    = FUSEDKERNEL::pack_nr;

  /** Get ic loop communicator */
  thread_communicator &ic_comm = *Comm4th.comm;

  /** Compute loop ranges for each thread */
  auto Loop3rd = Comm4th.DistributeOver1DGangs(        0, nc,      NR );
  auto Pack3rd = Comm4th.DistributeOver1DGangs(        0, nc, PACK_NR );
  auto Loop2nd = Comm4th.DistributeOver1DThreads(      0, mc,      MR );
  auto Pack2nd = Comm4th.DistributeOver1DThreads(      0, mc, PACK_MR );

  /** Loop 3rd (jr loop) */
  for ( int j  = get<0>( Loop3rd ), jp  = get<0>( Pack3rd );
            j  < get<1>( Loop3rd );
            j += get<2>( Loop3rd ), jp += get<2>( Pack3rd ) )
  {
    struct aux_s<TA, TB, TC, TV> aux;
    aux.pc       = pc;
    aux.b_next   = packB;
    aux.do_packC = 0;

    /** Loop 2nd (ir loop) */
    for ( int i  = get<0>( Loop2nd ), ip  = get<0>( Pack2nd );
              i  < get<1>( Loop2nd );
              i += get<2>( Loop2nd ), ip += get<2>( Pack2nd ) )
    {
      /**
       *  These auxiluary infos are used to access data in the closure of
       *  opkernel and opreduce.
       */
      aux.m = m;
      aux.n = n;
      aux.i = ic + i;
      aux.j = jc + j;
      aux.b = batchId;

      /**
       *  Encapsulate edge case information.
       */ 
      aux.ib = std::min( mc - i, MR );
      aux.jb = std::min( nc - j, NR );

      /** 
       * Prepare the intermediate semiring rank-k update 
       */
      aux.V = V + i * rs_v + j * cs_v;
      aux.ldv = cs_v;

      if ( i + MR >= mc ) 
      {
        aux.b_next += ic_comm.GetNumThreads() * PACK_NR * kc;
      }

      if ( aux.jb == NR && aux.ib == MR )
      {
        fusedkernel
        (
          kc,
          &packA[ ip * kc ],
          &packB[ jp * kc ],
          C,
          &V[ i * rs_v + j * cs_v ], rs_v, cs_v,
          &aux
        );
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
          aux.V = vtmp;
          aux.ldv = MR;
        }
        fusedkernel
        (
          kc,
          &packA[ ip * kc ],
          &packB[ jp * kc ],
          C,
          vtmp, 1, MR,
          &aux
        );
      }
    }
  }

}; /** end fused_macro_kernel() */




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
  typename SEMIRINGKERNEL, typename MICROKERNEL>
void gnbx_internal
(
  Worker &thread,
  int batchId, int m, int n, int k, int k_stra,
  TA& A, 
  TB& B, 
  TC& C,
  TV* V, int rs_v, int cs_v,
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel
)
{
  /** Get all block sizes */
  const static int MR         = SEMIRINGKERNEL::mr;
  const static int NR         = SEMIRINGKERNEL::nr;
  const static int PACK_MR    = SEMIRINGKERNEL::pack_mr;
  const static int PACK_NR    = SEMIRINGKERNEL::pack_nr;
  const static int ALIGN_SIZE = SEMIRINGKERNEL::align_size;
  const static int PACK_MC    = ( MC / MR ) * PACK_MR;
  const static int PACK_NC    = ( NC / NR ) * PACK_NR;

  /** Create subcommunicators for each loop */
  auto CommGLB = thread.Split();
  auto Comm6th = CommGLB.Split();
  auto Comm5th = Comm6th.Split();
  auto Comm4th = Comm5th.Split();


  /** Adjuest nc and pack_nc if the 6th loop is parallelized */
  int nc = CommGLB.BalanceOver1DGangs( n, NC, NR );
  int pack_nc = ( nc / NR ) * PACK_NR;



  //printf( "CommGLB %s tid %d gid %d ngangs %d\n", CommGLB.comm->name.data(), CommGLB.tid, CommGLB.gid, CommGLB.comm->GetNumGroups() );
  //printf( "Comm6th %s tid %d gid %d ngangs %d\n", Comm6th.comm->name.data(), Comm6th.tid, Comm6th.gid, Comm6th.comm->GetNumGroups() );
  //printf( "Comm5th %s tid %d gid %d ngangs %d\n", Comm5th.comm->name.data(), Comm5th.tid, Comm5th.gid, Comm5th.comm->GetNumGroups() );
  //printf( "Comm4th %s tid %d gid %d ngangs %d\n", Comm4th.comm->name.data(), Comm4th.tid, Comm4th.gid, Comm4th.comm->GetNumGroups() );
  //fflush( stdout );

  /** 
   *  Allocate packing buffers:
   *
   *  packA is shared over Comm4th
   *  packB is shared over Comm5th
   */
  auto *packA = Comm4th.AllocateSharedMemory<ALIGN_SIZE, TPACKA>( KC * ( PACK_MC + 1 ) );
  auto *packB = Comm5th.AllocateSharedMemory<ALIGN_SIZE, TPACKB>( KC * ( pack_nc + 1 ) );

  /** Compute loop ranges for each thread */
  auto Loop6th = CommGLB.DistributeOver1DGangs(      0, n, nc );
  auto Loop5th = Comm6th.DistributeOver1DGangs( k_stra, k, KC );
  auto Loop4th = Comm5th.DistributeOver1DGangs(      0, m, MC );

  /** Comm6th is used inside the 6th loop (i.e. jc loop) */
  for ( int jc  = get<0>( Loop6th );
            jc  < get<1>( Loop6th );
            jc += get<2>( Loop6th ) )
  {
    auto jb = std::min( n - jc, nc );


    /** Comm5th is used inside the 6th loop (i.e. pc loop) */
    for ( int pc  = get<0>( Loop5th );
              pc  < get<1>( Loop5th );
              pc += get<2>( Loop5th ) )
    {
      auto pb = std::min( k - pc, KC );
      auto is_the_last_pc_iteration = ( pc + KC >= k );
      auto LooppkB = Comm5th.DistributeOver1DThreads( 0, jb,      NR );
      auto PackpkB = Comm5th.DistributeOver1DThreads( 0, jb, PACK_NR );

      for ( int j  = get<0>( LooppkB ), jp  = get<0>( PackpkB );
                j  < get<1>( LooppkB );
                j += get<2>( LooppkB ), jp += get<2>( PackpkB ) )
      {
        /** packB and typecast from TB to TPACKB  */
        B.Pack( 
            k, pc, pb, 
            n, jc + j, std::min( jb - j, NR ), 
            &packB[ jp * pb ] );
      }
      Comm5th.Barrier();


      /** Comm4th is used inside the 6th loop (i.e. pc loop) */
      for ( int ic  = get<0>( Loop4th );
                ic  < get<1>( Loop4th );
                ic += get<2>( Loop4th ) )
      {
        auto &ic_comm = *thread.ic_comm;
        auto ib = std::min( m - ic, MC );
        auto LooppkA = Comm4th.DistributeOver1DThreads( 0, ib, MR );
        auto PackpkA = Comm4th.DistributeOver1DThreads( 0, ib, PACK_MR );

        for ( int i  = get<0>( LooppkA ), ip  = get<0>( PackpkA );
                  i  < get<1>( LooppkA );
                  i += get<2>( LooppkA ), ip += get<2>( PackpkA ) )
        {
          /** packA and typecast from TA to TPACKA  */
          A.Pack( 
              m, ic + i, std::min( ib - i, MR ), 
              k, pc, pb, 
              &packA[ ip * pb ] );
        }
        Comm4th.Barrier();

        if ( is_the_last_pc_iteration )                    // fused_macro_kernel
        {
          fused_macro_kernel<KC>
          (
            Comm4th,
            m, n, 
            ic, jc, pc,
            ib, jb, pb,
            packA, 
            packB, 
            &C,
            V + ic * rs_v + jc * cs_v, rs_v, cs_v,
            batchId,
            microkernel
          );

        }
        else                                               // semiring rank-k update
        {
          rank_k_macro_kernel<KC>
          (  
            Comm4th, 
            ic, jc, pc,
            ib, jb, pb,
            packA,
            packB,
            V + ic * rs_v + jc * cs_v, rs_v, cs_v,
            semiringkernel
          );
        }
        Comm4th.Barrier();
      }                                                    // end 4th loop
      Comm5th.Barrier();
    }                                                      // end 5th loop
    Comm6th.Barrier();
  }                                                        // end 6th loop
  CommGLB.Barrier();

  /** Free packing buffer */
  Comm4th.FreeSharedMemory( packA );
  Comm5th.FreeSharedMemory( packB );

}; /** end gnbx_internal() */





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
  typename SEMIRINGKERNEL, typename MICROKERNEL>
void gnbx
(
  int batchId, int m, int n, int k,
  TA& A, 
  TB& B, 
  TC& C,
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel
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

  /** Early return if possible */
  if ( m == 0 || n == 0 || k == 0 ) return;


  TV *V = NULL;
  int rs_v = 0;
  int cs_v = 0;


  if ( k > KC && !is_same<TC, MatrixLike<PACK_MR, TV, TV>>::value )
  {
    //printf( "here m %d n %d\n", m, n );
    V = hmlp_malloc<ALIGN_SIZE, TV>( m * n );
    rs_v = 1;
    cs_v = m;
  }
  else
  {
    /** Directly use C for intermediate semiring rank-k update */
    V = reinterpret_cast<TV*>( C.X );
    rs_v = C.rs;
    cs_v = C.cs;
  }


  int k_stra = 0; 
  if ( USE_STRASSEN )
  {
    assert( typeid(TPACKA) == typeid(TPACKB) );
    assert( typeid(TC) == typeid(TV) );
    k_stra = k - k % KC;

    if ( k_stra == k ) k_stra -= KC;
  }

  int jc_nt = 1, pc_nt = 1, ic_nt = 1, jr_nt = 1;
  if ( omp_get_num_threads() == 1 && omp_get_max_threads() > 1 )
  {
    /** Check the environment variable. */
    jc_nt = hmlp_read_nway_from_env( "KS_JC_NT" );
    ic_nt = hmlp_read_nway_from_env( "KS_IC_NT" );
    jr_nt = hmlp_read_nway_from_env( "KS_JR_NT" );
  }

  /** allocate tree communicator */
  thread_communicator my_comm( jc_nt, pc_nt, ic_nt, jr_nt );

  #pragma omp parallel num_threads( my_comm.GetNumThreads() ) 
  {
    Worker thread( &my_comm );

    /** TODO:  */
    thread.InitWithCommunicator( &my_comm, omp_get_thread_num(), 0 );

    //if ( USE_STRASSEN )
    //{
    //  strassen::strassen_internal
    //  <MC, NC, KC, MR, NR,
    //  PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    //  USE_STRASSEN,
    //  SEMIRINGKERNEL, SEMIRINGKERNEL,
    //  TA, TPACKA, TB, TPACKB, TC, TV>
    //  (
    //    thread,
    //    m, n, k_stra,
    //    A, packakernel,
    //    B, packbkernel,
    //    V, ldv,
    //    semiringkernel, semiringkernel,
    //    nc, pack_nc,
    //    packA_buff,
    //    packB_buff
    //  );
    //}

    gnbx_internal<MC, NC, KC, TPACKA, TPACKB>
    (
      thread,
      batchId, m, n, k, k_stra,
      A, 
      B, 
      C, 
      V, rs_v, cs_v,
      semiringkernel, microkernel
    );
  }                                                        // end omp parallel

  if ( k > KC && !is_same<TC, MatrixLike<PACK_MR, TV, TV>>::value )
  {
    hmlp_free( V );
  }
};                                                         // end gkmx





/**
 *  @beief  
 */ 
template<
  int MR, int NR, int MC, int NC, int KC,
  typename TPACKA, typename TPACKB, typename TPACKC, typename TV,
  typename     TA, typename     TB, typename     TC,
  typename OPKERNEL, typename OP1, typename OP2>
void gnbx
(
  int batchId, int m, int n, int k,
  TA& A, 
  TB& B, 
  TC& C,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV
)
{
  semiring_mrxnr<MR, NR, OP1, OP2, TPACKA, TPACKB, TV, TV> semiringkernel;
  gnbx_mrxnr<MR, NR, OPKERNEL, OP1, OP2, TPACKA, TPACKB, TC, TPACKC, TV> gkrmkernel;

  semiringkernel.op1 = op1;
  semiringkernel.op2 = op2;
  semiringkernel.initV = initV;

  gkrmkernel.op1 = op1;
  gkrmkernel.op2 = op2;
  gkrmkernel.opkernel = opkernel;
  gkrmkernel.initV = initV;

  gnbx<MC, NC, KC, TPACKA, TPACKB, TV>
    ( batchId, m, n, k, A, B, C, semiringkernel, gkrmkernel );

}; /** end gnbx() */

}; /** end namespace gnbx */
}; /** end namespace hmlp */

#endif /** define GNBX_HPP */
