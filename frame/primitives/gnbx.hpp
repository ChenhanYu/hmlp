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
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
//#include <hmlp_runtime.hpp>

/** for USE_STRASSEN */
//#include <primitives/strassen.hpp>

/** reference microkernels */
#include <packing.hpp>
#include <semiring_mrxnr.hpp>
#include <fused_mrxnr.hpp>



namespace hmlp
{
namespace gnbx
{







/**
 *  @brief Macro kernel contains the 3rd and 2nd loops. Depending on the
 *         configuration of the communicator, the 3rd loop may be parallelized.
 *         b_next is the prefetch pointer.
 */ 
template<
  int KC, int MR, int NR, int PACK_MR, int PACK_NR,
  typename SEMIRINGKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void rank_k_macro_kernel
(
  Worker &thread,
  int ic, int jc, int pc,
  int  m, int n,  int  k,
  TA *packA,
  TB *packB,
  TV *V, int ldv,
  SEMIRINGKERNEL semiringkernel
)
{
  thread_communicator &ic_comm = *thread.ic_comm;

  auto loop3rd = GetRange( 0, n,      NR, thread.jr_id, ic_comm.GetNumThreads() );
  auto pack3rd = GetRange( 0, n, PACK_NR, thread.jr_id, ic_comm.GetNumThreads() );
  auto loop2nd = GetRange( 0, m,      MR );
  auto pack2nd = GetRange( 0, m, PACK_MR );

  for ( int j   = loop3rd.beg(), jp  = pack3rd.beg(); 
            j   < loop3rd.end();
            j  += loop3rd.inc(), jp += pack3rd.inc() )     // beg 3rd loop
  {
    struct aux_s<TA, TB, TC, TV> aux;
    aux.pc       = pc;
    aux.b_next   = packB;
    aux.do_packC = 0;
    aux.jb       = std::min( n - j, NR );

    for ( int i  = loop2nd.beg(), ip  = pack2nd.beg(); 
              i  < loop2nd.end(); 
              i += loop2nd.inc(), ip += pack2nd.inc() )    // beg 2nd loop
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
          &V[ j * ldv + i ], ldv,
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
              vtmp[ jj * MR + ii ] = V[ ( j + jj ) * ldv + i + ii ];
        }

        semiringkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          vtmp, MR,
          &aux
        );

        for ( auto jj = 0; jj < aux.jb; jj ++ )
          for ( auto ii = 0; ii < aux.ib; ii ++ )
            V[ ( j + jj ) * ldv + i + ii ] = vtmp[ jj * MR + ii ];
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
template<
int KC, int MR, int NR, int PACK_MR, int PACK_NR,
bool REUSE_C,
typename FUSEDKERNEL,
typename TA, typename TB, typename TC, typename TV>
void fused_macro_kernel
(
  Worker &thread,
  int ic, int jc, int pc,
  int  m,  int n,  int k,
  TA *packA,
  TB *packB,
  TC *C, int ldc,
  TV *V, int ldv,
  int batchId,
  FUSEDKERNEL fusedkernel
)
{
  thread_communicator &ic_comm = *thread.ic_comm;

  auto loop3rd = GetRange( 0, n,      NR, thread.jr_id, ic_comm.GetNumThreads() );
  auto pack3rd = GetRange( 0, n, PACK_NR, thread.jr_id, ic_comm.GetNumThreads() );
  auto loop2nd = GetRange( 0, m,      MR );
  auto pack2nd = GetRange( 0, m, PACK_MR );

  for ( int j   = loop3rd.beg(), jp  = pack3rd.beg(); 
            j   < loop3rd.end();
            j  += loop3rd.inc(), jp += pack3rd.inc() )     // beg 3rd loop
  {
    struct aux_s<TA, TB, TC, TV> aux;
    aux.pc       = pc;
    aux.b_next   = packB;
    aux.do_packC = 0;

    for ( int i  = loop2nd.beg(), ip  = pack2nd.beg(); 
              i  < loop2nd.end(); 
              i += loop2nd.inc(), ip += pack2nd.inc() )    // beg 2nd loop
    {
      // These auxiluary infos are used to access data in the closure of
      // opkernel and opreduce.
      aux.i = ic + i;
      aux.j = jc + j;
      aux.b = batchId;

      aux.ib = std::min( m - i, MR );
      aux.jb = std::min( n - j, NR );

      aux.V = V + j * ldv + i;
      aux.ldv = ldv;

      if ( i + MR >= m ) 
      {
        aux.b_next += ic_comm.GetNumThreads() * PACK_NR * k;
      }

      if ( aux.jb == NR && aux.ib == MR )                 
      {
        fusedkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          &C[ j * ldc + i ], ldc,
          &aux
        );
      }
      else                                                 // corner case
      {
        TC ctmp[ MR * NR ];
        TV vtmp[ MR * NR ];

        if ( pc ) // initilize ctmp
        {
          if ( REUSE_C )
          {
            for ( auto jj = 0; jj < aux.jb; jj ++ )
              for ( auto ii = 0; ii < aux.ib; ii ++ )
                ctmp[ jj * MR + ii ] = C[ ( j + jj ) * ldc + i + ii ];
          }
          else
          {
            for ( auto jj = 0; jj < aux.jb; jj ++ )
              for ( auto ii = 0; ii < aux.ib; ii ++ )
                vtmp[ jj * MR + ii ] = V[ ( j + jj ) * ldv + i + ii ];
            aux.V = vtmp;
            aux.ldv = MR;
          }
        }

        fusedkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          ctmp, MR,
          &aux
        );

        for ( auto jj = 0; jj < aux.jb; jj ++ )
          for ( auto ii = 0; ii < aux.ib; ii ++ )
            C[ ( j + jj ) * ldc + i + ii ] = ctmp[ jj * MR + ii ];

      }
    }                                                      // end 2nd loop
  }                                                        // end 3rd loop
};                                                         // end fused_macro_kernel





/**
 *  @breif This function contains the loop body of the 6th to 4th loops,
 *         including all packing and unpacking routines. Notice that this
 *         function is executed by all threads in the root communicator.
 *         To access each thread in different level of communicators, use
 *         their ids.
 */ 
template<
  int MC, 
  int NC, 
  int KC, 
  int MR, 
  int NR, 
  int PACK_MC, 
  int PACK_NC, 
  int PACK_MR, 
  int PACK_NR, 
  int ALIGN_SIZE,
  bool USE_STRASSEN, 
  bool REUSE_C,
  typename PACKAKERNEL, typename PACKBKERNEL,
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TPACKA, 
  typename TB, typename TPACKB, 
  typename TC, typename TV>
void gnbx_internal
(
  Worker &thread,
  int m, int n, int k, int k_stra,
  TA *A, PACKAKERNEL packakernel,
  TB *B, PACKBKERNEL packbkernel,
  TC *C, int ldc,
  TV *V, int ldv,
  int batchId,
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel,
  int nc, int pack_nc,
  TPACKA *packA, 
  TPACKB *packB 
)
{
  /** get comm.Split = jc_comm, jc_comm.Split = pc_comm, ic_comm */



  packA  += ( thread.jc_id * thread.ic_nt                ) * PACK_MC * KC
          + ( thread.ic_id                               ) * PACK_MC * KC;
  packB  += ( thread.jc_id                               ) * pack_nc * KC;

  /** Comm6th.DistributeOver1DGang( 0, n, nc ) */
  auto loop6th = GetRange( 0,      n, nc, thread.jc_id, thread.jc_nt );
  auto loop5th = GetRange( k_stra, k, KC );
  /** Comm4th.DistributeOver1DGang( 0, m, MC ) */
  auto loop4th = GetRange( 0,      m, MC, thread.ic_id, thread.ic_nt );

  for ( int jc  = loop6th.beg(); 
            jc  < loop6th.end(); 
            jc += loop6th.inc() )                          // beg 6th loop 
  {
    /** Comm6th */
    auto &jc_comm = *thread.jc_comm;
    auto jb = std::min( n - jc, nc );

    for ( int pc  = loop5th.beg();
              pc  < loop5th.end();
              pc += loop5th.inc() )
    {
      /** Comm5th */
      auto &pc_comm = *thread.pc_comm;
      auto pb = std::min( k - pc, KC );
      auto is_the_last_pc_iteration = ( pc + KC >= k );
      /** Comm5th.DistributeOver1DThread( 0, jb, NR ) */
      auto looppkB = GetRange( 0, jb,      NR, thread.ic_jr, pc_comm.GetNumThreads() ); 
      /** Comm5th.DistributeOver1DThread( 0, jb, PACK_NR ) */
      auto packpkB = GetRange( 0, jb, PACK_NR, thread.ic_jr, pc_comm.GetNumThreads() ); 

      for ( int j   = looppkB.beg(), jp  = packpkB.beg(); 
                j   < looppkB.end(); 
                j  += looppkB.inc(), jp += packpkB.inc() ) 
      {
        /** packB and typecast from TB to TPACKB  */
        packbkernel( 
            k, pc, pb, 
            n, jc + j, std::min( jb - j, NR ), 
            B, &packB[ jp * pb ] );
      }
      /** Comm5th.Barrier() */
      pc_comm.Barrier();

      for ( int ic  = loop4th.beg(); 
                ic  < loop4th.end(); 
                ic += loop4th.inc() )                      // beg 4th loop
      {
        /** Comm4th */
        auto &ic_comm = *thread.ic_comm;
        auto ib = std::min( m - ic, MC );
        /** Comm5th.DistributeOver1DThread( 0, ib, MR ) */
        auto looppkA = GetRange( 0, ib,      MR, thread.jr_id, thread.jr_nt ); 
        /** Comm5th.DistributeOver1DThread( 0, ib, PACK_MR ) */
        auto packpkA = GetRange( 0, ib, PACK_MR, thread.jr_id, thread.jr_nt ); 

        for ( int i   = looppkA.beg(), ip  = packpkA.beg();  
                  i   < looppkA.end(); 
                  i  += looppkA.inc(), ip += packpkA.inc() )     
        {
          /** packA and typecast from TA to TPACKA  */
          packakernel( 
              k, pc, pb, 
              m, ic + i, std::min( ib - i, MR ), 
              A, &packA[ ip * pb ] );
        }
        /** Comm4th.Barrier() */
        ic_comm.Barrier();

        if ( is_the_last_pc_iteration )                    // fused_macro_kernel
        {
          fused_macro_kernel<
            KC, MR, NR, PACK_MR, PACK_NR, 
            REUSE_C, 
            MICROKERNEL, 
            TPACKA, TPACKB, TC, TV>
          (
            thread, 
            ic, jc, pc,
            ib, jb, pb,
            packA, 
            packB, 
            C + jc * ldc + ic, ldc,
            V + jc * ldv + ic, ldv, // if REUSE_C, then V = C.
            batchId,
            microkernel
          );
        }
        else                                               // semiring rank-k update
        {
          rank_k_macro_kernel<
            KC, MR, NR, PACK_MR, PACK_NR, 
            SEMIRINGKERNEL, 
            TPACKA, TPACKB, TC, TV>
          (  
            thread, 
            ic, jc, pc,
            ib, jb, pb,
            packA,
            packB,
            //C + jc * ldc + ic, ldc, 
            V + jc * ldv + ic, ldv, 
            semiringkernel
          );
        }
        /** Comm4th.Barrier() */
        ic_comm.Barrier();                                 // sync all jr_id!!
      }                                                    // end 4th loop
      /** Comm5th.Barrier() */
      pc_comm.Barrier();
    }                                                      // end 5th loop
  }                                                        // end 6th loop
};                                                         // end gnbx_internal





/**
 *  @breif This is the main routine of gkmx. All packing buffers are
 *         managed here. The communicator and the parallel section 
 *         start here.
 *
 */ 
template<
  int MC, 
  int NC, 
  int KC, 
  int MR, 
  int NR, 
  int PACK_MC, 
  int PACK_NC, 
  int PACK_MR, 
  int PACK_NR, 
  int ALIGN_SIZE,
  bool USE_STRASSEN = false, 
  bool REUSE_C,
  typename PACKAKERNEL, typename PACKBKERNEL,
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TPACKA, 
  typename TB, typename TPACKB, 
  typename TC, typename TV = TC>
void gnbx
(
  int m, int n, int k,
  TA *A, PACKAKERNEL packakernel,
  TB *B, PACKBKERNEL packbkernel,
  TC *C, int ldc,
  int batchId,
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel
)
{
  int jc_nt = 1, pc_nt = 1, ic_nt = 1, jr_nt = 1;
  int k_stra = 0; 
  int ldv = 0;
  int nc = NC, pack_nc = PACK_NC;
  char *str;

  TPACKA *packA_buff = NULL;
  TPACKB *packB_buff = NULL;
  TV *V = NULL;

  /** Early return if possible */
  if ( m == 0 || n == 0 || k == 0 ) return;

  /** type checking (currently assume TC == TV) */
  if ( typeid(TC) != typeid(TV) && k > KC )
  {
    printf( "gnbx: currently k(%d) must be smaller than %d when TC != TV\n", k, KC );
    exit( 1 );
  }

  if ( omp_get_num_threads() == 1 && omp_get_max_threads() > 1 )
  {
    /** Check the environment variable. */
    jc_nt = hmlp_read_nway_from_env( "KS_JC_NT" );
    ic_nt = hmlp_read_nway_from_env( "KS_IC_NT" );
    jr_nt = hmlp_read_nway_from_env( "KS_JR_NT" );
  }

  if ( jc_nt > 1 )
  {
    nc = ( ( n - 1 ) / ( NR * jc_nt ) + 1 ) * NR;
    pack_nc = ( nc / NR ) * PACK_NR;
  }

  /** allocate packing memory */
  packA_buff  = hmlp_malloc<ALIGN_SIZE, TPACKA>( KC * ( PACK_MC + 1 ) * jc_nt * ic_nt );
  packB_buff  = hmlp_malloc<ALIGN_SIZE, TPACKB>( KC * ( pack_nc + 1 ) * jc_nt         ); 


  /** allocate V if k > KC */
  if ( k > KC && !std::is_same<TC, TV>::value && !REUSE_C )
  {
    V = hmlp_malloc<ALIGN_SIZE, TV>( m * n );
    ldv = m;
  }
  else // TODO: do not free V in this case.
  {
    V = reinterpret_cast<TV*>( C );
    ldv = ldc;
  }

  /** allocate tree communicator */
  thread_communicator my_comm( jc_nt, pc_nt, ic_nt, jr_nt );


  if ( USE_STRASSEN )
  {
    assert( typeid(TA) == typeid(TB) );
    assert( typeid(TC) == typeid(TV) );
    k_stra = k - k % KC;

    if ( k_stra == k ) k_stra -= KC;

    if ( k_stra )
    {
      #pragma omp parallel for
      for ( int i = 0; i < n * ldv; i ++ ) V[ i ] = 0.0;
    }
  }


  #pragma omp parallel num_threads( my_comm.GetNumThreads() ) 
  {
    Worker thread( &my_comm );

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

    gnbx_internal<
      MC, NC, KC, MR, NR, 
      PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
      USE_STRASSEN, REUSE_C,
      PACKAKERNEL, PACKBKERNEL,
      SEMIRINGKERNEL, MICROKERNEL,
      TA, TPACKA, TB, TPACKB, TC, TV>
    (
      thread,
      m, n, k, k_stra,
      A, packakernel,
      B, packbkernel,
      C, ldc,
      V, ldv,
      batchId,
      semiringkernel, microkernel,
      nc, pack_nc,
      packA_buff,
      packB_buff
    );
  }                                                        // end omp parallel

  hmlp_free( packA_buff );
  hmlp_free( packB_buff );
  //hmlp_free( V );
};                                                         // end gkmx





/**
 *  @beief  
 */ 
template<
  int MC            = 104, 
  int NC            = 1024, 
  int KC            = 256, 
  int MR            = 8, 
  int NR            = 4, 
  int PACK_MC       = 104, 
  int PACK_NC       = 1024, 
  int PACK_MR       = 8, 
  int PACK_NR       = 4, 
  int ALIGN_SIZE    = 32,
  bool USE_STRASSEN = false,
  bool REUSE_C = false,
  typename OPKERNEL, typename OP1, typename OP2,
  typename TA, typename TB, typename TC, typename TV>
void gkmm
(
  int m, int n, int k,
  TA *A, int lda,
  TB *B, int ldb,
  TC *C, int ldc,
  int batchId,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV
)
{
  semiring_mrxnr<MR, NR, OP1, OP2, TA, TB, TC, TV> semiringkernel;
  gkmm_mrxnr<MR, NR, OPKERNEL, OP1, OP2, TA, TB, TC, TV> gkmmkernel;

  semiringkernel.op1 = op1;
  semiringkernel.op2 = op2;
  semiringkernel.initV = initV;

  gkmmkernel.op1 = op1;
  gkmmkernel.op2 = op2;
  gkmmkernel.opkernel = opkernel;
  gkmmkernel.initV = initV;

  gkmx
  <MC, NC, KC, MR, NR, PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
  USE_STRASSEN, REUSE_C,
  semiring_mrxnr<MR, NR, OP1, OP2, TA, TB, TC, TV>,
  gkmm_mrxnr<MR, NR, OPKERNEL, OP1, OP2, TA, TB, TC, TV>,
  TA, TB, TC, TV>
  (
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    batchId,
    semiringkernel, gkmmkernel
  );
};






/**
 *  @beief Implement GKRM with GKMX template. Notice that OPREDUCE 
 *         is handled inside fusedkernel. Updating microkernel has 
 *         to be atomic if jc_nt or jr_nt is not 1. We may be atomic
 *         update.
 *         
 */ 
template<
  int MC            = 104, 
  int NC            = 1024, 
  int KC            = 256, 
  int MR            = 8, 
  int NR            = 4, 
  int PACK_MC       = 104, 
  int PACK_NC       = 1024, 
  int PACK_MR       = 8, 
  int PACK_NR       = 4, 
  int ALIGN_SIZE    = 32,
  bool USE_STRASSEN = false,
  typename PACKAKERNEL, typename PACKBKERNEL,
  typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
  typename TA, typename TPACKA, typename TB, typename TPACKB, 
  typename TC, typename TV = TC>
void gnbx
(
  int m, int n, int k,
  TA *A, PACKAKERNEL packakernel,
  TB *B, PACKBKERNEL packbkernel,
  TC *C, int ldc,
  int batchId,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, 
  OPREDUCE opreduce, TC initC
)
{
  semiring_mrxnr<MR, NR, OP1, OP2, TA, TB, TC, TV> semiringkernel;
  gkrm_mrxnr<MR, NR, OPKERNEL, OP1, OP2, OPREDUCE, TA, TB, TC, TV> gkrmkernel;

  semiringkernel.op1 = op1;
  semiringkernel.op2 = op2;
  semiringkernel.initV = initV;
  
  gkrmkernel.op1 = op1;
  gkrmkernel.op2 = op2;
  gkrmkernel.opkernel = opkernel;
  gkrmkernel.initV = initV;
  gkrmkernel.opreduce = opreduce;
  gkrmkernel.initC = initC;

  gnbx<
    MC, NC, KC, MR, NR, 
    PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    USE_STRASSEN,
    semiring_mrxnr<MR, NR, OP1, OP2, TA, TB, TC, TV>,
    gkmm_mrxnr<MR, NR, OPKERNEL, OP1, OP2, TA, TB, TC, TV>,
    TA, TPACKA, TB, TPACKB, TC, TV>
  (
    m, n, k,
    A, packakernel,
    B, packbkernel,
    C, 0, // TODO: is there a better way to do this?
    batchId,
    semiringkernel, gkrmkernel
  );

}; /** end gnbx() */



}; /** end namespace gnbx */
}; /** end namespace hmlp */

#endif /** define GNBX_HPP */
