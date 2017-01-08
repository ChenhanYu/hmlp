#ifndef GKMX_HPP
#define GKMX_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>

#include <hmlp.h>
#include <hmlp_internal.hpp>
#include <hmlp_packing.hpp>
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>

// For USE_STRASSEN
#include <strassen.hpp>

// reference microkernels 
#include <semiring_mrxnr.hpp>
#include <fused_mrxnr.hpp>

//#define GKMX_CONFIG \


namespace hmlp
{
namespace gkmx
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
}                                                          // end rank_k_macro_kernel


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
          //&C[ ( j / NR ) * ldc + i ], ldc, // for conv_relu_pool
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
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void gkmx_internal
(
  Worker &thread,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k, int k_stra,
  TA *A, int lda,
  TB *B, int ldb,
  TC *C, int ldc,
  TV *V, int ldv,
  int batchId,
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel,
  int nc, int pack_nc,
  TA *packA, 
  TB *packB 
)
{
  packA  += ( thread.jc_id * thread.ic_nt                ) * PACK_MC * KC
          + ( thread.ic_id                               ) * PACK_MC * KC;
  packB  += ( thread.jc_id                               ) * pack_nc * KC;

  auto loop6th = GetRange( 0,      n, nc, thread.jc_id, thread.jc_nt );
  auto loop5th = GetRange( k_stra, k, KC );
  auto loop4th = GetRange( 0,      m, MC, thread.ic_id, thread.ic_nt );

  for ( int jc  = loop6th.beg(); 
            jc  < loop6th.end(); 
            jc += loop6th.inc() )                          // beg 6th loop 
  {
    auto &jc_comm = *thread.jc_comm;
    auto jb = std::min( n - jc, nc );

    for ( int pc  = loop5th.beg();
              pc  < loop5th.end();
              pc += loop5th.inc() )
    {
      auto &pc_comm = *thread.pc_comm;
      auto pb = std::min( k - pc, KC );
      auto is_the_last_pc_iteration = ( pc + KC >= k );
      auto looppkB = GetRange( 0, jb,      NR, thread.ic_jr, pc_comm.GetNumThreads() ); 
      auto packpkB = GetRange( 0, jb, PACK_NR, thread.ic_jr, pc_comm.GetNumThreads() ); 

      for ( int j   = looppkB.beg(), jp  = packpkB.beg(); 
                j   < looppkB.end(); 
                j  += looppkB.inc(), jp += packpkB.inc() ) 
      {
        if ( transB == HMLP_OP_N )
        {
          pack2D<true, PACK_NR>                            // packB
          (
            std::min( jb - j, NR ), pb, 
            &B[ ( jc + j ) * ldb + pc ], ldb, &packB[ jp * pb ] 
          );
        }
        else
        {
          pack2D<false, PACK_NR>                           // packB (transB)
          (
            std::min( jb - j, NR ), pb, 
            &B[ pc * ldb + ( jc + j ) ], ldb, &packB[ jp * pb ] 
          );
        }
      }
      pc_comm.Barrier();

      for ( int ic  = loop4th.beg(); 
                ic  < loop4th.end(); 
                ic += loop4th.inc() )                      // beg 4th loop
      {
        auto &ic_comm = *thread.ic_comm;
        auto ib = std::min( m - ic, MC );
        auto looppkA = GetRange( 0, ib,      MR, thread.jr_id, thread.jr_nt ); 
        auto packpkA = GetRange( 0, ib, PACK_MR, thread.jr_id, thread.jr_nt ); 

        for ( int i   = looppkA.beg(), ip  = packpkA.beg();  
                  i   < looppkA.end(); 
                  i  += looppkA.inc(), ip += packpkA.inc() )     
        {
          if ( transA == HMLP_OP_N )
          {
            pack2D<false, PACK_MR>                         // packA 
            ( 
              std::min( ib - i, MR ), pb,
              &A[ pc * lda + ( ic + i ) ], lda, &packA[ ip * pb ] 
            );
          }
          else
          {
            pack2D<true, PACK_MR>                          // packA (transA)
            ( 
              std::min( ib - i, MR ), pb,
              &A[ ( ic + i ) * lda + pc ], lda, &packA[ ip * pb ] 
            );
          }
        }
        ic_comm.Barrier();

        if ( is_the_last_pc_iteration )                    // fused_macro_kernel
        {
          fused_macro_kernel
          <KC, MR, NR, PACK_MR, PACK_NR, REUSE_C, MICROKERNEL, TA, TB, TC, TV>
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
          rank_k_macro_kernel
          <KC, MR, NR, PACK_MR, PACK_NR, SEMIRINGKERNEL, TA, TB, TC, TV>
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
        ic_comm.Barrier();                                 // sync all jr_id!!
      }                                                    // end 4th loop
      pc_comm.Barrier();
    }                                                      // end 5th loop
  }                                                        // end 6th loop
}                                                          // end gkmx_internal





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
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV = TC>
void gkmx
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  TA *A, int lda,
  TB *B, int ldb,
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

  TA *packA_buff = NULL;
  TB *packB_buff = NULL;
  TV *V = NULL;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  // type checking (currently assume TC == TV)
  if ( typeid(TC) != typeid(TV) && k > KC )
  {
    printf( "gkmx: currently k(%d) must be smaller than %d when TC != TV\n", k, KC );
    exit( 1 );
  }

  if ( omp_get_num_threads() == 1 && omp_get_max_threads() > 1 )
  {
    // Check the environment variable.
    jc_nt = hmlp_read_nway_from_env( "KS_JC_NT" );
    ic_nt = hmlp_read_nway_from_env( "KS_IC_NT" );
    jr_nt = hmlp_read_nway_from_env( "KS_JR_NT" );
  }

  if ( jc_nt > 1 )
  {
    nc = ( ( n - 1 ) / ( NR * jc_nt ) + 1 ) * NR;
    pack_nc = ( nc / NR ) * PACK_NR;
  }

  // allocate packing memory
  packA_buff  = hmlp_malloc<ALIGN_SIZE, TA>( KC * ( PACK_MC + 1 ) * jc_nt * ic_nt );
  packB_buff  = hmlp_malloc<ALIGN_SIZE, TB>( KC * ( pack_nc + 1 ) * jc_nt         ); 


  // allocate V if k > KC
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

  // allocate tree communicator
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

    if ( USE_STRASSEN )
    {
      strassen::strassen_internal
      <MC, NC, KC, MR, NR,
      PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
      USE_STRASSEN,
      SEMIRINGKERNEL, SEMIRINGKERNEL,
      TA, TB, TC, TV>
      (
        thread,
        transA, transB,
        m, n, k_stra,
        A, lda,
        B, ldb,
        V, ldv,
        semiringkernel, semiringkernel,
        nc, pack_nc,
        packA_buff,
        packB_buff
      );
    }

    gkmx_internal
    <MC, NC, KC, MR, NR, 
    PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    USE_STRASSEN, REUSE_C,
    SEMIRINGKERNEL, MICROKERNEL,
    TA, TB, TC, TV>
    (
      thread,
      transA, transB,
      m, n, k, k_stra,
      A, lda,
      B, ldb,
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
  hmlpOperation_t transA, hmlpOperation_t transB,
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
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    batchId,
    semiringkernel, gkmmkernel
  );
};


/**
 *  @brief batched interface with array of arrays
 *
 *  TODO: the problem is how to manage thread here? Do I want to use omp
 *  nested? or there is a better way to deal with this.
 *
 */ 
template<
  int MC, int NC, int KC, int MR, int NR, 
  int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
  bool USE_STRASSEN, bool REUSE_C,
  typename OPKERNEL, typename OP1, typename OP2,
  typename TA, typename TB, typename TC, typename TV>
void gkmm
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  TA *Aarray[], int lda,
  TB *Barray[], int ldb,
  TC *Carray[], int ldc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV
)
{
  #pragma omp parallel for
  for ( auto b = 0; b < batchSize; b ++ )
  {
    gkmm
    <MC, NC, KC, MR, NR, PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    USE_STRASSEN,
    OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
    (
      transA, transB,
      m, n, k, 
      Aarray[ b ], lda,
      Barray[ b ], ldb,
      Carray[ b ], ldc,
      b,
      opkernel, op1, op2, initV
    );
  }
}; // end gkmm


/**
 *  @brief batched interface with strides
 *
 *  TODO: the problem is how to manage thread here? Do I want to use omp
 *  nested? or there is a better way to deal with this.
 *
 */ 
template<
  int MC, 
  int NC, 
  int KC, int MR, int NR, 
  int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
  bool USE_STRASSEN, bool REUSE_C,
  typename OPKERNEL, typename OP1, typename OP2,
  typename TA, typename TB, typename TC, typename TV>
void gkmm
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  TA *Aarray, int lda, int loa, 
  TB *Barray, int ldb, int lob,
  TC *Carray, int ldc, int loc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV
)
{
  #pragma omp parallel for
  for ( auto b = 0; b < batchSize; b ++ )
  {
    gkmm
    <MC, NC, KC, MR, NR, PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    USE_STRASSEN, REUSE_C,
    OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
    (
      transA, transB,
      m, n, k, 
      Aarray + b * loa, lda,
      Barray + b * lob, ldb,
      Carray + b * loc, ldc,
      b,
      opkernel, op1, op2, initV
    );
  }
}; // end gkmm



















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
  typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
  typename TA, typename TB, typename TC, typename TV = TC>
void gkrm
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  TA *A, int lda,
  TB *B, int ldb,
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

  gkmx
  <MC, NC, KC, MR, NR, PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
  USE_STRASSEN,
  semiring_mrxnr<MR, NR, OP1, OP2, TA, TB, TC, TV>,
  gkmm_mrxnr<MR, NR, OPKERNEL, OP1, OP2, TA, TB, TC, TV>,
  TA, TB, TC, TV>
  (
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, 0, // TODO: is there a better way to do this?
    batchId,
    semiringkernel, gkrmkernel
  );
}; // end gkrm




/**
 *  @breif This is a simple triple loop reference.
 */
template<
  typename OPKERNEL, typename OP1, typename OP2,
  typename TA, typename TB, typename TC, typename TV = TC>
void gkmm_ref
(
 hmlpOperation_t transA, hmlpOperation_t transB,
 int m, int n, int k,
 TA *A, int lda,
 TB *B, int ldb,
 TC *C, int ldc,
 OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV 
)
{
  for ( int i = 0; i < m; i ++ )
  { 
    for ( int j = 0; j < n; j ++ )
    {
      auto v = initV;
      for ( int p = 0; p < k; p ++ )
      {
        TA a;
        TB b;
        if ( transA == HMLP_OP_N ) a = A[ p * lda + i ];
        else                       a = A[ i * lda + p ];
        if ( transB == HMLP_OP_N ) b = B[ j * ldb + p ];
        else                       b = B[ p * ldb + j ];
        v = op1( v, op2( a, b ) );
      }
      C[ j * ldc + i ] = opkernel( v ); 
    }
  }
}; // end gkmm_ref


/**
 *  @breif This is a simple triple loop reference.
 *
 *  TODO: ldc is strange here, assuming that C is a vector.
 */ 
template<
  typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
  typename TA, typename TB, typename TC, typename TV = TC>
void gkrm_ref
(
 hmlpOperation_t transA, hmlpOperation_t transB,
 int m, int n, int k,
 TA *A, int lda,
 TB *B, int ldb,
 TC *C, int ldc,
 int batchId,
 OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, 
 OPREDUCE opreduce, TC initC
 )
{
  for ( int i = 0; i < m; i ++ )
  { 
    auto c = initC;
    for ( int j = 0; j < n; j ++ )
    {
      auto v = initV;
      for ( int p = 0; p < k; p ++ )
      {
        TA a;
        TB b;
        if ( transA == HMLP_OP_N ) a = A[ p * lda + i ];
        else                       a = A[ i * lda + p ];
        if ( transB == HMLP_OP_N ) b = B[ j * ldb + p ];
        else                       b = B[ p * ldb + j ];
        v = op1( v, op2( a, b ) );
      }
      c = opreduce( c, opkernel( v ) ); 
    }
    C[ i ] = c;
  }
}; // end gkrm_ref


}; // end namespace gkmx
}; // end namespace hmlp

#endif // define GKMX_HPP
