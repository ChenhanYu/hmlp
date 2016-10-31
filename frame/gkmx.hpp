#ifndef GKMX_HPP
#define GKMX_HPP

#include <hmlp.h>
#include <hmlp_internal.hpp>
#include <hmlp_packing.hpp>
#include <hmlp_util.hpp>
#include <hmlp_thread_communicator.hpp>
#include <hmlp_thread_info.hpp>
#include <hmlp_runtime.hpp>

namespace hmlp
{
namespace gkmx
{

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/**
 *
 */ 
template<
  int KC, int MR, int NR, int PACK_MR, int PACK_NR,
  typename SEMIRINGKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void rank_k_macro_kernel
(
  worker &thread,
  int ic, int jc, int pc,
  int  m, int n,  int  k,
  TA *packA,
  TB *packB,
  TV *C, int ldc,
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
    aux.jb       = min( n - j, NR );

    for ( int i  = loop2nd.beg(), ip  = pack2nd.beg(); 
              i  < loop2nd.end(); 
              i += loop2nd.inc(), ip += pack2nd.inc() )    // beg 2nd loop
    {
      aux.ib = min( m - i, MR );
      if ( aux.ib != MR ) 
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
          &C[ j * ldc + i ], ldc,
          &aux
        );
      }
      else                                                 // corner case
      {
        // TODO: this should be initC.
        TV ctmp[ MR * NR ] = { (TV)0.0 };
        semiringkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          ctmp, MR,
          &aux
        );
        if ( pc )
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
          {
            for ( auto ii = 0; ii < aux.ib; ii ++ )
            {
              C[ ( j + jj ) * ldc + i + ii ] += ctmp[ jj * MR + ii ];
            }
          }
        }
        else 
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
          {
            for ( auto ii = 0; ii < aux.ib; ii ++ )
            {
              C[ ( j + jj ) * ldc + i + ii ] = ctmp[ jj * MR + ii ];
            }
          }
        }
      }
    }                                                      // end 2nd loop
  }                                                        // end 3rd loop
}                                                          // end rank_k_macro_kernel

/**
 *
 */ 
template<int KC, int MR, int NR, int PACK_MR, int PACK_NR,
    typename MICROKERNEL,
    typename TA, typename TB, typename TC, typename TV>
void fused_macro_kernel
(
  worker &thread,
  int ic, int jc, int pc,
  int  m,  int n,  int k,
  TA *packA,
  TB *packB,
  TV *C, int ldc,
  MICROKERNEL microkernel
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
    aux.jb       = min( n - j, NR );

    for ( int i  = loop2nd.beg(), ip  = pack2nd.beg(); 
              i  < loop2nd.end(); 
              i += loop2nd.inc(), ip += pack2nd.inc() )    // beg 2nd loop
    {
      aux.ib = min( m - i, MR );
      if ( i + MR >= m ) 
      {
        // aux.b_next += ic_comm.GetNumThreads() * PACK_NR * k;
        aux.b_next += PACK_NR * k;
      }

      if ( aux.jb == NR && aux.ib == MR )                 
      {
        microkernel
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
        // TODO: this should be initC.
        TV ctmp[ MR * NR ] = { (TV)0.0 };
        microkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          ctmp, MR,
          &aux
        );

        if ( pc )
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
          {
            for ( auto ii = 0; ii < aux.ib; ii ++ )
            {
              C[ ( j + jj ) * ldc + i + ii ] += ctmp[ jj * MR + ii ];
            }
          }
        }
        else 
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
          {
            for ( auto ii = 0; ii < aux.ib; ii ++ )
            {
              C[ ( j + jj ) * ldc + i + ii ] = ctmp[ jj * MR + ii ];
            }
          }
        }
      }
    }                                                      // end 2nd loop
  }                                                        // end 3rd loop
}                                                          // end fused_macro_kernel



/*
 *
 */ 
template<
  int MC, int NC, int KC, int MR, int NR, 
  int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
  bool USE_STRASSEN,
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void gkmx_internal
(
  worker &thread,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  TA *A, int lda,
  TB *B, int ldb,
  TC *C, int ldc,
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

  auto loop6th = GetRange( 0, n, nc, thread.jc_id, thread.jc_nt );
  auto loop5th = GetRange( 0, k, KC );
  auto loop4th = GetRange( 0, m, MC, thread.ic_id, thread.ic_nt );

  for ( int jc  = loop6th.beg(); 
            jc  < loop6th.end(); 
            jc += loop6th.inc() )                          // beg 6th loop 
  {
    auto &jc_comm = *thread.jc_comm;
    auto jb = min( n - jc, nc );

    for ( int pc  = loop5th.beg();
              pc  < loop5th.end();
              pc += loop5th.inc() )
    {
      auto &pc_comm = *thread.pc_comm;
      auto pb = min( k - pc, KC );
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
            min( jb - j, NR ), pb, 
            &B[ ( jc + j ) * ldb + pc ], ldb, &packB[ jp * pb ] 
          );
        }
        else
        {
          pack2D<false, PACK_NR>                           // packB (transB)
          (
            min( jb - j, NR ), pb, 
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
        auto ib = min( m - ic, MC );
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
              min( ib - i, MR ), pb,
              &A[ pc * lda + ( ic + i ) ], lda, &packA[ ip * pb ] 
            );
          }
          else
          {
            pack2D<true, PACK_MR>                          // packA (transA)
            ( 
              min( ib - i, MR ), pb,
              &A[ ( ic + i ) * lda + pc ], lda, &packA[ ip * pb ] 
            );
          }
        }
        ic_comm.Barrier();

        if ( is_the_last_pc_iteration )                    // fused_macro_kernel
        {
          fused_macro_kernel
          <KC, MR, NR, PACK_MR, PACK_NR, MICROKERNEL, TA, TB, TC, TV>
          (
            thread, 
            ic, jc, pc,
            ib, jb, pb,
            packA, 
            packB, 
            C + jc * ldc + ic, ldc, 
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
            C + jc * ldc + ic, ldc, 
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
 *
 *
 */ 
template<
  int MC, int NC, int KC, int MR, int NR, 
  int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
  bool USE_STRASSEN,
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void gkmx
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  TA *A, int lda,
  TB *B, int ldb,
  TC *C, int ldc,
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel
)
{
  int jc_nt = 1, pc_nt = 1, ic_nt = 1, jr_nt = 1;
  int nc = NC, pack_nc = PACK_NC;
  char *str;

  TA *packA_buff = NULL;
  TB *packB_buff = NULL;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  // Check the environment variable.
  str = getenv( "KS_JC_NT" );
  if ( str ) jc_nt = (int)strtol( str, NULL, 10 );
  str = getenv( "KS_IC_NT" );
  if ( str ) ic_nt = (int)strtol( str, NULL, 10 );
  str = getenv( "KS_JR_NT" );
  if ( str ) jr_nt = (int)strtol( str, NULL, 10 );


  if ( jc_nt > 1 )
  {
    nc = ( ( n - 1 ) / ( NR * jc_nt ) + 1 ) * NR;
    pack_nc = ( nc / NR ) * PACK_NR;
  }

  // allocate packing memory
  packA_buff  = hmlp_malloc<ALIGN_SIZE, TA>( KC, ( PACK_MC + 1 ) * jc_nt * ic_nt,         sizeof(TA) );
  packB_buff  = hmlp_malloc<ALIGN_SIZE, TB>( KC, ( pack_nc + 1 ) * jc_nt,                 sizeof(TB) ); 

  // allocate tree communicator
  thread_communicator my_comm( jc_nt, pc_nt, ic_nt, jr_nt );


  #pragma omp parallel num_threads( my_comm.GetNumThreads() ) 
  {
    worker thread( &my_comm );

    if ( USE_STRASSEN )
    {
      printf( "gkmx: strassen algorithms haven't been implemented." );
      exit( 1 );
    }

    gkmx_internal
    <MC, NC, KC, MR, NR, 
    PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    USE_STRASSEN,
    SEMIRINGKERNEL, MICROKERNEL,
    TA, TB, TC, TB>
    (
      thread,
      transA, transB,
      m, n, k,
      A, lda,
      B, ldb,
      C, ldc,
      semiringkernel, microkernel,
      nc, pack_nc,
      packA_buff,
      packB_buff
    );
  }                                                        // end omp  

  hmlp_free( packA_buff );
  hmlp_free( packB_buff );
}                                                          // end gkmx


}; // end namespace gkmx
}; // end namespace hmlp

#endif // define GKMX_HPP
