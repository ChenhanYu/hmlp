#ifndef STRASSEN_HPP
#define STRASSEN_HPP

#define STRAPRIM( A0,A1,gamma,B0,B1,delta,C0,C1,alpha0,alpha1 ) \
    straprim \
    <MC, NC, KC, MR, NR,  \
    PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE, \
    USE_STRASSEN, \
    RANK_SEMIRINGKERNEL, RANK_MICROKERNEL, \
    STRA_SEMIRINGKERNEL, STRA_MICROKERNEL, \
    TA, TB, TC, TB> \
    ( \
      thread, \
      transA, transB, \
      m, n, k, \
      A0, A1, lda, gamma, \
      B0, B1, ldb, delta, \
      C0, C1, ldc, alpha0, alpha1, \
      rank_semiringkernel, rank_microkernel, \
      stra_semiringkernel, stra_microkernel, \
      nc, pack_nc, \
      packA_buff, \
      packB_buff \
    ); \

#include <hmlp.h>
#include <hmlp_internal.hpp>
#include <hmlp_packing.hpp>
#include <hmlp_util.hpp>
#include <hmlp_thread_communicator.hpp>
#include <hmlp_thread_info.hpp>
#include <hmlp_runtime.hpp>

#include <gkmx.hpp>

namespace hmlp
{
namespace strassen
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
  TV *C0, TV *C1, int ldc, TV alpha0, TV alpha1,
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
          &C0[ j * ldc + i ], &C1[ j * ldc + i ], ldc, alpha0, alpha1,
          &aux
        );
      }
      else                                                 // corner case
      {
        // TODO: this should be initC.
        TV ctmp[ MR * NR ] = { (TV)0.0 };

        //rank_k_int_d8x4 rankk_semiringkernel;
        //rankk_semiringkernel
        semiringkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          //ctmp, MR,
          ctmp, NULL, MR, 1, 0,
          &aux
        );
        if ( pc )
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
          {
            for ( auto ii = 0; ii < aux.ib; ii ++ )
            {
              C0[ ( j + jj ) * ldc + i + ii ] += alpha0 * ctmp[ jj * MR + ii ];
              C1[ ( j + jj ) * ldc + i + ii ] += alpha1 * ctmp[ jj * MR + ii ];
            }
          }
        }
        else 
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
          {
            for ( auto ii = 0; ii < aux.ib; ii ++ )
            {
              C0[ ( j + jj ) * ldc + i + ii ] = alpha0 * ctmp[ jj * MR + ii ];
              C1[ ( j + jj ) * ldc + i + ii ] = alpha1 * ctmp[ jj * MR + ii ];
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
  TV *C0, TV *C1, int ldc, TV alpha0, TV alpha1,
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
      if ( aux.ib != MR ) 
      {
        aux.b_next += ic_comm.GetNumThreads() * PACK_NR * k;
      }

      if ( aux.jb == NR && aux.ib == MR )                 
      {
        microkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          &C0[ j * ldc + i ], &C1[ j * ldc + i ], ldc, alpha0, alpha1,
          &aux
        );
      }
      else                                                 // corner case
      {
        // TODO: this should be initC.
        TV ctmp[ MR * NR ] = { (TV)0.0 };

        //rank_k_int_d8x4 rankk_microkernel;
        //rankk_microkernel
        microkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          //ctmp, MR,
          ctmp, NULL, MR, 1, 0,
          &aux
        );

        if ( pc )
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
          {
            for ( auto ii = 0; ii < aux.ib; ii ++ )
            {
              C0[ ( j + jj ) * ldc + i + ii ] += alpha0 * ctmp[ jj * MR + ii ];
              C1[ ( j + jj ) * ldc + i + ii ] += alpha1 * ctmp[ jj * MR + ii ];
            }
          }
        }
        else 
        {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
          {
            for ( auto ii = 0; ii < aux.ib; ii ++ )
            {
              C0[ ( j + jj ) * ldc + i + ii ] = alpha0 * ctmp[ jj * MR + ii ];
              C1[ ( j + jj ) * ldc + i + ii ] = alpha1 * ctmp[ jj * MR + ii ];
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
  typename RANK_SEMIRINGKERNEL, typename RANK_MICROKERNEL,
  typename STRA_SEMIRINGKERNEL, typename STRA_MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void straprim
(
  worker &thread,
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  TA *A0, TA *A1, int lda, TA gamma,
  TB *B0, TB *B1, int ldb, TB delta,
  TC *C0, TC *C1, int ldc, TC alpha0, TC alpha1,
  RANK_SEMIRINGKERNEL rank_semiringkernel,
  RANK_MICROKERNEL rank_microkernel,
  STRA_SEMIRINGKERNEL stra_semiringkernel,
  STRA_MICROKERNEL stra_microkernel,
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

          if ( delta == 0 || B1 == NULL ) {
            pack2D<true, PACK_NR>                            // packB
            (
              min( jb - j, NR ), pb, 
              &B0[ ( jc + j ) * ldb + pc ], ldb, &packB[ jp * pb ] 
            );
          } else {
            pack2D<true, PACK_NR>                            // packB
            (
              min( jb - j, NR ), pb, 
              &B0[ ( jc + j ) * ldb + pc ], &B1[ ( jc + j ) * ldb + pc ], ldb, delta, &packB[ jp * pb ] 
            );
          }

        }
        else
        {
          if ( delta == 0 || B1 == NULL ) {
            pack2D<false, PACK_NR>                           // packB (transB)
            (
              min( jb - j, NR ), pb, 
              &B0[ pc * ldb + ( jc + j ) ], ldb, &packB[ jp * pb ] 
            );
          } else {
            pack2D<false, PACK_NR>                           // packB (transB)
            (
              min( jb - j, NR ), pb, 
              &B0[ pc * ldb + ( jc + j ) ], &B1[ pc * ldb + ( jc + j ) ], ldb, delta, &packB[ jp * pb ] 
            );
          }

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

            if ( gamma == 0 || A1 == NULL ) {
              pack2D<false, PACK_MR>                         // packA 
              ( 
                min( ib - i, MR ), pb,
                &A0[ pc * lda + ( ic + i ) ], lda, &packA[ ip * pb ] 
              );
            } else {
              pack2D<false, PACK_MR>                         // packA 
              ( 
                min( ib - i, MR ), pb,
                &A0[ pc * lda + ( ic + i ) ], &A1[ pc * lda + ( ic + i ) ], lda, gamma, &packA[ ip * pb ] 
              );
            }

          }
          else
          {

            if ( gamma == 0 || A1 == NULL ) {
              pack2D<true, PACK_MR>                          // packA (transA)
              ( 
                min( ib - i, MR ), pb,
                &A0[ ( ic + i ) * lda + pc ], lda, &packA[ ip * pb ] 
              );
            } else {
              pack2D<true, PACK_MR>                          // packA (transA)
              ( 
                min( ib - i, MR ), pb,
                &A0[ ( ic + i ) * lda + pc ], &A1[ ( ic + i ) * lda + pc ], lda, gamma, &packA[ ip * pb ] 
              );
            }

          }
        }
        ic_comm.Barrier();

        if ( is_the_last_pc_iteration )                    // fused_macro_kernel
        {
          if ( alpha1 == 0 || C1 == NULL ) {

            hmlp::gkmx::fused_macro_kernel
            <KC, MR, NR, PACK_MR, PACK_NR, RANK_MICROKERNEL, TA, TB, TC, TV>
            (
              thread, 
              ic, jc, pc,
              ib, jb, pb,
              packA, 
              packB, 
              C0 + jc * ldc + ic, ldc,
              rank_microkernel
            );

          } else {
            fused_macro_kernel
            <KC, MR, NR, PACK_MR, PACK_NR, STRA_MICROKERNEL, TA, TB, TC, TV>
            (
              thread, 
              ic, jc, pc,
              ib, jb, pb,
              packA, 
              packB, 
              C0 + jc * ldc + ic,
              C1 + jc * ldc + ic, ldc, alpha0, alpha1,
              stra_microkernel
            );
          }

        }
        else                                               // semiring rank-k update
        {

          if ( alpha1 == 0 || C1 == NULL ) {
            hmlp::gkmx::rank_k_macro_kernel
            <KC, MR, NR, PACK_MR, PACK_NR, RANK_SEMIRINGKERNEL, TA, TB, TC, TV>
            (  
              thread, 
              ic, jc, pc,
              ib, jb, pb,
              packA,
              packB,
              C0 + jc * ldc + ic, ldc,
              rank_semiringkernel
            );
          } else {

            rank_k_macro_kernel
            <KC, MR, NR, PACK_MR, PACK_NR, STRA_SEMIRINGKERNEL, TA, TB, TC, TV>
            (  
              thread, 
              ic, jc, pc,
              ib, jb, pb,
              packA,
              packB,
              C0 + jc * ldc + ic,
              C1 + jc * ldc + ic, ldc, alpha0, alpha1,
              stra_semiringkernel
            );

          }

        }
        ic_comm.Barrier();                                 // sync all jr_id!!
      }                                                    // end 4th loop
      pc_comm.Barrier();
    }                                                      // end 5th loop
  }                                                        // end 6th loop
}                                                          // end strassen_internal





/**
 *
 *
 */ 
template<
  int MC, int NC, int KC, int MR, int NR, 
  int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
  bool USE_STRASSEN,
  typename RANK_SEMIRINGKERNEL, typename RANK_MICROKERNEL,
  typename STRA_SEMIRINGKERNEL, typename STRA_MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void strassen
(
  hmlpOperation_t transA, hmlpOperation_t transB,
  int m, int n, int k,
  TA *A, int lda,
  TB *B, int ldb,
  TC *C, int ldc,
  RANK_SEMIRINGKERNEL rank_semiringkernel,
  RANK_MICROKERNEL rank_microkernel,
  STRA_SEMIRINGKERNEL stra_semiringkernel,
  STRA_MICROKERNEL stra_microkernel
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



  int ms, ks, ns;
  int md, kd, nd;
  int mr, kr, nr;

  mr = m % ( 2 ), kr = k % ( 2 ), nr = n % ( 2 );
  md = m - mr, kd = k - kr, nd = n - nr;

  // Partition code.
  ms=md, ks=kd, ns=nd;
  double *A00, *A01, *A10, *A11;
  hmlp_acquire_mpart( ms, ks, A, lda, 2, 2, 0, 0, &A00 );
  hmlp_acquire_mpart( ms, ks, A, lda, 2, 2, 0, 1, &A01 );
  hmlp_acquire_mpart( ms, ks, A, lda, 2, 2, 1, 0, &A10 );
  hmlp_acquire_mpart( ms, ks, A, lda, 2, 2, 1, 1, &A11 );

  double *B00, *B01, *B10, *B11;
  hmlp_acquire_mpart( ms, ks, B, lda, 2, 2, 0, 0, &B00 );
  hmlp_acquire_mpart( ms, ks, B, lda, 2, 2, 0, 1, &B01 );
  hmlp_acquire_mpart( ms, ks, B, lda, 2, 2, 1, 0, &B10 );
  hmlp_acquire_mpart( ms, ks, B, lda, 2, 2, 1, 1, &B11 );

  double *C00, *C01, *C10, *C11;
  hmlp_acquire_mpart( ms, ks, C, lda, 2, 2, 0, 0, &C00 );
  hmlp_acquire_mpart( ms, ks, C, lda, 2, 2, 0, 1, &C01 );
  hmlp_acquire_mpart( ms, ks, C, lda, 2, 2, 1, 0, &C10 );
  hmlp_acquire_mpart( ms, ks, C, lda, 2, 2, 1, 1, &C11 );

  #pragma omp parallel num_threads( my_comm.GetNumThreads() ) 
  {
    worker thread( &my_comm );

    if ( USE_STRASSEN )
    {
      printf( "strassen: strassen algorithms haven't been implemented." );
      exit( 1 );
    }

    // M1: C00 = 1*C00+1*(A00+A11)(B00+B11); C11 = 1*C11+1*(A00+A11)(B00+B11)
    STRAPRIM( A00, A11, 1, B00, B11, 1, C00, C11, 1, 1 )
    // M2: C10 = 1*C10+1*(A10+A11)B00; C11 = 1*C11-1*(A10+A11)B00
    STRAPRIM( A10, A11, 1, B00, NULL, 0, C10, C11, 1, 1 )
    // M3: C01 = 1*C01+1*A00(B01-B11); C11 = 1*C11+1*A00(B01-B11)
    STRAPRIM( A00, NULL, 0, B01, B11, -1, C01, C11, 1, 1 )
    // M4: C00 = 1*C00+1*A11(B10-B00); C10 = 1*C10+1*A11(B10-B00)
    STRAPRIM( A11, NULL, 0, B10, B00, -1, C00, C10, 1, 1 )
    // M5: C00 = 1*C00-1*(A00+A01)B11; C01 = 1*C01+1*(A00+A01)B11
    STRAPRIM( A00, A01, 1, B11, NULL, 0, C00, C01, 1, 1 )
    // M6: C11 = 1*C11+(A10-A00)(B00+B01)
    STRAPRIM( A10, A00, -1, B00, B01, 1, C11, NULL, 1, 0 )
    // M7: C00 = 1*C00+(A01-A11)(B10+B11)
    STRAPRIM( A01, A11, -1, B10, B11, 1, C00, NULL, 1, 0 )

    //hmlp_dynamic_peeling( m, n, k, XA, lda, XB, ldb, XC, ldc, 2 * DGEMM_MR, 2, 2 * DGEMM_NR );

  }                                                        // end omp  

}                                                          // end strassen


}; // end namespace strassen
}; // end namespace hmlp

#endif // define STRASSEN_HPP


