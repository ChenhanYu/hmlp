#ifndef GSKNN_HXX
#define GSKNN_HXX

#ifdef USE_VML
#include <mkl.h>
#endif

#include <math.h>
#include <vector>

#include <hmlp.h>
#include <hmlp_internal.hpp>
#include <hmlp_blas_lapack.h>
#include <hmlp_packing.hpp>
#include <hmlp_util.hpp>
#include <hmlp_thread_communicator.hpp>
//#include <hmlp_thread_info.hpp>
#include <hmlp_runtime.hpp>

// For USE_STRASSEN
#include <strassen.hpp>

namespace hmlp
{
namespace gsknn
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
  TC *packC, int ldc,
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
          &packC[ j * ldc + i ], ldc,
          &aux
        );
      }
      else
      {
        double c[ MR * NR ] __attribute__((aligned(32)));
        double *cbuff = c;
        if ( pc ) {
          for ( auto jj = 0; jj < aux.jb; jj ++ )
            for ( auto ii = 0; ii < aux.ib; ii ++ )
              cbuff[ jj * MR + ii ] = packC[ ( j + jj ) * ldc + i + ii ];
        }
        semiringkernel
        (
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          cbuff, MR,
          &aux
        );
        for ( auto jj = 0; jj < aux.jb; jj ++ )
          for ( auto ii = 0; ii < aux.ib; ii ++ )
            packC[ ( j + jj ) * ldc + i + ii ] = cbuff[ jj * MR + ii ];
      }
    }                                                      // end 2nd loop
  }                                                        // end 3rd loop
}                                                          // end rank_k_macro_kernel

/**
 *
 */
template<
  int KC, int MR, int NR, int PACK_MR, int PACK_NR,
  typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void fused_macro_kernel
(
  worker &thread,
  int pc,
  int  m,  int n,  int k,  int r,
  TA *packA, TA *packA2,
  TB *packB, TB *packB2,
  int *bmap,
  TV *D,  int *I,  int ldr,
  TC *packC, int ldc,
  MICROKERNEL microkernel
)
{
  double c[ MR * NR ] __attribute__((aligned(32)));
  double *cbuff = c;
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
    aux.ldr      = ldr;
    aux.jb       = min( n - j, NR );

    for ( int i  = loop2nd.beg(), ip  = pack2nd.beg();
              i  < loop2nd.end();
              i += loop2nd.inc(), ip += pack2nd.inc() )    // beg 2nd loop
    {
      aux.ib = min( m - i, MR );
      aux.I  = I + i * ldr;
      aux.D  = D + i * ldr;
      if ( i + MR >= m )
      {
        aux.b_next += ic_comm.GetNumThreads() * PACK_NR * k;
      }
      if ( pc ) {
        for ( auto jj = 0; jj < aux.jb; jj ++ )
          for ( auto ii = 0; ii < aux.ib; ii ++ )
            cbuff[ jj * MR + ii ] = packC[ ( j + jj ) * ldc + i + ii ];
      }
      microkernel
      (
        k,
        r,
        packA  + ip * k,
        packA2 + ip,
        packB  + jp * k,
        packB2 + jp,
        cbuff,
        &aux,
        bmap   + j
      );
      if ( pc ) {
        for ( auto jj = 0; jj < aux.jb; jj ++ )
          for ( auto ii = 0; ii < aux.ib; ii ++ )
            packC[ ( j + jj ) * ldc + i + ii ] = cbuff[ jj * MR + ii ];
      }
    }                                                      // end 2nd loop
  }                                                        // end 3rd loop
}                                                          // end fused_macro_kernel


/**
 *
 */
template<
  int MC, int NC, int KC, int MR, int NR,
  int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
  bool USE_STRASSEN,
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void gsknn_internal
(
  worker &thread,
  int m, int n, int k, int k_stra, int r,
  TA *A, TA *A2, int *amap,
  TB *B, TB *B2, int *bmap,
  TV *D,         int *I,
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel,
  TA *packA, TA *packA2,
  TB *packB, TB *packB2,
  TC *packC, int ldpackc, int padn,
  int ldr
)
{

  packA  += ( thread.jc_id * thread.ic_nt                ) * PACK_MC * KC
          + ( thread.ic_id                               ) * PACK_MC * KC;
  packA2 += ( thread.jc_id * thread.ic_nt + thread.ic_id ) * PACK_MC;
  packB  += ( thread.jc_id                               ) * PACK_NC * KC;
  packB2 += ( thread.jc_id                               ) * PACK_NC;

  auto loop6th = GetRange( 0, n, NC );
  auto loop5th = GetRange( k_stra, k, KC );
  auto loop4th = GetRange( 0, m, MC, thread.ic_id, thread.ic_nt );

  for ( int jc  = loop6th.beg();
            jc  < loop6th.end();
            jc += loop6th.inc() )                          // beg 6th loop
  {
    auto jb = min( n - jc, NC );

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
        pack2D<true, PACK_NR>                              // packB
        (
          min( jb - j, NR ), pb,
          &B[ pc ], k, &bmap[ jc + j ], &packB[ jp * pb ]
        );


        if ( is_the_last_pc_iteration )
        {

          pack2D<true, PACK_NR>                           // packB2
          (
            min( jb - j, NR ), 1,
            &B2[ 0 ], 1, &bmap[ jc + j ], &packB2[ jp * 1 ]
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

        auto looppkA = GetRange( 0, ib,      MR, thread.jr_id, 1 );
        auto packpkA = GetRange( 0, ib, PACK_MR, thread.jr_id, 1 );

        for ( int i   = looppkA.beg(), ip  = packpkA.beg();
                  i   < looppkA.end();
                  i  += looppkA.inc(), ip += packpkA.inc() )
        {
          pack2D<true, PACK_MR>                            // packA
          (
            min( ib - i, MR ), pb,
            &A[ pc ], k, &amap[ ic + i ], &packA[ ip * pb ]
          );

          if ( is_the_last_pc_iteration )
          {
            pack2D<true, PACK_MR>                        // packA2
            (
              min( ib - i, MR ), 1,
              &A2[ 0 ], 1, &amap[ ic + i ], &packA2[ ip * 1 ]
            );

          }
        }


        ic_comm.Barrier();
        if ( pc + KC  < k ) {
          rank_k_macro_kernel
          <KC, MR, NR, PACK_MR, PACK_NR, SEMIRINGKERNEL, TA, TB, TC, TV>
          (
            thread,
            ic, jc, pc,
            ib, jb, pb,
            packA,
            packB,
            packC + jc * ldpackc + ic,
            ldpackc,
            semiringkernel
          );
        }
        else {
          fused_macro_kernel
          <KC, MR, NR, PACK_MR, PACK_NR, MICROKERNEL, TA, TB, TC, TV>
          (
            thread,
            pc,
            ib, jb, pb, r,
            packA, packA2,
            packB, packB2, bmap + jc,
            D + ic * ldr,  I + ic * ldr,  ldr,
            packC + jc * ldpackc + ic,
            ldpackc,
            microkernel
          );
        }

        ic_comm.Barrier();                                 // sync all jr_id!!

      }                                                    // end 4th loop
      pc_comm.Barrier();
    }                                                      // end 5th loop
  }                                                        // end 6th loop
}                                                          // end gsknn_internal





/**
 *
 */
template<
  int MC, int NC, int KC, int MR, int NR,
  int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
  bool USE_STRASSEN,
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void gsknn(
    int m, int n, int k, int r,
    TA *A, TA *A2, int *amap,
    TB *B, TB *B2, int *bmap,
    TV *D,         int *I,
    SEMIRINGKERNEL semiringkernel,
    MICROKERNEL microkernel
    )
{
  int ic_nt = 1;
  int k_stra = 0;
  int ldpackc = 0, padn = 0;
  int ldr = 0;
  char *str;

  TA *packA_buff = NULL, *packA2_buff = NULL;
  TB *packB_buff = NULL, *packB2_buff = NULL;
  TC *packC_buff = NULL;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  // Check the environment variable.
  str = getenv( "KS_IC_NT" );
  if ( str ) ic_nt = (int)strtol( str, NULL, 10 );

  ldpackc = m;
  ldr = r;

  // allocate packing memory
  packA_buff  = hmlp_malloc<ALIGN_SIZE, TA>( KC, ( PACK_MC + 1 ) * ic_nt,         sizeof(TA) );
  packB_buff  = hmlp_malloc<ALIGN_SIZE, TB>( KC, ( PACK_NC + 1 ),                 sizeof(TB) );
  packA2_buff = hmlp_malloc<ALIGN_SIZE, TA>(  1, ( PACK_MC + 1 ) * ic_nt,         sizeof(TA) );
  packB2_buff = hmlp_malloc<ALIGN_SIZE, TB>(  1, ( PACK_NC + 1 ),                 sizeof(TB) );
  if ( k > KC ) {
    packC_buff = hmlp_malloc<ALIGN_SIZE, TC>(  m, n, sizeof(TC) );
  }

  // allocate tree communicator
  thread_communicator my_comm( 1, 1, ic_nt, 1 );

  if ( USE_STRASSEN )
  {
    k_stra = k - k % KC;

    if ( k_stra == k ) k_stra -= KC;

    if ( k_stra )
    {
      #pragma omp parallel for
      for ( int i = 0; i < m * n; i ++ ) packC_buff[ i ] = 0.0;
    }

  }

  #pragma omp parallel num_threads( my_comm.GetNumThreads() )
  {
    worker thread( &my_comm );

    if ( USE_STRASSEN && k > KC )
    {
      strassen::strassen_internal
      <MC, NC, KC, MR, NR,
      PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
      USE_STRASSEN,
      SEMIRINGKERNEL, SEMIRINGKERNEL,
      TA, TB, TC, TV>
      (
        thread,
        HMLP_OP_T, HMLP_OP_N,
        m, n, k_stra,
        A, k, amap,
        B, k, bmap,
        packC_buff, ldpackc,
        semiringkernel, semiringkernel,
        NC, PACK_NC,
        packA_buff,
        packB_buff
      );
    }

    gsknn_internal
    <MC, NC, KC, MR, NR, PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    USE_STRASSEN,
    SEMIRINGKERNEL, MICROKERNEL,
    TA, TB, TC, TB>
    (
      thread,
      m, n, k, k_stra, r,
      A, A2, amap,
      B, B2, bmap,
      D,     I,
      semiringkernel, microkernel,
      packA_buff, packA2_buff,
      packB_buff, packB2_buff,
      packC_buff, ldpackc, padn,
      ldr
    );

  }                                                        // end omp region

  hmlp_free( packA_buff );
  hmlp_free( packB_buff );
  hmlp_free( packA2_buff );
  hmlp_free( packB2_buff );
  hmlp_free( packC_buff );
}                                                          // end gsknn


/**
 *
 */
template<typename T>
void gsknn_ref
(
  int m, int n, int k, int r,
  T *A, T *A2, int *amap,
  T *B, T *B2, int *bmap,
  T *D,        int *I
)
{
  int    i, j, p;
  double beg, time_collect, time_dgemm, time_square, time_heap;
  std::vector<T> packA, packB, C;
  double fneg2 = -2.0, fzero = 0.0, fone = 1.0;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  packA.resize( k * m );
  packB.resize( k * n );
  C.resize( m * n );

  // Collect As from A and B.
  beg = omp_get_wtime();
  #pragma omp parallel for private( p )
  for ( i = 0; i < m; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
      packA[ i * k + p ] = A[ amap[ i ] * k + p ];
    }
  }
  #pragma omp parallel for private( p )
  for ( j = 0; j < n; j ++ ) {
    for ( p = 0; p < k; p ++ ) {
      packB[ j * k + p ] = B[ bmap[ j ] * k + p ];
    }
  }
  time_collect = omp_get_wtime() - beg;

  // Compute the inner-product term.
  beg = omp_get_wtime();
  #ifdef USE_BLAS
    xgemm
    (
      "T", "N",
      m, n, k,
      fone,         packA.data(), k,
                    packB.data(), k,
      fzero,        C.data(),     m
    );
  #else
    #pragma omp parallel for private( i, p )
    for ( j = 0; j < n; j ++ ) {
      for ( i = 0; i < m; i ++ ) {
        C[ j * m + i ] = 0.0;
        for ( p = 0; p < k; p ++ ) {
          C[ j * m + i ] += packA[ i * k + p ] * packB[ j * k + p ];
        }
      }
    }
  #endif
  time_dgemm = omp_get_wtime() - beg;

  beg = omp_get_wtime();
  #pragma omp parallel for private( i )
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < m; i ++ ) {
      C[ j * m + i ] *= -2.0;
      C[ j * m + i ] += A2[ amap[ i ] ];
      C[ j * m + i ] += B2[ bmap[ j ] ];
    }
  }
  time_square = omp_get_wtime() - beg;

  // Pure C Max Heap implementation.
  beg = omp_get_wtime();
  #pragma omp parallel for schedule( dynamic )
  for ( j = 0; j < n; j ++ ) {
    heap_select<T>( m, r, &C[ j * m ], amap, &D[ j * r ], &I[ j * r ] );
  }
  time_heap = omp_get_wtime() - beg;

} // end void gsknn_ref


}; // end namespace gsknn
}; // end namespace hmlp

#endif // define GSKNN_HXX
