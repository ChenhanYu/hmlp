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
#include <hmlp_thread_info.hpp>
#include <hmlp_runtime.hpp>

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
  TC *D,  int *I,  int ldr,
  TV *packC, int ldc,
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
    struct aux_d<TA, TB, TC, TV> aux;
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
      microkernel
      (
        k,
        r,
        packA  + ip * k,
        packA2 + ip,
        packB  + jp * k,
        packB2 + jp,
        packC  + j * ldc + i * NR,                         // packed
        &aux,
        bmap   + j
      );
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
  int m, int n, int k, int r,
  TA *A, TA *A2, int *amap,
  TB *B, TB *B2, int *bmap,
  TC *D,         int *I,
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel,
  TA *packA, TA *packA2,
  TB *packB, TB *packB2,
  TV *packC, int ldpackc, int padn,
  int ldr
)
{
  packA  += ( thread.jc_id * thread.ic_nt                ) * PACK_MC * KC
          + ( thread.ic_id                               ) * PACK_MC * KC;
  packA2 += ( thread.jc_id * thread.ic_nt + thread.ic_id ) * PACK_MC;
  packB  += ( thread.jc_id                               ) * PACK_NC * KC;
  packB2 += ( thread.jc_id                               ) * PACK_NC;
  // packC  += ( thread.jc_id                               ) * ldpackc * padn;

  auto loop6th = GetRange( 0, n, NC, thread.jc_id, thread.jc_nt );
  auto loop5th = GetRange( 0, k, KC );
  auto loop4th = GetRange( 0, m, MC, thread.ic_id, thread.ic_nt );

  for ( int jc  = loop6th.beg();
            jc  < loop6th.end();
            jc += loop6th.inc() )                          // beg 6th loop
  {
    auto &jc_comm = *thread.jc_comm;
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

          pack2D<true, PACK_NR>                          // packB2
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

        auto looppkA = GetRange( 0, ib,      MR, thread.jr_id, thread.jr_nt );
        auto packpkA = GetRange( 0, ib, PACK_MR, thread.jr_id, thread.jr_nt );

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


        fused_macro_kernel
        <KC, MR, NR, PACK_MR, PACK_NR, MICROKERNEL, TA, TB, TC, TV>
        (
          thread,
          pc,
          ib, jb, pb, r,
          packA, packA2,
          packB, packB2, bmap + jc,
          D + ic * ldr,  I + ic * ldr,  ldr,
          packC + ic * padn,                             // packed
          ( ( ib - 1 ) / MR + 1 ) * MR,                  // packed ldc
          microkernel
        );
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
    TC *D,         int *I,
    SEMIRINGKERNEL semiringkernel,
    MICROKERNEL microkernel
    )
{
  int ic_nt = 1;
  int ldpackc = 0, padn = 0;
  int ldr = 0;
  char *str;

  TA *packA_buff = NULL, *packA2_buff = NULL;
  TB *packB_buff = NULL, *packB2_buff = NULL;
  TV *packC_buff = NULL;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  // Check the environment variable.
  str = getenv( "KS_IC_NT" );
  if ( str ) ic_nt = (int)strtol( str, NULL, 10 );

  ldr = r;

  // allocate packing memory
  packA_buff  = hmlp_malloc<ALIGN_SIZE, TA>( KC, ( PACK_MC + 1 ) * ic_nt,         sizeof(TA) );
  packB_buff  = hmlp_malloc<ALIGN_SIZE, TB>( KC, ( PACK_NC + 1 ),                 sizeof(TB) );
  packA2_buff = hmlp_malloc<ALIGN_SIZE, TA>(  1, ( PACK_MC + 1 ) * ic_nt,         sizeof(TA) );
  packB2_buff = hmlp_malloc<ALIGN_SIZE, TB>(  1, ( PACK_NC + 1 ),                 sizeof(TB) );

  // Temporary bufferm <TV> to store the semi-ring rank-k update
  if ( k > KC )
  {
    ldpackc  = ( ( m - 1 ) / PACK_MR + 1 ) * PACK_MR;
    padn = PACK_NC;
    if ( n < PACK_NC ) padn = ( ( n - 1 ) / PACK_NR + 1 ) * PACK_NR ;
    packC_buff = hmlp_malloc<ALIGN_SIZE, TV>( ldpackc, padn, sizeof(TV) );
  }

  // allocate tree communicator
  thread_communicator my_comm( ic_nt, ic_nt, ic_nt, ic_nt );


  #pragma omp parallel num_threads( my_comm.GetNumThreads() )
  {
    worker thread( &my_comm );

    if ( USE_STRASSEN )
    {
      printf( "gsknn: strassen algorithms haven't been implemented." );
      exit( 1 );
    }

    gsknn_internal
    <MC, NC, KC, MR, NR, PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    USE_STRASSEN,
    SEMIRINGKERNEL, MICROKERNEL,
    TA, TB, TC, TB>
    (
      thread,
      m, n, k, r,
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
  double beg, time_heap;
  std::vector<T> packA, packB, C;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  packA.resize( k * m );
  packB.resize( k * n );
  C.resize( m * n );

  // Pure C Max Heap implementation.
  beg = omp_get_wtime();
  #pragma omp parallel for schedule( dynamic )
  for ( j = 0; j < n; j ++ ) {
    // heapSelect_d( m, r, &C[ j * m ], alpha, &D[ j * r ], &I[ j * r ] );
  }
  time_heap = omp_get_wtime() - beg;

} // end void gsknn_ref


}; // end namespace gsknn
}; // end namespace hmlp

#endif // define GSKNN_HXX
