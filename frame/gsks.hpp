#ifndef GSKS_HXX
#define GSKS_HXX

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
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>

namespace hmlp
{
namespace gsks
{

#define min( i, j ) ( (i)<(j) ? (i): (j) )
#define KS_RHS 1

/**
 *
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
  TV *packC, int ldc,
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
    aux.do_packC = 1;
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
      semiringkernel
      (
        k,
        &packA[ ip * k ],
        &packB[ jp * k ],
        //&packC[ j * ldc + i * NR ], ldc,
        &packC[ j * ldc + i * NR ], MR,
        &aux
      );
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
  ks_t *kernel,
  Worker &thread,
  int ic, int jc, int pc,
  int  m,  int n,  int k,
  TC *packu,
  TA *packA, TA *packA2, TV *packAh,
  TB *packB, TB *packB2, TV *packBh,
  TC *packw,
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
    struct aux_s<TA, TB, TC, TV> aux;
    aux.pc       = pc;
    aux.b_next   = packB;
    aux.do_packC = 1;
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
      microkernel
      (
        kernel,
        k,
        KS_RHS,
        packu  + ip * KS_RHS,
        packA  + ip * k,
        packA2 + ip,
        packB  + jp * k,
        packB2 + jp,
        packw  + jp * KS_RHS,
        packC  + j * ldc + i * NR, MR,                     // packed
        &aux
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
  bool USE_L2NORM, bool USE_VAR_BANDWIDTH, bool USE_STRASSEN,
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void gsks_internal
(
  Worker &thread,
  ks_t *kernel,
  int m, int n, int k,
  TC *u,         int *umap, 
  TA *A, TA *A2, int *amap,
  TB *B, TB *B2, int *bmap,
  TC *w,         int *wmap, 
  SEMIRINGKERNEL semiringkernel,
  MICROKERNEL microkernel,
  int nc, int pack_nc,
  TC *packu,
  TA *packA, TA *packA2, TA *packAh,
  TB *packB, TB *packB2, TB *packBh,
  TC *packw,
  TV *packC, int ldpackc, int padn
)
{
  packu  += ( thread.jc_id * thread.ic_nt * thread.jr_nt ) * PACK_MC * KS_RHS
          + ( thread.ic_id * thread.jr_nt + thread.jr_id ) * PACK_MC * KS_RHS;
  packA  += ( thread.jc_id * thread.ic_nt                ) * PACK_MC * KC
          + ( thread.ic_id                               ) * PACK_MC * KC;
  packA2 += ( thread.jc_id * thread.ic_nt + thread.ic_id ) * PACK_MC;
  packAh += ( thread.jc_id * thread.ic_nt + thread.ic_id ) * PACK_MC;
  packB  += ( thread.jc_id                               ) * pack_nc * KC;
  packB2 += ( thread.jc_id                               ) * pack_nc;
  packBh += ( thread.jc_id                               ) * pack_nc;
  packw  += ( thread.jc_id                               ) * pack_nc;
  packC  += ( thread.jc_id                               ) * ldpackc * padn;

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
        pack2D<true, PACK_NR>                              // packB
        (
          min( jb - j, NR ), pb, 
          &B[ pc ], k, &bmap[ jc + j ], &packB[ jp * pb ] 
        );


        if ( is_the_last_pc_iteration )
        {
          pack2D<true, PACK_NR, true>                      // packw
          (
            min( jb - j, NR ), 1, 
            &w[ 0 ], 1, &wmap[ jc + j ], &packw[ jp * 1 ] 
          );

          if ( USE_L2NORM )
          {
            pack2D<true, PACK_NR>                          // packB2
            (
              min( jb - j, NR ), 1, 
              &B2[ 0 ], 1, &bmap[ jc + j ], &packB2[ jp * 1 ] 
            );
          }

          if ( USE_VAR_BANDWIDTH )
          {
            pack2D<true, PACK_NR>                          // packBh
            (
              min( jb - j, NR ), 1, 
              kernel->hj, 1, &bmap[ jc + j ], &packBh[ jp * 1 ] 
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
          pack2D<true, PACK_MR>                            // packA 
          ( 
            min( ib - i, MR ), pb,
            &A[ pc ], k, &amap[ ic + i ], &packA[ ip * pb ] 
          );

          if ( is_the_last_pc_iteration )               
          {
            if ( USE_L2NORM )
            {
              pack2D<true, PACK_MR>                        // packA2
              (
                min( ib - i, MR ), 1, 
                &A2[ 0 ], 1, &amap[ ic + i ], &packA2[ ip * 1 ] 
              );
            }

            if ( USE_VAR_BANDWIDTH )                       // variable bandwidths
            {
              pack2D<true, PACK_MR>                        // packAh
              (
                min( ib - i, MR ), 1, 
                kernel->hi, 1, &amap[ ic + i ], &packAh[ ip * 1 ] 
              );
            }
          }
        }

        if ( is_the_last_pc_iteration )                    // Initialize packu to zeros.
        {
          for ( auto i = 0, ip = 0; i < ib; i += MR, ip += PACK_MR )
          {
            for ( auto ir = 0; ir < min( ib - i, MR ); ir ++ )
            {
              packu[ ip + ir ] = 0.0;
            }
          }
        }
        ic_comm.Barrier();


        if ( is_the_last_pc_iteration )                    // fused_macro_kernel
        {
          fused_macro_kernel
          <KC, MR, NR, PACK_MR, PACK_NR, MICROKERNEL, TA, TB, TC, TV>
          (
            kernel,
            thread, 
            ic, jc, pc,
            ib, jb, pb,
            packu,
            packA, packA2, packAh,
            packB, packB2, packBh,
            packw,
            packC + ic * padn,                             // packed
            ( ( ib - 1 ) / MR + 1 ) * MR,                  // packed ldc
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
            packC + ic * padn,                             // packed
            ( ( ib - 1 ) / MR + 1 ) * MR,                  // packed ldc
            semiringkernel
          );
        }
        ic_comm.Barrier();                                 // sync all jr_id!!

        if ( is_the_last_pc_iteration )
        {
          for ( auto i = 0, ip = 0; i < ib; i += MR, ip += PACK_MR )
          {
            for ( auto ir = 0; ir < min( ib - i, MR ); ir ++ )
            {
              TC *uptr = &( u[ umap[ ic + i + ir ] ] );
              #pragma omp atomic update                    // concurrent write
              *uptr += packu[ ip + ir ];
            }
          }
          ic_comm.Barrier();                               // sync all jr_id!!
        }
      }                                                    // end 4th loop
      pc_comm.Barrier();
    }                                                      // end 5th loop
  }                                                        // end 6th loop
}                                                          // end gsks_internal





/**
 *
 */ 
template<
  int MC, int NC, int KC, int MR, int NR, 
  int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
  bool USE_L2NORM, bool USE_VAR_BANDWIDTH, bool USE_STRASSEN,
  typename SEMIRINGKERNEL, typename MICROKERNEL,
  typename TA, typename TB, typename TC, typename TV>
void gsks(
    ks_t *kernel,
    int m, int n, int k,
    TC *u,         int *umap,
    TA *A, TA *A2, int *amap,
    TB *B, TB *B2, int *bmap,
    TC *w,         int *wmap,
    SEMIRINGKERNEL semiringkernel,
    MICROKERNEL microkernel
    )
{
  int jc_nt = 1, pc_nt = 1, ic_nt = 1, jr_nt = 1;
  int ldpackc = 0, padn = 0, nc = NC, pack_nc = PACK_NC;
  char *str;

  TC *packu_buff = NULL;
  TA *packA_buff = NULL, *packA2_buff = NULL, *packAh_buff = NULL;
  TB *packB_buff = NULL, *packB2_buff = NULL, *packBh_buff = NULL;
  TC *packw_buff = NULL;
  TV *packC_buff = NULL;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  // Check the environment variable.
  jc_nt = hmlp_read_nway_from_env( "KS_JC_NT" );
  ic_nt = hmlp_read_nway_from_env( "KS_IC_NT" );
  jr_nt = hmlp_read_nway_from_env( "KS_JR_NT" );

  if ( jc_nt > 1 )
  {
    nc = ( ( n - 1 ) / ( NR * jc_nt ) + 1 ) * NR;
    pack_nc = ( nc / NR ) * PACK_NR;
  }

  // allocate packing memory
  {
    packA_buff  = hmlp_malloc<ALIGN_SIZE, TA>( KC, ( PACK_MC + 1 ) * jc_nt * ic_nt,         sizeof(TA) );
    packB_buff  = hmlp_malloc<ALIGN_SIZE, TB>( KC, ( pack_nc + 1 ) * jc_nt,                 sizeof(TB) ); 
    packu_buff  = hmlp_malloc<ALIGN_SIZE, TC>(  1, ( PACK_MC + 1 ) * jc_nt * ic_nt * jr_nt, sizeof(TC) );
    packw_buff  = hmlp_malloc<ALIGN_SIZE, TC>(  1, ( pack_nc + 1 ) * jc_nt,                 sizeof(TC) ); 
  }

  // allocate extra packing buffer
  if ( USE_L2NORM )
  {
    packA2_buff = hmlp_malloc<ALIGN_SIZE, TA>(  1, ( PACK_MC + 1 ) * jc_nt * ic_nt,         sizeof(TA) );
    packB2_buff = hmlp_malloc<ALIGN_SIZE, TB>(  1, ( pack_nc + 1 ) * jc_nt,                 sizeof(TB) ); 
  }

  if ( USE_VAR_BANDWIDTH )
  {
    packAh_buff = hmlp_malloc<ALIGN_SIZE, TA>(  1, ( PACK_MC + 1 ) * jc_nt * ic_nt,         sizeof(TA) );
    packBh_buff = hmlp_malloc<ALIGN_SIZE, TB>(  1, ( pack_nc + 1 ) * jc_nt,                 sizeof(TB) ); 
  }

  // Temporary bufferm <TV> to store the semi-ring rank-k update
  if ( k > KC )
  {
    ldpackc  = ( ( m - 1 ) / PACK_MR + 1 ) * PACK_MR;
    padn = pack_nc;
    if ( n < nc ) padn = ( ( n - 1 ) / PACK_NR + 1 ) * PACK_NR ;
    packC_buff = hmlp_malloc<ALIGN_SIZE, TV>( ldpackc, padn * jc_nt, sizeof(TV) );
  }

  // allocate tree communicator
  thread_communicator my_comm( jc_nt, pc_nt, ic_nt, jr_nt );


  #pragma omp parallel num_threads( my_comm.GetNumThreads() ) 
  {
    Worker thread( &my_comm );

    if ( USE_STRASSEN )
    {
      printf( "gsks: strassen algorithms haven't been implemented." );
      exit( 1 );
    }

    gsks_internal
    <MC, NC, KC, MR, NR, PACK_MC, PACK_NC, PACK_MR, PACK_NR, ALIGN_SIZE,
    USE_L2NORM, USE_VAR_BANDWIDTH, USE_STRASSEN,
    SEMIRINGKERNEL, MICROKERNEL,
    TA, TB, TC, TB>
    (
      thread,
      kernel,
      m, n, k,
      u,     umap,
      A, A2, amap,
      B, B2, bmap,
      w,     wmap,
      semiringkernel, microkernel,
      nc, pack_nc,
      packu_buff,
      packA_buff, packA2_buff, packAh_buff,
      packB_buff, packB2_buff, packBh_buff,
      packw_buff,
      packC_buff, ldpackc, padn
    );




/*
    TC *packu = NULL;
    TA *packA = NULL, *packA2 = NULL, *packAh = NULL;
    TB *packB = NULL, *packB2 = NULL, *packBh = NULL;
    TC *packw = NULL;
    TV *packC = NULL;

    packu  = packu_buff  + ( thread.jc_id * ic_nt * jr_nt + thread.ic_id * jr_nt + thread.jr_id ) * PACK_MC * KS_RHS;
    packA  = NULL;
    packA2 = packA2_buff + ( thread.jc_id * ic_nt + thread.ic_id ) * PACK_MC;
    packAh = packAh_buff + ( thread.jc_id * ic_nt + thread.ic_id ) * PACK_MC;
    packB  = packB_buff  + ( thread.jc_id                        ) * pack_nc * KC;
    packB2 = packB2_buff + ( thread.jc_id                        ) * pack_nc;
    packBh = packBh_buff + ( thread.jc_id                        ) * pack_nc;
    packw  = packw_buff  + ( thread.jc_id                        ) * pack_nc;
    packC  = packC_buff  + ( thread.jc_id                        ) * ldpackc * padn;

    for ( auto jc  = thread.jc_id * nc; 
               jc  < n; 
               jc += jc_nt * nc )                          // beg 6th loop 
    {
      auto &jc_comm = *thread.jc_comm;
      auto jb = min( n - jc, nc );

      for ( auto pc = 0; pc < k; pc += KC )                // beg 5th loop 
      {
        auto &pc_comm = *thread.pc_comm;
        auto pb = min( k - pc, KC );
        auto is_the_last_pc_iteration = ( pc + KC >= k );

        packA = packA_buff + thread.jc_id * ic_nt * PACK_MC * KC 
                           + thread.ic_id         * PACK_MC * pb;

        for ( auto j   = thread.ic_jr * NR, 
                   jp  = thread.ic_jr * PACK_NR; 
                   j   < jb; 
                   j  += pc_comm.GetNumThreads() * NR, 
                   jp += pc_comm.GetNumThreads() * PACK_NR ) 
        {
          pack2D<true, PACK_NR>                            // packB
          (
            min( jb - j, NR ), pb, 
            &B[ pc ], k, &bmap[ jc + j ], &packB[ jp * pb ] 
          );


          if ( is_the_last_pc_iteration )
          {
            pack2D<true, PACK_NR, true>                    // packw
            (
              min( jb - j, NR ), 1, 
              &w[ 0 ], 1, &wmap[ jc + j ], &packw[ jp * 1 ] 
            );

            if ( USE_L2NORM )
            {
              pack2D<true, PACK_NR>                        // packB2
              (
                min( jb - j, NR ), 1, 
                &B2[ 0 ], 1, &bmap[ jc + j ], &packB2[ jp * 1 ] 
              );
            }

            if ( USE_VAR_BANDWIDTH )
            {
              pack2D<true, PACK_NR>                        // packBh
              (
                min( jb - j, NR ), 1, 
                kernel->hj, 1, &bmap[ jc + j ], &packBh[ jp * 1 ] 
              );
            }
          }
        }
        pc_comm.Barrier();

        for ( auto ic  = thread.ic_id * MC; 
                   ic  < m; 
                   ic += ic_nt * MC )                      // beg 4th loop
        {
          auto &ic_comm = *thread.ic_comm;
          auto ib = min( m - ic, MC );

          for ( auto i   = thread.jr_id * MR, 
                     ip  = thread.jr_id * PACK_MR; 
                     i   < ib; 
                     i  += jr_nt * MR, 
                     ip += jr_nt * PACK_MR )     
          {
            pack2D<true, PACK_MR>                          // packA 
            ( 
              min( ib - i, MR ), pb,
              &A[ pc ], k, &amap[ ic + i ], &packA[ ip * pb ] 
            );

            if ( is_the_last_pc_iteration )               
            {
              if ( USE_L2NORM )
              {
                pack2D<true, PACK_MR>                      // packA2
                (
                  min( ib - i, MR ), 1, 
                  &A2[ 0 ], 1, &amap[ ic + i ], &packA2[ ip * 1 ] 
                );
              }

              if ( USE_VAR_BANDWIDTH )                     // variable bandwidths
              {
                pack2D<true, PACK_MR>                      // packAh
                (
                  min( ib - i, MR ), 1, 
                  kernel->hi, 1, &amap[ ic + i ], &packAh[ ip * 1 ] 
                );
              }
            }
          }

          if ( is_the_last_pc_iteration )                  // Initialize packu to zeros.
          {
            for ( auto i = 0, ip = 0; i < ib; i += MR, ip += PACK_MR )
            {
              for ( auto ir = 0; ir < min( ib - i, MR ); ir ++ )
              {
                packu[ ip + ir ] = 0.0;
              }
            }
          }
          ic_comm.Barrier();


          if ( is_the_last_pc_iteration )                  // fused_macro_kernel
          {
            fused_macro_kernel
            <KC, MR, NR, PACK_MR, PACK_NR, MICROKERNEL, TA, TB, TC, TV>
            (
              kernel,
              thread, 
              ic, jc, pc,
              ib, jb, pb,
              packu,
              packA, packA2, packAh,
              packB, packB2, packBh,
              packw,
              packC + ic * padn,                           // packed
              ( ( ib - 1 ) / MR + 1 ) * MR,                // packed ldc
              microkernel
            );
          }
          else                                             // semiring rank-k update
          {
            rank_k_macro_kernel
            <KC, MR, NR, PACK_MR, PACK_NR, SEMIRINGKERNEL, TA, TB, TC, TV>
            (  
              thread, 
              ic, jc, pc,
              ib, jb, pb,
              packA,
              packB,
              packC + ic * padn,                           // packed
              ( ( ib - 1 ) / MR + 1 ) * MR,                // packed ldc
              semiringkernel
            );
          }
          ic_comm.Barrier();                               // sync all jr_id!!

          if ( is_the_last_pc_iteration )
          {
            for ( auto i = 0, ip = 0; i < ib; i += MR, ip += PACK_MR )
            {
              for ( auto ir = 0; ir < min( ib - i, MR ); ir ++ )
              {
                TC *uptr = &( u[ umap[ ic + i + ir ] ] );
                #pragma omp atomic update                  // concurrent write
                *uptr += packu[ ip + ir ];
              }
            }
            ic_comm.Barrier();                             // sync all jr_id!!
          }
        }                                                  // end 4th loop
        pc_comm.Barrier();
      }                                                    // end 5th loop
    }                                                      // end 6th loop
  */
  }                                                        // end omp region

  hmlp_free( packA_buff );
  hmlp_free( packB_buff );
  hmlp_free( packu_buff );
  hmlp_free( packw_buff );
  if ( USE_L2NORM )
  {
    hmlp_free( packA2_buff );
    hmlp_free( packB2_buff );
  }
}                                                          // end gsks


/**
 *
 */ 
template<typename T>
void gsks_ref
(
  ks_t *kernel,
  int m, int n, int k,
  T *u,        int *umap,
  T *A, T *A2, int *amap,
  T *B, T *B2, int *bmap,
  T *w,        int *wmap 
)
{
  int nrhs = KS_RHS;
  T rank_k_scale, fone = 1.0, fzero = 0.0;
  std::vector<T> packA, packB, C, packu, packw;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  packA.resize( k * m );
  packB.resize( k * n );
  C.resize( m * n );
  packu.resize( m );
  packw.resize( n );

  switch ( kernel->type )
  {
    case KS_GAUSSIAN:
      rank_k_scale = -2.0;
      break;
    default:
      exit( 1 );
  }

  /*
   *  Collect packA and packu
   */ 
  #pragma omp parallel for
  for ( int i = 0; i < m; i ++ ) 
  {
    for ( int p = 0; p < k; p ++ ) 
    {
      packA[ i * k + p ] = A[ amap[ i ] * k + p ];
    }
    for ( int p = 0; p < KS_RHS; p ++ ) 
    {
      packu[ p * m + i ] = u[ umap[ i ] * KS_RHS + p ];
    }
  }

  /*
   *  Collect packB and packw
   */ 
  #pragma omp parallel for
  for ( int j = 0; j < n; j ++ ) 
  {
    for ( int p = 0; p < k; p ++ ) 
    {
      packB[ j * k + p ] = B[ bmap[ j ] * k + p ];
    }
    for ( int p = 0; p < KS_RHS; p ++ ) 
    {
      packw[ p * n + j ] = w[ wmap[ j ] * KS_RHS + p ];
    }
  }

  /*
   *  C = -2.0 * A^T * B (GEMM)
   */ 
#ifdef USE_BLAS
  xgemm
  ( 
    "T", "N", 
    m, n, k, 
    rank_k_scale, packA.data(), k,
                  packB.data(), k, 
    fzero,        C.data(),     m 
  );
#else
  #pragma omp parallel for
  for ( int j = 0; j < n; j ++ ) 
  {
    for ( int i = 0; i < m; i ++ ) 
    {
      C[ j * m + i ] = 0.0;
      for ( int p = 0; p < k; p ++ ) 
      {
        C[ j * m + i ] += packA[ i * k + p ] * packB[ j * k + p ];
      }
    }
  }
  #pragma omp parallel for
  for ( int j = 0; j < n; j ++ ) 
  {
    for ( int i = 0; i < m; i ++ ) 
    {
      C[ j * m + i ] *= rank_k_scale;
    }
  }
#endif

  switch ( kernel->type ) 
  {
    case KS_GAUSSIAN:
      #pragma omp parallel for
      for ( int j = 0; j < n; j ++ ) 
      {
        for ( int i = 0; i < m; i ++ ) 
        {
          C[ j * m + i ] += A2[ amap[ i ] ];
          C[ j * m + i ] += B2[ bmap[ j ] ];
          C[ j * m + i ] *= kernel->scal;
        }
#ifdef USE_VML
        vdExp( m, C.data() + j * m, C.data() + j * m );
#else
        for ( int i = 0; i < m; i ++ ) 
        {
          C[ j * m + i ] = exp( C[ j * m + i ] );
        }
#endif
      }
      break;
    default:
      exit( 1 );
  }

  /*
   *  Kernel Summation
   */ 
#ifdef USE_BLAS
  xgemm
  (
    "N", "N", 
    m, nrhs, n, 
    fone, C.data(),     m,
          packw.data(), n, 
    fone, packu.data(), m
  );
#else
  #pragma omp parallel for
  for ( int i = 0; i < m; i ++ ) 
  {
    for ( int j = 0; j < nrhs; j ++ ) 
    {
      for ( int p = 0; p < n; p ++ ) 
      {
        packu[ j * m + i ] += C[ p * m + i ] * packw[ j * n + p ];
      }
    }
  }
#endif

  /*
   *  Assemble packu back
   */ 
  #pragma omp parallel for
  for ( int i = 0; i < m; i ++ ) 
  {
    for ( int p = 0; p < KS_RHS; p ++ ) 
    {
      u[ umap[ i ] * KS_RHS + p ] = packu[ p * m + i ];
    }
  }

} // end void gsks_ref


}; // end namespace gsks
}; // end namespace hmlp

#endif // define GSKS_HXX
