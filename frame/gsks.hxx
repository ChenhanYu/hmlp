#ifndef GSKS_HXX
#define GSKS_HXX

#ifdef USE_VML
#include <mkl.h>
#endif

#include <math.h>
#include <vector>

#include <hmlp.h>
#include <hmlp_internal.hxx>
#include <hmlp_blas_lapack.h>
#include <hmlp_packing.hxx>
#include <hmlp_util.hxx>
#include <hmlp_thread_communicator.hpp>
#include <hmlp_thread_info.hpp>


#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define KS_RHS 1

using namespace hmlp;


template<int KC, int MR, int NR, int PACK_MR, int PACK_NR,
    typename SEMIRINGKERNEL,
    typename TA, typename TB, typename TC, typename TV>
void rank_k_macro_kernel(
        worker &thread,
        int ic, int jc, int pc,
        int  m, int n,  int  k,
        TA *packA,
        TB *packB,
        TV *packC, int ldc,
        SEMIRINGKERNEL semiringkernel
        )
{
  thread_communicator &ic_comm = *thread.ic_comm;
  struct aux_s<TA, TB, TC, TV> aux;

  aux.pc       = pc;
  aux.b_next   = packB;
  aux.do_packC = 1;

  for ( int j  =  thread.jr_id * NR,
            jp =  thread.jr_id * PACK_NR; 
            j  <  n;
            j  += ic_comm.GetNumThreads() * NR, 
            jp += ic_comm.GetNumThreads() * PACK_NR )      // beg 3rd loop
  {
    for ( int i  = 0, ip = 0; 
              i  < m; 
              i += MR, ip += PACK_MR )                     // beg 2nd loop
    {
      if ( i + MR >= m ) 
      {
        aux.b_next += ic_comm.GetNumThreads() * PACK_NR * k;
      }
      semiringkernel
      (
        k,
        &packA[ ip * k ],
        &packB[ jp * k ],
        &packC[ j * ldc + i * NR ], ldc,
        &aux
      );
    }                                                      // end 2nd loop
  }                                                        // end 3rd loop
}

template<int KC, int MR, int NR, int PACK_MR, int PACK_NR,
    typename MICROKERNEL,
    typename TA, typename TB, typename TC, typename TV>
void fused_macro_kernel(
        ks_t *kernel,
        worker &thread,
        int ic, int jc, int pc,
        int m, int n, int k,
        TC *packu,
        TA *packA, TA *packA2, TV *packAh,
        TB *packB, TB *packB2, TV *packBh,
        TC *packw,
        TV *packC, int ldc,
        MICROKERNEL microkernel
    )
{
  thread_communicator &ic_comm = *thread.ic_comm;
  struct aux_s<TA, TB, TC, TV> aux;

  aux.pc       = pc;
  aux.b_next   = packB;
  aux.do_packC = 1;

  for ( int j  =  thread.jr_id * NR,
            jp =  thread.jr_id * PACK_NR; 
            j  <  n;
            j  += ic_comm.GetNumThreads() * NR, 
            jp += ic_comm.GetNumThreads() * PACK_NR )      // beg 3rd loop
  {
    for ( int i  = 0, ip = 0; 
              i  < m; 
              i += MR, ip += PACK_MR )                     // beg 2nd loop
    {
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
        packC  + j * ldc + i * NR,                         // packed
        &aux
      );
    }                                                      // end 2nd loop
  }                                                        // end 3rd loop
}


template<
    int MC, int NC, int KC, int MR, int NR, 
    int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int ALIGN_SIZE,
    bool pack_norm, bool pack_bandwidth,
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
  {
    packA_buff  = hmlp_malloc<ALIGN_SIZE, TA>( KC, ( PACK_MC + 1 ) * jc_nt * ic_nt,         sizeof(TA) );
    packB_buff  = hmlp_malloc<ALIGN_SIZE, TB>( KC, ( pack_nc + 1 ) * jc_nt,                 sizeof(TB) ); 
    packu_buff  = hmlp_malloc<ALIGN_SIZE, TC>(  1, ( PACK_MC + 1 ) * jc_nt * ic_nt * jr_nt, sizeof(TC) );
    packw_buff  = hmlp_malloc<ALIGN_SIZE, TC>(  1, ( pack_nc + 1 ) * jc_nt,                 sizeof(TC) ); 
  }

  // allocate extra packing buffer
  if ( pack_norm )
  {
    packA2_buff = hmlp_malloc<ALIGN_SIZE, TA>(  1, ( PACK_MC + 1 ) * jc_nt * ic_nt,         sizeof(TA) );
    packB2_buff = hmlp_malloc<ALIGN_SIZE, TB>(  1, ( pack_nc + 1 ) * jc_nt,                 sizeof(TB) ); 
  }

  if ( pack_bandwidth )
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
    worker thread( &my_comm );

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


    for ( int jc  = thread.jc_id * nc; 
              jc  < n; 
              jc += jc_nt * nc )                           // beg 6th loop 
    {
      thread_communicator &jc_comm = *thread.jc_comm;
      int jb = min( n - jc, nc );

      for ( int pc = 0; pc < k; pc += KC )                 // beg 5th loop 
      {
        thread_communicator &pc_comm = *thread.pc_comm;
        bool is_the_last_pc_iteration = ( pc + KC >= k );
        int pb = min( k - pc, KC );

        packA = packA_buff + thread.jc_id * ic_nt * PACK_MC * KC 
                           + thread.ic_id         * PACK_MC * pb;

        for ( int j   = thread.ic_jr * NR, 
                  jp  = thread.ic_jr * PACK_NR; 
                  j   < jb; 
                  j  += pc_comm.GetNumThreads() * NR, 
                  jp += pc_comm.GetNumThreads() * PACK_NR ) 
        {
          packB_kcxnc<PACK_NR> (
              min( jb - j, NR ),
              pb, &B[ pc ], k, 
              &bmap[ jc + j ], &packB[ jp * pb ] );

          if ( is_the_last_pc_iteration )
          {
            for ( int jr = 0; jr < NR; jr ++ )
            {
              packw[ jp + jr ] = 0.0;
            }
            for ( int jr = 0; jr < min( jb - j, NR ); jr ++ )
            {
              if ( 1 )               packw[ jp + jr ] =          w[ wmap[ jc + j + jr ] ];
              if ( pack_norm )      packB2[ jp + jr ] =         B2[ bmap[ jc + j + jr ] ];
              if ( pack_bandwidth ) packBh[ jp + jr ] = kernel->hj[ bmap[ jc + j + jr ] ];
            }
          }
        }                           
        pc_comm.Barrier();

        for ( int ic  = thread.ic_id * MC; 
                  ic  < m; 
                  ic += ic_nt * MC )                       // beg 4th loop
        {
          thread_communicator &ic_comm = *thread.ic_comm;
          int ib = min( m - ic, MC );

          for ( int i   = thread.jr_id * MR, 
                    ip  = thread.jr_id * PACK_MR; 
                    i   < ib; 
                    i  += jr_nt * MR, 
                    ip += jr_nt * PACK_MR )     
          {
            packA_kcxmc<PACK_MR> 
            ( 
              min( ib - i, MR ), pb,
              &A[ pc ], k, &amap[ ic + i ], 
              &packA[ ip * pb ] 
            );

            if ( is_the_last_pc_iteration )               
            {
              for ( int ir = 0; ir < min( ib - i, MR ); ir ++ )
              {
                if ( pack_norm )                           // l2-norm
                {
                  packA2[ ip + ir ] =         A2[ amap[ ic + i + ir ] ];
                }
                if ( pack_bandwidth )                      // variable bandwidths
                {
                  packAh[ ip + ir ] = kernel->hi[ amap[ ic + i + ir ] ];
                }
              }
            }
          }

          if ( is_the_last_pc_iteration )                  // Initialize packu to zeros.
          {
            for ( int i = 0, ip = 0; i < ib; i += MR, ip += PACK_MR )
            {
              for ( int ir = 0; ir < min( ib - i, MR ); ir ++ )
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
            for ( int i = 0, ip = 0; i < ib; i += MR, ip += PACK_MR )
            {
              for ( int ir = 0; ir < min( ib - i, MR ); ir ++ )
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
  }                                                        // end omp region
}


template<typename T>
void gsks_ref(
    ks_t *kernel,
    int m, int n, int k,
    T *u,        int *umap,
    T *A, T *A2, int *amap,
    T *B, T *B2, int *bmap,
    T *w,        int *wmap )
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
  // Need to make a template for this case.
  //dgemm_( "T", "N", &m, &n, &k, &rank_k_scale,
  //    packA.data(), &k, packB.data(), &k, &fzero, C.data(), &m );
  xgemm( 
      "T", "N", m, n, k, rank_k_scale,
      packA.data(), k,
      packB.data(), k, fzero,
      C.data(),     m 
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
  //dgemm_( "N", "N", &m, &nrhs, &n, &fone,
  //    C.data(), &m, packw.data(), &n, &fone, packu.data(), &m );
  xgemm(
      "N", "N", m, nrhs, n, fone,
      C.data(),     m,
      packw.data(), n, fone,
      packu.data(), m
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


#endif // define GSKS_HXX
