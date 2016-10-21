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
void rank_k_macro_kernel_new(
        worker &thread,
        int ic, int jc, int pc,
        int m, int n, int k,
        TA *packA,
        TB *packB,
        TV *packC, int ldc,
        SEMIRINGKERNEL semiringkernel
        )
{
  struct aux_s<TA, TB, TC, TV> aux;

  thread_communicator &ic_comm = *thread.ic_comm;


  aux.pc       = pc;
  aux.b_next   = packB;
  aux.do_packC = 1;

  //printf( "ic %d, jc %d, pc %d, ldc %d, jr_id %d, m %d, n %d\n", ic, jc, pc, ldc, jr_id, m, n );

  //for ( int i = 0; i < 8; i ++ ) printf( "%lf ", packA[ i ] );
  //printf( "\n" );
  //for ( int j = 0; j < 4; j ++ ) printf( "%lf ", packB[ j ] );
  //printf( "\n" );

  //printf( "ic_comm.GetNumThreads() %d jr_id %d n %d\n", ic_comm.GetNumThreads(), thread.jr_id, n );


  for ( int j  =  thread.jr_id * NR,
            jp =  thread.jr_id * PACK_NR; 
            j  <  n;
            j  += ic_comm.GetNumThreads() * NR, 
            jp += ic_comm.GetNumThreads() * PACK_NR ) 
  {
    for ( int i = 0, ip = 0; i < m; i += MR, ip += PACK_MR ) 
    {
      if ( i + MR >= m ) 
      {
        aux.b_next += ic_comm.GetNumThreads() * PACK_NR * k;
      }

      //printf( "ip %d, jp %d, i %d, j %d, k %d, NR %d, PACK_NR %d\n", ip, jp, i, j, k, NR, PACK_NR );

      semiringkernel(
          k,
          &packA[ ip * k ],
          &packB[ jp * k ],
          &packC[ j * ldc + i * NR ], ldc,
          &aux
          );
    }
  }
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
            jp += ic_comm.GetNumThreads() * PACK_NR ) 
  {
    for ( int i = 0, ip = 0; i < m; i += MR, ip += PACK_MR ) 
    {
      if ( i + MR >= m ) 
      {
        aux.b_next += ic_comm.GetNumThreads() * PACK_NR * k;
      }

      microkernel(
          kernel,
          k,
          KS_RHS,
          packu  + ip * KS_RHS,
          packA  + ip * k,
          packA2 + ip,
          packB  + jp * k,
          packB2 + jp,
          packw  + jp * KS_RHS,
          packC  + j * ldc + i * NR, // packed
          &aux
          );
    }
  }
}





template<
int MC, int NC, int KC, int MR, int NR, 
int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int SIMD_ALIGN_SIZE,
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
  int ldpackc, padn;
  char *str;

  TA *packA  = NULL;
  TB *packB  = NULL;
  TV *packC  = NULL;

  TA *packA2 = NULL;
  TB *packB2 = NULL;
  TC *packu  = NULL;
  TC *packw  = NULL;

  TV *packAh = NULL;
  TV *packBh = NULL;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  // Check the environment variable.
  str = getenv( "KS_IC_NT" );
  if ( str ) ic_nt = (int)strtol( str, NULL, 10 );
  str = getenv( "KS_JR_NT" );
  if ( str ) jr_nt = (int)strtol( str, NULL, 10 );

  // allocate packing memory
  packA  = hmlp_malloc<SIMD_ALIGN_SIZE, TA>(    KC, ( PACK_MC + 1 ) * ic_nt, sizeof(TA) );
  packB  = hmlp_malloc<SIMD_ALIGN_SIZE, TB>(    KC, ( PACK_NC + 1 )        , sizeof(TB) ); 

  // allocate extra packing buffer
  packA2 = hmlp_malloc<SIMD_ALIGN_SIZE, TA>(     1, ( PACK_MC + 1 ) * ic_nt, sizeof(TA) );
  packB2 = hmlp_malloc<SIMD_ALIGN_SIZE, TB>(     1, ( PACK_NC + 1 )        , sizeof(TB) ); 
  packu  = hmlp_malloc<SIMD_ALIGN_SIZE, TC>( jr_nt, ( PACK_MC + 1 ) * ic_nt, sizeof(TC) );
  packw  = hmlp_malloc<SIMD_ALIGN_SIZE, TC>(     1, ( PACK_NC + 1 )        , sizeof(TC) ); 


  if ( pack_bandwidth )
  {
  }





  ldpackc  = ( ( m - 1 ) / PACK_MR + 1 ) * PACK_MR;
  padn = NC;
  if ( n < NC ) 
  {
    padn = ( ( n - 1 ) / PACK_NR + 1 ) * PACK_NR;
  }

  // Temporary bufferm <TV> to store the semi-ring rank-k update
  packC = hmlp_malloc<SIMD_ALIGN_SIZE, TV>( ldpackc, padn, sizeof(TV) ); 

  // allocate tree communicator
  thread_communicator my_comm( jc_nt, pc_nt, ic_nt, jr_nt );

  #pragma omp parallel num_threads( my_comm.GetNumThreads() ) 
  {
    worker thread( &my_comm );

    for ( int jc = 0; jc < n; jc += NC )         // beg 6th loop 
    {
      thread_communicator &jc_comm = *thread.jc_comm;
      int jb = min( n - jc, NC );

      for ( int pc = 0; pc < k; pc += KC )       // beg 5th loop 
      {
        thread_communicator &pc_comm = *thread.pc_comm;
        int pb = min( k - pc, KC );

        for ( int j   = thread.tid * NR, 
                  jp  = thread.tid * PACK_NR; 
                  j   < jb; 
                  j  += jc_comm.GetNumThreads() * NR, 
                  jp += jc_comm.GetNumThreads() * PACK_NR )  // packB [ num_threads ] threads
        {
          packB_kcxnc<PACK_NR> (
              min( jb - j, NR ),
              pb, &B[ pc ], k, 
              &bmap[ jc + j ], &packB[ jp * pb ] );

          if ( pc + KC >= k )
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
       // std::cout << "pc_comm.Barrier()\n";

        for ( int ic  = thread.ic_id * MC; 
                  ic  < m; 
                  ic += ic_nt * MC )             // beg 4th loop
        {
          thread_communicator &ic_comm = *thread.ic_comm;
          int ib = min( m - ic, MC );
          //std::cout << "ic_comm" << tid << ", " << ic_id << ", " << ic_comm.GetNumThreads() << "\n";

          for ( int i   = thread.jr_id * MR, 
                    ip  = thread.jr_id * PACK_MR; 
                    i   < ib; 
                    i  += jr_nt * MR, 
                    ip += jr_nt * PACK_MR )      // packA [ jr_nt ] threads
          {
            packA_kcxmc<PACK_MR> ( 
                min( ib - i, MR ), pb,
                &A[ pc ], k, &amap[ ic + i ], 
                &packA[ thread.ic_id * PACK_MC * pb + ip * pb ] );

            if ( pc + KC >= k )
            {
              for ( int ir = 0; ir < min( ib - i, MR ); ir ++ )
              {
                if ( pack_norm ) 
                {
                  packA2[ thread.ic_id * PACK_MC + ip + ir ] =         A2[ amap[ ic + i + ir ] ];
                }
                if ( pack_bandwidth ) 
                {
                  packAh[ thread.ic_id * PACK_MC + ip + ir ] = kernel->hi[ amap[ ic + i + ir ] ];
                }
              }
            }
          }

          if ( pc + KC >= k )
          {
            for ( int i = 0, ip = 0; i < ib; i += MR, ip += PACK_MR )
            {
              for ( int ir = 0; ir < min( ib - i, MR ); ir ++ )
              {
                packu[ ( thread.ic_id * jr_nt + thread.jr_id ) * PACK_MC + ip + ir ] = 0.0;
              }
            }
          }
          ic_comm.Barrier();
          //std::cout << "ic_comm.Barrier()" << tid << "\n";

          if ( pc + KC < k )                     // semiring rank-k update
          {
            rank_k_macro_kernel_new
              <KC, MR, NR, PACK_MR, PACK_NR, SEMIRINGKERNEL, TA, TB, TC, TV>
              (
               thread, 
               ic, jc, pc,
               ib, jb, pb,
               packA + thread.ic_id * PACK_MC * pb,
               packB,
               packC + ic * padn,            // packed
               ( ( ib - 1 ) / MR + 1 ) * MR, // packed ldc
               semiringkernel
              );
          }
          else 
          {                                      // fused_macro_kernel
            fused_macro_kernel
              <KC, MR, NR, PACK_MR, PACK_NR, MICROKERNEL, TA, TB, TC, TV>
              (
               kernel,
               thread, 
               ic, jc, pc,
               ib, jb, pb,
               packu  + ( thread.ic_id * jr_nt + thread.jr_id ) * PACK_MC * KS_RHS,
               packA  + thread.ic_id * PACK_MC * pb,
               packA2 + thread.ic_id * PACK_MC,
               packAh + thread.ic_id * PACK_MC,
               packB,
               packB2,
               packBh,
               packw,
               packC + ic * padn,            // packed
               ( ( ib - 1 ) / MR + 1 ) * MR, // packed ldc
               microkernel
              );
          }
          ic_comm.Barrier();                     // sync all jr_id!!
          //std::cout << "ic_comm.Barrier() #2\n";

          if ( pc + KC >= k )
          {
            for ( int i = 0, ip = 0; i < ib; i += MR, ip += PACK_MR )
            {
              for ( int ir = 0; ir < min( ib - i, MR ); ir ++ )
              {
                TC *uptr = &( u[ umap[ ic + i + ir ] ] );

                //printf( "ic_id %d jr_id %d u %lf packu %lf\n", thread.ic_id, thread.jr_id,
                //    *uptr,  packu[ ( thread.ic_id * jr_nt + thread.jr_id ) * PACK_MC + ip + ir ] );

                #pragma omp atomic update
                *uptr += packu[ ( thread.ic_id * jr_nt + thread.jr_id ) * PACK_MC + ip + ir ];
              }
            }
          }
          ic_comm.Barrier();                     // sync all jr_id!!

        }                                        // end 4th loop
        pc_comm.Barrier();
        //std::cout << "pc_comm.Barrier() #2\n";
      }                                          // end 5th loop
    }                                            // end 6th loop
  }
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
