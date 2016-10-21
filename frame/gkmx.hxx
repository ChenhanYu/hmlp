#ifndef GKMX_HXX
#define GKMX_HXX

#include <hmlp_internal.hxx>
#include <hmlp_packing.hxx>
#include <hmlp_util.hxx>
#include <hmlp_thread_communicator.hpp>
#include <hmlp_thread_info.hpp>


namespace gkmx
{

#define min( i, j ) ( (i)<(j) ? (i): (j) )


using namespace hmlp;


template<int KC, int MR, int NR, int PACK_MR, int PACK_NR,
    typename SEMIRINGKERNEL,
    typename TA, typename TB, typename TC, typename TV>
    void rank_k_macro_kernel_new(
        //thread_communicator &ic_comm,
        worker &thread,
        //int jr_id,
        int ic, int jc, int pc,
        int m, int n, int k,
        TA *packA,
        TB *packB,
        TV *packC, int ldc,
        SEMIRINGKERNEL semiringkernel
        )
{
  //int i, j, ip, jp;
  struct aux_s<TA, TB, TC, TV> aux;

  thread_communicator &ic_comm = *thread.ic_comm;


  aux.pc     = pc;
  aux.b_next = packB;

  //printf( "ic %d, jc %d, pc %d, ldc %d, jr_id %d, m %d, n %d\n", ic, jc, pc, ldc, jr_id, m, n );

  //for ( int i = 0; i < 8; i ++ ) printf( "%lf ", packA[ i ] );
  //printf( "\n" );
  //for ( int j = 0; j < 4; j ++ ) printf( "%lf ", packB[ j ] );
  //printf( "\n" );

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

      if ( min( MR, m - i ) < MR || min( NR, n - j ) < NR ) 
      {
        semiringkernel(
            k,
            &packA[ ip * k ],
            &packB[ jp * k ],
            &packC[ j * ldc + i ], ldc,
            &aux
            );
      }
      else
      {
        semiringkernel(
            k,
            &packA[ ip * k ],
            &packB[ jp * k ],
            &packC[ j * ldc + i ], ldc,
            &aux
            );
      }
    }
  }
}












template<
int MC, int NC, int KC, int MR, int NR, 
int PACK_MC, int PACK_NC, int PACK_MR, int PACK_NR, int SIMD_ALIGN_SIZE,
typename SEMIRINGKERNEL, typename MICROKERNEL,
typename TA, typename TB, typename TC, typename TV>
void gkmx_new(
        int m, int n, int k,
        TA *A, int lda, int *amap,
        TB *B, int ldb, int *bmap,
        TC *C, int ldc,
        SEMIRINGKERNEL semiringkernel,
        MICROKERNEL microkernel
    )
{
  int jc_nt = 1, pc_nt = 1, ic_nt = 1, jr_nt = 1;
  int nc, pack_nc;
  int ldpackc, padn;
  char *str;

  TA *packA = NULL;
  TB *packB = NULL;
  TV *packC = NULL;

  // Early return if possible
  if ( m == 0 || n == 0 || k == 0 ) return;

  // Check the environment variable.
  str = getenv( "KS_JC_NT" );
  if ( str ) jc_nt = (int)strtol( str, NULL, 10 );
  str = getenv( "KS_IC_NT" );
  if ( str ) ic_nt = (int)strtol( str, NULL, 10 );
  str = getenv( "KS_JR_NT" );
  if ( str ) jr_nt = (int)strtol( str, NULL, 10 );


  // Decide NC according to n and jc_nt.
  if ( jc_nt > 1 )
  {
    nc      = ( n - 1 ) / jc_nt + 1;
    pack_nc = ( ( NC - 1 ) / PACK_NR + 1 ) * PACK_NR;
  }


  // allocate packing memory
  packA  = hmlp_malloc<SIMD_ALIGN_SIZE, TA>(  KC, ( PACK_MC + 1 ) * ic_nt * jc_nt, sizeof(TA) );
  packB  = hmlp_malloc<SIMD_ALIGN_SIZE, TB>(  KC, ( PACK_NC + 1 )         * jc_nt, sizeof(TB) ); 

  ldpackc  = ( ( m - 1 ) / PACK_MR + 1 ) * PACK_MR;
  padn = NC;
  if ( n < NC ) 
  {
    padn = ( ( n - 1 ) / PACK_NR + 1 ) * PACK_NR;
  }

  // Temporary bufferm <TV> to store the semi-ring rank-k update
  packC = hmlp_malloc<SIMD_ALIGN_SIZE, TV>( ldpackc, padn * jc_nt, sizeof(TV) ); 

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
          }

          ic_comm.Barrier();
          //std::cout << "ic_comm.Barrier()" << tid << "\n";

          //if ( pc + KC < k )                     // semiring rank-k update
          if ( 0 )                     // semiring rank-k update
          {
            rank_k_macro_kernel_new
              <MC, NC, KC, MR, NR, SEMIRINGKERNEL, TA, TB, TC, TV>
              (
               thread, ic, jc, pc,
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
            rank_k_macro_kernel_new
              <KC, MR, NR, PACK_MR, PACK_NR, SEMIRINGKERNEL, TA, TB, TC, TV>
              (
               thread, ic, jc, pc,
               ib, jb, pb,
               packA + thread.ic_id * PACK_MC * pb,
               packB,
               C + ic,        
               ldc,
               semiringkernel
              );
          }

          ic_comm.Barrier();                     // sync all jr_id!!
          //std::cout << "ic_comm.Barrier() #2\n";

        }                                        // end 4th loop
        pc_comm.Barrier();
        //std::cout << "pc_comm.Barrier() #2\n";
      }                                          // end 5th loop
    }                                            // end 6th loop
  }
};









};



#endif // define GKMX_HXX
