#include <stdio.h>
#include <math.h>

/** self-defined vector type */
#include <avx_type.h> 
/** HMLP */
#include <hmlp.h>
#include <hmlp_internal.hpp>



struct gsks_gaussian_int_s8x8
{
  inline GSKS_OPERATOR(float) const
  {
    printf( "not implemented yet\n" );
    exit( 1 );
  };
};




struct gsks_gaussian_int_d8x4 
{
  const size_t mr         =  8;
  const size_t nr         =  4;
  const size_t pack_mr    =  8;
  const size_t pack_nr    =  4;
  const size_t align_size = 32;
  const bool   row_major  = false;


  //inline void operator()
  //(
  //  kernel_s<double> *ker,
  //  int k,
  //  int rhs,
  //  double *u,
  //  double *a, double *aa, 
  //  double *b, double *bb,
  //  double *w,
  //  double *c, int ldc,
  //  aux_s<double, double, double, double> *aux 
  //) const 

  inline GSKS_OPERATOR(double) const
  {
    int    i, rhs_left;
    double neg2 = -2.0;
    double dzero = 0.0;
    double alpha = ker->scal;

    v4df_t    c03_0,    c03_1,    c03_2,    c03_3;
    v4df_t    c47_0,    c47_1,    c47_2,    c47_3;
    v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
    v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
    v4df_t u03, u47;
    v4df_t a03, a47, A03, A47; // prefetched A 
    v4df_t b0, b1, b2, b3, B0; // prefetched B
    v4df_t c_tmp, aa_tmp, bb_tmp, w_tmp;

    /** rank-k update segment */
    #include "component/rank_k_int_d8x4.hpp"

    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );

    if ( aux->pc )
    {
      tmpc03_0.v = _mm256_load_pd( (double*)( c      ) );
      tmpc47_0.v = _mm256_load_pd( (double*)( c + 4  ) );

      tmpc03_1.v = _mm256_load_pd( (double*)( c + 1 * ldc     ) );
      tmpc47_1.v = _mm256_load_pd( (double*)( c + 1 * ldc + 4 ) );

      tmpc03_2.v = _mm256_load_pd( (double*)( c + 2 * ldc  ) );
      tmpc47_2.v = _mm256_load_pd( (double*)( c + 2 * ldc + 4 ) );

      tmpc03_3.v = _mm256_load_pd( (double*)( c + 3 * ldc ) );
      tmpc47_3.v = _mm256_load_pd( (double*)( c + 3 * ldc + 4 ) );


      c03_0.v    = _mm256_add_pd( tmpc03_0.v, c03_0.v );
      c47_0.v    = _mm256_add_pd( tmpc47_0.v, c47_0.v );

      c03_1.v    = _mm256_add_pd( tmpc03_1.v, c03_1.v );
      c47_1.v    = _mm256_add_pd( tmpc47_1.v, c47_1.v );

      c03_2.v    = _mm256_add_pd( tmpc03_2.v, c03_2.v );
      c47_2.v    = _mm256_add_pd( tmpc47_2.v, c47_2.v );

      c03_3.v    = _mm256_add_pd( tmpc03_3.v, c03_3.v );
      c47_3.v    = _mm256_add_pd( tmpc47_3.v, c47_3.v );
    }

    // Scale -2
    aa_tmp.v = _mm256_broadcast_sd( &neg2 );
    c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
    c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
    c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
    c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
    c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
    c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
    c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
    c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );


    aa_tmp.v = _mm256_load_pd( (double*)aa );
    c03_0.v  = _mm256_add_pd( aa_tmp.v, c03_0.v );
    c03_1.v  = _mm256_add_pd( aa_tmp.v, c03_1.v );
    c03_2.v  = _mm256_add_pd( aa_tmp.v, c03_2.v );
    c03_3.v  = _mm256_add_pd( aa_tmp.v, c03_3.v );


    aa_tmp.v = _mm256_load_pd( (double*)( aa + 4 ) );
    c47_0.v  = _mm256_add_pd( aa_tmp.v, c47_0.v );
    c47_1.v  = _mm256_add_pd( aa_tmp.v, c47_1.v );
    c47_2.v  = _mm256_add_pd( aa_tmp.v, c47_2.v );
    c47_3.v  = _mm256_add_pd( aa_tmp.v, c47_3.v );


    // Prefetch u
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );

    bb_tmp.v = _mm256_broadcast_sd( (double*)bb );
    c03_0.v  = _mm256_add_pd( bb_tmp.v, c03_0.v );
    c47_0.v  = _mm256_add_pd( bb_tmp.v, c47_0.v );

    bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 1 ) );
    c03_1.v  = _mm256_add_pd( bb_tmp.v, c03_1.v );
    c47_1.v  = _mm256_add_pd( bb_tmp.v, c47_1.v );

    bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 2 ) );
    c03_2.v  = _mm256_add_pd( bb_tmp.v, c03_2.v );
    c47_2.v  = _mm256_add_pd( bb_tmp.v, c47_2.v );

    bb_tmp.v = _mm256_broadcast_sd( (double*)( bb + 3 ) );
    c03_3.v  = _mm256_add_pd( bb_tmp.v, c03_3.v );
    c47_3.v  = _mm256_add_pd( bb_tmp.v, c47_3.v );


    // Check if there is any illegle value 
    c_tmp.v  = _mm256_broadcast_sd( &dzero );
    c03_0.v  = _mm256_max_pd( c_tmp.v, c03_0.v );
    c03_1.v  = _mm256_max_pd( c_tmp.v, c03_1.v );
    c03_2.v  = _mm256_max_pd( c_tmp.v, c03_2.v );
    c03_3.v  = _mm256_max_pd( c_tmp.v, c03_3.v );
    c47_0.v  = _mm256_max_pd( c_tmp.v, c47_0.v );
    c47_1.v  = _mm256_max_pd( c_tmp.v, c47_1.v );
    c47_2.v  = _mm256_max_pd( c_tmp.v, c47_2.v );
    c47_3.v  = _mm256_max_pd( c_tmp.v, c47_3.v );


    aa_tmp.v = _mm256_broadcast_sd( &alpha );
    c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
    c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
    c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
    c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
    c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
    c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
    c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
    c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );


    /** preload u03, u47 */
    u03.v    = _mm256_load_pd( (double*)u );
    u47.v    = _mm256_load_pd( (double*)( u + 4 ) );


    /** prefetch u and w */
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 8 ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

    /** c = exp( c ) */
    #include "component/exp_int_d8x4.hpp"

    /** multiple rhs kernel summation */
    #include "component/weighted_sum_int_d8x4.hpp"

  }; /** end inline void operator */
}; /** end struct gaussian_ref_d8x4 */
