#include <stdio.h>
#include <math.h>
#include <immintrin.h> // AVX

#include <hmlp.h>
#include <hmlp_internal.hpp>
#include <avx_type.h> // self-defined vector type

// #define DEBUG_MICRO 1

struct variable_bandwidth_gaussian_ref_d8x4 
{
  inline void operator()
  (
    //ks_t *kernel,
    kernel_s<double> *kernel,
    int k,
    int nrhs,
    double *u,
    double *a, double *a2, 
    double *b, double *b2,
    double *w,
    double *c, int ldc,
    aux_s<double, double, double, double> *aux 
  ) const 
  {
    double c_reg[ 8 * 4 ] = { 0.0 };

    for ( int p = 0; p < k; p ++ ) 
    {
      #pragma unroll
      for ( int j = 0; j < 4; j ++ )
      {
        #pragma unroll
        for ( int i = 0; i < 8; i ++ ) 
        {
          c_reg[ j * 8 + i ] += a[ p * 8 + i ] * b[ p * 4 + j ];
        }
      }
    }

    if ( aux->pc ) 
    {
      #pragma unroll
      for ( int j = 0; j < 4; j ++ )
      {
        #pragma unroll
        for ( int i = 0; i < 8; i ++ ) 
        {
          c_reg[ j * 8 + i ] += c[ j * ldc + i ];
        }
      }
    }

#ifdef DEBUG_MICRO
    printf( "variable_bandwidth_gaussian_ref_d8x4: c_reg\n" );
    for ( int i = 0; i < 8; i ++ ) 
    {
      for ( int j = 0; j < 4; j ++ )
      {
        //printf( "%E (%E) ", c_reg[ j * 8 + i ], c[ j * 8 + i ] );
        printf( "%E ", c_reg[ j * 8 + i ] );
      }
      printf( "\n" );
    }
#endif

    #pragma unroll
    for ( int j = 0; j < 4; j ++ )
    {
      #pragma unroll
      for ( int i = 0; i < 8; i ++ ) 
      {
        c_reg[ j * 8 + i ] *= -2.0;
        c_reg[ j * 8 + i ] += a2[ i ] + b2[ j ];
        c_reg[ j * 8 + i ] *= -0.5;
        c_reg[ j * 8 + i ] *= aux->hi[ i ];
        c_reg[ j * 8 + i ] *= aux->hj[ j ];
        c_reg[ j * 8 + i ]  = exp( c_reg[ j * 8 + i ] );
      }
    }

    #pragma unroll
    for ( int j = 0; j < 4; j ++ )
    {
      #pragma unroll
      for ( int i = 0; i < 8; i ++ ) 
      {
        u[ i ] += c_reg[ j * 8 + i ] * w[ j ];
      }
    }    

  }; // end inline void operator
}; // end struct variable_bandwidth_gaussian_ref_d8x4



struct variable_bandwidth_gaussian_int_d8x4 
{
  inline void operator()
  (
    //ks_t *ker,
    kernel_s<double> *ker,
    int k,
    int rhs,
    double *u,
    double *a, double *aa, 
    double *b, double *bb,
    double *w,
    double *c, int ldc,
    aux_s<double, double, double, double> *aux 
  ) const 
  {
    int    i, rhs_left;
    double neg2 = -2.0;
    double neghalf = -0.5;
    double dzero = 0.0;

    v4df_t    c03_0,    c03_1,    c03_2,    c03_3;
    v4df_t    c47_0,    c47_1,    c47_2,    c47_3;
    v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
    v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
    v4df_t u03, u47;
    v4df_t a03, a47, A03, A47; // prefetched A 
    v4df_t b0, b1, b2, b3, B0; // prefetched B
    v4df_t c_tmp, aa_tmp, bb_tmp, w_tmp;

    // Rank-k update segment
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


    aa_tmp.v = _mm256_broadcast_sd( &neghalf );
    c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
    c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
    c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
    c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
    c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );
    c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );
    c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );
    c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );


    u03.v    = _mm256_load_pd( (double*)u );
    u47.v    = _mm256_load_pd( (double*)( u + 4 ) );

    // Scale columns with hj
    aa_tmp.v = _mm256_broadcast_sd( aux->hj + 0 );
    c03_0.v  = _mm256_mul_pd( aa_tmp.v, c03_0.v );
    c47_0.v  = _mm256_mul_pd( aa_tmp.v, c47_0.v );

    aa_tmp.v = _mm256_broadcast_sd( aux->hj + 1 );
    c03_1.v  = _mm256_mul_pd( aa_tmp.v, c03_1.v );
    c47_1.v  = _mm256_mul_pd( aa_tmp.v, c47_1.v );

    aa_tmp.v = _mm256_broadcast_sd( aux->hj + 2 );
    c03_2.v  = _mm256_mul_pd( aa_tmp.v, c03_2.v );
    c47_2.v  = _mm256_mul_pd( aa_tmp.v, c47_2.v );

    aa_tmp.v = _mm256_broadcast_sd( aux->hj + 3 );
    c03_3.v  = _mm256_mul_pd( aa_tmp.v, c03_3.v );
    c47_3.v  = _mm256_mul_pd( aa_tmp.v, c47_3.v );

    // Scale rows with hi
    u03.v    = _mm256_load_pd( aux->hi + 0 );
    u47.v    = _mm256_load_pd( aux->hi + 4 );

    c03_0.v  = _mm256_mul_pd( u03.v, c03_0.v );
    c03_1.v  = _mm256_mul_pd( u03.v, c03_1.v );
    c03_2.v  = _mm256_mul_pd( u03.v, c03_2.v );
    c03_3.v  = _mm256_mul_pd( u03.v, c03_3.v );
    c47_0.v  = _mm256_mul_pd( u47.v, c47_0.v );
    c47_1.v  = _mm256_mul_pd( u47.v, c47_1.v );
    c47_2.v  = _mm256_mul_pd( u47.v, c47_2.v );
    c47_3.v  = _mm256_mul_pd( u47.v, c47_3.v );

    // Preload u03, u47
    u03.v    = _mm256_load_pd( (double*)u );
    u47.v    = _mm256_load_pd( (double*)( u + 4 ) );


    // Prefetch u and w
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 8 ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

    // c = exp( c );
    #include "component/exp_int_d8x4.hpp"

    // Multiple rhs kernel summation.
    #include "component/weighted_sum_int_d8x4.hpp"

  }; // end inline void operator
}; // end struct variable_bandwidth_gaussian_ref_d8x4
