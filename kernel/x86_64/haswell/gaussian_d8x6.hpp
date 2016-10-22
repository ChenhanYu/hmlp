#include <stdio.h>
#include <math.h>
#include <immintrin.h> // AVX

#include <hmlp.h>
#include <hmlp_internal.hxx>
#include <avx_type.h> // self-defined vector type


struct gaussian_int_d8x6
{
  inline void operator()(
      ks_t *ker,
      int k,
      int nrhs,
      double *u,
      double *a, double *aa, 
      double *b, double *bb,
      double *w,
      double *c,
      aux_s<double, double, double, double> *aux ) const 
  {
    int    i;
    double alpha = ker->scal;
    // 16 registers.
    v4df_t c03_0, c03_1, c03_2, c03_3, c03_4, c03_5;
    v4df_t c47_0, c47_1, c47_2, c47_3, c47_4, c47_5;
    v4df_t a03, a47, b0, b1;

    #include <rank_k_int_d8x6.h>
    #include <sq2nrm_int_d8x6.h>

    // Scale before the kernel evaluation
    a03.v   = _mm256_broadcast_sd( &alpha );
    c03_0.v = _mm256_mul_pd( a03.v, c03_0.v );
    c03_1.v = _mm256_mul_pd( a03.v, c03_1.v );
    c03_2.v = _mm256_mul_pd( a03.v, c03_2.v );
    c03_3.v = _mm256_mul_pd( a03.v, c03_3.v );
    c03_4.v = _mm256_mul_pd( a03.v, c03_4.v );
    c03_5.v = _mm256_mul_pd( a03.v, c03_5.v );

    c47_0.v = _mm256_mul_pd( a03.v, c47_0.v );
    c47_1.v = _mm256_mul_pd( a03.v, c47_1.v );
    c47_2.v = _mm256_mul_pd( a03.v, c47_2.v );
    c47_3.v = _mm256_mul_pd( a03.v, c47_3.v );
    c47_4.v = _mm256_mul_pd( a03.v, c47_4.v );
    c47_5.v = _mm256_mul_pd( a03.v, c47_5.v );

    // Prefetch u, w
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

    // c = exp( c )
    c03_0.v = _mm256_exp_pd( c03_0.v );
    c03_1.v = _mm256_exp_pd( c03_1.v );
    c03_2.v = _mm256_exp_pd( c03_2.v );
    c03_3.v = _mm256_exp_pd( c03_3.v );
    c03_4.v = _mm256_exp_pd( c03_4.v );
    c03_5.v = _mm256_exp_pd( c03_5.v );

    c47_0.v = _mm256_exp_pd( c47_0.v );
    c47_1.v = _mm256_exp_pd( c47_1.v );
    c47_2.v = _mm256_exp_pd( c47_2.v );
    c47_3.v = _mm256_exp_pd( c47_3.v );
    c47_4.v = _mm256_exp_pd( c47_4.v );
    c47_5.v = _mm256_exp_pd( c47_5.v );

    // Preload u03, u47
    a03.v    = _mm256_load_pd( (double*)  u       );
    a47.v    = _mm256_load_pd( (double*)( u + 4 ) );

    // Multiple rhs weighted sum.
    #include<weighted_sum_int_d8x6.h>

    //if ( u[ 0 ] != u[ 0 ] ) printf( "u[ 0 ] nan\n" );
    //if ( u[ 1 ] != u[ 1 ] ) printf( "u[ 1 ] nan\n" );
    //if ( u[ 2 ] != u[ 2 ] ) printf( "u[ 2 ] nan\n" );
    //if ( u[ 3 ] != u[ 3 ] ) printf( "u[ 3 ] nan\n" );
    //if ( u[ 4 ] != u[ 4 ] ) printf( "u[ 4 ] nan\n" );
    //if ( u[ 5 ] != u[ 5 ] ) printf( "u[ 5 ] nan\n" );
    //if ( u[ 6 ] != u[ 6 ] ) printf( "u[ 6 ] nan\n" );
    //if ( u[ 7 ] != u[ 7 ] ) printf( "u[ 7 ] nan\n" );

    //if ( w[ 0 ] != w[ 0 ] ) printf( "w[ 0 ] nan\n" );
    //if ( w[ 1 ] != w[ 1 ] ) printf( "w[ 1 ] nan\n" );
    //if ( w[ 2 ] != w[ 2 ] ) printf( "w[ 2 ] nan\n" );
    //if ( w[ 3 ] != w[ 3 ] ) printf( "w[ 3 ] nan\n" );
    //if ( w[ 4 ] != w[ 4 ] ) printf( "w[ 4 ] nan\n" );
    //if ( w[ 5 ] != w[ 5 ] ) printf( "w[ 5 ] nan\n" );
  }
};
