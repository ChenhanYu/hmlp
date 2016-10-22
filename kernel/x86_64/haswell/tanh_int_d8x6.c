#include <math.h>
#include <immintrin.h> // AVX
#include <ks.h>
#include <gsks_internal.h>
#include <avx_type.h>


void tanh_int_s16x6(
    int    k,
    int    rhs,
    //float  *h,
    float  *u,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *w,
    float  *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  printf( "tanh_int_s16x6 not yet implemented.\n" );
}

void tanh_int_d8x6(
    int    k,
    int    rhs,
    //double *h,
    double *u,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  int    i;
  double scal = ker->scal;
  double cons = ker->cons;
  // 16 registers.
  v4df_t c03_0, c03_1, c03_2, c03_3, c03_4, c03_5;
  v4df_t c47_0, c47_1, c47_2, c47_3, c47_4, c47_5;
  v4df_t a03, a47, b0, b1;

  #include <rank_k_int_d8x6.h>

  // Prefetch u, w
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

  // c = c * scal
  a03.v   = _mm256_broadcast_sd( &scal );
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

  // c = c + cons
  a03.v   = _mm256_broadcast_sd( &cons );
  c03_0.v = _mm256_add_pd( a03.v, c03_0.v );
  c03_1.v = _mm256_add_pd( a03.v, c03_1.v );
  c03_2.v = _mm256_add_pd( a03.v, c03_2.v );
  c03_3.v = _mm256_add_pd( a03.v, c03_3.v );
  c03_4.v = _mm256_add_pd( a03.v, c03_4.v );
  c03_5.v = _mm256_add_pd( a03.v, c03_5.v );

  c47_0.v = _mm256_add_pd( a03.v, c47_0.v );
  c47_1.v = _mm256_add_pd( a03.v, c47_1.v );
  c47_2.v = _mm256_add_pd( a03.v, c47_2.v );
  c47_3.v = _mm256_add_pd( a03.v, c47_3.v );
  c47_4.v = _mm256_add_pd( a03.v, c47_4.v );
  c47_5.v = _mm256_add_pd( a03.v, c47_5.v );

  // c = tanh( c );
  c03_0.v  = _mm256_tanh_pd( c03_0.v );
  c03_1.v  = _mm256_tanh_pd( c03_1.v );
  c03_2.v  = _mm256_tanh_pd( c03_2.v );
  c03_3.v  = _mm256_tanh_pd( c03_3.v );
  c03_4.v  = _mm256_tanh_pd( c03_4.v );
  c03_5.v  = _mm256_tanh_pd( c03_5.v );

  c47_0.v  = _mm256_tanh_pd( c47_0.v );
  c47_1.v  = _mm256_tanh_pd( c47_1.v );
  c47_2.v  = _mm256_tanh_pd( c47_2.v );
  c47_3.v  = _mm256_tanh_pd( c47_3.v );
  c47_4.v  = _mm256_tanh_pd( c47_4.v );
  c47_5.v  = _mm256_tanh_pd( c47_5.v );
  
  // Preload u03, u47
  a03.v    = _mm256_load_pd( (double*)  u       );
  a47.v    = _mm256_load_pd( (double*)( u + 4 ) );

  // Multiple rhs weighted sum.
  #include<weighted_sum_int_d8x6.h>
}
