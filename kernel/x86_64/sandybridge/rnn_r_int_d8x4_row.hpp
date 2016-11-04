#include <stdio.h>
#include <math.h>
#include <immintrin.h> // AVX

#include <hmlp.h>
#include <hmlp_util.hpp>
#include <hmlp_internal.hpp>
#include <gsknn.hpp>
#include <avx_type.h> // self-defined vector type

struct rnn_r_int_d8x4_row
{
  inline void operator()
  (
   int k,
   int r,
   double *a, double *aa,
   double *b, double *bb,
   double *c,
   aux_d<double, double, double, double> *aux,
   int *bmap
  ) const
  {
  int    i, j, ldr;
  int    *I = aux->I;
  double *D = aux->D;

  double neg2 = -2.0;
  double dzero = 0.0;
  v4df_t c03_0, c03_1, c03_2, c03_3;
  v4df_t c47_0, c47_1, c47_2, c47_3;
  v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
  v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
  v4df_t c_tmp;
  v4df_t a03, a47;
  v4df_t A03, A47; // prefetched A

  v4df_t b0, b1, b2, b3;
  v4df_t B0; // prefetched B
  v4df_t aa_tmp, bb_tmp;


  int k_iter = k / 2;
  int k_left = k % 2;

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );
  __asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( c +  3 ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( c + 11 ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( c + 19 ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( c + 27 ) );


  c03_0.v = _mm256_setzero_pd();
  c03_1.v = _mm256_setzero_pd();
  c03_2.v = _mm256_setzero_pd();
  c03_3.v = _mm256_setzero_pd();
  c47_0.v = _mm256_setzero_pd();
  c47_1.v = _mm256_setzero_pd();
  c47_2.v = _mm256_setzero_pd();
  c47_3.v = _mm256_setzero_pd();


  // Load a03
  a03.v = _mm256_load_pd(      (double*)a         );
  // Load a47
  a47.v = _mm256_load_pd(      (double*)( a + 4 ) );
  // Load ( b0, b1, b2, b3 )
  b0.v  = _mm256_load_pd(      (double*)b         );

  for ( i = 0; i < k_iter; ++i ) {
    __asm__ volatile( "prefetcht0 192(%0)    \n\t" : :"r"(a) );

    // Preload A03
    A03.v = _mm256_load_pd(      (double*)( a + 8 ) );

    c_tmp.v = _mm256_mul_pd( a03.v  , b0.v    );
    c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b0.v    );
    c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );

    // Preload A47
    A47.v = _mm256_load_pd(      (double*)( a + 12 ) );

    // Shuffle b ( 1, 0, 3, 2 )
    b1.v  = _mm256_shuffle_pd( b0.v, b0.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b1.v    );
    c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b1.v    );
    c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    // Preload B0
    B0.v  = _mm256_load_pd(      (double*)( b + 4 ) );

    c_tmp.v = _mm256_mul_pd( a03.v  , b2.v    );
    c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b2.v    );
    c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b3.v    );
    c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b3.v    );
    c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );


    // Iteration #1
    __asm__ volatile( "prefetcht0 512(%0)    \n\t" : :"r"(a) );

    // Preload a03 ( next iteration )
    a03.v = _mm256_load_pd(      (double*)( a + 16 ) );

    c_tmp.v = _mm256_mul_pd( A03.v  , B0.v    );
    c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );

    b1.v  = _mm256_shuffle_pd( B0.v, B0.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( A47.v  , B0.v    );
    c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );
    c_tmp.v = _mm256_mul_pd( A03.v  , b1.v    );
    c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );

    // Preload a47 ( next iteration )
    a47.v = _mm256_load_pd(      (double*)( a + 20 ) );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    c_tmp.v = _mm256_mul_pd( A47.v  , b1.v    );
    c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );
    c_tmp.v = _mm256_mul_pd( A03.v  , b2.v    );
    c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( A47.v  , b2.v    );
    c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );

    // Load b0 ( next iteration )
    b0.v  = _mm256_load_pd(      (double*)( b + 8 ) );

    c_tmp.v = _mm256_mul_pd( A03.v  , b3.v    );
    c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );
    c_tmp.v = _mm256_mul_pd( A47.v  , b3.v    );
    c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );

    a += 16;
    b += 8;
  }

  for ( i = 0; i < k_left; ++i ) {
    a03.v = _mm256_load_pd(      (double*)a         );

    a47.v = _mm256_load_pd(      (double*)( a + 4 ) );

    b0.v  = _mm256_load_pd(      (double*)b         );

    c_tmp.v = _mm256_mul_pd( a03.v  , b0.v    );
    c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b0.v    );
    c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );

    // Shuffle b ( 1, 0, 3, 2 )
    b1.v  = _mm256_shuffle_pd( b0.v, b0.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b1.v    );
    c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b1.v    );
    c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );

    // Permute b ( 3, 2, 1, 0 )
    b2.v  = _mm256_permute2f128_pd( b1.v, b1.v, 0x1 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b2.v    );
    c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b2.v    );
    c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );

    // Shuffle b ( 3, 2, 1, 0 )
    b3.v  = _mm256_shuffle_pd( b2.v, b2.v, 0x5 );

    c_tmp.v = _mm256_mul_pd( a03.v  , b3.v    );
    c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );
    c_tmp.v = _mm256_mul_pd( a47.v  , b3.v    );
    c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );

    a += 8;
    b += 4;
  }


  // Prefetch aa and bb
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );


  tmpc03_0.v = _mm256_blend_pd( c03_0.v, c03_1.v, 0x6 );
  tmpc03_1.v = _mm256_blend_pd( c03_1.v, c03_0.v, 0x6 );

  tmpc03_2.v = _mm256_blend_pd( c03_2.v, c03_3.v, 0x6 );
  tmpc03_3.v = _mm256_blend_pd( c03_3.v, c03_2.v, 0x6 );

  tmpc47_0.v = _mm256_blend_pd( c47_0.v, c47_1.v, 0x6 );
  tmpc47_1.v = _mm256_blend_pd( c47_1.v, c47_0.v, 0x6 );

  tmpc47_2.v = _mm256_blend_pd( c47_2.v, c47_3.v, 0x6 );
  tmpc47_3.v = _mm256_blend_pd( c47_3.v, c47_2.v, 0x6 );

  c03_0.v    = _mm256_permute2f128_pd( tmpc03_0.v, tmpc03_2.v, 0x30 );
  c03_3.v    = _mm256_permute2f128_pd( tmpc03_2.v, tmpc03_0.v, 0x30 );

  c03_1.v    = _mm256_permute2f128_pd( tmpc03_1.v, tmpc03_3.v, 0x30 );
  c03_2.v    = _mm256_permute2f128_pd( tmpc03_3.v, tmpc03_1.v, 0x30 );

  c47_0.v    = _mm256_permute2f128_pd( tmpc47_0.v, tmpc47_2.v, 0x30 );
  c47_3.v    = _mm256_permute2f128_pd( tmpc47_2.v, tmpc47_0.v, 0x30 );

  c47_1.v    = _mm256_permute2f128_pd( tmpc47_1.v, tmpc47_3.v, 0x30 );
  c47_2.v    = _mm256_permute2f128_pd( tmpc47_3.v, tmpc47_1.v, 0x30 );


  if ( aux->pc ) {
    c_tmp.v = _mm256_load_pd( c +  0 );
    c03_0.v = _mm256_add_pd( c_tmp.v, c03_0.v );

    c_tmp.v = _mm256_load_pd( c +  4 );
    c47_0.v = _mm256_add_pd( c_tmp.v, c47_0.v );

    c_tmp.v = _mm256_load_pd( c +  8 );
    c03_1.v = _mm256_add_pd( c_tmp.v, c03_1.v );

    c_tmp.v = _mm256_load_pd( c + 12 );
    c47_1.v = _mm256_add_pd( c_tmp.v, c47_1.v );

    c_tmp.v = _mm256_load_pd( c + 16 );
    c03_2.v = _mm256_add_pd( c_tmp.v, c03_2.v );

    c_tmp.v = _mm256_load_pd( c + 20 );
    c47_2.v = _mm256_add_pd( c_tmp.v, c47_2.v );

    c_tmp.v = _mm256_load_pd( c + 24 );
    c03_3.v = _mm256_add_pd( c_tmp.v, c03_3.v );

    c_tmp.v = _mm256_load_pd( c + 28 );
    c47_3.v = _mm256_add_pd( c_tmp.v, c47_3.v );
  }


  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( I ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( D ) );

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


  // Transpose c03/c47 _0, _1, _2, _3 to be the row vector
  tmpc03_0.v = _mm256_shuffle_pd( c03_0.v, c03_1.v, 0x0 );
  tmpc03_1.v = _mm256_shuffle_pd( c03_0.v, c03_1.v, 0xF );

  tmpc03_2.v = _mm256_shuffle_pd( c03_2.v, c03_3.v, 0x0 );
  tmpc03_3.v = _mm256_shuffle_pd( c03_2.v, c03_3.v, 0xF );

  tmpc47_0.v = _mm256_shuffle_pd( c47_0.v, c47_1.v, 0x0 );
  tmpc47_1.v = _mm256_shuffle_pd( c47_0.v, c47_1.v, 0xF );

  tmpc47_2.v = _mm256_shuffle_pd( c47_2.v, c47_3.v, 0x0 );
  tmpc47_3.v = _mm256_shuffle_pd( c47_2.v, c47_3.v, 0xF );

  c03_0.v    = _mm256_permute2f128_pd( tmpc03_0.v, tmpc03_2.v, 0x20 );
  c03_2.v    = _mm256_permute2f128_pd( tmpc03_0.v, tmpc03_2.v, 0x31 );

  c03_1.v    = _mm256_permute2f128_pd( tmpc03_1.v, tmpc03_3.v, 0x20 );
  c03_3.v    = _mm256_permute2f128_pd( tmpc03_1.v, tmpc03_3.v, 0x31 );

  c47_0.v    = _mm256_permute2f128_pd( tmpc47_0.v, tmpc47_2.v, 0x20 );
  c47_2.v    = _mm256_permute2f128_pd( tmpc47_0.v, tmpc47_2.v, 0x31 );

  c47_1.v    = _mm256_permute2f128_pd( tmpc47_1.v, tmpc47_3.v, 0x20 );
  c47_3.v    = _mm256_permute2f128_pd( tmpc47_1.v, tmpc47_3.v, 0x31 );


  ldr = aux->ldr;


  aa_tmp.v = _mm256_broadcast_sd( D );
  b0.v     = _mm256_cmp_pd( c03_0.v, aa_tmp.v, 0x1 );
  if ( !_mm256_testz_pd( b0.v, b0.v ) ) {
    _mm256_store_pd( c     , c03_0.v );
    hmlp::heap_select<double>( aux->jb, r, c + 0, bmap, D + 0 * ldr, I + 0 * ldr );
  }

  if ( aux->ib > 1 ) {
    aa_tmp.v = _mm256_broadcast_sd( D + ldr );
    b0.v     = _mm256_cmp_pd( c03_1.v, aa_tmp.v, 0x1 );
    if ( !_mm256_testz_pd( b0.v, b0.v ) ) {
      _mm256_store_pd( c + 4, c03_1.v );
      hmlp::heap_select<double>( aux->jb, r, c + 4, bmap, D + 1 * ldr, I + 1 * ldr );
    }
  }

  if ( aux->ib > 2 ) {
    aa_tmp.v = _mm256_broadcast_sd( D + 2 * ldr );
    b0.v     = _mm256_cmp_pd( c03_2.v, aa_tmp.v, 0x1 );
    if ( !_mm256_testz_pd( b0.v, b0.v ) ) {
      _mm256_store_pd( c + 8, c03_2.v );
      hmlp::heap_select<double>( aux->jb, r, c + 8, bmap, D + 2 * ldr, I + 2 * ldr );
    }
  }

  if ( aux->ib > 3 ) {
    aa_tmp.v = _mm256_broadcast_sd( D + 3 * ldr );
    b0.v     = _mm256_cmp_pd( c03_3.v, aa_tmp.v, 0x1 );
    if ( !_mm256_testz_pd( b0.v, b0.v ) ) {
      _mm256_store_pd( c + 12, c03_3.v );
      hmlp::heap_select<double>( aux->jb, r, c + 12, bmap, D + 3 * ldr, I + 3 * ldr );
    }
  }

  if ( aux->ib > 4 ) {
    aa_tmp.v = _mm256_broadcast_sd( D + 4 * ldr );
    b0.v     = _mm256_cmp_pd( c47_0.v, aa_tmp.v, 0x1 );
    if ( !_mm256_testz_pd( b0.v, b0.v ) ) {
      _mm256_store_pd( c + 16, c47_0.v );
      hmlp::heap_select<double>( aux->jb, r, c + 16, bmap, D + 4 * ldr, I + 4 * ldr );
    }
  }

  if ( aux->ib > 5 ) {
    aa_tmp.v = _mm256_broadcast_sd( D + 5 * ldr );
    b0.v     = _mm256_cmp_pd( c47_1.v, aa_tmp.v, 0x1 );
    if ( !_mm256_testz_pd( b0.v, b0.v ) ) {
      _mm256_store_pd( c + 20, c47_1.v );
      hmlp::heap_select<double>( aux->jb, r, c + 20, bmap, D + 5 * ldr, I + 5 * ldr );
    }
  }

  if ( aux->ib > 6 ) {
    aa_tmp.v = _mm256_broadcast_sd( D + 6 * ldr );
    b0.v     = _mm256_cmp_pd( c47_2.v, aa_tmp.v, 0x1 );
    if ( !_mm256_testz_pd( b0.v, b0.v ) ) {
      _mm256_store_pd( c + 24, c47_2.v );
      hmlp::heap_select<double>( aux->jb, r, c + 24, bmap, D + 6 * ldr, I + 6 * ldr );
    }
  }

  if ( aux->ib > 7 ) {
    aa_tmp.v = _mm256_broadcast_sd( D + 7 * ldr );
    b0.v     = _mm256_cmp_pd( c47_3.v, aa_tmp.v, 0x1 );
    if ( !_mm256_testz_pd( b0.v, b0.v ) ) {
      _mm256_store_pd( c + 28, c47_3.v );
      hmlp::heap_select<double>( aux->jb, r, c + 28, bmap, D + 7 * ldr, I + 7 * ldr );
    }
  }
  }
};
