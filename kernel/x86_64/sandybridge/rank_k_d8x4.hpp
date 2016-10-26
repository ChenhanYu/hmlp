#include <stdio.h>
#include <immintrin.h> // AVX

#include <hmlp_internal.hpp>
#include <avx_type.h> // self-defined vector type


// #define DEBUG_MICRO 1

struct rank_k_ref_d8x4 
{
  inline void operator()( 
      int k, 
      double *a, 
      double *b, 
      double *c, int ldc, 
      aux_s<double, double, double, double> *aux ) const 
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
          c[ j * ldc + i ] += c_reg[ j * 8 + i ];
        }
      }
    }
    else 
    {
      #pragma unroll
      for ( int j = 0; j < 4; j ++ )
      {
        #pragma unroll
        for ( int i = 0; i < 8; i ++ ) 
        {
          c[ j * ldc + i ] = c_reg[ j * 8 + i ];
        }
      }
    }

#ifdef DEBUG_MICRO
    printf( "rank_k_ref_d8x4:" );
    for ( int i = 0; i < 8; i ++ ) 
    {
      for ( int j = 0; j < 4; j ++ )
      { 
        printf( "%E ", c[ j * ldc + i ] );
      }
      printf( "\n" );
    }
#endif
  }
};


struct rank_k_int_d8x4 
{
  inline void operator()( 
      int k, 
      double *a, 
      double *b, 
      double *c, int ldc, 
      aux_s<double, double, double, double> *aux ) const 
  {
    int i;
    v4df_t    c03_0,    c03_1,    c03_2,    c03_3;
    v4df_t    c47_0,    c47_1,    c47_2,    c47_3;
    v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
    v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
    //v4df_t u03, u47;
    v4df_t a03, a47, A03, A47; // prefetched A 
    v4df_t b0, b1, b2, b3, B0; // prefetched B
    v4df_t c_tmp, aa_tmp, bb_tmp, w_tmp;


    // Rank-k update segment
    #include "rank_k_int_d8x4.segment"

    // Store back
    if ( aux->pc ) 
    {
      // packed
      if ( aux->do_packC ) 
      {
        tmpc03_0.v = _mm256_load_pd( (double*)( c      ) );
        tmpc47_0.v = _mm256_load_pd( (double*)( c + 4  ) );

        tmpc03_1.v = _mm256_load_pd( (double*)( c + 8  ) );
        tmpc47_1.v = _mm256_load_pd( (double*)( c + 12 ) );

        tmpc03_2.v = _mm256_load_pd( (double*)( c + 16 ) );
        tmpc47_2.v = _mm256_load_pd( (double*)( c + 20 ) );

        tmpc03_3.v = _mm256_load_pd( (double*)( c + 24 ) );
        tmpc47_3.v = _mm256_load_pd( (double*)( c + 28 ) );
      }
      else 
      {
        tmpc03_0.v = _mm256_load_pd( (double*)( c      ) );
        tmpc47_0.v = _mm256_load_pd( (double*)( c + 4  ) );

        tmpc03_1.v = _mm256_load_pd( (double*)( c + 1 * ldc     ) );
        tmpc47_1.v = _mm256_load_pd( (double*)( c + 1 * ldc + 4 ) );

        tmpc03_2.v = _mm256_load_pd( (double*)( c + 2 * ldc  ) );
        tmpc47_2.v = _mm256_load_pd( (double*)( c + 2 * ldc + 4 ) );

        tmpc03_3.v = _mm256_load_pd( (double*)( c + 3 * ldc ) );
        tmpc47_3.v = _mm256_load_pd( (double*)( c + 3 * ldc + 4 ) );
      }


      c03_0.v    = _mm256_add_pd( tmpc03_0.v, c03_0.v );
      c47_0.v    = _mm256_add_pd( tmpc47_0.v, c47_0.v );

      c03_1.v    = _mm256_add_pd( tmpc03_1.v, c03_1.v );
      c47_1.v    = _mm256_add_pd( tmpc47_1.v, c47_1.v );

      c03_2.v    = _mm256_add_pd( tmpc03_2.v, c03_2.v );
      c47_2.v    = _mm256_add_pd( tmpc47_2.v, c47_2.v );

      c03_3.v    = _mm256_add_pd( tmpc03_3.v, c03_3.v );
      c47_3.v    = _mm256_add_pd( tmpc47_3.v, c47_3.v );
    }

#ifdef DEBUG_MICRO
    printf( "rank_k_int_d8x4:\n" );
    printf( "%E %E %E %E\n", c03_0.d[ 0 ], c03_1.d[ 0 ], c03_2.d[ 0 ], c03_3.d[ 0 ] );
    printf( "%E %E %E %E\n", c03_0.d[ 1 ], c03_1.d[ 1 ], c03_2.d[ 1 ], c03_3.d[ 1 ] );
    printf( "%E %E %E %E\n", c03_0.d[ 2 ], c03_1.d[ 2 ], c03_2.d[ 2 ], c03_3.d[ 2 ] );
    printf( "%E %E %E %E\n", c03_0.d[ 3 ], c03_1.d[ 3 ], c03_2.d[ 3 ], c03_3.d[ 3 ] );

    printf( "%E %E %E %E\n", c47_0.d[ 0 ], c47_1.d[ 0 ], c47_2.d[ 0 ], c47_3.d[ 0 ] );
    printf( "%E %E %E %E\n", c47_0.d[ 1 ], c47_1.d[ 1 ], c47_2.d[ 1 ], c47_3.d[ 1 ] );
    printf( "%E %E %E %E\n", c47_0.d[ 2 ], c47_1.d[ 2 ], c47_2.d[ 2 ], c47_3.d[ 2 ] );
    printf( "%E %E %E %E\n", c47_0.d[ 3 ], c47_1.d[ 3 ], c47_2.d[ 3 ], c47_3.d[ 3 ] );
#endif


    if ( aux->do_packC ) 
    {
      // packed
      _mm256_store_pd( (double*)( c      ), c03_0.v );
      _mm256_store_pd( (double*)( c + 4  ), c47_0.v );

      _mm256_store_pd( (double*)( c + 8  ), c03_1.v );
      _mm256_store_pd( (double*)( c + 12 ), c47_1.v );

      _mm256_store_pd( (double*)( c + 16 ), c03_2.v );
      _mm256_store_pd( (double*)( c + 20 ), c47_2.v );

      _mm256_store_pd( (double*)( c + 24 ), c03_3.v );
      _mm256_store_pd( (double*)( c + 28 ), c47_3.v );
    }
    else 
    {
      _mm256_store_pd( (double*)( c      ), c03_0.v );
      _mm256_store_pd( (double*)( c + 4  ), c47_0.v );

      _mm256_store_pd( (double*)( c + 1 * ldc     ), c03_1.v );
      _mm256_store_pd( (double*)( c + 1 * ldc + 4 ), c47_1.v );

      _mm256_store_pd( (double*)( c + 2 * ldc     ), c03_2.v );
      _mm256_store_pd( (double*)( c + 2 * ldc + 4 ), c47_2.v );

      _mm256_store_pd( (double*)( c + 3 * ldc     ), c03_3.v );
      _mm256_store_pd( (double*)( c + 3 * ldc + 4 ), c47_3.v );
    }
  }
};
