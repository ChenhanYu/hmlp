#include <stdio.h>
#include <immintrin.h> // AVX

#include <hmlp_internal.hxx>
#include <avx_type.h> // self-defined vector type


struct rank_k_int_d24x8
{
  inline void operator()( 
      int k, 
      double *a, 
      double *b, 
      double *c, int ldc, 
      aux_s<double, double, double, double> *aux ) const 
  {
    int    i;
    double neg2 = -2.0;

    // 24 avx512 registers
    v8df_t c07_0, c07_1, c07_2, c07_3, c07_4, c07_5, c07_6, c07_7;
    v8df_t c15_0, c15_1, c15_2, c15_3, c15_4, c15_5, c15_6, c15_7;
    v8df_t c23_0, c23_1, c23_2, c23_3, c23_4, c23_5, c23_6, c23_7;

    // 8 avx512 registers
    v8df_t a07, a15, a23;
    v8df_t A07, A15, A23;
    v8df_t b0, b1;


    //__asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );
    //__asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( c ) );
    __asm__ volatile( "prefetcht0 192(%0)  \n\t" : :"r"( c ) );
    __asm__ volatile( "prefetcht0 384(%0)  \n\t" : :"r"( c ) );
    __asm__ volatile( "prefetcht0 576(%0)  \n\t" : :"r"( c ) );
    __asm__ volatile( "prefetcht0 768(%0)  \n\t" : :"r"( c ) );
    __asm__ volatile( "prefetcht0 960(%0)  \n\t" : :"r"( c ) );
    __asm__ volatile( "prefetcht0 1152(%0)  \n\t" : :"r"( c ) );
    __asm__ volatile( "prefetcht0 1344(%0)  \n\t" : :"r"( c ) );


    #include "rank_k_int_d24x8.h"



    //  printf( "%lf, %lf, %lf, %lf\n", c07_0.d[0], c07_1.d[0], c07_2.d[0], c07_3.d[0], c07_4.d[0], c07_5.d[0], c07_6.d[0], c07_7.d[0] );
    //  printf( "%lf, %lf, %lf, %lf\n", c07_0.d[1], c07_1.d[1], c07_2.d[1], c07_3.d[1], c07_4.d[1], c07_5.d[1], c07_6.d[1], c07_7.d[1] );
    //  printf( "%lf, %lf, %lf, %lf\n", c07_0.d[2], c07_1.d[2], c07_2.d[2], c07_3.d[2], c07_4.d[2], c07_5.d[2], c07_6.d[2], c07_7.d[2] );
    //  printf( "%lf, %lf, %lf, %lf\n", c07_0.d[3], c07_1.d[3], c07_2.d[3], c07_3.d[3], c07_4.d[3], c07_5.d[3], c07_6.d[3], c07_7.d[3] );
    //  printf( "%lf, %lf, %lf, %lf\n", c07_0.d[4], c07_1.d[4], c07_2.d[4], c07_3.d[4], c07_4.d[4], c07_5.d[4], c07_6.d[4], c07_7.d[4] );
    //  printf( "%lf, %lf, %lf, %lf\n", c07_0.d[5], c07_1.d[5], c07_2.d[5], c07_3.d[5], c07_4.d[5], c07_5.d[5], c07_6.d[5], c07_7.d[5] );
    //  printf( "%lf, %lf, %lf, %lf\n", c07_0.d[6], c07_1.d[6], c07_2.d[6], c07_3.d[6], c07_4.d[6], c07_5.d[6], c07_6.d[6], c07_7.d[6] );
    //  printf( "%lf, %lf, %lf, %lf\n", c07_0.d[7], c07_1.d[7], c07_2.d[7], c07_3.d[7], c07_4.d[7], c07_5.d[7], c07_6.d[7], c07_7.d[7] );


    // nonpacked
    /*
       _mm256_store_pd( (double*)( c               ), c03_0.v );
       _mm256_store_pd( (double*)( c + 4           ), c47_0.v );

       _mm256_store_pd( (double*)( c + ldc * 1     ), c03_1.v );
       _mm256_store_pd( (double*)( c + ldc * 1 + 4 ), c47_1.v );

       _mm256_store_pd( (double*)( c + ldc * 2     ), c03_2.v );
       _mm256_store_pd( (double*)( c + ldc * 2 + 4 ), c47_2.v );

       _mm256_store_pd( (double*)( c + ldc * 3     ), c03_3.v );
       _mm256_store_pd( (double*)( c + ldc * 3 + 4 ), c47_3.v );
       */

    // packed
    if ( 0 ) {
      _mm512_stream_pd( c       , c07_0.v );
      _mm512_stream_pd( c +   8 , c15_0.v );
      _mm512_stream_pd( c +  16 , c23_0.v );

      _mm512_stream_pd( c +  24 , c07_1.v );
      _mm512_stream_pd( c +  32 , c15_1.v );
      _mm512_stream_pd( c +  40 , c23_1.v );

      _mm512_stream_pd( c +  48 , c07_2.v );
      _mm512_stream_pd( c +  56 , c15_2.v );
      _mm512_stream_pd( c +  64 , c23_2.v );

      _mm512_stream_pd( c +  72 , c07_3.v );
      _mm512_stream_pd( c +  80 , c15_3.v );
      _mm512_stream_pd( c +  88 , c23_3.v );

      //_mm512_stream_pd( c +  96 , c07_4.v );
      //_mm512_stream_pd( c + 104 , c15_4.v );
      //_mm512_stream_pd( c + 112 , c23_4.v );

      //_mm512_stream_pd( c + 120 , c07_5.v );
      //_mm512_stream_pd( c + 128 , c15_5.v );
      //_mm512_stream_pd( c + 136 , c23_5.v );

      //_mm512_stream_pd( c + 144 , c07_6.v );
      //_mm512_stream_pd( c + 152 , c15_6.v );
      //_mm512_stream_pd( c + 160 , c23_6.v );

      //_mm512_stream_pd( c + 168 , c07_7.v );
      //_mm512_stream_pd( c + 176 , c15_7.v );
      //_mm512_stream_pd( c + 184 , c23_7.v );

      _mm512_store_pd( c +  96 , c07_4.v );
      _mm512_store_pd( c + 104 , c15_4.v );
      _mm512_store_pd( c + 112 , c23_4.v );

      _mm512_store_pd( c + 120 , c07_5.v );
      _mm512_store_pd( c + 128 , c15_5.v );
      _mm512_store_pd( c + 136 , c23_5.v );

      _mm512_store_pd( c + 144 , c07_6.v );
      _mm512_store_pd( c + 152 , c15_6.v );
      _mm512_store_pd( c + 160 , c23_6.v );

      _mm512_store_pd( c + 168 , c07_7.v );
      _mm512_store_pd( c + 176 , c15_7.v );
      _mm512_store_pd( c + 184 , c23_7.v );
    }
    else {
      _mm512_store_pd( c       , c07_0.v );
      _mm512_store_pd( c +   8 , c15_0.v );
      _mm512_store_pd( c +  16 , c23_0.v );

      _mm512_store_pd( c +  24 , c07_1.v );
      _mm512_store_pd( c +  32 , c15_1.v );
      _mm512_store_pd( c +  40 , c23_1.v );

      _mm512_store_pd( c +  48 , c07_2.v );
      _mm512_store_pd( c +  56 , c15_2.v );
      _mm512_store_pd( c +  64 , c23_2.v );

      _mm512_store_pd( c +  72 , c07_3.v );
      _mm512_store_pd( c +  80 , c15_3.v );
      _mm512_store_pd( c +  88 , c23_3.v );

      _mm512_store_pd( c +  96 , c07_4.v );
      _mm512_store_pd( c + 104 , c15_4.v );
      _mm512_store_pd( c + 112 , c23_4.v );

      _mm512_store_pd( c + 120 , c07_5.v );
      _mm512_store_pd( c + 128 , c15_5.v );
      _mm512_store_pd( c + 136 , c23_5.v );

      _mm512_store_pd( c + 144 , c07_6.v );
      _mm512_store_pd( c + 152 , c15_6.v );
      _mm512_store_pd( c + 160 , c23_6.v );

      _mm512_store_pd( c + 168 , c07_7.v );
      _mm512_store_pd( c + 176 , c15_7.v );
      _mm512_store_pd( c + 184 , c23_7.v );
    }

    //printf( "ldc = %d\n", ldc );
    //printf( "%lf, %lf, %lf, %lf\n", c[0], c[ ldc + 0], c[ ldc * 2 + 0], c[ ldc * 3 + 0] );
    //printf( "%lf, %lf, %lf, %lf\n", c[1], c[ ldc + 1], c[ ldc * 2 + 1], c[ ldc * 3 + 1] );
    //printf( "%lf, %lf, %lf, %lf\n", c[2], c[ ldc + 2], c[ ldc * 2 + 2], c[ ldc * 3 + 2] );
    //printf( "%lf, %lf, %lf, %lf\n", c[3], c[ ldc + 3], c[ ldc * 2 + 3], c[ ldc * 3 + 3] );
    //printf( "%lf, %lf, %lf, %lf\n", c[4], c[ ldc + 4], c[ ldc * 2 + 4], c[ ldc * 3 + 4] );
    //printf( "%lf, %lf, %lf, %lf\n", c[5], c[ ldc + 5], c[ ldc * 2 + 5], c[ ldc * 3 + 5] );
    //printf( "%lf, %lf, %lf, %lf\n", c[6], c[ ldc + 6], c[ ldc * 2 + 6], c[ ldc * 3 + 6] );
    //printf( "%lf, %lf, %lf, %lf\n", c[7], c[ ldc + 7], c[ ldc * 2 + 7], c[ ldc * 3 + 7] );
  }
};
