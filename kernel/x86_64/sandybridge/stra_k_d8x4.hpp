#include <stdio.h>
#include <immintrin.h> // AVX

#include <hmlp_internal.hpp>
#include <avx_type.h> // self-defined vector type

// #define DEBUG_MICRO 1

struct stra_k_ref_d8x4 
{
  inline void operator()( 
      int k, 
      double *a, 
      double *b, 
      int len,
      double **c, int ldc, double *alpha,
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

    for ( int t = 0; t < len; t ++ )
    {
      #pragma unroll
      for ( int j = 0; j < 4; j ++ )
      {
        #pragma unroll
        for ( int i = 0; i < 8; i ++ ) 
        {
          c[t][ j * ldc + i ] += alpha[t] * c_reg[ j * 8 + i ];
        }
      }
    }

  }
};

//struct stra_k_ref_d8x4 
//{
//  inline void operator()( 
//      int k, 
//      double *a, 
//      double *b, 
//      int len,
//      double *c0, double *c1, int ldc, double alpha0, double alpha1,
//      aux_s<double, double, double, double> *aux ) const 
//  {
//    double c_reg[ 8 * 4 ] = { 0.0 };
//
//    for ( int p = 0; p < k; p ++ ) 
//    {
//      #pragma unroll
//      for ( int j = 0; j < 4; j ++ )
//      {
//        #pragma unroll
//        for ( int i = 0; i < 8; i ++ ) 
//        {
//          c_reg[ j * 8 + i ] += a[ p * 8 + i ] * b[ p * 4 + j ];
//        }
//      }
//    }
//
//    //if ( aux->pc ) 
//    {
//      #pragma unroll
//      for ( int j = 0; j < 4; j ++ )
//      {
//        #pragma unroll
//        for ( int i = 0; i < 8; i ++ ) 
//        {
//          c0[ j * ldc + i ] += alpha0 * c_reg[ j * 8 + i ];
//          c1[ j * ldc + i ] += alpha1 * c_reg[ j * 8 + i ];
//        }
//      }
//    }
//    //else 
//    //{
//    //  #pragma unroll
//    //  for ( int j = 0; j < 4; j ++ )
//    //  {
//    //    #pragma unroll
//    //    for ( int i = 0; i < 8; i ++ ) 
//    //    {
//    //      c0[ j * ldc + i ] = alpha0 * c_reg[ j * 8 + i ];
//    //      c1[ j * ldc + i ] = alpha1 * c_reg[ j * 8 + i ];
//    //    }
//    //  }
//    //}
//
//#ifdef DEBUG_MICRO
//    printf( "stra_k_ref_d8x4:" );
//    for ( int i = 0; i < 8; i ++ ) 
//    {
//      for ( int j = 0; j < 4; j ++ )
//      { 
//        printf( "%E ", c0[ j * ldc + i ] );
//        printf( "%E ", c1[ j * ldc + i ] );
//      }
//      printf( "\n" );
//    }
//#endif
//  }
//};


//struct stra_k_int_d8x4 
//{
//  inline void operator()( 
//      int k, 
//      double *a, 
//      double *b, 
//      double *c0, double *c1, int ldc, 
//      aux_s<double, double, double, double> *aux ) const 
//  {
//    int i;
//    v4df_t    c03_0,    c03_1,    c03_2,    c03_3;
//    v4df_t    c47_0,    c47_1,    c47_2,    c47_3;
//    v4df_t tmpc03_0, tmpc03_1, tmpc03_2, tmpc03_3;
//    v4df_t tmpc47_0, tmpc47_1, tmpc47_2, tmpc47_3;
//    //v4df_t u03, u47;
//    v4df_t a03, a47, A03, A47; // prefetched A 
//    v4df_t b0, b1, b2, b3, B0; // prefetched B
//    v4df_t c_tmp, aa_tmp, bb_tmp, w_tmp;
//
//
//    // Rank-k update segment
//    #include "stra_k_int_d8x4.segment"
//
//    // Store back
//    if ( aux->pc ) 
//    {
//      // packed
//      if ( aux->do_packC ) 
//      {
//        tmpc03_0.v = _mm256_load_pd( (double*)( c      ) );
//        tmpc47_0.v = _mm256_load_pd( (double*)( c + 4  ) );
//
//        tmpc03_1.v = _mm256_load_pd( (double*)( c + 8  ) );
//        tmpc47_1.v = _mm256_load_pd( (double*)( c + 12 ) );
//
//        tmpc03_2.v = _mm256_load_pd( (double*)( c + 16 ) );
//        tmpc47_2.v = _mm256_load_pd( (double*)( c + 20 ) );
//
//        tmpc03_3.v = _mm256_load_pd( (double*)( c + 24 ) );
//        tmpc47_3.v = _mm256_load_pd( (double*)( c + 28 ) );
//      }
//      else 
//      {
//        tmpc03_0.v = _mm256_load_pd( (double*)( c      ) );
//        tmpc47_0.v = _mm256_load_pd( (double*)( c + 4  ) );
//
//        tmpc03_1.v = _mm256_load_pd( (double*)( c + 1 * ldc     ) );
//        tmpc47_1.v = _mm256_load_pd( (double*)( c + 1 * ldc + 4 ) );
//
//        tmpc03_2.v = _mm256_load_pd( (double*)( c + 2 * ldc  ) );
//        tmpc47_2.v = _mm256_load_pd( (double*)( c + 2 * ldc + 4 ) );
//
//        tmpc03_3.v = _mm256_load_pd( (double*)( c + 3 * ldc ) );
//        tmpc47_3.v = _mm256_load_pd( (double*)( c + 3 * ldc + 4 ) );
//      }
//
//
//      c03_0.v    = _mm256_add_pd( tmpc03_0.v, c03_0.v );
//      c47_0.v    = _mm256_add_pd( tmpc47_0.v, c47_0.v );
//
//      c03_1.v    = _mm256_add_pd( tmpc03_1.v, c03_1.v );
//      c47_1.v    = _mm256_add_pd( tmpc47_1.v, c47_1.v );
//
//      c03_2.v    = _mm256_add_pd( tmpc03_2.v, c03_2.v );
//      c47_2.v    = _mm256_add_pd( tmpc47_2.v, c47_2.v );
//
//      c03_3.v    = _mm256_add_pd( tmpc03_3.v, c03_3.v );
//      c47_3.v    = _mm256_add_pd( tmpc47_3.v, c47_3.v );
//    }
//
//#ifdef DEBUG_MICRO
//    printf( "stra_k_int_d8x4:\n" );
//    printf( "%E %E %E %E\n", c03_0.d[ 0 ], c03_1.d[ 0 ], c03_2.d[ 0 ], c03_3.d[ 0 ] );
//    printf( "%E %E %E %E\n", c03_0.d[ 1 ], c03_1.d[ 1 ], c03_2.d[ 1 ], c03_3.d[ 1 ] );
//    printf( "%E %E %E %E\n", c03_0.d[ 2 ], c03_1.d[ 2 ], c03_2.d[ 2 ], c03_3.d[ 2 ] );
//    printf( "%E %E %E %E\n", c03_0.d[ 3 ], c03_1.d[ 3 ], c03_2.d[ 3 ], c03_3.d[ 3 ] );
//
//    printf( "%E %E %E %E\n", c47_0.d[ 0 ], c47_1.d[ 0 ], c47_2.d[ 0 ], c47_3.d[ 0 ] );
//    printf( "%E %E %E %E\n", c47_0.d[ 1 ], c47_1.d[ 1 ], c47_2.d[ 1 ], c47_3.d[ 1 ] );
//    printf( "%E %E %E %E\n", c47_0.d[ 2 ], c47_1.d[ 2 ], c47_2.d[ 2 ], c47_3.d[ 2 ] );
//    printf( "%E %E %E %E\n", c47_0.d[ 3 ], c47_1.d[ 3 ], c47_2.d[ 3 ], c47_3.d[ 3 ] );
//#endif
//
//
//    if ( aux->do_packC ) 
//    {
//      // packed
//      _mm256_store_pd( (double*)( c      ), c03_0.v );
//      _mm256_store_pd( (double*)( c + 4  ), c47_0.v );
//
//      _mm256_store_pd( (double*)( c + 8  ), c03_1.v );
//      _mm256_store_pd( (double*)( c + 12 ), c47_1.v );
//
//      _mm256_store_pd( (double*)( c + 16 ), c03_2.v );
//      _mm256_store_pd( (double*)( c + 20 ), c47_2.v );
//
//      _mm256_store_pd( (double*)( c + 24 ), c03_3.v );
//      _mm256_store_pd( (double*)( c + 28 ), c47_3.v );
//    }
//    else 
//    {
//      _mm256_store_pd( (double*)( c      ), c03_0.v );
//      _mm256_store_pd( (double*)( c + 4  ), c47_0.v );
//
//      _mm256_store_pd( (double*)( c + 1 * ldc     ), c03_1.v );
//      _mm256_store_pd( (double*)( c + 1 * ldc + 4 ), c47_1.v );
//
//      _mm256_store_pd( (double*)( c + 2 * ldc     ), c03_2.v );
//      _mm256_store_pd( (double*)( c + 2 * ldc + 4 ), c47_2.v );
//
//      _mm256_store_pd( (double*)( c + 3 * ldc     ), c03_3.v );
//      _mm256_store_pd( (double*)( c + 3 * ldc + 4 ), c47_3.v );
//    }
//  }
//};


struct stra_k_asm_d8x4 
{
  inline void operator()( 
      int k, 
      double *a, 
      double *b, 
      int len,
      double **my_c_list, int c_ldc, double *alpha_list,
      aux_s<double, double, double, double> *aux )
      //aux_s<double, double, double, double> *aux ) const 
  {
    //printf( "enter micro\n" );
    unsigned long long len_c = (unsigned long long)(len);
    unsigned long long ldc   = (unsigned long long)(c_ldc);

	double*   b_next = (double *)(aux->b_next);
    unsigned long long k_iter = (unsigned long long)k / 4;
    unsigned long long k_left = (unsigned long long)k % 4;

    double** c_list = my_c_list;

    //printf( "k_iter:%lu,k_left:%lu\n", k_iter, k_left );

    //int i, j;
    //printf( "c0:\n" );
    //for ( i = 0; i < 8; i ++ ) {
    //  for ( j = 0; j < 4; j ++ ) {
    //    printf( "%lf ", c_list[0][j * (int)ldc + i] );
    //  }
    //  printf( "\n" );
    //}
    //printf( "c1:\n" );
    //for ( i = 0; i < 8; i ++ ) {
    //  for ( j = 0; j < 4; j ++ ) {
    //    printf( "%lf ", c_list[1][j * (int)ldc + i] );
    //  }
    //  printf( "\n" );
    //}
    ////printf( "c_list[0][0]: %lf\n", c_list[0][0] );
    ////printf( "c_list[1][0]: %lf\n", c_list[1][0] );

    //printf( "before asm\n" );

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
    "movq                %2, %%rax               \n\t" // load address of a.              ( v )
    "movq                %3, %%rbx               \n\t" // load address of b.              ( v )
    "movq                %8, %%r15               \n\t" // load address of b_next.         ( v )
    "addq          $-4 * 64, %%r15               \n\t" //                                 ( ? )
    "                                            \n\t"
    "vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // initialize loop by pre-loading
    "vmovapd   0 * 32(%%rbx), %%ymm2             \n\t" // elements of a and b.
    "vpermilpd  $0x5, %%ymm2, %%ymm3             \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq                %7, %%rdi               \n\t" // load ldc
    "leaq        (,%%rdi,8), %%rdi               \n\t" // ldc * sizeof(double)
    "                                            \n\t"
    "movq                %6, %%rcx               \n\t" // load address of c_list[ 0 ]
    "                                            \n\t"
	"movq      %5, %%rsi                         \n\t" // i = len;                        ( v )
    "                                            \n\t"
    ".DPREFETCHLOOP:                             \n\t"
    "                                            \n\t"
	"movq       0 * 8(%%rcx),  %%rdx             \n\t" // load address of c_list[ i ]: rdx = c_list[ i ] ( address )
    "                                            \n\t"
	"testq  %%rdx, %%rdx                         \n\t" // check rdx via logical AND.      ( v )
	"je     .DC1NULL                             \n\t" // if rdx == 0, jump to code that  ( v )
    "leaq   (%%rdx,%%rdi,2), %%r11               \n\t" // load address of c_list[ i ] + 2 * ldc;
    "prefetcht0   3 * 8(%%rdx)                   \n\t" // prefetch c_list[ i ] + 0 * ldc
    "prefetcht0   3 * 8(%%rdx,%%rdi)             \n\t" // prefetch c_list[ i ] + 1 * ldc
    "prefetcht0   3 * 8(%%r11)                   \n\t" // prefetch c_list[ i ] + 2 * ldc
    "prefetcht0   3 * 8(%%r11,%%rdi)             \n\t" // prefetch c_list[ i ] + 3 * ldc
    "                                            \n\t"
    ".DC1NULL:                                   \n\t" // if C1 == NULL, code to jump
    "                                            \n\t"
	"addq              $1 * 8,  %%rcx            \n\t" // c_list += 8
    "                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DPREFETCHLOOP                       \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
	"vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \n\t" // set ymm8 to 0                   ( v )
	"vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
	"vxorpd    %%ymm10, %%ymm10, %%ymm10         \n\t"
	"vxorpd    %%ymm11, %%ymm11, %%ymm11         \n\t"
	"vxorpd    %%ymm12, %%ymm12, %%ymm12         \n\t"
	"vxorpd    %%ymm13, %%ymm13, %%ymm13         \n\t"
	"vxorpd    %%ymm14, %%ymm14, %%ymm14         \n\t"
	"vxorpd    %%ymm15, %%ymm15, %%ymm15         \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;                     ( v )
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.        ( v )
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that    ( v )
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"addq         $4 * 4 * 8,  %%r15             \n\t" // b_next += 4*4 (unroll x nr)     ( v )
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 0
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t" // ymm6 ( c_tmp0 ) = ymm0 ( a03 ) * ymm2( b0 )
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t" // ymm4 ( b0x3_0 )
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t" // ymm7 ( c_tmp1 ) = ymm0 ( a03 ) * ymm3( b0x5 )
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t" // ymm5 ( b0x3_1 )
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t" // ymm15 ( c_03_0 ) += ymm6( c_tmp0 )
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t" // ymm13 ( c_03_1 ) += ymm7( c_tmp1 )
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t" // prefetch a03 for iter 1
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 1
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   2 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 1
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"prefetcht0   0 * 32(%%r15)                  \n\t" // prefetch b_next[0*4]
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmovapd   3 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 1
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  18 * 32(%%rax)                  \n\t" // prefetch a for iter 9  ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   2 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 2 
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   4 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 2
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vmovapd   5 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 2
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  20 * 32(%%rax)                  \n\t" // prefetch a for iter 10 ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   3 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 3
	"addq         $4 * 4 * 8,  %%rbx             \n\t" // b += 4*4 (unroll x nr)
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   6 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 3
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"prefetcht0   2 * 32(%%r15)                  \n\t" // prefetch b_next[2*4]
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmovapd   7 * 32(%%rax),  %%ymm1            \n\t" // preload a47 for iter 3
	"addq         $4 * 8 * 8,  %%rax             \n\t" // a += 4*8 (unroll x mr)
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t" // prefetch a for iter 11 ( ? )
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   0 * 32(%%rbx),  %%ymm2            \n\t" // preload b for iter 4
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t" // preload a03 for iter 4
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t" // preload a47 
	"addq         $8 * 1 * 8,  %%rax             \n\t" // a += 8 (1 x mr)
	"vmulpd           %%ymm0,  %%ymm2, %%ymm6    \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2, %%ymm4    \n\t"
	"vmulpd           %%ymm0,  %%ymm3, %%ymm7    \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3, %%ymm5    \n\t"
	"vaddpd           %%ymm15, %%ymm6, %%ymm15   \n\t"
	"vaddpd           %%ymm13, %%ymm7, %%ymm13   \n\t"
	"                                            \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t" // prefetch a03 for iter 7 later ( ? )
	"vmulpd           %%ymm1,  %%ymm2, %%ymm6    \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t"
	"addq         $4 * 1 * 8,  %%rbx             \n\t" // b += 4 (1 x nr)
	"vmulpd           %%ymm1,  %%ymm3, %%ymm7    \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6, %%ymm14   \n\t"
	"vaddpd           %%ymm12, %%ymm7, %%ymm12   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4, %%ymm6    \n\t"
	"vmulpd           %%ymm0,  %%ymm5, %%ymm7    \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t"
	"vaddpd           %%ymm11, %%ymm6, %%ymm11   \n\t"
	"vaddpd           %%ymm9,  %%ymm7, %%ymm9    \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4, %%ymm6    \n\t"
	"vmulpd           %%ymm1,  %%ymm5, %%ymm7    \n\t"
	"vaddpd           %%ymm10, %%ymm6, %%ymm10   \n\t"
	"vaddpd           %%ymm8,  %%ymm7, %%ymm8    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab11    ab10    ab13    ab12  
	"                                            \n\t" //   ab22    ab23    ab20    ab21
	"                                            \n\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab51    ab50    ab53    ab52  
	"                                            \n\t" //   ab62    ab63    ab60    ab61
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"vmovapd          %%ymm15, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm15, %%ymm13, %%ymm15  \n\t"
	"vshufpd    $0xa, %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm11, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm11, %%ymm9,  %%ymm11  \n\t"
	"vshufpd    $0xa, %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm14, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm14, %%ymm12, %%ymm14  \n\t"
	"vshufpd    $0xa, %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm10, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm10, %%ymm8,  %%ymm10  \n\t"
	"vshufpd    $0xa, %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab01  ( ab00  ( ab03  ( ab02
	"                                            \n\t" //   ab11    ab10    ab13    ab12  
	"                                            \n\t" //   ab23    ab22    ab21    ab20
	"                                            \n\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab41  ( ab40  ( ab43  ( ab42
	"                                            \n\t" //   ab51    ab50    ab53    ab52  
	"                                            \n\t" //   ab63    ab62    ab61    ab60
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"vmovapd           %%ymm15, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm15, %%ymm11, %%ymm15 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm11, %%ymm11 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm13, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm13, %%ymm9,  %%ymm13 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm9,  %%ymm9  \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm14, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm14, %%ymm10, %%ymm14 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm10, %%ymm10 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm12, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm12, %%ymm8,  %%ymm12 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm8,  %%ymm8  \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm9:   ymm11:  ymm13:  ymm15:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab10    ab11    ab12    ab13  
	"                                            \n\t" //   ab20    ab21    ab22    ab23
	"                                            \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
	"                                            \n\t"
	"                                            \n\t" // ymm8:   ymm10:  ymm12:  ymm14:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab50    ab51    ab52    ab53  
	"                                            \n\t" //   ab60    ab61    ab62    ab63
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )  ab73 )
	"                                            \n\t"
	"                                            \n\t"
    "movq         %4, %%rax                      \n\t" // load address of alpha_list[ 0 ]
    "movq         %6, %%rcx                      \n\t" // load address of c_list[ 0 ]
	"                                            \n\t"
    "                                            \n\t"
	"movq      %5, %%rsi                         \n\t" // i = len;                        ( v )
    "                                            \n\t"
    ".DSTORELOOP:                                \n\t"
    "                                            \n\t"
	"movq       0 * 8(%%rcx),  %%rdx             \n\t" // rdx = c_list[ i ] ( address )
    "                                            \n\t"
	//"movq       0 * 8(%%rax),  %%rbx             \n\t" // load address of alpha_list[ i ]
	//"vbroadcastsd    (%%rbx), %%ymm6             \n\t" // load alpha_list[ 1 ] and duplicate
	"vbroadcastsd    (%%rax), %%ymm6             \n\t" // load alpha_list[ i ] and duplicate
    "                                            \n\t"
    "                                            \n\t"
	//"jmp              .DDONE                      \n\t"
	"vmovapd    0 * 32(%%rdx),  %%ymm0           \n\t" // ymm0 = c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm9,  %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm9( ab0_3:0 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmovapd    1 * 32(%%rdx),  %%ymm3           \n\t" // ymm3 = c_list[1]( 4:7, 0 )
	"vmulpd            %%ymm6,  %%ymm8,  %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm8( ab4_7:0 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rdx),  %%ymm0           \n\t" // ymm0 = c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm11, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm11( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmovapd    1 * 32(%%rdx),  %%ymm3           \n\t" // ymm3 = c_list[1]( 4:7, 0 )
	"vmulpd            %%ymm6,  %%ymm10, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm10( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rdx),  %%ymm0           \n\t" // ymm0 = c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm13, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm13( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmovapd    1 * 32(%%rdx),  %%ymm3           \n\t" // ymm3 = c_list[1]( 4:7, 0 )
	"vmulpd            %%ymm6,  %%ymm12, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm12( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
	"addq              %%rdi,   %%rdx            \n\t"
    "                                            \n\t"
	"vmovapd    0 * 32(%%rdx),  %%ymm0           \n\t" // ymm0 = c_list[1]( 0:3, 0 )
	"vmulpd            %%ymm6,  %%ymm15, %%ymm1  \n\t" // scale by alpha2, ymm1 = ymm6( alpha2 ) * ymm15( ab0_3:1 )
	"vaddpd            %%ymm1,  %%ymm0,  %%ymm1  \n\t" // ymm1 = ymm0 + ymm1
	"vmovapd           %%ymm1,  0 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 0:3, 0 )
	"vmovapd    1 * 32(%%rdx),  %%ymm3           \n\t" // ymm3 = c_list[1]( 4:7, 0 )
	"vmulpd            %%ymm6,  %%ymm14, %%ymm2  \n\t" // scale by alpha2, ymm2 = ymm6( alpha2 ) * ymm14( ab4_7:1 )
	"vaddpd            %%ymm2,  %%ymm3,  %%ymm2  \n\t" // ymm2 = ymm3 + ymm2
	"vmovapd           %%ymm2,  1 * 32(%%rdx)    \n\t" // and store back to memory: c_list[1]( 4:7, 0 )
    "                                            \n\t"
    "                                            \n\t"
	"addq              $1 * 8,  %%rcx            \n\t" // c_list += 8
	"addq              $1 * 8,  %%rax            \n\t" // alpha_list += 8
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DSTORELOOP                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
    ".DDONE:                                     \n\t"
	"                                            \n\t"
	: // output operands (none)
	: // input operands
	  "m" (k_iter),             // 0
	  "m" (k_left),             // 1
	  "m" (a),                  // 2
	  "m" (b),                  // 3
	  "m" (alpha_list),         // 4
	  "m" (len_c),              // 5
	  "m" (c_list),             // 6
      "m" (ldc),                // 7
	  "m" (b_next)              // 8
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx",  "rdi", "rsi",
      "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);

    //printf( "leave micro\n" );
  }
};
