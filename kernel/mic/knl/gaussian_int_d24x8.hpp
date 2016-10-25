#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h> // AVX

#include <hmlp.h>
#include <hmlp_internal.hxx>
#include <avx_type.h> // self-defined vector type

// #define DEBUG_MICRO 1


// void gaussian_ref_d24x8(
//     int    k,
//     int    rhs,
//     //double *h,
//     double *u,
//     double *aa,
//     double *a,
//     double *bb,
//     double *b,
//     double *w,
//     double *c,
//     ks_t   *ker,
//     aux_t  *aux
//     )
// {
//   int    i, j, p;
//   double K[ 24 * 8 ] = {{ 0.0 }};
// 
//   #include <rank_k_ref_d24x8.h>
// 
//   // Gaussian kernel
//   for ( j = 0; j < 8; j ++ ) {
// 	for ( i = 0; i < 24; i ++ ) { 
// 	  K[ j * 24 + i ] = aa[ i ] - 2.0 * K[ j * 24 + i ] + bb[ j ];
//       K[ j * 24+ i ] = exp( ker->scal * K[ j * 24 + i ] );
// 	  u[ i ] += K[ j * 24 + i ] * w[ j ];
// 	}
//   }
// }
// 
// 
// void gaussian_int_s48x8(
//     int    k,
//     int    rhs,
//     //float  *h,
//     float  *u,
//     float  *aa,
//     float  *a,
//     float  *bb,
//     float  *b,
//     float  *w,
//     float  *c,
//     ks_t   *ker,
//     aux_t  *aux
//     )
// {
//   printf( "gaussian_int_s48x8 not yet implemented.\n" );
// }





struct gaussian_int_d24x8
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

    // 24 avx512 registers
    v8df_t c07_0, c07_1, c07_2, c07_3, c07_4, c07_5, c07_6, c07_7;
    v8df_t c15_0, c15_1, c15_2, c15_3, c15_4, c15_5, c15_6, c15_7;
    v8df_t c23_0, c23_1, c23_2, c23_3, c23_4, c23_5, c23_6, c23_7;

    // 8 avx512 registers
    v8df_t a07, a15, a23;
    v8df_t A07, A15, A23;
    v8df_t b0, b1;



    //printf( "a\n" );
    //printf( "%E, %E, %E, %E, %E, %E, %E, %E\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7] );
    //printf( "b\n" );
    //printf( "%E, %E, %E, %E, %E, %E, %E, %E\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7] );

    #include <rank_k_int_d24x8.h>

//    #pragma omp critical 
//    {
//      printf( "c (tid %d)\n", omp_get_thread_num() );
//      printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[0], c07_1.d[0], c07_2.d[0], c07_3.d[0], c07_4.d[0], c07_5.d[0], c07_6.d[0], c07_7.d[0] );
//      printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[1], c07_1.d[1], c07_2.d[1], c07_3.d[1], c07_4.d[1], c07_5.d[1], c07_6.d[1], c07_7.d[1] );
//      printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[2], c07_1.d[2], c07_2.d[2], c07_3.d[2], c07_4.d[2], c07_5.d[2], c07_6.d[2], c07_7.d[2] );
//      printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[3], c07_1.d[3], c07_2.d[3], c07_3.d[3], c07_4.d[3], c07_5.d[3], c07_6.d[3], c07_7.d[3] );
//      printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[4], c07_1.d[4], c07_2.d[4], c07_3.d[4], c07_4.d[4], c07_5.d[4], c07_6.d[4], c07_7.d[4] );
//      printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[5], c07_1.d[5], c07_2.d[5], c07_3.d[5], c07_4.d[5], c07_5.d[5], c07_6.d[5], c07_7.d[5] );
//      printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[6], c07_1.d[6], c07_2.d[6], c07_3.d[6], c07_4.d[6], c07_5.d[6], c07_6.d[6], c07_7.d[6] );
//      printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[7], c07_1.d[7], c07_2.d[7], c07_3.d[7], c07_4.d[7], c07_5.d[7], c07_6.d[7], c07_7.d[7] );
//    }

    #include <sq2nrm_int_d24x8.h>

    //printf( "sq2\n" );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[0], c07_1.d[0], c07_2.d[0], c07_3.d[0], c07_4.d[0], c07_5.d[0], c07_6.d[0], c07_7.d[0] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[1], c07_1.d[1], c07_2.d[1], c07_3.d[1], c07_4.d[1], c07_5.d[1], c07_6.d[1], c07_7.d[1] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[2], c07_1.d[2], c07_2.d[2], c07_3.d[2], c07_4.d[2], c07_5.d[2], c07_6.d[2], c07_7.d[2] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[3], c07_1.d[3], c07_2.d[3], c07_3.d[3], c07_4.d[3], c07_5.d[3], c07_6.d[3], c07_7.d[3] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[4], c07_1.d[4], c07_2.d[4], c07_3.d[4], c07_4.d[4], c07_5.d[4], c07_6.d[4], c07_7.d[4] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[5], c07_1.d[5], c07_2.d[5], c07_3.d[5], c07_4.d[5], c07_5.d[5], c07_6.d[5], c07_7.d[5] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[6], c07_1.d[6], c07_2.d[6], c07_3.d[6], c07_4.d[6], c07_5.d[6], c07_6.d[6], c07_7.d[6] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[7], c07_1.d[7], c07_2.d[7], c07_3.d[7], c07_4.d[7], c07_5.d[7], c07_6.d[7], c07_7.d[7] );


    // Scale before the kernel evaluation
    a07.v   = _mm512_set1_pd( alpha );
    c07_0.v = _mm512_mul_pd( a07.v, c07_0.v );
    c07_1.v = _mm512_mul_pd( a07.v, c07_1.v );
    c07_2.v = _mm512_mul_pd( a07.v, c07_2.v );
    c07_3.v = _mm512_mul_pd( a07.v, c07_3.v );
    c07_4.v = _mm512_mul_pd( a07.v, c07_4.v );
    c07_5.v = _mm512_mul_pd( a07.v, c07_5.v );
    c07_6.v = _mm512_mul_pd( a07.v, c07_6.v );
    c07_7.v = _mm512_mul_pd( a07.v, c07_7.v );

    c15_0.v = _mm512_mul_pd( a07.v, c15_0.v );
    c15_1.v = _mm512_mul_pd( a07.v, c15_1.v );
    c15_2.v = _mm512_mul_pd( a07.v, c15_2.v );
    c15_3.v = _mm512_mul_pd( a07.v, c15_3.v );
    c15_4.v = _mm512_mul_pd( a07.v, c15_4.v );
    c15_5.v = _mm512_mul_pd( a07.v, c15_5.v );
    c15_6.v = _mm512_mul_pd( a07.v, c15_6.v );
    c15_7.v = _mm512_mul_pd( a07.v, c15_7.v );

    c23_0.v = _mm512_mul_pd( a07.v, c23_0.v );
    c23_1.v = _mm512_mul_pd( a07.v, c23_1.v );
    c23_2.v = _mm512_mul_pd( a07.v, c23_2.v );
    c23_3.v = _mm512_mul_pd( a07.v, c23_3.v );
    c23_4.v = _mm512_mul_pd( a07.v, c23_4.v );
    c23_5.v = _mm512_mul_pd( a07.v, c23_5.v );
    c23_6.v = _mm512_mul_pd( a07.v, c23_6.v );
    c23_7.v = _mm512_mul_pd( a07.v, c23_7.v );

    //printf( "-1/(2h^2)\n" );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[0], c07_1.d[0], c07_2.d[0], c07_3.d[0], c07_4.d[0], c07_5.d[0], c07_6.d[0], c07_7.d[0] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[1], c07_1.d[1], c07_2.d[1], c07_3.d[1], c07_4.d[1], c07_5.d[1], c07_6.d[1], c07_7.d[1] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[2], c07_1.d[2], c07_2.d[2], c07_3.d[2], c07_4.d[2], c07_5.d[2], c07_6.d[2], c07_7.d[2] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[3], c07_1.d[3], c07_2.d[3], c07_3.d[3], c07_4.d[3], c07_5.d[3], c07_6.d[3], c07_7.d[3] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[4], c07_1.d[4], c07_2.d[4], c07_3.d[4], c07_4.d[4], c07_5.d[4], c07_6.d[4], c07_7.d[4] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[5], c07_1.d[5], c07_2.d[5], c07_3.d[5], c07_4.d[5], c07_5.d[5], c07_6.d[5], c07_7.d[5] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[6], c07_1.d[6], c07_2.d[6], c07_3.d[6], c07_4.d[6], c07_5.d[6], c07_6.d[6], c07_7.d[6] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[7], c07_1.d[7], c07_2.d[7], c07_3.d[7], c07_4.d[7], c07_5.d[7], c07_6.d[7], c07_7.d[7] );





    // Prefetch u, w
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );
    __asm__ volatile( "prefetcht0 64(%0)   \n\t" : :"r"( u ) );
    __asm__ volatile( "prefetcht0 128(%0)  \n\t" : :"r"( u ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

    // c = exp( c )
    if ( 0 ) {
      #include "exp_int_d24x8.h"
    }
    else {
      c07_0.v = _mm512_exp_pd( c07_0.v );
      c07_1.v = _mm512_exp_pd( c07_1.v );
      c07_2.v = _mm512_exp_pd( c07_2.v );
      c07_3.v = _mm512_exp_pd( c07_3.v );
      c07_4.v = _mm512_exp_pd( c07_4.v );
      c07_5.v = _mm512_exp_pd( c07_5.v );
      c07_6.v = _mm512_exp_pd( c07_6.v );
      c07_7.v = _mm512_exp_pd( c07_7.v );

      c15_0.v = _mm512_exp_pd( c15_0.v );
      c15_1.v = _mm512_exp_pd( c15_1.v );
      c15_2.v = _mm512_exp_pd( c15_2.v );
      c15_3.v = _mm512_exp_pd( c15_3.v );
      c15_4.v = _mm512_exp_pd( c15_4.v );
      c15_5.v = _mm512_exp_pd( c15_5.v );
      c15_6.v = _mm512_exp_pd( c15_6.v );
      c15_7.v = _mm512_exp_pd( c15_7.v );

      c23_0.v = _mm512_exp_pd( c23_0.v );
      c23_1.v = _mm512_exp_pd( c23_1.v );
      c23_2.v = _mm512_exp_pd( c23_2.v );
      c23_3.v = _mm512_exp_pd( c23_3.v );
      c23_4.v = _mm512_exp_pd( c23_4.v );
      c23_5.v = _mm512_exp_pd( c23_5.v );
      c23_6.v = _mm512_exp_pd( c23_6.v );
      c23_7.v = _mm512_exp_pd( c23_7.v );
    }

    //printf( "exp\n" );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[0], c07_1.d[0], c07_2.d[0], c07_3.d[0], c07_4.d[0], c07_5.d[0], c07_6.d[0], c07_7.d[0] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[1], c07_1.d[1], c07_2.d[1], c07_3.d[1], c07_4.d[1], c07_5.d[1], c07_6.d[1], c07_7.d[1] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[2], c07_1.d[2], c07_2.d[2], c07_3.d[2], c07_4.d[2], c07_5.d[2], c07_6.d[2], c07_7.d[2] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[3], c07_1.d[3], c07_2.d[3], c07_3.d[3], c07_4.d[3], c07_5.d[3], c07_6.d[3], c07_7.d[3] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[4], c07_1.d[4], c07_2.d[4], c07_3.d[4], c07_4.d[4], c07_5.d[4], c07_6.d[4], c07_7.d[4] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[5], c07_1.d[5], c07_2.d[5], c07_3.d[5], c07_4.d[5], c07_5.d[5], c07_6.d[5], c07_7.d[5] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[6], c07_1.d[6], c07_2.d[6], c07_3.d[6], c07_4.d[6], c07_5.d[6], c07_6.d[6], c07_7.d[6] );
    //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[7], c07_1.d[7], c07_2.d[7], c07_3.d[7], c07_4.d[7], c07_5.d[7], c07_6.d[7], c07_7.d[7] );



    // Preload u03, u47
    a07.v    = _mm512_load_pd( u      );
    a15.v    = _mm512_load_pd( u +  8 );
    a23.v    = _mm512_load_pd( u + 16 );

    // Multiple rhs weighted sum.
    #include<weighted_sum_int_d24x8.h>

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
