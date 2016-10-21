#include <stdio.h>
#include <math.h>
#include <immintrin.h> // AVX

#include <hmlp.h>
#include <hmlp_internal.hxx>
#include <avx_type.h> // self-defined vector type

// #define DEBUG_MICRO 1

struct gaussian_ref_d8x4 
{
  inline void operator()(
      ks_t *kernel,
      int k,
      int nrhs,
      double *u,
      double *a, double *a2, 
      double *b, double *b2,
      double *w,
      double *c,
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
          c_reg[ j * 8 + i ] += c[ j * 8 + i ];
        }
      }
    }

#ifdef DEBUG_MICRO
      printf( "gaussian_ref_d8x4 debug\n" );
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
        c_reg[ j * 8 + i ] *= kernel->scal;
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

  } // end inline void operator

}; // end struct gaussian_ref_d8x4
