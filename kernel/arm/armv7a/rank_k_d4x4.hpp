#include <stdio.h>
#include <hmlp_internal.hpp>

// #define DEBUG_MICRO 1

void bli_sgemm_opt_4x4
(
  dim_t               k,
  float*    restrict alpha,
  float*    restrict a,
  float*    restrict b,
  float*    restrict beta,
  float*    restrict c, inc_t rs_c, inc_t cs_c,
  aux_s<float, float, float, float> *aux
);

void bli_dgemm_opt_4x4
(
  dim_t               k,
  double*    restrict alpha,
  double*    restrict a,
  double*    restrict b,
  double*    restrict beta,
  double*    restrict c, inc_t rs_c, inc_t cs_c,
  aux_s<double, double, double, double> *aux
);


//struct rank_k_ref_d8x4 
//{
//  inline void operator()
//  ( 
//    int k, 
//    double *a, 
//    double *b, 
//    double *c, int ldc, 
//    aux_s<double, double, double, double> *aux 
//  ) const 
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
//    if ( aux->pc ) 
//    {
//      #pragma unroll
//      for ( int j = 0; j < 4; j ++ )
//      {
//        #pragma unroll
//        for ( int i = 0; i < 8; i ++ ) 
//        {
//          c[ j * ldc + i ] += c_reg[ j * 8 + i ];
//        }
//      }
//    }
//    else 
//    {
//      #pragma unroll
//      for ( int j = 0; j < 4; j ++ )
//      {
//        #pragma unroll
//        for ( int i = 0; i < 8; i ++ ) 
//        {
//          c[ j * ldc + i ] = c_reg[ j * 8 + i ];
//        }
//      }
//    }
//
//#ifdef DEBUG_MICRO
//    printf( "rank_k_ref_d8x4:" );
//    for ( int i = 0; i < 8; i ++ ) 
//    {
//      for ( int j = 0; j < 4; j ++ )
//      { 
//        printf( "%E ", c[ j * ldc + i ] );
//      }
//      printf( "\n" );
//    }
//#endif
//  }
//};



struct rank_k_asm_s4x4
{
  // Strassen interface
  inline void operator()
  ( 
      int k, 
      float *a, 
      float *b, 
      int len,
      float **c, int ldc, float *alpha,
      aux_s<float, float, float, float> *aux 
  ) const 
  {
    float c_reg[ 4 * 4 ] = { 0.0 };

    for ( int p = 0; p < k; p ++ ) 
    {
      #pragma unroll
      for ( int j = 0; j < 4; j ++ )
      {
        #pragma unroll
        for ( int i = 0; i < 4; i ++ ) 
        {
          c_reg[ j * 4 + i ] += a[ p * 4 + i ] * b[ p * 4 + j ];
        }
      }
    }

    for ( int t = 0; t < len; t ++ )
    {
      #pragma unroll
      for ( int j = 0; j < 4; j ++ )
      {
        #pragma unroll
        for ( int i = 0; i < 4; i ++ ) 
        {
          c[ t ][ j * ldc + i ] += alpha[ t ] * c_reg[ j * 4 + i ];
        }
      }
    }
  }; // end inline void operator()

  inline void operator()
  ( 
    int k, 
    float *a, 
    float *b, 
    float *c, int ldc, 
      aux_s<float, float, float, float> *aux 
  ) const 
  {
    float alpha = 1.0;
    float beta  = aux->pc ? 1.0 : 0.0;
    bli_sgemm_opt_4x4
    (
      k,
      &alpha,
      a,
      b,
      &beta,
      c, 1, ldc,
      aux
    );
  }; // end inline void operator()

}; // end struct rank_k_asm_d4x4












struct rank_k_asm_d4x4
{
  // Strassen interface
  inline void operator()
  ( 
      int k, 
      double *a, 
      double *b, 
      int len,
      double **c, int ldc, double *alpha,
      aux_s<double, double, double, double> *aux 
  ) const 
  {
    double c_reg[ 4 * 4 ] = { 0.0 };

    for ( int p = 0; p < k; p ++ ) 
    {
      #pragma unroll
      for ( int j = 0; j < 4; j ++ )
      {
        #pragma unroll
        for ( int i = 0; i < 4; i ++ ) 
        {
          c_reg[ j * 4 + i ] += a[ p * 4 + i ] * b[ p * 4 + j ];
        }
      }
    }

    for ( int t = 0; t < len; t ++ )
    {
      #pragma unroll
      for ( int j = 0; j < 4; j ++ )
      {
        #pragma unroll
        for ( int i = 0; i < 4; i ++ ) 
        {
          c[ t ][ j * ldc + i ] += alpha[ t ] * c_reg[ j * 4 + i ];
        }
      }
    }
  }; // end inline void operator()

  inline void operator()
  ( 
    int k, 
    double *a, 
    double *b, 
    double *c, int ldc, 
    aux_s<double, double, double, double> *aux 
  ) const 
  {
    double alpha = 1.0;
    double beta  = aux->pc ? 1.0 : 0.0;
    bli_dgemm_opt_4x4
    (
      k,
      &alpha,
      a,
      b,
      &beta,
      c, 1, ldc,
      aux
    );
  }; // end inline void operator()

}; // end struct rank_k_asm_d4x4
