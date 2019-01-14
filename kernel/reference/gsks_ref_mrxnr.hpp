#ifndef GSKS_REF_MRXNR_HPP
#define GSKS_REF_MRXNR_HPP

#include <KernelMatrix.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{

template<int MR, int NR, typename T>
struct gsks_ref_mrxnr 
{
  inline void operator()
  (
    kernel_s<T, T> *kernel,
    int k,
    int nrhs,
    T *u,
    T *a, T *a2, 
    T *b, T *b2,
    T *w,
    T *c, int ldc,
    aux_s<T, T, T, T> *aux 
  ) const 
  {
    /** use an MR-by-NR static buffer */
    T c_reg[ MR * NR ] = { 0.0 };

    /** rank-k update */
    for ( int p = 0; p < k; p ++ ) 
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma unroll
        for ( int i = 0; i < MR; i ++ ) 
          c_reg[ j * MR + i ] += a[ p * MR + i ] * b[ p * NR + j ];

    /** accumulate the previous rank-k update */
    if ( aux->pc ) 
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma unroll
        for ( int i = 0; i < MR; i ++ ) 
          c_reg[ j * MR + i ] += c[ j * ldc + i ];
    }

    switch ( kernel->type )
    {
      case GAUSSIAN:
      {
        #pragma unroll
        for ( int j = 0; j < NR; j ++ )
        {
          #pragma unroll
          for ( int i = 0; i < MR; i ++ ) 
          {
            c_reg[ j * MR + i ] *= -2.0;
            c_reg[ j * MR + i ] += a2[ i ] + b2[ j ];
            c_reg[ j * MR + i ] *= kernel->scal;
            c_reg[ j * MR + i ]  = std::exp( c_reg[ j * MR + i ] );
          }
        }
        break;
      }
      default:
      {
        exit( 1 );
      }
    }

    /** matrix-vector multiplication */
    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
      #pragma unroll
      for ( int i = 0; i < MR; i ++ ) 
        u[ i ] += c_reg[ j * MR + i ] * w[ j ];

  }; /** end inline void operator */
}; /** end struct gsks_ref_mrxnr */

}; /** end namespace hmlp */

#endif /** define GSKS_REF_MRXNR_HPP */
