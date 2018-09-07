#ifndef SEMIRING_MRXNR_HPP
#define SEMIRING_MRXNR_HPP

#include <hmlp_internal.hpp>


template<
int MR, int NR,
typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
struct semiring_mrxnr 
{
  const static size_t mr         = MR;
  const static size_t nr         = NR;
  const static size_t pack_mr    = MR;
  const static size_t pack_nr    = NR;
  const static size_t align_size = 32;

  OP1 op1;
  OP2 op2;
  TV initV;

  /** Strassen interface: ignore op1, op2 and initV */
  inline void operator()
  ( 
    int k, 
    TA *a, 
    TB *b,
    int len,
    TV **v_list, int ldv, TV *alpha_list, 
    aux_s<TA, TB, TC, TV> *aux 
  ) const 
  {
    TV regV[ MR * NR ] = { 0.0 };

    // semiring rank-k update
    for ( int p = 0; p < k; p ++ )
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma simd
        for ( int i = 0; i < MR; i ++ )
          regV[ j * MR + i ] += a[ p * MR + i ] * b[ p * NR + j ];
    }

    // store back
    for ( int t = 0; t < len; t ++ )
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
      {
        #pragma simd
        for ( int i = 0; i < MR; i ++ ) 
        {
          v_list[ t ][ j * ldv + i ] += alpha_list[ t ] * regV[ j * MR + i ];
        }
      }
    }
  };

  /** Non-Strassen interface */
  inline void operator()
  ( 
    dim_t k, 
    TA *a, 
    TB *b, 
    TV *v, int rs_c, int cs_c,
    aux_s<TA, TB, TC, TV> *aux 
  ) const 
  {
    TV regV[ MR * NR ];

    if ( !aux->pc ) // Initialize
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma simd
        for ( int i = 0; i < MR; i ++ )
          regV[ j * MR + i ] = initV;
    }
    else // accumulate
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma simd
        for ( int i = 0; i < MR; i ++ )
          regV[ j * MR + i ] = v[ j * cs_c + i * rs_c ];
    }
    
    // semiring rank-k update
    for ( int p = 0; p < k; p ++ )
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma simd
        for ( int i = 0; i < MR; i ++ )
          regV[ j * MR + i ] = 
            op1( regV[ j * MR + i ], op2( a[ p * MR + i ], b[ p * NR + j ] ) );
        
    }

    // store back
    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
      #pragma simd
      for ( int i = 0; i < MR; i ++ )
        v[ j * cs_c + i * rs_c ] = regV[ j * MR + i ];

  };
};


#endif /** define SEMIRING_MRXNR_HPP */
