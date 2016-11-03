template<
int MR, int NR,
typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
struct semiring_mrxnr 
{
  OP1 op1;
  OP2 op2;
  TV initV;

  inline void operator()
  ( 
    int k, 
    TA *a, 
    TB *b, 
    TV *c, int ldc, 
    aux_s<TA, TB, TC, TV> *aux 
  ) const 
  {
    TV regC[ MR * NR ];

    if ( !aux->pc ) // Initialize
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma simd
        for ( int i = 0; i < MR; i ++ )
          regC[ j * MR + i ] = initV;
    }
    else // accumulate
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma simd
        for ( int i = 0; i < MR; i ++ )
          regC[ j * MR + i ] = c[ j * ldc + i ];
    }
    
    // semiring rank-k update
    for ( int p = 0; p < k; p ++ )
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma simd
        for ( int i = 0; i < MR; i ++ )
          regC[ j * MR + i ] = 
            op1( regC[ j * MR + i ], op2( a[ p * MR + i ], b[ p * NR + j ] ) );
        
    }

    // store back
    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
      #pragma simd
      for ( int i = 0; i < MR; i ++ )
        c[ j * ldc + i ] = regC[ j * MR + i ];

  };
};


