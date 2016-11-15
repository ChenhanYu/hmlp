
/**
 *  @brief This kernel takes opkernel, op1 and op2 to implement an MR-by-NR
 *         GKMM operation.
 *
 */ 
template<
int MR, int NR,
typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
struct gkmm_mrxnr 
{
  OPKERNEL opkernel;
  OP1 op1;
  OP2 op2;
  TV initV;

  inline void operator()
  ( 
    int k, 
    TA *a, 
    TB *b, 
    TC *c, int ldc, 
    TV *v, int ldv, 
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
          regV[ j * MR + i ] = v[ j * ldv + i ];
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

    // kernel transformation and store back
    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
      #pragma simd
      for ( int i = 0; i < MR; i ++ )
        c[ j * ldc + i ] = opkernel( regV[ j * MR + i ], aux->i, aux->j, aux->b );
  };


  inline void operator()
  ( 
    int k, 
    TA *a, 
    TB *b, 
    TV *v, int ldv, 
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
          regV[ j * MR + i ] = v[ j * ldv + i ];
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

    // kernel transformation and store back
    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
      #pragma simd
      for ( int i = 0; i < MR; i ++ )
        v[ j * ldv + i ] = opkernel( regV[ j * MR + i ], aux->i, aux->j, aux->b );
  };
};


template<
int MR, int NR,
typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV>
struct gkrm_mrxnr 
{
  OPKERNEL opkernel;
  OP1 op1;
  OP2 op2;
  TV initV;
  OPKERNEL opkreduce;
  TC initC;

  inline void operator()
  ( 
    int k, 
    TA *a, 
    TB *b, 
    TC *c, int ldc, // ldc is redundant here
    TV *v, int ldv, 
    aux_s<TA, TB, TC, TV> *aux 
  ) const 
  {
    TV regV[ MR * NR ];
    TC regC[ MR ];

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
          regV[ j * MR + i ] = v[ j * ldv + i ];
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

    // Initialize
    #pragma simd
    for ( int i = 0; i < MR; i ++ )
      regC[ i ] = initC;

    // kernel transformation and reduction
    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
      #pragma simd
      for ( int i = 0; i < MR; i ++ )
        regC[ i ] = opreduce( regC[ i ], opkernel( regV[ j * MR + i ], aux->i, aux->j, aux->b ), aux->i, aux->j, aux->b );

    #pragma simd
    for ( int i = 0; i < MR; i ++ )
      c[ i ] = regC[ i ];

  };
};
