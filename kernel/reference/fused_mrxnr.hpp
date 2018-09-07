#ifndef FUSED_MRXNR_HPP
#define FUSED_MRXNR_HPP

/**
 *  @brief This kernel takes opkernel, op1 and op2 to implement an MR-by-NR
 *         GKMM operation.
 *
 */ 
template<int MR, int NR,
typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
struct gkmm_mrxnr 
{
  const static size_t mr         = MR;
  const static size_t nr         = NR;
  const static size_t pack_mr    = MR;
  const static size_t pack_nr    = NR;
  const static size_t align_size = 32;

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

    // kernel transformation and store back
    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
      #pragma simd
      for ( int i = 0; i < MR; i ++ )
        v[ j * cs_c + i * rs_c ] = opkernel( regV[ j * MR + i ], aux->i, aux->j, aux->b );
  };
};


template<
int MR, int NR,
typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV>
struct gkrm_mrxnr 
{
  const static size_t mr         = MR;
  const static size_t nr         = NR;
  const static size_t pack_mr    = MR;
  const static size_t pack_nr    = NR;
  const static size_t align_size = 32;

  OPKERNEL opkernel;
  OP1 op1;
  OP2 op2;
  TV initV;
  OPKERNEL opreduce;
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

    // Here we need omp atomic update
    for ( int i = 0; i < MR; i ++ )
    {
      TC *cptr = c + i;
#ifdef USE_INTEL
      #pragma omp atomic update
#else
			#pragma omp critical
#endif
      *c = opreduce( *c, regC[ i ] );
    }
  };

}; /** end struct gkrm_mrxnr */



template<
int MR, int NR,
typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TPACKC, typename TV>
struct gnbx_mrxnr 
{
  const static size_t mr         = MR;
  const static size_t nr         = NR;
  const static size_t pack_mr    = MR;
  const static size_t pack_nr    = NR;
  const static size_t align_size = 32;

  OPKERNEL opkernel;
  OP1 op1;
  OP2 op2;
  TV initV;

  inline void operator()
  ( 
    int k, 
    TA *a, 
    TB *b, 
    TC *c,
    TV *v, int rs_v, int cs_v, 
    aux_s<TA, TB, TC, TV> *aux 
  ) const 
  {
    TV     regV[ MR * NR ];
    TPACKC regC[ MR * NR ];

    if ( !aux->pc )
    {
      for ( int j = 0; j < NR; j ++ )
        for ( int i = 0; i < MR; i ++ )
          regV[ j * MR + i ] = initV;
    }
    else 
    {
      for ( int j = 0; j < aux->jb; j ++ )
        for ( int i = 0; i < aux->ib; i ++ )
          regV[ j * MR + i ] = v[ j * cs_v + i * rs_v ];
    }

    /** 
     *  Semiring rank-k update
     */  
    for ( int p = 0; p < k; p ++ )
    {
      #pragma unroll
      for ( int j = 0; j < NR; j ++ )
        #pragma simd
        for ( int i = 0; i < MR; i ++ )
          regV[ j * MR + i ] = op1( regV[ j * MR + i ], op2( a[ p * MR + i ], b[ p * NR + j ] ) );
    }

    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
      #pragma simd
      for ( int i = 0; i < MR; i ++ )
        regC[ j * MR + i ] = opkernel( regV[ j * MR + i ], aux->i + i, aux->j + j, aux->b );

    /**
     *  Store back
     */ 
    c->Unpack( aux->m, aux->i, aux->ib, aux->n, aux->j, aux->jb, regC );
  };

}; /** end struct gnbx_mrxnr */

#endif /** define FUSED_MRXNR_HPP */
