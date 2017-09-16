template<int MR, int NR, typename T>
struct gsknn_ref_mrxnr
{
  inline void operator()
  (
    int k,
    int r,
    T *a, T *a2,
    T *b, T *b2,
    T *c,
    aux_s<T, T, T, T> *aux,
    int *bmap
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

    /** 2-norm */
    #pragma unroll
    for ( int j = 0; j < NR; j ++ )
    {
      #pragma unroll
      for ( int i = 0; i < MR; i ++ ) 
      {
        c_reg[ j * MR + i ] *= -2.0;
        c_reg[ j * MR + i ] += a2[ i ] + b2[ j ];
      }
    }




  }; /** end inline void operator */
}; /** end struct gsknn_ref_mrxnr */
