#ifndef HMLP_PACKING_HXX
#define HMLP_PACKING_HXX

/**
 *
 */ 
template<int PACK_MR, typename TA>
inline void packA_kcxmc(
    int m, int k,
    TA *A, int lda, int *amap, TA *packA )
{
  int i, p;
  TA *a_pntr[ PACK_MR ];

  for ( i = 0; i < m; i ++ )       a_pntr[ i ] = A + lda * amap[ i ];
  for ( i = m; i < PACK_MR; i ++ ) a_pntr[ i ] = A + lda * amap[ 0 ];
  for ( p = 0; p < k; p ++ ) 
  {
    for ( i = 0; i < PACK_MR; i ++ ) 
    {
      *packA ++ = *a_pntr[ i ] ++;
    }
  }
}

/**
 *
 */ 
template<int PACK_NR, typename TB>
inline void packB_kcxnc(
    int n, int k, 
	TB *B, int ldb, int *bmap, TB *packB )
{
  int    j, p; 
  TB *b_pntr[ PACK_NR ];

  for ( j = 0; j < n; j ++ )       b_pntr[ j ] = B + ldb * bmap[ j ];
  for ( j = n; j < PACK_NR; j ++ ) b_pntr[ j ] = B + ldb * bmap[ 0 ];
  for ( p = 0; p < k; p ++ ) 
  {
    for ( j = 0; j < PACK_NR; j ++ ) 
    {
      *packB ++ = *b_pntr[ j ] ++;
    }
  }
}

/**
 *
 */ 
template<int PACK_NR, typename TC>
inline void packw_rhsxnc(
    int n, int rhs,
    TC *w, int ldw, int *wmap, TC *packw )
{
  int j, p;
  TC *w_pntr[ PACK_NR ];

  for ( j = 0; j < n; j ++ ) w_pntr[ j ] = w + ldw * wmap[ j ];
  
  for ( p = 0; p < rhs; p ++ ) 
  {
    for ( j = 0; j < n; j ++ ) 
    {
      *packw ++ = *w_pntr[ j ] ++;
    }
    for ( j = n; j < PACK_NR; j ++ ) 
    {
      *packw ++ = 0.0;
    }
  }
}

/**
 *
 */ 
template<int PACK_MR, typename TC>
inline void packu_rhsxmc(
    int m, int rhs,
    TC *u, int ldu, int *umap, TC *packu )
{
  int i, p;
  TC *u_pntr[ PACK_MR ];

  for ( i = 0; i < m; i ++ )  u_pntr[ i ] = u + ldu * umap[ i ];
  for ( p = 0; p < rhs; p ++ ) 
  {
    for ( i = 0; i < m; i ++ ) 
    {
      *packu ++ = *u_pntr[ i ] ++;
    }
    for ( i = m; i < PACK_MR; i ++ ) 
    {
      packu ++;
    }
  }
}




#endif // define HMLP_PACKING_HXX
