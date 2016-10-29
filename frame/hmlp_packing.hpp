#ifndef HMLP_PACKING_HPP
#define HMLP_PACKING_HPP

#include <stdio.h>

namespace hmlp
{




template<bool TRANS, int FOLD, bool ZEROPAD=false, typename T>
inline void pack2D
(
  int m, int n,
  T *X0, T *X1, int ldx, T gamma, int *xmap, T *packX
)
{
  T *x0_pntr[ FOLD ];
  T *x1_pntr[ FOLD ];

  if ( TRANS )
  {
    for ( auto i = 0; i < m; i ++ )
    {
      x0_pntr[ i ] = X0 + ldx * xmap[ i ];
      x1_pntr[ i ] = X1 + ldx * xmap[ i ];
    }
    for ( auto i = m; i < FOLD; i ++ )
    {
      x0_pntr[ i ] = X0 + ldx * xmap[ 0 ];
      x1_pntr[ i ] = X1 + ldx * xmap[ 0 ];
    }
    for ( auto j = 0; j < n; j ++ ) 
    {
      for ( auto i = 0; i < m; i ++ )
      {
        *packX ++ = (*x0_pntr[ i ] ++) + gamma * (*x1_pntr[ i ] ++) ;
      }
      for ( auto i = m; i < FOLD; i ++ )
      {
        if ( ZEROPAD ) *packX ++ = (T)0.0;
        else           *packX ++ = (*x0_pntr[ i ] ++) + gamma * (*x1_pntr[ i ] ++) ;
      }
    }
  }
  else 
  {
    //printf( "pack2D(): TRANS = false not yet implemented yet.\n" );
    for ( auto i = 0; i < m; i ++ )
    {
      x0_pntr[ i ] = X0 + xmap[ i ];
      x1_pntr[ i ] = X1 + xmap[ i ];
    }
    for ( auto i = m; i < FOLD; i ++ )
    {
      x0_pntr[ i ] = X0 + xmap[ 0 ];
      x0_pntr[ i ] = X1 + xmap[ 0 ];
    }
    for ( auto j = 0; j < n; j ++ )
    {
      for ( auto i = 0; i < m; i ++ )
      {
        *packX = *x0_pntr[ i ] + gamma * *x1_pntr[ i ];
        packX ++;
        x0_pntr[ i ] += ldx;
        x1_pntr[ i ] += ldx;
      }
      for ( auto i = m; i < FOLD; i ++ )
      {
        if ( ZEROPAD ) *packX ++ = (T)0.0;
        else
        {
          *packX = (*x0_pntr[ i ]) + gamma * (*x1_pntr[ i ]);
          *packX ++; 
          x0_pntr[ i ] += ldx;
          x1_pntr[ i ] += ldx;
        }
      }
    }
  }
}

/**
 *
 */ 
template<bool TRANS, int FOLD, bool ZEROPAD=false, typename T>
inline void pack2D
(
  int m, int n,
  T *X0, T *X1, int ldx, T gamma, T *packX
)
{
  int xmap[ FOLD ];
  for ( int i = 0; i < FOLD; i ++ ) xmap[ i ] = i;
  pack2D<TRANS, FOLD, ZEROPAD, T>
  (
    m, n, 
    X0, X1, ldx, gamma, xmap, packX
  );
}



/**
 *
 */ 
template<bool TRANS, int FOLD, bool ZEROPAD=false, typename T>
inline void pack2D
(
  int m, int n,
  T *X, int ldx, int *xmap, T *packX
)
{
  T *x_pntr[ FOLD ];

  if ( TRANS )
  {
    for ( auto i = 0; i < m; i ++ )
    {
      x_pntr[ i ] = X + ldx * xmap[ i ];
    }
    for ( auto i = m; i < FOLD; i ++ )
    {
      x_pntr[ i ] = X + ldx * xmap[ 0 ];
    }
    for ( auto j = 0; j < n; j ++ ) 
    {
      for ( auto i = 0; i < m; i ++ )
      {
        *packX ++ = *x_pntr[ i ] ++;
      }
      for ( auto i = m; i < FOLD; i ++ )
      {
        if ( ZEROPAD ) *packX ++ = (T)0.0;
        else           *packX ++ = *x_pntr[ i ] ++;
      }
    }
  }
  else 
  {
    //printf( "pack2D(): TRANS = false not yet implemented yet.\n" );
    for ( auto i = 0; i < m; i ++ )
    {
      x_pntr[ i ] = X + xmap[ i ];
    }
    for ( auto i = m; i < FOLD; i ++ )
    {
      x_pntr[ i ] = X + xmap[ 0 ];
    }
    for ( auto j = 0; j < n; j ++ )
    {
      for ( auto i = 0; i < m; i ++ )
      {
        *packX = *x_pntr[ i ];
        packX ++;
        x_pntr[ i ] += ldx;
      }
      for ( auto i = m; i < FOLD; i ++ )
      {
        if ( ZEROPAD ) *packX ++ = (T)0.0;
        else
        {
          *packX = *x_pntr[ i ];
          *packX ++; 
          x_pntr[ i ] += ldx;
        }
      }
    }
  }
}

/**
 *
 */ 
template<bool TRANS, int FOLD, bool ZEROPAD=false, typename T>
inline void pack2D
(
  int m, int n,
  T *X, int ldx, T *packX
)
{
  int xmap[ FOLD ];
  for ( int i = 0; i < FOLD; i ++ ) xmap[ i ] = i;
  pack2D<TRANS, FOLD, ZEROPAD, T>
  (
    m, n, 
    X, ldx, xmap, packX
  );
}




/**
 *
 */ 
template<int PACK_MR, typename TA>
inline void packA_kcxmc(
    int m, int k,
    TA *A, int lda, int *amap, TA *packA )
{
  TA *a_pntr[ PACK_MR ];

  for ( auto i = 0; i < m; i ++ )       a_pntr[ i ] = A + lda * amap[ i ];
  for ( auto i = m; i < PACK_MR; i ++ ) a_pntr[ i ] = A + lda * amap[ 0 ];
  for ( auto p = 0; p < k; p ++ ) 
  {
    for ( auto i = 0; i < PACK_MR; i ++ ) 
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



};
#endif // define HMLP_PACKING_HPP
