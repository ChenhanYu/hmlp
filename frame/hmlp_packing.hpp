/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  


#ifndef HMLP_PACKING_HPP
#define HMLP_PACKING_HPP

#include <stdio.h>

//#define DEBUG_PACKING 1

namespace hmlp
{



/**
 *  @biref This is the im2col_gpu() functiobn from 
 *
 *         BVLC/caffe/blob/master/src/caffe/util/im2col.cpp.
 *
 *         We slightly modify it.
 *
 */ 
//template<typename T>
//void im2col
//(
//  const T* data_im,  size_t channels,
//  size_t height,     size_t width, 
//  size_t kernel_h,   size_t kernel_w,
//  size_t pad_h,      size_t pad_w,
//  size_t stride_h,   size_t stride_w,
//  size_t dilation_h, size_t dilation_w,
//  T* data_col
//) 
//{
//  size_t output_h = ( height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
//  size_t output_w = (  width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
//  size_t channel_size = height * width;
//
//  /** loop over channel, data_im += channel_size */
//  for ( int channel = channels; channel --; data_im += channel_size ) 
//  {
//    for ( int kernel_row = 0; kernel_row < kernel_h; kernel_row ++ ) 
//    {
//      for ( int kernel_col = 0; kernel_col < kernel_w; kernel_col ++) 
//      {
//        int input_row = -pad_h + kernel_row * dilation_h;
//        for ( int output_rows = output_h; output_rows; output_rows--) 
//        {
//          /** zero-padding */
//          if ( !is_a_ge_zero_and_a_lt_b( input_row, height ) ) 
//          {
//            for ( int output_cols = output_w; output_cols; output_cols -- ) 
//            {
//              *(data_col++) = 0;
//            }
//          } 
//          else 
//          {
//            int input_col = -pad_w + kernel_col * dilation_w;
//            for ( int output_col = output_w; output_col; output_col-- ) 
//            {
//              if ( is_a_ge_zero_and_a_lt_b( input_col, width ) ) 
//              {
//                *(data_col++) = data_im[ input_row * width + input_col ];
//              } 
//              else 
//              {
//                *(data_col++) = 0;
//              }
//              input_col += stride_w;
//            }
//          }
//          input_row += stride_h;
//        }
//      }
//    }
//  }
//
//}; /** end im2col() */





template<typename T>
inline void im2col
(
  int m, int n,                           // packing buffer size
  T* packX,
  T* X,
  int w0, int h0, int d0, int s, int p,   // Image size 
  int w1, int h1
)
{
  int nx = ( w0 - w1 + 2 * p ) / s + 1;

  #pragma omp parallel for
  for ( auto y0 = -1 * p; y0 <= h0 - h1 + p; y0 += s )
  {
    for ( auto x0 = -1 * p; x0 <= w0 - w1 + p; x0 += s )
    {
      auto i = ( ( y0 + p ) / s ) * nx + ( x0 + p ) / s;

      //printf( "x0 %d y0 %d i %d\n", x0, y0, i );

      for ( auto j = 0, z = 0, x = 0, y = 0; j < n; j ++ )
      {
        auto x1 = x0 + x;
        auto y1 = y0 + y;

        if ( 0 <= x1 && x1 < w0 && 0 <= y1 && y1 < h0 ) 
        {
          packX[ i * n + j ] = X[ y1 * w0 * d0 + x1 * d0 + z ];
        }
        else // zero-paging
        {
          packX[ i * n + j ] = 0.0;
        }

        z ++;
        if ( z >= d0 ) 
        {
          z = 0; x ++;
        }
        if ( x >= w1 ) 
        {
          x = 0; y ++;
        }
      }

    }
  }
}; // end im2col()




/**
 *  @brief pack image into 2D packed buffer. Notice that here X is d leading.
 */ 
template<int FOLD, bool ZEROPAD=true, typename T>
inline void pack2Dimg
(
  int m, int n,                           // packing buffer size
  T* packX,
  int x0, int y0, int offset,             // Image pointers
  T *X,                                   // Image
  int w0, int h0, int d0, int s, int p,   // Image size 
  int w1, int h1
  )
{
  //int x, x1, y, y1, z;

  for ( auto i = 0; i < m; i ++ )
  {
    // Compute the current x, y, z.
    for ( auto j =  0,
               z = ( offset % d0 ),
               x = ( offset / d0 ) % w1,
               y = ( offset / d0 ) / w1;
               j < n; j ++ )
    {
      auto x1 = x0 + x;
      auto y1 = y0 + y;

      if ( 0 <= x1 && x1 < w0 && 0 <= y1 && y1 < h0 ) 
      {
        packX[ j * FOLD + i ] = X[ y1 * w0 * d0 + x1 * d0 + z ];
      }
      else // zero-paging
      {
        packX[ j * FOLD + i ] = 0.0;
      }

      //printf( "( y, x, z ) = ( %2d, %2d, %2d ) %5.2lf\n", y1, x1, z, packX[ j * FOLD + i ] );

               z ++;
      if ( z >= d0 ) 
      {
        z = 0; x ++;
      }
      if ( x >= w1 ) 
      {
        x = 0; y ++;
      }
    }

    // move to the next window
                   x0 += s;               
    if ( ( x0 + w1 ) > ( w0 + p ) ) 
    {
      x0 = -1 * p; y0 += s;
    }
  }
}; // end pack2Dimg()




/**
 *  @brief This is the default packing routine for GKMX, GSKS, 
 *         GSKNN and STRASSEN.
 */ 
template<bool TRANS, int FOLD, bool ZEROPAD=false, typename T>
inline void pack2D
(
  int m, int n,
  T *X0, T *X1, int ldx, T gamma, int *xmap, T *packX
)
{
  //printf( "X0[0]: %lf, X1[0]: %lf\n", X0[0], X1[0] );
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
        //*packX ++ = (*x0_pntr[ i ] ++) + gamma * (*x1_pntr[ i ] ++) ;

        *packX = ( *x0_pntr[ i ] ) + gamma * ( *x1_pntr[ i ] ) ;
        //printf( "TRANS:*x0_pntr[i]:%lf, gamma:%lf, x1_pntr[i]:%lf,packX:%lf\n",*x0_pntr[i], gamma, *x1_pntr[i], *packX);
        packX ++;
        x0_pntr[ i ] += 1;
        x1_pntr[ i ] += 1;
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
      x1_pntr[ i ] = X1 + xmap[ 0 ];
    }

    for ( auto j = 0; j < n; j ++ )
    {

      for ( auto i = 0; i < m; i ++ )
      {
        *packX = *x0_pntr[ i ] + gamma * *x1_pntr[ i ];
        //printf( "NOTRANS:*x0_pntr[i]:%lf, gamma:%lf, x1_pntr[i]:%lf,packX:%lf\n",*x0_pntr[i], gamma, *x1_pntr[i], *packX);
        packX ++;
        x0_pntr[ i ] += ldx;
        x1_pntr[ i ] += ldx;
      }
    //printf( "ldx: %d\n" , ldx );
    //printf( "m:%d,FOLD:%d\n", m, FOLD );
      for ( auto i = m; i < FOLD; i ++ )
      {

        //printf( "i: %d\n", i );
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
}; // end pack2D()


/**
 *  @brief
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
}; // end pack2D()



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
};

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
};



}; // end namespace hmlp

#endif // define HMLP_PACKING_HPP
