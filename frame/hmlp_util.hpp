#ifndef HMLP_UTIL_HPP
#define HMLP_UTIL_HPP

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

namespace hmlp
{

template<int ALIGN_SIZE, typename T>
T *hmlp_malloc( int m, int n, int size )
{
  T *ptr;
  int err;
#ifdef HMLP_MIC_AVX512
  err = hbw_posix_memalign( (void**)&ptr, (size_t)ALIGN_SIZE, size * m * n );
#else
  err = posix_memalign( (void**)&ptr, (size_t)ALIGN_SIZE, size * m * n );
#endif

  if ( err )
  {
    printf( "hmlp_malloc(): posix_memalign() failures\n" );
    exit( 1 );
  }

  return ptr;
};

template<int ALIGN_SIZE, typename T>
T *hmlp_malloc( int n )
{
  return hmlp_malloc<ALIGN_SIZE, T>( n, 1, sizeof(T) );
};

template<typename T>
void hmlp_free( T *ptr )
{
#ifdef HMLP_MIC_AVX512
  if ( ptr ) hbw_free( ptr );
  ptr = NULL;
#else
  if ( ptr ) free( ptr );
  ptr = NULL;
#endif
}

// Split into m x n, get the subblock starting from i th row and j th column.
template<typename T>
void hmlp_acquire_mpart(
        hmlpOperation_t transX,
        int m,
        int n,
        T *src_buff,
        int lda,
        int x,
        int y,
        int i,
        int j,
        T **dst_buff
        )
{
    //printf( "m: %d, n: %d, lda: %d, x: %d, y: %d, i: %d, j: %d\n", m, n, lda, x, y, i, j );
  if ( transX == HMLP_OP_N ) {
    *dst_buff = &src_buff[ ( m / x * i ) + ( n / y * j ) * lda ]; //src( m/x*i, n/y*j )
  } else {
    *dst_buff = &src_buff[ ( m / x * i ) * lda + ( n / y * j ) ]; //src( m/x*i, n/y*j )
    /* x,y,i,j split partition dimension and id after the transposition */
  }
}


template<typename T>
void hmlp_printmatrix(
    T *A,
    int    lda,
    int    m,
    int    n
    )
{
  int    i, j;
  for ( i = 0; i < m; i ++ ) {
    for ( j = 0; j < n; j ++ ) {
      printf("%lf\t", A[j * lda + i]);  //Assume T is double...
    }
    printf("\n");
  }
}


//__host__ __device__
static inline int hmlp_ceildiv( int x, int y )
{
    return ( x + y - 1 ) / y;
}

template<typename T>
inline void swap( T *x, int i, int j ) {
  T tmp = x[ i ];
  x[ i ] = x[ j ];
  x[ j ] = tmp;
}

template<typename T>
inline void heap_adjust(
    T *D,
    int    s,
    int    n,
    int    *I
    )
{
  int    j;

  while ( 2 * s + 1 < n ) {
    j = 2 * s + 1;
    if ( ( j + 1 ) < n ) {
      if ( D[ j ] < D[ j + 1 ] ) j ++;
    }
    if ( D[ s ] < D[ j ] ) {
      swap<T>( D, s, j );
      swap<int>( I, s, j );
      s = j;
    }
    else break;
  }
}

template<typename T>
inline void heap_select(
    int    m,
    int    r,
    T *x,
    int    *alpha,
    T *D,
    int    *I
    )
{
  int    i;

  for ( i = 0; i < m; i ++ ) {
    if ( x[ i ] > D[ 0 ] ) {
      continue;
    }
    else {
      D[ 0 ] = x[ i ];
      I[ 0 ] = alpha[ i ];
      heap_adjust<T>( D, 0, r, I );
    }
  }
}

template<typename T>
void bubble_sort(
    int    n,
    T *D,
    int    *I
    )
{
  int    i, j;

  for ( i = 0; i < n - 1; i ++ ) {
    for ( j = 0; j < n - 1 - i; j ++ ) {
      if ( D[ j ] > D[ j + 1 ] ) {
        swap<T>( D, j, j + 1 );
        swap<int>( I, j, j + 1 );
      }
    }
  }
}

}; // end namespace hmlp

#endif // define HMLP_UTIL_HPP
