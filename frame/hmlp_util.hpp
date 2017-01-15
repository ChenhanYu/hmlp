#ifndef HMLP_UTIL_HPP
#define HMLP_UTIL_HPP

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

namespace hmlp
{


/**
 *  @brief The default function to allocate memory for HMLP.
 *         Memory allocated by this function is aligned. Most of the
 *         HMLP primitives require memory alignment.
 */
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

/**
 *  @brief Another interface.
 */ 
template<int ALIGN_SIZE, typename T>
T *hmlp_malloc( int n )
{
  return hmlp_malloc<ALIGN_SIZE, T>( n, 1, sizeof(T) );
};

/**
 *  @brief Free the aligned memory.
 */ 
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

/**
 *  @brief Split into m x n, get the subblock starting from ith row 
 *         and jth column. (for STRASSEN)
 */         
template<typename T>
void hmlp_acquire_mpart
(
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
  if ( transX == HMLP_OP_N ) 
  {
    *dst_buff = &src_buff[ ( m / x * i ) + ( n / y * j ) * lda ]; //src( m/x*i, n/y*j )
  } 
  else 
  {
    *dst_buff = &src_buff[ ( m / x * i ) * lda + ( n / y * j ) ]; //src( m/x*i, n/y*j )
    /* x,y,i,j split partition dimension and id after the transposition */
  }
};

template<typename T>
T hmlp_norm
(
  int m, int n,
  T *A, int lda
)
{
  T nrm2 = 0.0;
  for ( int j = 0; j < n; j ++ )
  {
    for ( int i = 0; i < m; i ++ )
    {
      nrm2 += A[ j * lda + i ] * A[ j * lda + i ];
    }
  }
  return std::sqrt( nrm2 );
}; // end hmlp_norm()


template<bool IGNOREZERO=false, bool COLUMNINDEX=false, typename T>
void hmlp_printmatrix
(
  int m, int n,
  T *A, int lda
)
{
  if ( COLUMNINDEX )
  {
    for ( int j = 0; j < n; j ++ )
    {
      if ( j % 5 == 0 || j == 0 || j == n - 1 ) 
      {
        printf( "col[%4d] ", j );
      }
      else
      {
        printf( "          " );
      }
    }
    printf( "\n" );
    printf( "===========================================================\n" );
  }
  printf( "A = [\n" );
  for ( int i = 0; i < m; i ++ ) 
  {
    for ( int j = 0; j < n; j ++ ) 
    {
      // Cast into double precision.
      //printf( "%13E ", (double) A[ j * lda + i ] );
      if ( IGNOREZERO )
      {
        if ( std::fabs( A[ j * lda + i ] ) < 1E-15 )
        {
          printf( "          " );
        }
        else
        {
          printf( "% .2E ", (double) A[ j * lda + i ] );
        }
      }
      else
      {
        printf( "% .2E ", (double) A[ j * lda + i ] );
      }
    }
    printf(";\n");
  }
  printf("];\n");
};


//__host__ __device__
static inline int hmlp_ceildiv( int x, int y )
{
    return ( x + y - 1 ) / y;
}

static inline int hmlp_read_nway_from_env( const char* env )
{
  int number = 1;
  char* str = getenv( env );
  if( str != NULL )
  {
    number = strtol( str, NULL, 10 );
  }
  return number;
};

/**
 *  @brief A swap function. Just in case we do not have one.
 *         (for GSKNN)
 */ 
template<typename T>
inline void swap( T *x, int i, int j ) 
{
  T tmp = x[ i ];
  x[ i ] = x[ j ];
  x[ j ] = tmp;
};

/**
 *  @brief This function is called after the root of the heap
 *         is replaced by an new candidate. We need to readjust
 *         such the heap condition is satisfied.
 */
template<typename T>
inline void heap_adjust
(
  T *D,
  int s,
  int n,
  int *I
)
{
  int j;

  while ( 2 * s + 1 < n ) 
  {
    j = 2 * s + 1;
    if ( ( j + 1 ) < n ) 
    {
      if ( D[ j ] < D[ j + 1 ] ) j ++;
    }
    if ( D[ s ] < D[ j ] ) 
    {
      swap<T>( D, s, j );
      swap<int>( I, s, j );
      s = j;
    }
    else break;
  }
}

template<typename T>
inline void heap_select
(
  int m,
  int r,
  T *x,
  int *alpha,
  T *D,
  int *I
)
{
  int i;

  for ( i = 0; i < m; i ++ ) 
  {
    if ( x[ i ] > D[ 0 ] ) 
    {
      continue;
    }
    else // Replace the root with the new query.
    {
      D[ 0 ] = x[ i ];
      I[ 0 ] = alpha[ i ];
      heap_adjust<T>( D, 0, r, I );
    }
  }
};

/**
 *  @brief A bubble sort for reference.
 */ 
template<typename T>
void bubble_sort
(
  int n,
  T *D,
  int *I
)
{
  int i, j;

  for ( i = 0; i < n - 1; i ++ ) 
  {
    for ( j = 0; j < n - 1 - i; j ++ ) 
    {
      if ( D[ j ] > D[ j + 1 ] ) 
      {
        swap<T>( D, j, j + 1 );
        swap<int>( I, j, j + 1 );
      }
    }
  }
};

}; // end namespace hmlp

#endif // define HMLP_UTIL_HPP
