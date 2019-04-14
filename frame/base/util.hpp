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


#ifndef HMLP_UTIL_HPP
#define HMLP_UTIL_HPP

/* Use STL templates. */
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <tuple>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstdint>
#include <iostream>
#include <exception>
/* Use OpenMP for threading. */
#include <omp.h>
/* HMLP public header. */
#include <hmlp.h>


#define HANDLE_ERROR( err ) (hmlp::handleError( (hmlpError_t)err, __FILE__, __LINE__ ))
#define RETURN_IF_ERROR( err ) { auto hmlp_err = hmlp::returnIfError( (hmlpError_t)err, __FILE__, __LINE__ );\
                                  if ( hmlp_err != HMLP_ERROR_SUCCESS ) return hmlp_err; }
#define HANDLE_EXCEPTION( e ) { std::cerr << e.what() << std::endl; exit( 1 ); }

/* Use STL namespace. */
using namespace std;


namespace hmlp
{

/** 
 *  \breif Handling runtime error with information. 
 */
void handleError( hmlpError_t error, const char* file, int line );

/** 
 *  \breif Handling runtime error with information. 
 */
hmlpError_t returnIfError( hmlpError_t error, const char* file, int line );


/**
 *  \brief The default function to allocate memory for HMLP.
 *         Memory allocated by this function is aligned. Most of the
 *         HMLP primitives require memory alignment.
 */
template<int ALIGN_SIZE, typename T>
T *hmlp_malloc( int m, int n, int size )
{
  T *ptr = NULL;
#ifdef HMLP_MIC_AVX512
  int err = hbw_posix_memalign( (void**)&ptr, (size_t)ALIGN_SIZE, size * m * n );
#else
  int err = posix_memalign( (void**)&ptr, (size_t)ALIGN_SIZE, size * m * n );
#endif

  if ( err )
  {
    printf( "hmlp_malloc(): posix_memalign() failures\n" );
    exit( 1 );
  }

  return ptr;
};

/** @brief Another interface. */ 
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
#else
  if ( ptr ) free( ptr );
#endif
};

template<typename T>
void hmlp_print_binary( T number )
{
  char binary[ 33 ];

  for ( int i = 31; i >= 0; i -- )
  {
    if ( i % 5 ) printf( " " );
    else         printf( "%d", i / 5 );
  }
  printf( "\n" );


  for ( size_t i = 0; i < sizeof(T) * 4; i ++ )
  {
    if ( number & 1 ) binary[ 31 - i ] = '1';
    else              binary[ 31 - i ] = '0';
    number >>= 1;
    if ( i == 31 ) break;
  }
  binary[ 32 ] = '\0';

  //size_t i = 0;

  //while ( number ) 
  //{
  //  if ( i < 32 && ( number & 1 )  ) binary[ i ] = '1';
  //  else                             binary[ i ] = '0';
  //  //if ( number & 1) printf( "1" );
  //  //else             printf( "0" );
  //  number >>= 1;
  //  i ++;
  //}
  //binary[ 32 ] = 3;
  //printf("\n");
  printf( "%s\n", binary );
};



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


/**
 *  \brief Compute the relative error given the root of square error (rse) and 2-norm (nrm).
 *  \param [in] T the compute type
 *  \param [in] rse the root of square error
 *  \param [in] nrm the 2-norm
 */
template<typename T>
T relative_error_from_rse_and_nrm( T rse, T nrm )
{
  try
  {
    /* Throw an exception. */
    if ( rse < 0 || nrm < 0 ) 
    {
      throw std::invalid_argument( "sse and ss should be non-negative" );
    }
    /* Return the relative error directlu. */
    if ( nrm ) return rse / nrm;
    /** Sum of square is zero but SSE is not. */
    if ( rse ) return 1;
    /* Both SSE and SS are zero. */
    return 0;
  }
  catch ( const exception & e )
  {
    HANDLE_EXCEPTION( e );
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


template<typename TA, typename TB>
TB hmlp_relative_error
(
  int m, int n,
  TA *A, int lda,
  TB *B, int ldb
)
{
  TB nrmB = 0.0;
  TB nrm2 = 0.0;
 
  std::tuple<int, int, TB> max_error( -1, -1, 0.0 );
  for ( int j = 0; j < n; j ++ )
  {
    for ( int i = 0; i < m; i ++ )
    {
      TA a = A[ j * lda + i ];
      TB b = B[ j * ldb + i ];
      TB r = a - b;
      nrmB += b * b;
      nrm2 += r * r;
      if ( r * r > std::get<2>( max_error ) )
      {
        max_error = std::make_tuple( i, j, r );
      }
    }
  }
  nrm2 = std::sqrt( nrm2 ) / std::sqrt( nrmB );

  if ( nrm2 > 1E-7 )
  {
    printf( "relative error % .2E maxinum elemenwise error % .2E ( %d, %d )\n",
        nrm2,
        std::get<2>( max_error ),
        std::get<0>( max_error ), std::get<1>( max_error ) );
  }

  return nrm2;
};


template<typename TA, typename TB>
TB hmlp_relative_error
(
  int m, int n,
  TA *A, int lda, int loa,
  TB *B, int ldb, int lob,
  int batchSize
)
{
  TB err = 0.0;
  for ( int b = 0; b < batchSize; b ++ )
  {
    err += 
    hmlp_relative_error
    ( 
      m, n, 
      A + b * loa, lda, 
      B + b * lob, ldb
    );
  }
  if ( err > 1E-7 )
  {
    printf( "average relative error % .2E\n", err / batchSize );
  }
  return err;
};



template<typename T>
int hmlp_count_error
(
  int m, int n,
  T *A, int lda,
  T *B, int ldb
)
{
  int error_count = 0;
  std::tuple<int, int> err_location( -1, -1 );

  for ( int j = 0; j < n; j ++ )
  {
    for ( int i = 0; i < m; i ++ )
    {
      T a = A[ j * lda + i ];
      T b = B[ j * ldb + i ];
      if ( a != b )
      {
        err_location = std::make_tuple( i, j );
        error_count ++;
      }
    }
  }
  if ( error_count )
  {
    printf( "total error count %d\n", error_count );
  }
  return error_count;
};


template<typename T>
int hmlp_count_error
(
  int m, int n,
  T *A, int lda, int loa,
  T *B, int ldb, int lob,
  int batchSize
)
{
  int error_count = 0;
  for ( int b = 0; b < batchSize; b ++ )
  {
    error_count += 
    hmlp_count_error
    ( 
      m, n, 
      A + b * loa, lda, 
      B + b * lob, ldb
    );
  }
  if ( error_count )
  {
    printf( "total error count %d\n", error_count );
  }
  return error_count;
};



template<bool IGNOREZERO=false, bool COLUMNINDEX=true, typename T>
void hmlp_printmatrix( int m, int n, T *A, int lda )
{
  if ( COLUMNINDEX )
  {
    for ( int j = 0; j < n; j ++ )
    {
      if ( j % 5 == 0 || j == 0 || j == n - 1 ) 
      {
        if ( is_same<T, pair<double, size_t>>::value || is_same<T, pair<float, size_t>>::value ) 
        {
          printf( "col[%10d] ", j );
        }
        else
        {
          printf( "col[%4d] ", j );
        }
      }
      else
      {
        if ( is_same<T, pair<double, size_t>>::value || is_same<T, pair<float, size_t>>::value ) 
        {
          printf( "                " );
        }
        else
        {
          printf( "          " );
        }
      }
    }
    printf( "\n" );
    if ( is_same<T, pair<double, size_t>>::value || is_same<T, pair<float, size_t>>::value ) 
    {
      printf( "===============================================================================\n" );
    }
    else
    {
      printf( "===========================================================\n" );
    }
  }
  printf( "A = [\n" );
  for ( int i = 0; i < m; i ++ ) 
  {
    for ( int j = 0; j < n; j ++ ) 
    {
      if ( is_same<T, pair<double, size_t>>::value ) 
      {
        auto* A_pair = reinterpret_cast<pair<double, size_t>*>( A );
        printf( "(% .1E,%5lu)", (double) A_pair[ j * lda + i ].first, A_pair[ j * lda + i ].second );
      }
      else if ( is_same<T, pair<float, size_t>>::value )
      {
        auto* A_pair = reinterpret_cast<pair<float, size_t>*>( A );
        printf( "(% .1E,%5lu)", (double) A_pair[ j * lda + i ].first, A_pair[ j * lda + i ].second );
      }
      else if ( is_same<T, double>::value )
      {
        auto* A_double = reinterpret_cast<double*>( A );
        if ( std::fabs( A_double[ j * lda + i ] ) < 1E-15 )
        {
          printf( "          " );
        }
        else
        {
          printf( "% .4E ", (double) A_double[ j * lda + i ] );
        }
      }
      else if ( is_same<T, double>::value )
      {
        auto* A_float = reinterpret_cast<float*>( A );
        if ( std::fabs( A_float[ j * lda + i ] ) < 1E-15 )
        {
          printf( "          " );
        }
        else
        {
          printf( "% .4E ", (double) A_float[ j * lda + i ] );
        }
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


template<typename T>
void HeapAdjust
(
  size_t s, size_t n,
  std::pair<T, size_t> *NN
)
{
  while ( 2 * s + 1 < n ) 
  {
    size_t j = 2 * s + 1;
    if ( ( j + 1 ) < n ) 
    {
      if ( NN[ j ].first < NN[ j + 1 ].first ) j ++;
    }
    if ( NN[ s ].first < NN[ j ].first ) 
    {
      std::swap( NN[ s ], NN[ j ] );
      s = j;
    }
    else break;
  }
}; /** end HeapAdjust() */

template<typename T>
void HeapSelect
(
  size_t n, size_t k,
  std::pair<T, size_t> *Query,
  std::pair<T, size_t> *NN
)
{
  for ( size_t i = 0; i < n; i ++ )
  {
    if ( Query[ i ].first > NN[ 0 ].first )
    {
      continue;
    }
    else // Replace the root with the new query.
    {
      NN[ 0 ] = Query[ i ];
      HeapAdjust<T>( 0, k, NN );
    }
  }
}; /** end HeapSelect() */


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
}; /** end bubble_sort() */


/**
 *
 *
 **/ 
class Statistic
{
  public:

    Statistic()
    {
      _num = 0;
      _max = std::numeric_limits<double>::min();
      _min = std::numeric_limits<double>::max();
      _avg = 0.0;
    };

    std::size_t _num;

    double _max;

    double _min;

    double _avg;

    void Update( double query )
    {
      // Compute the sum
      _avg = _num * _avg;
      _num += 1;
      _max = std::max( _max, query );
      _min = std::min( _min, query );
      _avg += query;
      _avg /= _num;
    };

    void Print()
    {
      printf( "num %5lu min %.1E max %.1E avg %.1E\n", _num, _min, _max, _avg );
    };
  
}; /** end class Statistic */

}; /** end namespace hmlp */

#endif /** define HMLP_UTIL_HPP */
