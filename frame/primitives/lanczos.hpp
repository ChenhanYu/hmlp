#ifndef LANCZOS_HPP
#define LANCZOS_HPP

#include <limits>
#include <functional>

#include <hmlp.h>
#include <Data.hpp>
#include <containers/KernelMatrix.hpp>

namespace hmlp
{


/**
 *  Implement a simple Lanczos algorithm for symmetric eigenpairs.
 *
 */
template<typename VIRTUALMATRIX, typename T>
void lanczos( 
    VIRTUALMATRIX &A, 
    size_t n, size_t r, size_t nkrylov, 
    std::vector<T> &Sigma, std::vector<T> &V )
{
  /** r <= nkrylov */
  assert( r > 0 && r <= nkrylov );

  /** symmetric tridiagonal matrix */
  hmlp::Data<T> alpha( nkrylov, 1, 0.0 );
  hmlp::Data<T>  beta( nkrylov, 1, 0.0 );

  /** initialize the Krylov subspace */
  hmlp::Data<T> U( n, nkrylov, 1.0 );;
  hmlp::Data<T> w( n, 1, 0.0 );

  /** update beta[ 0 ], although we don't use it */
  beta[ 0 ] = hmlp::xnrm2( n, U.columndata( 0 ), 1 );
  //for ( size_t i = 0; i < n; i ++ ) 
  //  beta[ 0 ] += U( i, (size_t)0 ) * U( i, (size_t)0 );
  //beta[ 0 ] = std::sqrt( beta[ 0 ] );

  /** normalization */
  for ( size_t i = 0; i < n; i ++ ) 
    U( i, (size_t)0 ) /= beta[ 0 ];

  /** w = A * U( :, 0 )*/
  //A.NormalizedMultiply( 1, w.data(), U.data() );
  for ( size_t i = 0; i < n; i ++ ) w[ i ] = U[ i ];
  A.Multiply( w );  
  
  /** update alpha[ 0 ] */
  for ( size_t i = 0; i < n; i ++ )
  {
    alpha[ 0 ] += w[ i ] * U( i, (size_t)0 );
  }

  /** update w */
  for ( size_t i = 0; i < n; i ++ )
  {
    w[ i ] = w[ i ] - alpha[ 0 ] * U( i, (size_t)0 );
  }

  /** building the Krylov subspace and form the tridiagonal system */
  for ( size_t iter = 1; iter < nkrylov; iter ++ )
  {

    /** update beta[ iter ] = nrm2( w ) */
    for ( size_t i = 0; i < n; i ++ )
    {
      beta[ iter ] += w[ i ] * w[ i ]; 
    }
    beta[ iter ] = std::sqrt( beta[ iter ] );

    if ( beta[ iter ] == 0.0 )
    {
      printf( "problemaic\n" );
      exit( 1 );
    }
    else
    {
      /** v = w / beta */
      for ( size_t i = 0; i < n; i ++ )
      {
        U( i, iter ) = w[ i ] / beta[ iter ];
      }
    }

    /** w = A * U( :, iter )*/
    //for ( size_t i = 0; i < n; i ++ ) w[ i ] = 0.0;
    //A.NormalizedMultiply( 1, w.data(), U.data() + iter * n );  
    for ( size_t i = 0; i < n; i ++ ) w[ i ] = U( i, iter );
    A.Multiply( w );  

    /** update alpha[ iter ] */
    for ( size_t i = 0; i < n; i ++ )
    {
      alpha[ iter ] += w[ i ] * U( i, iter );
    }

    /** update w */
    for ( size_t i = 0; i < n; i ++ )
    {
      w[ i ] = w[ i ] - alpha[ iter ] * U( i ,iter     );
      w[ i ] = w[ i ] -  beta[ iter ] * U( i ,iter - 1 );
    }

    //printf( "alpha[ iter ] = %E, beta[ iter ] = %E\n",
    //    alpha[ iter ], beta[ iter ] );

  }

  /** invoke xstev to compute eigenpairs of the tridiagonal system */
  hmlp::Data<T> D = alpha;
  hmlp::Data<T> E = beta;
  hmlp::Data<T> Z( nkrylov, nkrylov, 0.0 );
  hmlp::Data<T> work( 2 * nkrylov - 2, 1, 0.0 );
  xstev( 
    "Vectors", nkrylov, 
    D.data(), 
    E.data() + 1, 
    Z.data(), nkrylov, work.data() );


  //for ( size_t j = 0; j < nkrylov; j ++ )
  //{
  //  printf( "Sigma[ %lu ] = %E, residual = %E\n", 
  //      j, D[ j ], Z( nkrylov - 1, j ) * beta[ j ] );
  //}

  /** V' = Z' * U' (V = U * Z( :, (nkrylov - r) ) ) */
  xgemm( "Transpose", "Transpose",
      r, n, nkrylov,
      1.0, Z.columndata( nkrylov - r ), nkrylov,
           U.data(), n, 
      0.0, V.data(), r );

  /** eigenpairs are in ascending order */
  for ( size_t j = 0; j < r; j ++ )
  {
    Sigma[ j ] = alpha[ j ];
  }

}; /** end lanczos() */



}; /** end namespace hmlp */

#endif /** define LANCZOS_HPP */
