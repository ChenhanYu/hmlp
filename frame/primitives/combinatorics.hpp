#ifndef COMBINATORICS_HPP
#define COMBINATORICS_HPP

#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include <mpi/hmlp_mpi.hpp>

namespace hmlp
{
namespace combinatorics
{

template<typename T>
std::vector<T> Sum( size_t d, size_t n, std::vector<T> &X, std::vector<size_t> &gids )
{
  bool do_general_stride = ( gids.size() == n );

  /** assertion */
  if ( !do_general_stride ) assert( X.size() == d * n );

  /** declaration */
  int n_split = omp_get_max_threads();
  std::vector<T> sum( d, 0.0 );
  std::vector<T> temp( d * n_split, 0.0 );

  /** compute partial sum on each thread */
  #pragma omp parallel for num_threads( n_split )
  for ( int j = 0; j < n_split; j ++ )
    for ( int i = j; i < n; i += n_split )
      for ( int p = 0; p < d; p ++ )
        if ( do_general_stride )
          temp[ j * d + p ] += X[ gids[ i ] * d + p ];
        else
          temp[ j * d + p ] += X[ i * d + p ];

  /** reduce all temporary buffers */
  for ( int j = 0; j < n_split; j ++ )
    for ( int p = 0; p < d; p ++ )
      sum[ p ] += temp[ j * d + p ];

  return sum;

}; /** end Sum() */


/** TODO */
template<typename T>
std::vector<T> Sum( size_t d, size_t n, std::vector<T> &X,
    hmlp::mpi::Comm comm )
{
  size_t num_points_owned = X.size() / d;

  /** gids */
  std::vector<std::size_t> dummy_gids;

  /** local sum */
  auto local_sum = Sum( d, num_points_owned, X, dummy_gids );

  /** total sum */
  std::vector<T> total_sum( d, 0 );

  /** Allreduce */
  hmlp::mpi::Allreduce(
      local_sum.data(), total_sum.data(), d, MPI_SUM, comm );

  return total_sum;

}; /** end Sum() */




template<typename T>
std::vector<T> Mean( size_t d, size_t n, std::vector<T> &X, std::vector<size_t> &gids )
{
  /** sum over n points in d dimensions */
  auto mean = Sum( d, n, X, gids );

  /** compute average */
  for ( int p = 0; p < d; p ++ ) mean[ p ] /= n;

  return mean;

}; /** end Mean() */


/**
 *  @brief Compute the mean values. (alternative interface)
 *  
 */ 
template<typename T>
std::vector<T> Mean( size_t d, size_t n, std::vector<T> &X )
{
  /** assertion */
  assert( X.size() == d * n );

  /** gids */
  std::vector<std::size_t> dummy_gids;

  return Mean( d, n, X, dummy_gids );

}; /** end Mean() */




/** TODO */
/**
 *  @biref distributed mean values over d dimensions
 */ 
template<typename T>
std::vector<T> Mean( size_t d, size_t n, std::vector<T> &X, 
    hmlp::mpi::Comm comm )
{
  /** sum over n points in d dimensions */
  auto mean = Sum( d, n, X, comm );

  /** compute average */
  for ( int p = 0; p < d; p ++ ) mean[ p ] /= n;

  return mean;

}; /** end Mean() */



/**
 *  @brief Parallel prefix scan
 */ 
template<typename TA, typename TB>
void Scan( std::vector<TA> &A, std::vector<TB> &B )
{
  assert( A.size() == B.size() - 1 );

  /** number of threads */
  size_t p = omp_get_max_threads();

  /** problem size */
  size_t n = B.size();

  /** step size */
  size_t nb = n / p;

  /** private temporary buffer for each thread */
  std::vector<TB> sum( p, (TB)0 );

  /** B[ 0 ] = (TB)0 */
  B[ 0 ] = (TB)0;

  /** small problem size: sequential */
  if ( n < 100 * p ) 
  {
    size_t beg = 0;
    size_t end = n;
    for ( size_t j = beg + 1; j < end; j ++ ) 
      B[ j ] = B[ j - 1 ] + A[ j - 1 ];
    return;
  }

  /** parallel local scan */
  #pragma omp parallel for schedule( static )
  for ( size_t i = 0; i < p; i ++ ) 
  {
    size_t beg = i * nb;
    size_t end = beg + nb;
    /** deal with the edge case */
    if ( i == p - 1 ) end = n;
    if ( i != 0 ) B[ beg ] = (TB)0;
    for ( size_t j = beg + 1; j < end; j ++ ) 
    {
      B[ j ] = B[ j - 1 ] + A[ j - 1 ];
    }
  }

  /** sequential scan on local sum */
  for ( size_t i = 1; i < p; i ++ ) 
  {
    sum[ i ] = sum[ i - 1 ] + B[ i * nb - 1 ] + A[ i * nb - 1 ];
  }

  #pragma omp parallel for schedule( static )
  for ( size_t i = 1; i < p; i ++ ) 
  {
    size_t beg = i * nb;
    size_t end = beg + nb;
    /** deal with the edge case */
    if ( i == p - 1 ) end = n;
    TB sum_ = sum[ i ];
    for ( size_t j = beg; j < end; j ++ ) 
    {
      B[ j ] += sum_;
    }
  }

}; /** end Scan() */



/**
 *  @brief Select the kth element in x in the increasing order.
 *
 *  @para  
 *
 *  @TODO  The mean function is parallel, but the splitter is not.
 *         I need something like a parallel scan.
 */ 
template<typename T>
T Select( size_t n, size_t k, std::vector<T> &x )
{
  /** assertion */
  assert( k <= n && x.size() == n );

  /** early return */
  if ( n == 1 ) return x[ 0 ];

  std::vector<T> mean = Mean( (size_t)1, n, x );
  std::vector<T> lhs, rhs;
  std::vector<size_t> lflag( n, 0 );
  std::vector<size_t> rflag( n, 0 );
  std::vector<size_t> pscan( n + 1, 0 );

  /** mark flags */
  #pragma omp parallel for
  for ( size_t i = 0; i < n; i ++ )
  {
    if ( x[ i ] > mean[ 0 ] ) rflag[ i ] = 1;
    else                      lflag[ i ] = 1;
  }
  
  /** 
   *  prefix sum on flags of left hand side 
   *  input:  flags
   *  output: zero-base index
   **/
  Scan( lflag, pscan );

  /** resize left hand side */
  lhs.resize( pscan[ n ] );

  #pragma omp parallel for 
  for ( size_t i = 0; i < n; i ++ )
  {
	  if ( lflag[ i ] ) 
      lhs[ pscan[ i ] ] = x[ i ];
  }

  /** 
   *  prefix sum on flags of right hand side 
   *  input:  flags
   *  output: zero-base index
   **/
  Scan( rflag, pscan );

  /** resize right hand side */
  rhs.resize( pscan[ n ] );

  #pragma omp parallel for 
  for ( size_t i = 0; i < n; i ++ )
  {
	  if ( rflag[ i ] ) 
      rhs[ pscan[ i ] ] = x[ i ];
  }

  int nlhs = lhs.size();
  int nrhs = rhs.size();

  if ( nlhs == k || nlhs == n || nrhs == n ) 
  {
    return mean[ 0 ];
  }
  else if ( nlhs > k )
  {
    rhs.clear();
    return Select( nlhs, k, lhs );
  }
  else
  {
    lhs.clear();
    return Select( nrhs, k - nlhs, rhs );
  }

}; /** end Select() */



template<typename T>
T Select( size_t k, std::vector<T> &x, hmlp::mpi::Comm comm )
{
  /** declaration */
  std::vector<T> lhs, rhs;
  lhs.reserve( x.size() );
  rhs.reserve( x.size() );
  int n = 0;
  int num_points_owned = x.size();

  /** reduce to get the total problem size */
  hmlp::mpi::Allreduce( &num_points_owned, &n, 1, MPI_SUM, comm );

  /** TODO: mean value */
  std::vector<T> mean = Mean( (size_t)1, n, x, comm );

  for ( size_t i = 0; i < x.size(); i ++ )
  {
    if ( x[ i ] < mean[ 0 ] ) lhs.push_back( x[ i ] );
    else                      rhs.push_back( x[ i ] );
  }

  /** reduce nlhs and nrhs */
  int nlhs = 0;
  int nrhs = 0;
  int num_lhs_owned = lhs.size();
  int num_rhs_owned = rhs.size();
  hmlp::mpi::Allreduce( &num_lhs_owned, &nlhs, 1, MPI_SUM, comm );
  hmlp::mpi::Allreduce( &num_rhs_owned, &nrhs, 1, MPI_SUM, comm );

  if ( nlhs == k || n == 1 || n == nlhs || n == nrhs )
  {
    return mean[ 0 ];
  }
  else if ( nlhs > k )
  {
    rhs.clear();
    return Select( k, lhs, comm );
  }
  else
  {
    lhs.clear();
    return Select( k - nlhs, rhs, comm );
  }

}; /** end Select() */





}; /** end namespace combinatorics */
}; /** end namespace hmlp */

#endif /** define COMBINATORICS_HPP */
