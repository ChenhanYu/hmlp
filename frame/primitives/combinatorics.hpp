#ifndef COMBINATORICS_HPP
#define COMBINATORICS_HPP

/** Use STL, and vector. */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>

/** Use MPI support. */
#include <hmlp_mpi.hpp>

using namespace std;
using namespace hmlp;


namespace hmlp
{
namespace combinatorics
{


template<typename T>
vector<T> SampleWithoutReplacement( int l, vector<T> v )
{
  if ( l >= v.size() ) return v;
  random_device rd;
  std::mt19937 generator( rd() );
  shuffle( v.begin(), v.end(), generator );
  vector<T> ret( l );
  for ( int i = 0; i < l; i ++ ) ret[ i ] = v[ i ];
  return ret;
}; /** end SampleWithoutReplacement() */





#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#elif  HMLP_USE_CUDA
/** use pinned (page-lock) memory for NVIDIA GPUs */
template<class T, class Allocator = thrust::system::cuda::experimental::pinned_allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
vector<T> Sum( size_t d, size_t n, vector<T, Allocator> &X, vector<size_t> &gids )
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
#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#elif  HMLP_USE_CUDA
/** use pinned (page-lock) memory for NVIDIA GPUs */
template<class T, class Allocator = thrust::system::cuda::experimental::pinned_allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
vector<T> Sum( size_t d, size_t n, vector<T, Allocator> &X, mpi::Comm comm )
{
  size_t num_points_owned = X.size() / d;

  /** gids */
  vector<size_t> dummy_gids;

  /** local sum */
  auto local_sum = Sum( d, num_points_owned, X, dummy_gids );

  /** total sum */
  vector<T> total_sum( d, 0 );

  /** Allreduce */
  hmlp::mpi::Allreduce(
      local_sum.data(), total_sum.data(), d, MPI_SUM, comm );

  return total_sum;

}; /** end Sum() */




#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#elif  HMLP_USE_CUDA
/** use pinned (page-lock) memory for NVIDIA GPUs */
template<class T, class Allocator = thrust::system::cuda::experimental::pinned_allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
std::vector<T> Mean( size_t d, size_t n, std::vector<T, Allocator> &X, std::vector<size_t> &gids )
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



template<typename T>
vector<vector<size_t>> MedianThreeWaySplit( vector<T> &v, T tol )
{
  size_t n = v.size();
  auto median = Select( n, 0.5 * n, v );

  auto left = median;
  auto right = median;
  T perc = 0.0;

  while ( left == median || right == median )
  {
    if ( perc == 0.5 ) break;
    perc += 0.1;
    left = Select( n, ( 0.5 - perc ) * n, v );
    right = Select( n, ( 0.5 + perc ) * n, v );
  }

  /** Split indices of v into 3-way: lhs, rhs, and mid. */
  vector<vector<size_t>> three_ways( 3 );
  auto & lhs = three_ways[ 0 ];
  auto & rhs = three_ways[ 1 ];
  auto & mid = three_ways[ 2 ];
  for ( size_t i = 0; i < v.size(); i ++ )
  {
    //if ( std::fabs( v[ i ] - median ) < tol ) mid.push_back( i );
    if ( v[ i ] >= left && v[ i ] <= right ) 
    {
      mid.push_back( i );
    }
    else if ( v[ i ] < median ) 
    {
      lhs.push_back( i );
    }
    else rhs.push_back( i );
  }
  return three_ways;
}; /** end MedianTreeWaySplit() */



/** @brief Split values into two halfs accroding to the median. */ 
template<typename T>
vector<vector<size_t>> MedianSplit( vector<T> &v )
{
  auto three_ways = MedianThreeWaySplit( v, (T)1E-6 );
  vector<vector<size_t>> two_ways( 2 );
  two_ways[ 0 ] = three_ways[ 0 ];
  two_ways[ 1 ] = three_ways[ 1 ];  
  auto & lhs = two_ways[ 0 ];
  auto & rhs = two_ways[ 1 ];
  auto & mid = three_ways[ 2 ];
  for ( auto it : mid )
  {
    if ( lhs.size() < rhs.size() ) lhs.push_back( it );
    else rhs.push_back( it );
  }
  return two_ways;
}; /** end MedianSplit() */








}; /** end namespace combinatorics */
}; /** end namespace hmlp */

#endif /** define COMBINATORICS_HPP */
