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
std::vector<T> sampleWithoutReplacement( int l, std::vector<T> v )
{
  if ( l >= v.size() ) 
  {
    return v;
  }
  std::random_device rd;
  std::mt19937 generator( rd() );
  std::shuffle( v.begin(), v.end(), generator );
  vector<T> ret( v.begin(), v.begin() + l );
  //for ( int i = 0; i < l; i ++ ) ret[ i ] = v[ i ];
  return ret;
}; /* end SampleWithoutReplacement() */





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


template<class InputIt, class T>
hmlpError_t accumulate(InputIt first, InputIt last, T & sum_glb, mpi::Comm comm)
{
  /* Initialize global sum to zero. */
  sum_glb = static_cast<T>(0);
  /* TODO: replcae with std::reduce(std::execution::par, v.begin(), v.end());*/
  T sum_loc = std::accumulate(first, last, T(0));
  /* sum_glb = sum(sum_loc) */
  RETURN_IF_ERROR(hmlp::mpi::Allreduce( &sum_loc, &sum_glb, 1, MPI_SUM, comm ));
  return HMLP_ERROR_SUCCESS;
};




//#ifdef HMLP_MIC_AVX512
///** use hbw::allocator for Intel Xeon Phi */
//template<class T, class Allocator = hbw::allocator<T> >
//#elif  HMLP_USE_CUDA
///** use pinned (page-lock) memory for NVIDIA GPUs */
//template<class T, class Allocator = thrust::system::cuda::experimental::pinned_allocator<T> >
//#else
///** use default stl allocator */
//template<class T, class Allocator = std::allocator<T> >
//#endif
//std::vector<T> Mean( size_t d, size_t n, std::vector<T, Allocator> &X, std::vector<size_t> &gids )
//{
//  /** sum over n points in d dimensions */
//  auto mean = Sum( d, n, X, gids );
//
//  /** compute average */
//  for ( int p = 0; p < d; p ++ ) mean[ p ] /= n;
//
//  return mean;
//
//}; /** end Mean() */
//
//
///**
// *  @brief Compute the mean values. (alternative interface)
// *  
// */ 
//template<typename T>
//std::vector<T> Mean( size_t d, size_t n, std::vector<T> &X )
//{
//  /** assertion */
//  assert( X.size() == d * n );
//
//  /** gids */
//  std::vector<std::size_t> dummy_gids;
//
//  return Mean( d, n, X, dummy_gids );
//
//}; /** end Mean() */
//
//
//
//
///** TODO */
///**
// *  @biref distributed mean values over d dimensions
// */ 
//template<typename T>
//std::vector<T> Mean( uint64_t d, uint64_t n, std::vector<T> & X, hmlp::mpi::Comm comm )
//{
//  /** Sum over n points in d dimensions */
//  auto mean = Sum( d, n, X, comm );
//
//  /** Compute average */
//  for ( int p = 0; p < d; p ++ ) 
//  {
//    mean[ p ] /= n;
//  }
//
//  return mean;
//
//}; /** end Mean() */



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

  /** Early return */
  if ( n == 1 )
  {
    return x[ 0 ];
  }

  T mean = std::accumulate(x.begin(), x.end(), static_cast<T>(0)) / x.size();



  std::vector<T> lhs, rhs;
  std::vector<size_t> lflag( n, 0 );
  std::vector<size_t> rflag( n, 0 );
  std::vector<size_t> pscan( n + 1, 0 );

  /** mark flags */
  #pragma omp parallel for
  for ( size_t i = 0; i < n; i ++ )
  {
    if ( x[ i ] > mean ) rflag[ i ] = 1;
    else                 lflag[ i ] = 1;
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
    return mean;
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
hmlpError_t kthSelect(uint32_t k, const std::vector<T> & x, T & x_k, hmlp::mpi::Comm comm)
{
  /** declaration */
  std::vector<T> lhs, rhs;
  lhs.reserve( x.size() );
  rhs.reserve( x.size() );
  int32_t num_points_glb = 0;
  int32_t num_points_loc = x.size();

  /* Reduce to get the total problem size. */
  RETURN_IF_ERROR(hmlp::mpi::Allreduce( &num_points_loc, &num_points_glb, 1, MPI_SUM, comm));

  if (k > num_points_glb)
  {
    //std::cerr << k << "," << num_points_glb << std::endl;
    return HMLP_ERROR_INVALID_VALUE;
  }

  /* TODO: mean value */
  T mean = static_cast<T>(0);
  HANDLE_ERROR(hmlp::combinatorics::accumulate(x.begin(), x.end(), mean, comm));
  mean /= num_points_glb;
  //std::vector<T> mean = Mean( 1U, num_points_glb, x, comm );

  for ( uint64_t i = 0U; i < x.size(); i ++ )
  {
    if ( x[ i ] < mean )
    {
      lhs.push_back( x[ i ] );
    }
    else
    {
      rhs.push_back( x[ i ] );
    }
  }

  /** Reduce nlhs and nrhs. */
  int num_lhs_glb = 0;
  int num_rhs_glb = 0;
  int num_lhs_loc = lhs.size();
  int num_rhs_loc = rhs.size();
  RETURN_IF_ERROR(hmlp::mpi::Allreduce( &num_lhs_loc, &num_lhs_glb, 1, MPI_SUM, comm));
  RETURN_IF_ERROR(hmlp::mpi::Allreduce( &num_rhs_loc, &num_rhs_glb, 1, MPI_SUM, comm));

  if ( num_lhs_glb == k || num_points_glb == 1 || num_points_glb == num_lhs_glb || num_points_glb == num_rhs_glb )
  {
    /* Assign x_k to be the mean value. */
    x_k = mean;
    return HMLP_ERROR_SUCCESS;
  }
  else if ( num_lhs_glb > k )
  {
    rhs.clear();
    return kthSelect(k, lhs, x_k, comm);
  }
  else
  {
    lhs.clear();
    return kthSelect( k - num_lhs_glb, rhs, x_k, comm );
  }
};



template<typename T>
std::vector<std::vector<uint64_t>> MedianThreeWaySplit( std::vector<T> &v, T tol )
{
  uint64_t n = v.size();
  auto median = Select( n, 0.5 * n, v );

  auto left = median;
  auto right = median;
  T perc = 0.0;

  while ( left == median || right == median )
  {
    if ( perc == 0.5 ) 
    {
      break;
    }
    perc += 0.1;
    left = Select( n, ( 0.5 - perc ) * n, v );
    right = Select( n, ( 0.5 + perc ) * n, v );
  }

  /** Split indices of v into 3-way: lhs, rhs, and mid. */
  std::vector<std::vector<uint64_t>> three_ways( 3 );
  auto & lhs = three_ways[ 0 ];
  auto & rhs = three_ways[ 1 ];
  auto & mid = three_ways[ 2 ];
  for ( uint64_t i = 0U; i < v.size(); i ++ )
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
    else 
    {
      rhs.push_back( i );
    }
  }
  return three_ways;
}; /* end MedianTreeWaySplit() */



/** @brief Split values into two halfs accroding to the median. */ 
template<typename T>
std::vector<std::vector<uint64_t>> MedianSplit(std::vector<T> &v)
{
  auto three_ways = MedianThreeWaySplit( v, (T)1E-6 );
  std::vector<std::vector<uint64_t>> two_ways( 2 );
  two_ways[0] = three_ways[0 ];
  two_ways[1] = three_ways[1 ];  
  auto & lhs = two_ways[ 0 ];
  auto & rhs = two_ways[ 1 ];
  auto & mid = three_ways[ 2 ];
  for ( auto it : mid )
  {
    if ( lhs.size() < rhs.size() )
    {
      lhs.push_back( it );
    }
    else 
    {
      rhs.push_back( it );
    }
  }
  return two_ways;
}; /* end MedianSplit() */


/** @brief Split values into two halfs accroding to the median. */ 
template<typename T>
hmlpError_t medianSplit(const std::vector<T> & values, std::vector<std::vector<uint64_t>> & split, hmlp::mpi::Comm comm )
{
  int num_points_glb = 0;
  int num_points_owned = values.size();

  /* Initialize output. */
  split.resize( 2 );
  split[ 0 ].clear();
  split[ 1 ].clear();

  std::vector<uint64_t> middle;
  /* n = sum( num_points_owned ) over all MPI processes in comm */
  RETURN_IF_ERROR(mpi::Allreduce( &num_points_owned, &num_points_glb, 1, MPI_SUM, comm));
  /* Early return if the problem size is zero. */
  if ( num_points_glb == 0 ) 
  {
    return HMLP_ERROR_SUCCESS;
  }

  T median1 = static_cast<T>(0);
  RETURN_IF_ERROR(hmlp::combinatorics::kthSelect(0.5 * num_points_glb, values, median1, comm));
  T median0 = median1 ;
  T median2 = median1;
  float portion0 = 0.0;
  float portion2 = 0.0;
  while (median1 == median0 && portion0 < 0.5)
  {
    portion0 += 0.125;
    RETURN_IF_ERROR(hmlp::combinatorics::kthSelect((0.5 - portion0) * num_points_glb, values, median0, comm));
  }
  while (median1 == median2 && portion2 < 0.5)
  {
    portion2 += 0.125;
    RETURN_IF_ERROR(hmlp::combinatorics::kthSelect((0.5 + portion2) * num_points_glb, values, median2, comm));
  }

  if (portion0 > 0.125 || portion2 > 0.125)
  {
    printf( "[WARNING] middle gap contains [-%d,%d] of %d/%d points.\n", 
        (int)(portion0 * 100), (int)(portion2 * 100), (int)((portion0 + portion2) * num_points_owned), num_points_owned);
  }

  for ( uint64_t i = 0U; i < values.size(); i ++ )
  {
    auto v = values[ i ];
    if ( v >= median0 && v <= median2 ) 
    {
      middle.push_back( i );
    }
    else if ( v < median1 )
    {
      split[ 0 ].push_back( i );
    }
    else 
    {
      split[ 1 ].push_back( i );
    }
  }

  int nmid = 0;
  int nlhs = 0;
  int nrhs = 0;
  int num_mid_owned = middle.size();
  int num_lhs_owned = split[ 0 ].size();
  int num_rhs_owned = split[ 1 ].size();

  /** nmid = sum( num_mid_owned ) over all MPI processes in comm. */
  RETURN_IF_ERROR(mpi::Allreduce( &num_mid_owned, &nmid, 1, MPI_SUM, comm ));
  RETURN_IF_ERROR(mpi::Allreduce( &num_lhs_owned, &nlhs, 1, MPI_SUM, comm ));
  RETURN_IF_ERROR(mpi::Allreduce( &num_rhs_owned, &nrhs, 1, MPI_SUM, comm ));

  /** Assign points in the middle to left or right. */
  if ( nmid )
  {
    int nlhs_required = std::max( 0, num_points_glb / 2 - nlhs );
    int nrhs_required = std::max( 0, ( num_points_glb - num_points_glb / 2 ) - nrhs );

    /** Now decide the portion */
    double lhs_ratio = (double)nlhs_required / ( nlhs_required + nrhs_required );
    int nlhs_required_owned = num_mid_owned * lhs_ratio;
    int nrhs_required_owned = num_mid_owned - nlhs_required_owned;


    //printf( "\nrank %d [ %d %d ] nlhs %d mid %d (med40 %e med50 %e med60 %e) nrhs %d [ %d %d ]\n",
    //  //global_rank,
    //  0,
    //  nlhs_required_owned, nlhs_required,
    //  nlhs, nmid, med40, median, med60, nrhs,
    //  nrhs_required_owned, nrhs_required ); fflush( stdout );

    //assert( nlhs_required >= 0 && nrhs_required >= 0 );
    assert( nlhs_required_owned >= 0 && nrhs_required_owned >= 0 );

    for (uint64_t i = 0U; i < middle.size(); i ++ )
    {
      if ( i < nlhs_required_owned ) 
      {
        split[ 0 ].push_back( middle[ i ] );
      }
      else
      {
        split[ 1 ].push_back( middle[ i ] );
      }
    }
  }
  return HMLP_ERROR_SUCCESS;
};


}; /* end namespace combinatorics */
}; /* end namespace hmlp */

#endif /* define COMBINATORICS_HPP */
