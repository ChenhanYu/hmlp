#ifndef COMBINATORICS_HPP
#define COMBINATORICS_HPP
 
namespace hmlp
{
namespace combinatorics
{


template<typename T>
std::vector<T> Mean( int d, int n, std::vector<T> &X, std::vector<std::size_t> &lids )
{
  assert( lids.size() == n );
  int n_split = omp_get_max_threads();
  std::vector<T> mean( d, 0.0 );
  std::vector<T> temp( d * n_split, 0.0 );

  //printf( "n_split %d\n", n_split );

  #pragma omp parallel for num_threads( n_split )
  for ( int j = 0; j < n_split; j ++ )
  {
    for ( int i = j; i < n; i += n_split )
    {
      for ( int p = 0; p < d; p ++ )
      {
        temp[ j * d + p ] += X[ lids[ i ] * d + p ];
      }
    }
  }

  // Reduce all temporary buffers
  for ( int j = 0; j < n_split; j ++ )
  {
    for ( int p = 0; p < d; p ++ )
    {
      mean[ p ] += temp[ j * d + p ];
    }
  }

  for ( int p = 0; p < d; p ++ )
  {
    mean[ p ] /= n;
    //printf( "%5.2lf ", mean[ p ] );
  }
  //printf( "\n" );


  return mean;
}; // end Mean()



template<typename T>
std::vector<T> Mean( int d, int n, hmlp::Data<T> &X, std::vector<std::size_t> &lids )
{
  assert( lids.size() == n );
  int n_split = omp_get_max_threads();
  std::vector<T> mean( d, 0.0 );
  std::vector<T> temp( d * n_split, 0.0 );

  #pragma omp parallel for num_threads( n_split )
  for ( int j = 0; j < n_split; j ++ )
    for ( int i = j; i < n; i += n_split )
      for ( int p = 0; p < d; p ++ )
        temp[ j * d + p ] += X[ lids[ i ] * d + p ];

  /** reduce all temporary buffers */
  for ( int j = 0; j < n_split; j ++ )
    for ( int p = 0; p < d; p ++ )
      mean[ p ] += temp[ j * d + p ];

  for ( int p = 0; p < d; p ++ ) mean[ p ] /= n;

  return mean;
}; // end Mean()


/**
 *  @brief Compute the mean values. (alternative interface)
 *  
 */ 
template<typename T>
std::vector<T> Mean( int d, int n, hmlp::Data<T> &X )
{
  std::vector<std::size_t> lids( n );
  for ( int i = 0; i < n; i ++ ) lids[ i ] = i;
  return Mean( d, n, X, lids );
}; // end Mean()

template<typename T>
std::vector<T> Mean( int d, int n, std::vector<T> &X )
{
  std::vector<std::size_t> lids( n );
  for ( int i = 0; i < n; i ++ ) lids[ i ] = i;
  return Mean( d, n, X, lids );
}; // end Mean()



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

  std::vector<T> mean = Mean( 1, n, x );
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

  if ( lhs.size() == n || rhs.size() == n || lhs.size() == k ) 
  {
    return mean[ 0 ];
  }
  else if ( lhs.size() > k )
  {
    return Select( lhs.size(), k, lhs );
  }
  else
  {
    return Select( rhs.size(), k - lhs.size(), rhs );
  }

}; /** end Select() */








}; /** end namespace combinatorics */
}; /** end namespace hmlp */

#endif /** define COMBINATORICS_HPP */
