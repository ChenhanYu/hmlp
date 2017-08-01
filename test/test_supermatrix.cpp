#include <tuple>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <hmlp_blas_lapack.h>

#include <containers/data.hpp>
#include <primitives/gemm.hpp>


#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace hmlp::gemm;


template<bool TRANSA, bool TRANSB, typename T>
void test_gemm_view( size_t m, size_t n, size_t k )
{
  T alpha =  2.0;
  T beta  = -1.0;

  hmlp::Data<T> C( m, n, 1.0 ); 
  hmlp::Data<T> M( m, n, 1.0 );
  hmlp::Data<T> A, B;
  
  if ( TRANSA ) A.resize( k, m );
  else          A.resize( m, k );
  A.rand();
  if ( TRANSB ) B.resize( n, k );
  else          B.resize( k, n );
  B.rand();

  if ( TRANSA )
  {
    if ( TRANSB )
    {
      /** TT */
      printf( "TT\n" );
      xgemm( HMLP_OP_T, HMLP_OP_T, alpha, A, B, beta, C );
      hmlp::xgemm( "T", "T", m, n, k,
          alpha, A.data(), k, 
                 B.data(), n,
          beta,  M.data(), m );
    }
    else
    {
      /** TN */
      printf( "TN\n" );
      xgemm( HMLP_OP_T, HMLP_OP_N, alpha, A, B, beta, C );
      hmlp::xgemm( "T", "N", m, n, k,
          alpha, A.data(), k, 
                 B.data(), k,
          beta,  M.data(), m );
    }
  }
  else
  {
    if ( TRANSB )
    {
      /** NT */
      printf( "NT\n" );
      xgemm( HMLP_OP_N, HMLP_OP_T, alpha, A, B, beta, C );
      hmlp::xgemm( "N", "T", m, n, k,
          alpha, A.data(), m, 
                 B.data(), n,
          beta,  M.data(), m );
    }
    else
    {
      /** NN */
      printf( "NN\n" );
      xgemm( HMLP_OP_N, HMLP_OP_N, alpha, A, B, beta, C );
      hmlp::xgemm( "N", "N", m, n, k,
          alpha, A.data(), m, 
                 B.data(), k,
          beta,  M.data(), m );
    }
  }

  for ( size_t i = 0; i < m; i ++ )
  {
    for ( size_t j = 0; j < n; j ++ )
    {
      T tar = C( i, j );
      T src = M( i, j );

      if ( std::abs( ( tar - src ) / src ) > TOLERANCE )
      {
        printf( "i %lu j %lu\n", i, j );
      }
    }
  }


}; /** end test_gemm_view() */





int main( int argc, char *argv[] )
{
  size_t m, n, k;

  sscanf( argv[ 1 ], "%lu", &m );
  sscanf( argv[ 2 ], "%lu", &n );
  sscanf( argv[ 3 ], "%lu", &k );
  
  using T = double;

  /** NN */
  test_gemm_view<false, false, T>( m, n, k );
  /** NT */
  test_gemm_view<false,  true, T>( m, n, k );
  /** TN */
  test_gemm_view< true, false, T>( m, n, k );
  /** TT */
  test_gemm_view< true,  true, T>( m, n, k );

  return 0;
};
