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
#include <primitives/lowrank.hpp>
#include <primitives/gemm.hpp>


#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace hmlp::lowrank;


template<typename T>
void test_gemm_view( size_t m, size_t n, size_t k )
{
  hmlp::Data<T> A( m, k ); A.rand();
  hmlp::Data<T> B( k, n ); B.rand();
  hmlp::Data<T> A( m, n, 0.0 ); 




}; /** end test_gemm_view() */





int main( int argc, char *argv[] )
{
  int m, n, s;

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &s );
  
  test_skel<double>( m, n, s );

  hmlp::Data<double> X;

  X.resize( 3, 10 );

  std::cout << X.size() << std::endl;
 
  std::size_t arg1, arg2;
  std::tie( arg1, arg2 ) = X.shape();

  std::cout << arg1 << std::endl;
  std::cout << arg2 << std::endl;
  

  X.rand();
  X.Print();
  X.randn( 0.0, 1.0 );
  X.Print();



  return 0;
};
