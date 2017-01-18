/*
 * test_conv_relu_pool.c
 *
 * Chenhan D. Yu
 *
 * Department of Computer Science, University of Texas at Austin
 *
 * Purpose: 
 * this is the main function to exam the correctness between dgemm_tn()
 * and dgemm() from the BLAS library.
 *
 * Todo:
 *
 * Modification:
 *
 * */

#include <tuple>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <hmlp_blas_lapack.h>

#include <tree.hpp>
#include <iaskit.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace hmlp::tree;

// By default, we use binary tree.
#define N_CHILDREN 2



template<typename T>
struct gaussian
{
  inline hmlp::Data<T> operator()( hmlp::Data<T> &A, hmlp::Data<T> &B ) const 
  {
    std::vector<T> A2( A.num(), 0.0 );
    std::vector<T> B2( B.num(), 0.0 );
    hmlp::Data<T> Kab( A.num(), B.num() );

    assert( A.dim() == B.dim() );

    #pragma omp parallel for
    for ( int i = 0; i < A.num(); i ++ )
    {
      T nrm2 = 0.0;
      for ( int p = 0; p < A.dim(); p ++ )
      {
        T a = A[ i * A.dim() + p ];
        nrm2 += a * a;
      }
      A2[ i ] = nrm2;
    }

    #pragma omp parallel for
    for ( int j = 0; j < B.num(); j ++ )
    {
      T nrm2 = 0.0;
      for ( int p = 0; p < B.dim(); p ++ )
      {
        T b = B[ j * B.dim() + p ];
        nrm2 += b * b;
      }
      B2[ j ] = nrm2;
    }

    hmlp::xgemm
    ( 
      "T", "N", 
      A.num(), B.num(), A.dim(), 
      -2.0, A.data(),   A.dim(),
            B.data(),   B.dim(), 
       0.0, Kab.data(), Kab.dim()
    );

    for ( int j = 0; j < B.num(); j ++ )
    {
      for ( int i = 0; i < A.num(); i ++ )
      {
        Kab[ j * Kab.dim() + i ] = Kab[ j * Kab.dim() + i ] + A2[ i ] + B2[ j ];
        Kab[ j * Kab.dim() + i ] = Kab[ j * Kab.dim() + i ] / ( -2.0 * h * h );        
        Kab[ j * Kab.dim() + i ] = std::exp( Kab[ j * Kab.dim() + i ] );        
      }
    }

    return Kab;
  };

  T h = 0.3;
};


/* 
 * --------------------------------------------------------------------------
 * @brief  This is the test routine to exam the correctness of GSKS. XA and
 *         XB are d leading coordinate tables, and u, w have to be rhs
 *         leading. In this case, dgsks() doesn't need to know the size of
 *         nxa and nxb as long as those index map--amap, bmap, umap and wmap
 *         --are within the legal range.
 *
 * @param  *kernel gsks data structure
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * --------------------------------------------------------------------------
 */
template<typename T>
void test_tree( int d, int n )
{
  using KERNEL = gaussian<T>;
  using SETUP = hmlp::iaskit::Setup<KERNEL,T>;
  using SPLITTER = centersplit<N_CHILDREN, T>;
  using DATA = hmlp::iaskit::Data<T, KERNEL>;
  using NODE = Node<SETUP, SPLITTER, N_CHILDREN, DATA, T>;
  using TASK = hmlp::iaskit::Task<NODE>;
  
  double beg, dynamic_time, ref_time;
  std::vector<std::size_t> gids( n ), lids( n );

 
  // IMPORTANT: Must declare explcitly without "using"
  Tree<
    // SETUP
    hmlp::iaskit::Setup<gaussian<double>, double>,
    // NODE
    Node<
      hmlp::iaskit::Setup<gaussian<double>, double>,
      centersplit<2, double>, 
      N_CHILDREN, 
      hmlp::iaskit::Data<double, gaussian<double>>, 
      double
        >,
    // N_CHILDREN
    N_CHILDREN,
    // T
    double
      > tree;

  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  tree.setup.X.resize( d, n );
  tree.setup.X.rand();
  tree.setup.s = 1024;
  tree.setup.stol = 1E-3;
  for ( auto i = 0; i < n; i ++ ) 
  {
    gids[ i ] = i;
    lids[ i ] = i;
  }
  // ------------------------------------------------------------------------


  tree.TreePartition( 1024, 10, gids, lids );

  beg = omp_get_wtime();
  // Sekeletonization with dynamic scheduling (symbolic traversal).
  tree.TraverseUp<false, TASK>();
  // Execute all skeletonization tasks.
  hmlp_run();
  dynamic_time = omp_get_wtime() - beg;
  beg = omp_get_wtime();
  // Sekeletonization with level-by-level traversal.
  tree.TraverseUp<true,  TASK>();
  ref_time = omp_get_wtime() - beg;

  printf( "dynamic %5.2lfs level-by-level %5.2lfs\n", dynamic_time, ref_time );
};

int main( int argc, char *argv[] )
{
  int d, n;

  sscanf( argv[ 1 ], "%d", &d );
  sscanf( argv[ 2 ], "%d", &n );

  hmlp_init();
  
  test_tree<double>( d, n );

  hmlp_finalize();
  
  //printf( "finalize()\n" );

  return 0;
};
