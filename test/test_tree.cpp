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
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <hmlp_blas_lapack.h>

#include <tree.hpp>
//#include <skel.hpp>
#include <iaskit.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace hmlp::tree;

#define N_CHILDREN 2


//#define SPLITTER centersplit<N_CHILDREN,double>

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
  using SPLITTER = centersplit<N_CHILDREN, T>;
  using DATA = hmlp::iaskit::Data<T>;
  using NODE = Node<SPLITTER, N_CHILDREN, DATA, T>;
  //using NODE = hmlp::iaskit::iaskitNode<SPLITTER, N_CHILDREN, DATA, T>;
  //using TASK = hmlp::skeleton::Task<NODE>;
  //using TASK = hmlp::skel::Task<NODE>;
  using TASK = hmlp::iaskit::Task<NODE>;
  

  double ref_beg, ref_time, gkmx_beg, gkmx_time;



  //std::vector<T> X( d * n );
  hmlp::Data<T> X( d, n );
  std::vector<std::size_t> gids( n ), lids( n );


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  for ( auto i = 0; i < n; i ++ ) 
  {
    for ( auto p = 0; p < d; p ++ ) 
    {
      X[ i * d + p ] = (T)( rand() % 100 ) / 1000.0;	
    }
  }
  for ( auto i = 0; i < n; i ++ ) 
  {
    gids[ i ] = i;
    lids[ i ] = i;
  }
  // ------------------------------------------------------------------------
 
  // IMPORTANT: Must declare explcitly without "using"
  //Tree< centersplit<2, double>, N_CHILDREN, hmlp::Data<double>, double> tree;
  Tree< Node<centersplit<2, double>, N_CHILDREN, hmlp::iaskit::Data<double>, double>
    , N_CHILDREN, double> tree;

  //tree.TreePartition( d, n, 128, 10, X, gids, lids );

  //auto treelist = tree
  //<centersplit<2, double>, 2, double>.TreePartition
  //( d, n, 128, 10, X, gids, lids );

  //printf( "here\n" );


  //auto treelist = 
    
  tree.TreePartition( d, n, 128, 10, X, gids, lids );

  //tree.TraverseUp<false, hmlp::skeleton::Task<Node<SPLITTER, N_CHILDREN, T>>>();
  //tree.TraverseUp<false, hmlp::skeleton::Task<NODE>>();
  tree.TraverseUp<false, TASK>();
  hmlp_run();
  //tree.TraverseUp<true,  hmlp::skeleton::Task<Node<SPLITTER, N_CHILDREN, T>>>();
  tree.TraverseUp<true,  TASK>();
};

int main( int argc, char *argv[] )
{
  int d, n;

  sscanf( argv[ 1 ], "%d", &d );
  sscanf( argv[ 2 ], "%d", &n );

  hmlp_init();
  
  test_tree<double>( d, n );

  printf( "here\n" );

  hmlp_finalize();
  
  printf( "finalize()\n" );

  return 0;
};
