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
#include <spdaskit.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace hmlp::tree;

// By default, we use binary tree.
#define N_CHILDREN 2

template<typename T>
void test_spdaskit( int n )
{
  using SPDMATRIX = hmlp::spdaskit::SPDMatrix<T>;
  using SPLITTER = hmlp::spdaskit::centersplit<N_CHILDREN, T>;
  using SETUP = hmlp::spdaskit::Setup<SPDMATRIX,SPLITTER,T>;
  using DATA = hmlp::spdaskit::Data<T>;
  using NODE = Node<SETUP, N_CHILDREN, DATA, T>;
  using SKELTASK = hmlp::spdaskit::SkeletonizeTask<NODE>;
  using UPDATETASK = hmlp::spdaskit::UpdateWeightsTask<NODE>;
  
  double beg, dynamic_time, omptask_time, ref_time;


  printf( "n %d\n", n );

  // Original order of the matrix.
  std::vector<std::size_t> gids( n ), lids( n );

  // IMPORTANT: Must declare explcitly without "using"
  Tree<
    // SETUP
    hmlp::spdaskit::Setup<
      hmlp::spdaskit::SPDMatrix<double>, 
      hmlp::spdaskit::centersplit<2, double>, 
      double
      >,
    // NODE
    Node<
      hmlp::spdaskit::Setup<
        hmlp::spdaskit::SPDMatrix<double>, 
        hmlp::spdaskit::centersplit<2, double>, 
        double
        >,
      N_CHILDREN, 
      hmlp::spdaskit::Data<double>, 
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
  tree.setup.K.resize( n, n );
  tree.setup.K.randspd<false>();
  tree.setup.splitter.Kptr = &tree.setup.K; // The closure takes coordinates.
  tree.setup.s = 128;
  tree.setup.stol = 1E-3;
  for ( auto i = 0; i < n; i ++ ) 
  {
    gids[ i ] = i;
    lids[ i ] = i;
  }
  // ------------------------------------------------------------------------
  tree.setup.w.resize( 128, n );
  tree.setup.w.rand();
  // ------------------------------------------------------------------------


  tree.TreePartition( 512, 10, gids, lids );

  beg = omp_get_wtime();
  // Sekeletonization with dynamic scheduling (symbolic traversal).
  tree.TraverseUp<false, SKELTASK>();
  // Execute all skeletonization tasks.
  hmlp_run();
  dynamic_time = omp_get_wtime() - beg;
  beg = omp_get_wtime();
  // Sekeletonization with level-by-level traversal.
  tree.TraverseUp<true,  SKELTASK>();
  ref_time = omp_get_wtime() - beg;
  beg = omp_get_wtime();
  // Sekeletonization with omp task.
  tree.PostOrder<SKELTASK>( tree.treelist[ 0 ] );
  omptask_time = omp_get_wtime() - beg;
  

  printf( "dynamic %5.2lfs level-by-level %5.2lfs OpenMP task %5.2lfs\n", 
      dynamic_time, ref_time, omptask_time );

  beg = omp_get_wtime();
  // Sekeletonization with dynamic scheduling (symbolic traversal).
  tree.TraverseUp<false, UPDATETASK>();
  // Execute all skeletonization tasks.
  hmlp_run();
  dynamic_time = omp_get_wtime() - beg;
  beg = omp_get_wtime();
  // Sekeletonization with level-by-level traversal.
  tree.TraverseUp<true, UPDATETASK>();
  ref_time = omp_get_wtime() - beg;
  beg = omp_get_wtime();
  // Sekeletonization with omp task.
  tree.PostOrder<UPDATETASK>( tree.treelist[ 0 ] );
  omptask_time = omp_get_wtime() - beg;

  printf( "dynamic %5.2lfs level-by-level %5.2lfs OpenMP task %5.2lfs\n", 
      dynamic_time, ref_time, omptask_time );

  // Use the right most leaf node to test the accuracy.
  hmlp::Data<T> potentials;
  hmlp::spdaskit::Evaluate( tree.treelist[ 0 ], tree.treelist.back(), potentials );
  hmlp::spdaskit::ComputeError( tree.treelist.back(), potentials );

};

int main( int argc, char *argv[] )
{
  int n;

  sscanf( argv[ 1 ], "%d", &n );

  hmlp_init();
  
  test_spdaskit<double>( n );

  hmlp_finalize();
  
  return 0;
};
