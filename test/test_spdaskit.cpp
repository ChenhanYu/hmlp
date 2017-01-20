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

  using RKDTSPLITTER = hmlp::spdaskit::randomsplit<N_CHILDREN, T>;
  using RKDTSETUP = hmlp::spdaskit::Setup<SPDMATRIX,RKDTSPLITTER,T>;
  using RKDTNODE = Node<RKDTSETUP, N_CHILDREN, DATA, T>;
  using KNNTASK = hmlp::spdaskit::KNNTask<RKDTNODE>;
  
  double beg, dynamic_time, omptask_time, ref_time;


  printf( "n %d\n", n );

  // ------------------------------------------------------------------------
  // Original order of the matrix.
  // ------------------------------------------------------------------------
  SPDMATRIX K;
  K.resize( n, n );
  K.randspd<true>( 0.0, 1.0 );
  //K.Print();
  std::vector<std::size_t> gids( n ), lids( n );
  for ( auto i = 0; i < n; i ++ ) 
  {
    gids[ i ] = i;
    lids[ i ] = i;
  }
  // ------------------------------------------------------------------------



  // IMPORTANT: Must declare explcitly without "using"
  Tree<
    // SETUP
    hmlp::spdaskit::Setup<
      hmlp::spdaskit::SPDMatrix<double>, 
      hmlp::spdaskit::randomsplit<2, double>, 
      double
      >,
    // NODE
    Node<
      hmlp::spdaskit::Setup<
        hmlp::spdaskit::SPDMatrix<double>, 
        hmlp::spdaskit::randomsplit<2, double>, 
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
    > rkdt;

  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  rkdt.setup.K = &K;
  rkdt.setup.splitter.Kptr = rkdt.setup.K; // The closure takes coordinates.
  std::pair<T, std::size_t> initNN( 999999.9, n );
  // ------------------------------------------------------------------------

  printf( "here\n" );

  auto NN = rkdt.AllNearestNeighbor<KNNTASK>( 10, 128, 10, gids, lids, initNN );


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
  tree.setup.K = &K;
  tree.setup.splitter.Kptr = tree.setup.K; // The closure takes coordinates.
  tree.setup.NN = &NN;
  tree.setup.k = 128;
  tree.setup.s = 128;
  tree.setup.stol = 1E-3;
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
