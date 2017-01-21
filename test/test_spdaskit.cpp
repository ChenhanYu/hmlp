#include <tuple>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <hmlp_blas_lapack.h>
#include <hmlp_util.hpp>

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
void test_spdaskit( size_t n, size_t k, size_t s, size_t nrhs )
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
  
  double beg, dynamic_time, omptask_time, ref_time, ann_time, tree_time, nneval_time, nonneval_time;


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

  //printf( "here\n" );

  beg = omp_get_wtime();
  auto NN = rkdt.AllNearestNeighbor<KNNTASK>( 10, k, 10, gids, lids, initNN );
  ann_time = omp_get_wtime() - beg;


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
  tree.setup.k = k;
  tree.setup.s = s;
  tree.setup.stol = 1E-3;
  // ------------------------------------------------------------------------
  tree.setup.w.resize( nrhs, n );
  tree.setup.w.rand();
  // ------------------------------------------------------------------------


  beg = omp_get_wtime();
  tree.TreePartition( 512, 10, gids, lids );
  tree_time = omp_get_wtime() - beg;

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
  // Do not use NN pruning
  /*
  hmlp::Data<T> potentials;
  hmlp::spdaskit::Evaluate( tree.treelist[ 0 ], tree.treelist.back(), potentials );
  hmlp::spdaskit::ComputeError( tree.treelist.back(), potentials );
  */


  // Test evaluation with NN prunning.
  for ( size_t i = 0; i < 10; i ++ )
  {
    hmlp::Data<T> potentials;
    // Use NN pruning
    beg = omp_get_wtime();
    hmlp::spdaskit::Evaluate<false, true>( tree, i, potentials );
    nneval_time = omp_get_wtime() - beg;
    hmlp::spdaskit::ComputeError( tree, i, potentials );
    // Do not use NN pruning
    beg = omp_get_wtime();
    hmlp::spdaskit::Evaluate<false, false>( tree, i, potentials );
    nonneval_time = omp_get_wtime() - beg;
    hmlp::spdaskit::ComputeError( tree, i, potentials );
  }


  printf( "n %5lu k %4lu s %4lu nrhs %4lu ANN %5.3lf CONSTRUCT %5.3lf EVAL(NN) %5.3lf EVAL %5.3lf\n", 
      n, k, s, nrhs, ann_time, tree_time, nneval_time, nonneval_time );

};

int main( int argc, char *argv[] )
{
  size_t n, k, s, nrhs;

  sscanf( argv[ 1 ], "%lu", &n );
  sscanf( argv[ 2 ], "%lu", &k );
  sscanf( argv[ 3 ], "%lu", &s );
  sscanf( argv[ 4 ], "%lu", &nrhs );

  hmlp_init();
  
  test_spdaskit<double>( n, k, s, nrhs );

  hmlp_finalize();
 
  return 0;
};
