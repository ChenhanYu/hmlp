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
#include <limits>

#include <data.hpp>
#include <tree.hpp>
#include <spdaskit.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

//#define DUMP_ANALYSIS_DATA 1



using namespace hmlp::tree;

// By default, we use binary tree.
#define N_CHILDREN 2




template<bool ADAPTIVE, int TESTSUITS = -1, typename T>
void test_spdaskit( size_t n, size_t m, size_t k, size_t s, size_t nrhs )
{
  // Instantiation for the Spd-Askit tree.
  using SPDMATRIX = hmlp::spdaskit::SPDMatrix<T>;
  using SPLITTER = hmlp::spdaskit::centersplit<N_CHILDREN, T>;
  using SETUP = hmlp::spdaskit::Setup<SPDMATRIX,SPLITTER,T>;
  using DATA = hmlp::spdaskit::Data<T>;
  using NODE = Node<SETUP, N_CHILDREN, DATA, T>;
  using SKELTASK = hmlp::spdaskit::SkeletonizeTask<ADAPTIVE, NODE, T>;
  //Using UPDATETASK = hmlp::spdaskit::UpdateWeightsTask<NODE>;
  //Using SKELTOSKELTASK = hmlp::spdaskit::SkeletonsToSkeletonsTask<NODE>;
  //Using SKELTONODETASK = hmlp::spdaskit::SkeletonsToNodesTask<NODE, T>;

  // Instantiation for the randomisze Spd-Askit tree.
  using RKDTSPLITTER = hmlp::spdaskit::randomsplit<N_CHILDREN, T>;
  using RKDTSETUP = hmlp::spdaskit::Setup<SPDMATRIX,RKDTSPLITTER,T>;
  using RKDTNODE = Node<RKDTSETUP, N_CHILDREN, DATA, T>;
  using KNNTASK = hmlp::spdaskit::KNNTask<RKDTNODE, T>;
 
  // All timers.
  double beg, dynamic_time, omptask_time, ref_time, ann_time, tree_time;
  double nneval_time, nonneval_time, symbolic_evaluation_time;

  // Dummy instances for each task.
  SKELTASK skeltask;
  //SKELTOSKELTASK skeltoskeltask;
  //SKELTONODETASK skeltonodetask;
  //UPDATETASK updatetask;
  KNNTASK knntask;

  // ------------------------------------------------------------------------
  // Original order of the matrix.
  // ------------------------------------------------------------------------
  const bool USE_LOWRANK = true;
  SPDMATRIX K;
  K.resize( n, n );
  if ( TESTSUITS > 0 )
  {
    std::string testsuit = std::string( "K" ) + std::to_string( TESTSUITS ) 
                                              + std::string( ".dat" );
    K.read( n, n, testsuit );
  }
  else
  {
    K.randspd<USE_LOWRANK>( 0.0, 1.0 );
  }
  //K.Print();
  std::vector<std::size_t> gids( n ), lids( n );
  for ( auto i = 0; i < n; i ++ ) 
  {
    gids[ i ] = i;
    lids[ i ] = i;
  }
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Initialize randomized Spd-Askit tree.
  // ------------------------------------------------------------------------
  Tree<RKDTSETUP, RKDTNODE, N_CHILDREN, T> rkdt;
  rkdt.setup.K = &K;
  rkdt.setup.splitter.Kptr = rkdt.setup.K;
  std::pair<T, std::size_t> initNN( std::numeric_limits<T>::max(), n );
  // ------------------------------------------------------------------------

  const size_t n_iter = 20;
  const bool SORTED = true;
  beg = omp_get_wtime();
  auto NN = rkdt.AllNearestNeighbor<SORTED>( n_iter, k, 10, gids, lids, initNN, knntask );
  ann_time = omp_get_wtime() - beg;

 /*print neighbors for checking 
  int offset=20;
  printf("For globalID %d nearest neighbors are:\n", offset);
  for ( int ii = 0 ; ii < k ; ii++)
  {
    printf( "distance: %f node: %lu\n" ,
      NN.data()[ offset*k + ii ].first ,
      NN.data()[ offset*k + ii ].second );
  }*/


  // ------------------------------------------------------------------------
  // Initialize Spd-Askit tree using approximate center split.
  // ------------------------------------------------------------------------
  Tree<SETUP, NODE, N_CHILDREN, T> tree;
  tree.setup.K = &K;
  tree.setup.splitter.Kptr = tree.setup.K; // The closure takes coordinates.
  tree.setup.NN = &NN;
  tree.setup.k = k;
  tree.setup.s = s;
  tree.setup.stol = 1E-3;
  // ------------------------------------------------------------------------

  beg = omp_get_wtime();
  tree.TreePartition( m, 10, gids, lids );
  tree_time = omp_get_wtime() - beg;



  // ------------------------------------------------------------------------
  // Sekeletonization with dynamic scheduling (symbolic traversal).
  // ------------------------------------------------------------------------
  beg = omp_get_wtime();
  tree.TraverseUp<false, true>( skeltask );
  hmlp_run();
  dynamic_time = omp_get_wtime() - beg;
  printf( "dynamic %5.2lfs level-by-level %5.2lfs OpenMP task %5.2lfs\n", 
      dynamic_time, ref_time, omptask_time );
  // ------------------------------------------------------------------------
 

//  beg = omp_get_wtime();
//  // Sekeletonization with level-by-level traversal.
//  tree.TraverseUp<true>( skeltask );
//  ref_time = omp_get_wtime() - beg;
//  beg = omp_get_wtime();
//  // Sekeletonization with omp task.
//  tree.PostOrder( tree.treelist[ 0 ], skeltask );
//  omptask_time = omp_get_wtime() - beg;
  



  // ------------------------------------------------------------------------
  // NearFarNodes
  // IMPORTANT: this requires to know the fact of ``isskel''.
  // ------------------------------------------------------------------------
  const bool SYMMETRIC_PRUNE = true;
  const bool NNPRUNE = true;
  beg = omp_get_wtime();
  hmlp::spdaskit::NearFarNodes<SYMMETRIC_PRUNE, NNPRUNE>( tree );
  symbolic_evaluation_time = omp_get_wtime() - beg;
  hmlp::spdaskit::DrawInteraction<true>( tree );
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // ComputeAll
  // ------------------------------------------------------------------------
  hmlp::Data<T> w( nrhs, n );
  w.rand();
  beg = omp_get_wtime();
  auto u = hmlp::spdaskit::ComputeAll<true, true, true, NODE>( tree, w );
  dynamic_time = omp_get_wtime() - beg;
  printf( "dynamic %5.2lfs level-by-level %5.2lfs OpenMP task %5.2lfs\n", 
      dynamic_time, ref_time, omptask_time );
  // ------------------------------------------------------------------------


  // Test evaluation with NN prunning.
  for ( size_t i = 0; i < 10; i ++ )
  {
    hmlp::Data<T> potentials;
    // Use NN pruning
    beg = omp_get_wtime();
    hmlp::spdaskit::Evaluate<false, true>( tree, i, potentials );
    nneval_time = omp_get_wtime() - beg;
    auto nnerr = hmlp::spdaskit::ComputeError( tree, i, potentials );
    // Do not use NN pruning
    beg = omp_get_wtime();
    hmlp::spdaskit::Evaluate<false, false>( tree, i, potentials );
    nonneval_time = omp_get_wtime() - beg;
    auto nonnerr = hmlp::spdaskit::ComputeError( tree, i, potentials );

    for ( size_t p = 0; p < potentials.num(); p ++ )
    {
      potentials[ p ] = u( p, i );
    }
    auto fmmerr = hmlp::spdaskit::ComputeError( tree, i, potentials );

#ifdef DUMP_ANALYSIS_DATA
    printf( "@DATA\n" );
    printf( "%5lu, %E, %E\n", i, nnerr, nonnerr );
#endif
    printf( "gid %5lu relative error (NN) %E, relative error %E fmm error %E\n", i, nnerr, nonnerr, fmmerr );
  }

#ifdef DUMP_ANALYSIS_DATA
  hmlp::spdaskit::Summary<NODE> summary;
  tree.Summary( summary );
  summary.Print();
#endif
  printf( "n %5lu k %4lu s %4lu nrhs %4lu ANN %5.3lf CONSTRUCT %5.3lf EVAL(NN) %5.3lf EVAL %5.3lf SYMBOLIC EVAL %5.3lf\n", 
      n, k, s, nrhs, ann_time, tree_time, 
      nneval_time, nonneval_time, symbolic_evaluation_time );

};




int main( int argc, char *argv[] )
{
  size_t n, m, k, s, nrhs;

  const bool ADAPTIVE = false;
  const bool RANDOMMATRIX = true;

  sscanf( argv[ 1 ], "%lu", &n );
  sscanf( argv[ 2 ], "%lu", &m );
  sscanf( argv[ 3 ], "%lu", &k );
  sscanf( argv[ 4 ], "%lu", &s );
  sscanf( argv[ 5 ], "%lu", &nrhs );

  hmlp_init();
  
  if ( RANDOMMATRIX )
  {
    test_spdaskit<ADAPTIVE,-1, double>( n, m, k, s, nrhs );
  }
  else
  {
    n = 1024;
    test_spdaskit<ADAPTIVE, 1, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE, 2, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE, 3, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE, 4, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE, 5, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE, 6, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE, 7, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE, 8, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE, 9, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE,10, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE,11, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE,12, double>( n, m, k, s, nrhs );
    test_spdaskit<ADAPTIVE,13, double>( n, m, k, s, nrhs );
  }

  //test_spdaskit<ADAPTIVE, float>( n, m, k, s, nrhs );

  hmlp_finalize();
 
  return 0;
};
