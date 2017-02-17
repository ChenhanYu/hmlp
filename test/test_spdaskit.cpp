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

#define OMPCOMPARISON 0

// By default, we use binary tree.
#define N_CHILDREN 2



//template<bool ADAPTIVE, bool LEVELRESTRICTION, 
//  typename SPLITTER, typename RKDTSPLITTER, typename T,
//  typename SPDMATRIX>
//void test_spdaskit( SPDMATRIX &K, size_t n, size_t m, size_t k, size_t s, size_t nrhs )
//{
//  using DATA = hmlp::spdaskit::Data<T>;
//  using RKDTSETUP = hmlp::spdaskit::Setup<SPDMATRIX, RKDTSPLITTER, T>;
//  using RKDTNODE = Node<RKDTSETUP, N_CHILDREN, DATA, T>;
//  using KNNTASK = hmlp::spdaskit::KNNTask<RKDTNODE, T>;
//
//  double beg, ann_time;
//
//  KNNTASK knntask;
//
//  // ------------------------------------------------------------------------
//  // Initialize randomized Spd-Askit tree.
//  // ------------------------------------------------------------------------
//  Tree<RKDTSETUP, RKDTNODE, N_CHILDREN, T> rkdt;
//  rkdt.setup.K = &K;
//  rkdt.setup.splitter.Kptr = rkdt.setup.K;
//  std::pair<T, std::size_t> initNN( std::numeric_limits<T>::max(), n );
//  // ------------------------------------------------------------------------
//
//  // ------------------------------------------------------------------------
//  std::vector<std::size_t> gids( n ), lids( n );
//  for ( auto i = 0; i < n; i ++ ) 
//  {
//    gids[ i ] = i;
//    lids[ i ] = i;
//  }
//  // ------------------------------------------------------------------------
//
//  const size_t n_iter = 20;
//  const bool SORTED = true;
//  beg = omp_get_wtime();
//  hmlp::Data<std::pair<T, std::size_t>> NN;
//  NN = rkdt.template AllNearestNeighbor<SORTED>( n_iter, k, 10, gids, lids, initNN, knntask );
//  ann_time = omp_get_wtime() - beg;
//  printf( "ANN %5.3lfs\n", ann_time );
//
//  test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, T>( K, NN, n, m, k, s, nrhs );
//  //test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, T>
////	( K, NN, n, m, k, s, nrhs );
//};




template<bool ADAPTIVE, bool LEVELRESTRICTION, 
  typename SPLITTER, typename RKDTSPLITTER, 
  typename T,
  typename SPDMATRIX>
void test_spdaskit( 
	SPDMATRIX &K, hmlp::Data<std::pair<T, std::size_t>> &NN,
	size_t n, size_t m, size_t k, size_t s, size_t nrhs )
{
  // Instantiation for the Spd-Askit tree.
  //using SPLITTER = hmlp::spdaskit::centersplit<SPDMATRIX, N_CHILDREN, T>;
  using SETUP = hmlp::spdaskit::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA = hmlp::spdaskit::Data<T>;
  using NODE = Node<SETUP, N_CHILDREN, DATA, T>;
  using SKELTASK = hmlp::spdaskit::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T>;

  // Instantiation for the randomisze Spd-Askit tree.
  //using RKDTSPLITTER = hmlp::spdaskit::randomsplit<SPDMATRIX, N_CHILDREN, T>;
  using RKDTSETUP = hmlp::spdaskit::Setup<SPDMATRIX, RKDTSPLITTER, T>;
  using RKDTNODE = Node<RKDTSETUP, N_CHILDREN, DATA, T>;
  using KNNTASK = hmlp::spdaskit::KNNTask<RKDTNODE, T>;
 
  // All timers.
  double beg, dynamic_time, omptask45_time, omptask_time, ref_time, ann_time, tree_time, overhead_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

  // Dummy instances for each task.
  SKELTASK skeltask;
  KNNTASK knntask;

  // ------------------------------------------------------------------------
  // Original order of the matrix.
  // ------------------------------------------------------------------------
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

  printf( "AllNearestNeighbors ..." ); fflush( stdout );
  const size_t n_iter = 20;
  const bool SORTED = true;
  beg = omp_get_wtime();
  if ( NN.size() != n * k )
  {
    NN = rkdt.template AllNearestNeighbor<SORTED>( n_iter, k, 10, gids, lids, initNN, knntask );
  }
  else
  {
	printf( "not performed (precomputed) ..." ); fflush( stdout );
  }
  ann_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );

  // ------------------------------------------------------------------------
  // Initialize Spd-Askit tree using approximate center split.
  // ------------------------------------------------------------------------
  Tree<SETUP, NODE, N_CHILDREN, T> tree;
  tree.setup.K = &K;
  tree.setup.splitter.Kptr = tree.setup.K; // The closure takes coordinates.
  tree.setup.NN = &NN;
  tree.setup.m = m;
  tree.setup.k = k;
  tree.setup.s = s;
  tree.setup.stol = 1E-3;
  // ------------------------------------------------------------------------

  printf( "TreePartition ..." ); fflush( stdout );
  beg = omp_get_wtime();
  tree.TreePartition( gids, lids );
  tree_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );



  // ------------------------------------------------------------------------
  // Sekeletonization with dynamic scheduling (symbolic traversal).
  // ------------------------------------------------------------------------
  printf( "Skeletonization (HMLP Runtime) ..." ); fflush( stdout );
  beg = omp_get_wtime();
  tree.template TraverseUp<false, true, SKELTASK>( skeltask );
  overhead_time = omp_get_wtime() - beg;
  hmlp_run();
  dynamic_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );

  // Parallel level-by-level traversal.
  printf( "Skeletonization (Level-By-Level) ..." ); fflush( stdout );
  beg = omp_get_wtime();
  if ( OMPCOMPARISON ) tree.template TraverseUp<false, false, SKELTASK>( skeltask );
  ref_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );

  // Sekeletonization with omp task.
  printf( "Skeletonization (Recursive OpenMP tasks) ..." ); fflush( stdout );
  beg = omp_get_wtime();
  if ( OMPCOMPARISON ) tree.PostOrder<true>( tree.treelist[ 0 ], skeltask );
  omptask_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );

  // Sekeletonization with omp task.
  printf( "Skeletonization (OpenMP-4.5 Dependency tasks) ..." ); fflush( stdout );
  beg = omp_get_wtime();
  if ( OMPCOMPARISON ) tree.PostOrder<false>( tree.treelist[ 0 ], skeltask );
  omptask45_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );

  printf( "Runtime %5.2lfs (overhead %5.2lfs) level-by-level %5.2lfs OMP task %5.2lfs OMP-4.5 %5.2lfs\n", 
      dynamic_time, overhead_time, ref_time, omptask_time, omptask45_time ); fflush( stdout );
  // ------------------------------------------------------------------------
 

  // ------------------------------------------------------------------------
  // NearFarNodes
  // IMPORTANT: this requires to know the fact of ``isskel''.
  // ------------------------------------------------------------------------
  const bool SYMMETRIC_PRUNE = true;
  const bool NNPRUNE = true;
  printf( "NearFarNodes ..." ); fflush( stdout );
  beg = omp_get_wtime();
  hmlp::spdaskit::NearFarNodes<SYMMETRIC_PRUNE, NNPRUNE>( tree );
  symbolic_evaluation_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );
  hmlp::spdaskit::DrawInteraction<true>( tree );
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // ComputeAll
  // ------------------------------------------------------------------------
  printf( "ComputeAll (HMLP Runtime) ..." ); fflush( stdout );
  hmlp::Data<T> w( nrhs, n );
  w.rand();
  beg = omp_get_wtime();
  auto u = hmlp::spdaskit::ComputeAll<true, true, true, NODE>( tree, w );
  dynamic_time = omp_get_wtime() - beg;
  fmm_evaluation_time = dynamic_time;
  printf( "Done.\n" ); fflush( stdout );

  printf( "ComputeAll (Level-By-Level) ..." ); fflush( stdout );
  beg = omp_get_wtime();
  if ( OMPCOMPARISON ) u = hmlp::spdaskit::ComputeAll<false, true, true, NODE>( tree, w );
  ref_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );

  beg = omp_get_wtime();
  omptask_time = omp_get_wtime() - beg;

  beg = omp_get_wtime();
  omptask45_time = omp_get_wtime() - beg;

  printf( "Runtime %5.2lfs level-by-level %5.2lfs OMP task %5.2lfs OMP-4.5 %5.2lfs\n", 
      dynamic_time, ref_time, omptask_time, omptask45_time ); fflush( stdout );
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Test evaluation with NN prunning.
  // ------------------------------------------------------------------------
  std::size_t ntest = 100;
  T nnerr_avg = 0.0;
  T nonnerr_avg = 0.0;
  T fmmerr_avg = 0.0;
  printf( "================================================================\n" );
  for ( size_t i = 0; i < ntest; i ++ )
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

    for ( size_t p = 0; p < potentials.col(); p ++ )
    {
      potentials[ p ] = u( p, i );
    }
    auto fmmerr = hmlp::spdaskit::ComputeError( tree, i, potentials );

	// Only print 10 values.
	if ( i < 10 )
	{
#ifdef DUMP_ANALYSIS_DATA
      printf( "@DATA\n" );
      printf( "%5lu, %E, %E\n", i, nnerr, nonnerr );
#endif
      printf( "gid %5lu relative error (NN) %E, relative error %E fmm error %E\n", 
	  	i, nnerr, nonnerr, fmmerr );
	}
	nnerr_avg += nnerr;
	nonnerr_avg += nonnerr;
	fmmerr_avg += fmmerr;
  }
  printf( "================================================================\n" );
  printf( "          relative error (NN) %E, relative error %E fmm error %E\n", 
	  	nnerr_avg / ntest , nonnerr_avg / ntest, fmmerr_avg / ntest );
  printf( "================================================================\n" );
  // ------------------------------------------------------------------------

//#ifdef DUMP_ANALYSIS_DATA
  hmlp::spdaskit::Summary<NODE> summary;
  tree.Summary( summary );
  summary.Print();
//#endif
  printf( "n %5lu k %4lu s %4lu nrhs %4lu ANN %5.3lf CONSTRUCT %5.3lf EVAL(NN) %5.3lf EVAL %5.3lf EVAL(FMM) %5.3lf SYMBOLIC(FMM) %5.3lf\n", 
      n, k, s, nrhs, ann_time, tree_time,
      nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time );

};




int main( int argc, char *argv[] )
{
  const bool ADAPTIVE = true;
  const bool LEVELRESTRICTION = false;
  const bool RANDOMMATRIX = false;
  const bool USE_LOWRANK = true;
  const bool DENSETESTSUIT = false;
  const bool SPARSETESTSUIT = false;
  const bool GRAPHTESTSUIT = false;
  const bool OOCTESTSUIT = false;
  const bool KERNELTESTSUIT = true;

  std::string DATADIR( "/scratch/jlevitt/data/" );

  size_t n, m, d, k, s, nrhs;

  using T = double;
  //using SPLITTER = hmlp::spdaskit::centersplit<SPDMATRIX, N_CHILDREN, T>;

  sscanf( argv[ 1 ], "%lu", &n );
  sscanf( argv[ 2 ], "%lu", &m );
  sscanf( argv[ 3 ], "%lu", &k );
  sscanf( argv[ 4 ], "%lu", &s );
  sscanf( argv[ 5 ], "%lu", &nrhs );

  //T val[ 6 ] = { 1.0, 4.0, 5.0, 2.0, 3.0, 6.0 };
  //std::size_t row_ind[ 6 ] = { 0, 2, 2, 0, 1, 2 };
  //std::size_t col_ptr[ 4 ] = { 0, 2, 3, 6 };
  //hmlp::CSC<T> myK( (size_t)3, (size_t)3, (size_t)6, val, row_ind, col_ptr );
  //myK.Print();


  //hmlp::CSC<T> LF10( (size_t)18, (size_t)18, (size_t)50 );
  //std::string filename( "/Users/chenhan/Documents/Projects/hmlp/build/bin/LF10/LF10.mtx" );
  //LF10.readmtx<false>( filename );
  //LF10.Print();





  hmlp_init();
  
  if ( RANDOMMATRIX )
  {
    hmlp::spdaskit::SPDMatrix<T> K;
    hmlp::Data<std::pair<T, std::size_t>> NN;
    using SPLITTER = hmlp::spdaskit::centersplit<hmlp::spdaskit::SPDMatrix<T>, N_CHILDREN, T>;
    using RKDTSPLITTER = hmlp::spdaskit::randomsplit<hmlp::spdaskit::SPDMatrix<T>, N_CHILDREN, T>;
    K.resize( n, n );
    K.randspd<USE_LOWRANK>( 0.0, 1.0 );
    test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
  }
  
  if ( DENSETESTSUIT )
  {
    n = 4096;
    hmlp::spdaskit::SPDMatrix<T> K;
    hmlp::Data<std::pair<T, std::size_t>> NN;
    using SPLITTER = hmlp::spdaskit::centersplit<hmlp::spdaskit::SPDMatrix<T>, N_CHILDREN, T>;
    using RKDTSPLITTER = hmlp::spdaskit::randomsplit<hmlp::spdaskit::SPDMatrix<T>, N_CHILDREN, T>;
    K.resize( n, n );
    for ( size_t id = 1; id < 14; id ++ )
    {
      std::ostringstream id_stream;
      id_stream << id;
      std::string filename = DATADIR + std::string( "K" ) + id_stream.str()
                                                + std::string( ".dat" );
      K.read( n, n, filename );
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
    }
  }

  if ( SPARSETESTSUIT )
  {
	const bool SYMMETRIC = false;
	const bool LOWERTRIANGULAR = true;
    using SPLITTER = hmlp::spdaskit::centersplit<hmlp::CSC<SYMMETRIC, T>, N_CHILDREN, T>;
    using RKDTSPLITTER = hmlp::spdaskit::randomsplit<hmlp::CSC<SYMMETRIC, T>, N_CHILDREN, T>;
    //{
    //  std::string filename = DATADIR + std::string( "bcsstk10.mtx" );
    //  n = 1086;
    //  hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)11578 );
    //  K.readmtx<LOWERTRIANGULAR, false>( filename );
    //  //hmlp::Data<std::pair<T, std::size_t>> NN;
    //  hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
    //  test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
    //}
    {
      std::string filename = DATADIR + std::string( "inline_1.mtx" );
      n = 503712;
      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)18660027 );
      K.readmtx<LOWERTRIANGULAR, false>( filename );
      //hmlp::Data<std::pair<T, std::size_t>> NN;
      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
    }
    {
      std::string filename = DATADIR + std::string( "msdoor.mtx" );
      n = 415863;
      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)10328399 );
      K.readmtx<LOWERTRIANGULAR, false>( filename );
      //hmlp::Data<std::pair<T, std::size_t>> NN;
      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
    }
    //{
    //  std::string filename = DATADIR + std::string( "thermal2.mtx" );
    //  n = 1228045;
    //  hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)4904179 );
    //  K.readmtx<LOWERTRIANGULAR, false>( filename );
    //  //hmlp::Data<std::pair<T, std::size_t>> NN;
    //  hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
    //  test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
    //}
  }

  if ( GRAPHTESTSUIT )
  {
	const bool SYMMETRIC = false;
	const bool LOWERTRIANGULAR = true;
	const bool IJONLY = true;
    using SPLITTER = hmlp::spdaskit::centersplit<hmlp::CSC<SYMMETRIC, T>, N_CHILDREN, T>;
    using RKDTSPLITTER = hmlp::spdaskit::randomsplit<hmlp::CSC<SYMMETRIC, T>, N_CHILDREN, T>;
	{
      std::string filename = std::string( "ca-AstroPh.mtx" );
      n = 18772;
      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)198110 );
      K.readmtx<LOWERTRIANGULAR, false, IJONLY>( filename );
      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
	}
	{
      std::string filename = std::string( "email-Enron.mtx" );
      n = 36692;
      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)183831 );
      K.readmtx<LOWERTRIANGULAR, false, IJONLY>( filename );
      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
	}
	//{
    //  std::string filename = std::string( "as-Skitter.mtx" );
    //  n = 1696415;
    //  hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)11095298 );
    //  K.readmtx<LOWERTRIANGULAR, false, IJONLY>( filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
    //  test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
	//}
  }

  if ( OOCTESTSUIT )
  {
    using SPLITTER = hmlp::spdaskit::centersplit<hmlp::OOC<T>, N_CHILDREN, T>;
    using RKDTSPLITTER = hmlp::spdaskit::randomsplit<hmlp::OOC<T>, N_CHILDREN, T>;
    n = 4096;
    for ( size_t id = 1; id < 14; id ++ )
    {
      hmlp::Data<std::pair<T, std::size_t>> NN;
      std::ostringstream id_stream;
      id_stream << id;
      std::string filename = DATADIR + std::string( "K" ) + id_stream.str()
                                                + std::string( ".dat" );
      hmlp::OOC<T> K( n, n, filename );
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
    }
  }

  if ( KERNELTESTSUIT )
  {
    const bool SYMMETRIC = true;
    using SPLITTER = hmlp::spdaskit::centersplit<hmlp::Kernel<SYMMETRIC, T>, N_CHILDREN, T>;
    using RKDTSPLITTER = hmlp::spdaskit::randomsplit<hmlp::Kernel<SYMMETRIC, T>, N_CHILDREN, T>;
    {
      std::string filename = DATADIR + std::string( "covtype.100k.trn.X.bin" );
      n = 100000;
      d = 54;
      kernel_s<T> kernel;
      kernel.type = KS_GAUSSIAN;
      kernel.scal = -0.5;
      hmlp::Kernel<SYMMETRIC, T> K( n, n, d, kernel );
      K.read( n, d, filename );
      hmlp::Data<std::pair<T, std::size_t>> NN;
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>( K, NN, n, m, k, s, nrhs );
    }
  }

  hmlp_finalize();
 
  return 0;
};
