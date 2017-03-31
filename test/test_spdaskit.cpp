#include <tuple>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <iomanip>
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

#ifdef HMLP_USE_CUDA
#include <hmlp_gpu.hpp>
#endif

#ifdef HMLP_AVX512
/** this is for hbw_malloc() and hnw_free */
#include <hbwmalloc.h>
/** we need hbw::allocator<T> to replace std::allocator<T> */
#include <hbw_allocator.h>
/** MKL headers */
#include <mkl.h>
#endif

using namespace hmlp::tree;

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

#define OMPLEVEL 0
#define OMPRECTASK 0
#define OMPDAGTASK 0

/** by default, we use binary tree */
#define N_CHILDREN 2


/** This function tests for omp task depend.  */
void OpenMP45()
{
#pragma omp parallel
#pragma omp single
  {
    int a[3];
    #pragma omp task shared(a) depend(out:a[1])
    {
      a[1] = 1;
      printf( "me %d\n", 1 );
    }

    #pragma omp task shared(a) depend(out:a[2])
    {
      a[2] = 2;
      printf( "me %d\n", 2 );
    }

    #pragma omp task shared(a) depend(in:a[1],a[2]) depend(out:a[0])
    {
      a[0] = 0;
      printf( "me %d\n", 0 );
    }
    printf( "prepare to terminate\n" );
  }
  printf( "now  terminate\n" );
};



template<bool CACHENEARNODE, typename TREE, typename SKELTASK, typename PROJTASK, 
  typename NEARNODESTASK, typename CACHENEARNODESTASK>
void Compress( TREE &tree, SKELTASK &skeltask, PROJTASK &projtask,
    NEARNODESTASK &dummy, CACHENEARNODESTASK &cachenearnodestask )
{
  /** all timers */
  double beg, dynamic_time, omptask45_time, omptask_time, ref_time, ann_time, tree_time, overhead_time;

  /** NearNodes */
  auto *nearnodestask = new NEARNODESTASK();
  nearnodestask->Set( &tree );
  nearnodestask->Submit();

#ifdef HMLP_AVX512
  /** if we are using KNL, use nested omp construct */
  assert( omp_get_max_threads() == 68 );
  mkl_set_dynamic( 0 );
  mkl_set_num_threads( 4 );
  hmlp_set_num_workers( 17 );
  //printf( "here\n" );
#else
  mkl_set_dynamic( 0 );
  if ( omp_get_max_threads() > 8 )
  {
    mkl_set_num_threads( 2 );
    hmlp_set_num_workers( omp_get_max_threads() / 2 );
  }
  printf( "omp_get_max_threads() %d\n", omp_get_max_threads() );
#endif


  /** runtime */
  printf( "Skeletonization (HMLP Runtime) ..." ); fflush( stdout );
  const bool AUTODEPENDENCY = true;
  beg = omp_get_wtime();
  tree.template TraverseUp<AUTODEPENDENCY, true>( skeltask );
  nearnodestask->DependencyAnalysis();
  tree.template TraverseUnOrdered<AUTODEPENDENCY, true>( projtask );
  if ( CACHENEARNODE )
    tree.template TraverseLeafs<AUTODEPENDENCY, true>( cachenearnodestask );
  overhead_time = omp_get_wtime() - beg;
  hmlp_run();
  dynamic_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );


#ifdef HMLP_AVX512
  mkl_set_dynamic( 1 );
  mkl_set_num_threads( omp_get_max_threads() );
#else
  mkl_set_dynamic( 1 );
  mkl_set_num_threads( omp_get_max_threads() );
#endif


  /** parallel level-by-level traversal */
  beg = omp_get_wtime();
  if ( OMPLEVEL ) 
  {
    printf( "Skeletonization (Level-By-Level) ..." ); fflush( stdout );
    tree.template TraverseUp<false, false>( skeltask );
    printf( "Proj (Level-By-Level) ..." ); fflush( stdout );
    tree.template TraverseUnOrdered<false, false>( projtask );
    //printf( "NearNodes (Level-By-Level) ..." ); fflush( stdout );
    //nearnodestask->Execute( NULL );

    printf( "CacheNearNode (Level-By-Level) ..." ); fflush( stdout );
    if ( CACHENEARNODE )
      tree.template TraverseLeafs<false, false>( cachenearnodestask );
    printf( "Done.\n" ); fflush( stdout );
  }
  ref_time = omp_get_wtime() - beg;


  /** sekeletonization with omp task. */
  beg = omp_get_wtime();
  if ( OMPRECTASK ) 
  {
    printf( "Skeletonization (Recursive OpenMP tasks) ..." ); fflush( stdout );
    tree.template PostOrder<true>( tree.treelist[ 0 ], skeltask );
    tree.template TraverseUp<false, false>( projtask );
    nearnodestask->Execute( NULL );
    if ( CACHENEARNODE )
      tree.template TraverseLeafs<false, false>( cachenearnodestask );
    printf( "Done.\n" ); fflush( stdout );
  }
  omptask_time = omp_get_wtime() - beg;


  /** sekeletonization with omp task. */
  beg = omp_get_wtime();
  if ( OMPDAGTASK ) 
  {
    printf( "Skeletonization (OpenMP-4.5 Dependency tasks) ..." ); fflush( stdout );
    tree.template UpDown<true, true, false>( skeltask, projtask, projtask );
    //nearnodestask->Execute( NULL );
    if ( CACHENEARNODE )
      tree.template TraverseLeafs<false, false>( cachenearnodestask );
    printf( "Done.\n" ); fflush( stdout );
  }
  omptask45_time = omp_get_wtime() - beg;

  printf( "Runtime %5.2lfs (overhead %5.2lfs) level-by-level %5.2lfs OMP task %5.2lfs OMP-4.5 %5.2lfs\n", 
      dynamic_time, overhead_time, ref_time, omptask_time, omptask45_time ); fflush( stdout );

}; /** void Compress() */





template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  SplitScheme SPLIT,
  typename    SPLITTER, 
  typename    RKDTSPLITTER, 
  typename    T, 
  typename    SPDMATRIX>
void test_spdaskit( 
  hmlp::Data<T> *X,
  SPDMATRIX &K, hmlp::Data<std::pair<T, std::size_t>> &NN,
  SPLITTER splitter, RKDTSPLITTER rkdtsplitter,
  size_t n, size_t m, size_t k, size_t s, T stol, size_t nrhs )
{
  /** instantiation for the Spd-Askit tree */
  using SETUP = hmlp::spdaskit::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA = hmlp::spdaskit::Data<T>;
  using NODE = Node<SETUP, N_CHILDREN, DATA, T>;
  using SKELTASK = hmlp::spdaskit::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T>;
  using PROJTASK = hmlp::spdaskit::InterpolateTask<NODE, T>;

  /** instantiation for the randomisze Spd-Askit tree */
  using RKDTSETUP = hmlp::spdaskit::Setup<SPDMATRIX, RKDTSPLITTER, T>;
  using RKDTNODE = Node<RKDTSETUP, N_CHILDREN, DATA, T>;
  using KNNTASK = hmlp::spdaskit::KNNTask<3, SPLIT, RKDTNODE, T>;
 
  /** all timers */
  double beg, dynamic_time, omptask45_time, omptask_time, ref_time;
  double ann_time, tree_time, overhead_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

  /** dummy instances for each task */
  SKELTASK skeltask;
  PROJTASK projtask;
  KNNTASK knntask;

  /** original order of the matrix */
  std::vector<std::size_t> gids( n ), lids( n );
  for ( auto i = 0; i < n; i ++ ) 
  {
    gids[ i ] = i;
    lids[ i ] = i;
  }



  // ------------------------------------------------------------------------
  // Initialize randomized Spd-Askit tree.
  // ------------------------------------------------------------------------
  const size_t n_iter = 10;
  const bool SORTED = false;
  Tree<RKDTSETUP, RKDTNODE, N_CHILDREN, T> rkdt;
  rkdt.setup.X = X;
  rkdt.setup.K = &K;
  rkdt.setup.splitter = rkdtsplitter;
  std::pair<T, std::size_t> initNN( std::numeric_limits<T>::max(), n );
  // ------------------------------------------------------------------------

  printf( "AllNearestNeighbors ..." ); fflush( stdout );
  beg = omp_get_wtime();
  if ( NN.size() != n * k )
  {
    NN = rkdt.template AllNearestNeighbor<SORTED>
         ( n_iter, k, 10, gids, lids, initNN, knntask );
  }
  else
  {
    printf( "not performed (precomputed or k=0) ..." ); fflush( stdout );
  }
  ann_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );



  // ------------------------------------------------------------------------
  // Initialize Spd-Askit tree using approximate center split.
  // ------------------------------------------------------------------------
  Tree<SETUP, NODE, N_CHILDREN, T> tree;
  tree.setup.X = X;
  tree.setup.K = &K;
  tree.setup.splitter = splitter;
  tree.setup.NN = &NN;
  tree.setup.m = m;
  tree.setup.k = k;
  tree.setup.s = s;
  tree.setup.stol = stol;
  // ------------------------------------------------------------------------

  printf( "TreePartition ..." ); fflush( stdout );
  beg = omp_get_wtime();
  tree.TreePartition( gids, lids );
  tree_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );


  // ------------------------------------------------------------------------
  // Sekeletonization with dynamic scheduling (symbolic traversal).
  // ------------------------------------------------------------------------
  const int BUDGET = 60; /** 3% */
  const bool SYMMETRIC_PRUNE = true;
  const bool NNPRUNE = true;
  const bool CACHE = true;
  using NEARNODESTASK = hmlp::spdaskit::NearNodesTask<BUDGET, SYMMETRIC_PRUNE, NNPRUNE, Tree<SETUP, NODE, N_CHILDREN, T>>;
  using CACHENEARNODESTASK = hmlp::spdaskit::CacheNearNodesTask<NNPRUNE, NODE>;
  NEARNODESTASK nearnodestask;
  CACHENEARNODESTASK cachenearnodestask;
  // ------------------------------------------------------------------------
  
  /** compree the matrix  */
  Compress<CACHE>( tree, skeltask, projtask, nearnodestask, cachenearnodestask );

  /** FarNodes */
  printf( "FarNodes ..." ); fflush( stdout );
  beg = omp_get_wtime();
  hmlp::spdaskit::NearFarNodes<SYMMETRIC_PRUNE, NNPRUNE, CACHE, NODE>( tree );
  symbolic_evaluation_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );

  /** plot iteraction matrix */  
  auto exact_ratio = hmlp::spdaskit::DrawInteraction<true>( tree );


  // ------------------------------------------------------------------------
  // ComputeAll
  // ------------------------------------------------------------------------
#ifdef HMLP_AVX512
  /** if we are using KNL, use nested omp construct */
  assert( omp_get_max_threads() == 68 );
  mkl_set_dynamic( 0 );
  mkl_set_num_threads( 2 );
  hmlp_set_num_workers( 34 );
#else
  //mkl_set_dynamic( 0 );
  //mkl_set_num_threads( 2 );
  //hmlp_set_num_workers( omp_get_max_threads() / 2 );
  hmlp_set_num_workers( omp_get_max_threads() );
  printf( "omp_get_max_threads() %d\n", omp_get_max_threads() );
#endif

  printf( "ComputeAll (HMLP Runtime) ..." ); fflush( stdout );
  hmlp::Data<T> w( nrhs, n );
  w.rand();
  beg = omp_get_wtime();
  auto u = hmlp::spdaskit::ComputeAll<true, false, true, true, CACHE, NODE>( tree, w );
  dynamic_time = omp_get_wtime() - beg;
  fmm_evaluation_time = dynamic_time;
  printf( "Done.\n" ); fflush( stdout );

#ifdef HMLP_AVX512
  mkl_set_dynamic( 1 );
  mkl_set_num_threads( omp_get_max_threads() );
#else
  //mkl_set_dynamic( 1 );
  //mkl_set_num_threads( omp_get_max_threads() );
#endif


  /** omp level-by-level */
  beg = omp_get_wtime();
  if ( OMPLEVEL ) 
  {
    printf( "ComputeAll (Level-By-Level) ..." ); fflush( stdout );
    u = hmlp::spdaskit::ComputeAll<false, false, true, true, CACHE, NODE>( tree, w );
    printf( "Done.\n" ); fflush( stdout );
  }
  ref_time = omp_get_wtime() - beg;
  printf( "Done.\n" ); fflush( stdout );

  /** omp recu task */
  beg = omp_get_wtime();
  omptask_time = omp_get_wtime() - beg;

  /** omp recu task depend */
  beg = omp_get_wtime();
  if ( OMPDAGTASK )
  {
    u = hmlp::spdaskit::ComputeAll<false, true, true, true, CACHE, NODE>( tree, w );
  }
  omptask45_time = omp_get_wtime() - beg;

  printf( "Exact ratio %5.2lf Runtime %5.2lfs level-by-level %5.2lfs OMP task %5.2lfs OMP-4.5 %5.2lfs\n", 
      exact_ratio, dynamic_time, ref_time, omptask_time, omptask45_time ); fflush( stdout );
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
  // Decomment for latex tables
  /*
     printf( "TABLE: n %5lu m %4lu k %4lu s %4lu nrhs %4lu e(nntree) %E %%K %1.3lf ANN %5.3lf CONSTRUCT %5.2lf SKEL %5.2lf EVAL(NN) %5.3lf EVAL %5.3lf EVAL(FMM) %5.3lf SYMBOLIC(FMM) %5.3lf\n",
     n, m, k, s, nrhs, nnerr_avg / ntest, exact_ratio,ann_time, tree_time, skel_time,
     nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time );
     */

};



template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  SplitScheme SPLIT,
  typename    T, 
  typename    SPDMATRIX>
void test_spdaskit_setup( 
  hmlp::Data<T> *X,
  SPDMATRIX &K, hmlp::Data<std::pair<T, std::size_t>> &NN,
  size_t n, size_t m, size_t k, size_t s, T stol, size_t nrhs )
{
  switch ( SPLIT )
  {
    case SPLIT_POINT_DISTANCE:
    {
      assert( X );

      using SPLITTER = hmlp::tree::centersplit<N_CHILDREN, T>;
      using RKDTSPLITTER = hmlp::tree::randomsplit<N_CHILDREN, T>;
      SPLITTER splitter;
      splitter.Coordinate = X;
      RKDTSPLITTER rkdtsplitter;
      rkdtsplitter.Coordinate = X;
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLIT, SPLITTER, RKDTSPLITTER, T>
      ( X, K, NN, splitter, rkdtsplitter, n, m, k, s, stol, nrhs );
      break;
    }
    case SPLIT_KERNEL_DISTANCE:
    case SPLIT_ANGLE:
    {
      using SPLITTER = hmlp::spdaskit::centersplit<SPDMATRIX, N_CHILDREN, T, SPLIT>;
      using RKDTSPLITTER = hmlp::spdaskit::randomsplit<SPDMATRIX, N_CHILDREN, T, SPLIT>;
      SPLITTER splitter;
      splitter.Kptr = &K;
      RKDTSPLITTER rkdtsplitter;
      rkdtsplitter.Kptr = &K;
      test_spdaskit<ADAPTIVE, LEVELRESTRICTION, SPLIT, SPLITTER, RKDTSPLITTER, T>
      ( X, K, NN, splitter, rkdtsplitter, n, m, k, s, stol, nrhs );
      break;
    }
    default:
      exit( 1 );
  }
}



template<typename TMATRIX, typename SPLITTER, typename T>
//void OpenMP45Site( TMATRIX &K, T &dummy )
void OpenMP45Site( T &dummy )
{
  OpenMP45();
};


int main( int argc, char *argv[] )
{
  const bool ADAPTIVE = true;
  // const SplitScheme SPLIT = SPLIT_POINT_DISTANCE;
  //const SplitScheme SPLIT = SPLIT_KERNEL_DISTANCE;
  const SplitScheme SPLIT = SPLIT_ANGLE;
  const bool POINTDISTANCE = true;
  const bool KERNELDISTANCE = true;
  const bool ANGLE = true;
  const bool LEVELRESTRICTION = false;

  /** test suit options */
  const bool RANDOMMATRIX = false;
  const bool USE_LOWRANK = true;
  const bool DENSETESTSUIT = false;
  const bool SPARSETESTSUIT = false;
  const bool GRAPHTESTSUIT = false;
  const bool OOCTESTSUIT = false;
  const bool KERNELTESTSUIT = true;
  const bool SCALETESTSUIT = true;

  //std::string DATADIR( "/scratch/jlevitt/data/" );
  std::string DATADIR( "/work/02794/ych/data/" );
  //std::string DATADIR( "/users/chenhan/data/" );
  //std::string DATADIR( "/workspace/biros/sc17/" );

  size_t n, m, d, k, s, nrhs;

  using T = double;

  sscanf( argv[ 1 ], "%lu", &n );
  sscanf( argv[ 2 ], "%lu", &m );
  sscanf( argv[ 3 ], "%lu", &k );
  sscanf( argv[ 4 ], "%lu", &s );
  sscanf( argv[ 5 ], "%lu", &nrhs );
  T stol = 1E-5;
  //T stol = 0.0;


  //T val[ 6 ] = { 1.0, 4.0, 5.0, 2.0, 3.0, 6.0 };
  //std::size_t row_ind[ 6 ] = { 0, 2, 2, 0, 1, 2 };
  //std::size_t col_ptr[ 4 ] = { 0, 2, 3, 6 };
  //hmlp::CSC<T> myK( (size_t)3, (size_t)3, (size_t)6, val, row_ind, col_ptr );
  //myK.Print();


  //hmlp::CSC<T> LF10( (size_t)18, (size_t)18, (size_t)50 );
  //std::string filename( "/Users/chenhan/Documents/Projects/hmlp/build/bin/LF10/LF10.mtx" );
  //LF10.readmtx<false>( filename );
  //LF10.Print();


  //printf( "Not from a template function\n" );
  //OpenMP45();
  //printf( "template<typename float> function\n" );
  //{
  //  float dummy1;
  //  using TMATRIX = hmlp::spdaskit::SPDMatrix<T>;
  //  using SPLITTER = hmlp::spdaskit::centersplit<hmlp::spdaskit::SPDMatrix<T>, N_CHILDREN, T>;
  //  hmlp::spdaskit::SPDMatrix<T> K;
  //  OpenMP45Site<TMATRIX, SPLITTER>( dummy1 );
  //}
  //printf( "template<typename double> function\n" );
  //{
  //  double dummy2;
  //  using TMATRIX = hmlp::spdaskit::SPDMatrix<T>;
  //  using SPLITTER = hmlp::spdaskit::centersplit<hmlp::spdaskit::SPDMatrix<T>, N_CHILDREN, T>;
  //  hmlp::spdaskit::SPDMatrix<T> K;
  //  OpenMP45Site<TMATRIX, SPLITTER>( dummy2 );
  //}

//  auto gpu = hmlp::gpu::Nvidia( 0 );
//
//  hmlp::Data<T> tmp( 10, 10 );
//
//  /** test device_data() */
//  auto *ptr_d = tmp.device_data( &gpu );
//
//  /** */
//  tmp.PrefetchH2D( &gpu );
//  tmp.WaitPrefetch( &gpu );


  hmlp_init();
  
//  if ( RANDOMMATRIX )
//  {
//    hmlp::Data<T> *X = NULL;
//    hmlp::spdaskit::SPDMatrix<T> K;
//    K.resize( n, n );
//    K.randspd<USE_LOWRANK>( 0.0, 1.0 );
//    hmlp::Data<std::pair<T, std::size_t>> NN;
//    test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//    ( X, K, NN, n, m, k, s, stol, nrhs );
//  }
  
//  if ( DENSETESTSUIT )
//  {
//    using T = float;
//    // std::string DATADIR( "/scratch/sreiz/65536/" );
//    std::string DATADIR( "/work/02497/jll2752/maverick/data/" );
//    n = 16384;
//    for ( size_t id = 1; id <= 16; id ++ )
//    {
//      std::ostringstream id_stream;
//      id_stream << std::setw( 2 ) << std::setfill( '0' ) << id;
//      std::string filename = DATADIR + std::string( "K" ) + id_stream.str()
//                                                + std::string( "N16384.bin" );
//      hmlp::spdaskit::SPDMatrix<T> K;
//      K.resize( n, n );
//      K.read( n, n, filename );
//
//      std::string pointfilename;
//      switch ( id )
//      {
//        case 11:
//          printf( "Test for matrix K11 not yet implemented\n" );
//          // d = 1;
//          // pointfilename = DATADIR + std::string( "X1DN16384.points.bin" );
//          continue;
//        case 1:
//        case 2:
//        case 3:
//        case 12:
//        case 13:
//        case 14:
//        case 15:
//        case 16:
//          d = 2;
//          pointfilename = DATADIR + std::string( "X2DN16384.points.bin" );
//          break;
//        case 4:
//        case 5:
//        case 6:
//        case 7:
//        case 8:
//        case 9:
//        case 10:
//          d = 3;
//          pointfilename = DATADIR + std::string( "XKEN16384.points.bin" );
//          break;
//      }
//      hmlp::Data<T> X( d, n, pointfilename );
//      hmlp::Data<std::pair<T, std::size_t>> NN;
//
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( &X, K, NN, n, m, k, s, stol, nrhs );
//    }
//  }

//  if ( SPARSETESTSUIT )
//  {
//    const bool SYMMETRIC = false;
//    const bool LOWERTRIANGULAR = true;
//
//    {
//      std::string filename = DATADIR + std::string( "bcsstk10.mtx" );
//      n = 1086;
//      hmlp::Data<T> *X = NULL;
//      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)11578 );
//      K.readmtx<LOWERTRIANGULAR, false>( filename );
//      //hmlp::Data<std::pair<T, std::size_t>> NN;
//      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( X, K, NN, n, m, k, s, stol, nrhs );
//    }
//    {
//      std::string filename = DATADIR + std::string( "inline_1.mtx" );
//      n = 503712;
//      hmlp::Data<T> *X = NULL;
//      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)18660027 );
//      K.readmtx<LOWERTRIANGULAR, false>( filename );
//      //hmlp::Data<std::pair<T, std::size_t>> NN;
//      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( X, K, NN, n, m, k, s, stol, nrhs );
//    }
//    {
//      std::string filename = DATADIR + std::string( "msdoor.mtx" );
//      n = 415863;
//      hmlp::Data<T> *X = NULL;
//      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)10328399 );
//      K.readmtx<LOWERTRIANGULAR, false>( filename );
//      //hmlp::Data<std::pair<T, std::size_t>> NN;
//      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( X, K, NN, n, m, k, s, stol, nrhs );
//    }
//    {
//      std::string filename = DATADIR + std::string( "thermal2.mtx" );
//      n = 1228045;
//      hmlp::Data<T> *X = NULL;
//      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)4904179 );
//      K.readmtx<LOWERTRIANGULAR, false>( filename );
//      //hmlp::Data<std::pair<T, std::size_t>> NN;
//      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( X, K, NN, n, m, k, s, stol, nrhs );
//    }
//  }

//  if ( GRAPHTESTSUIT )
//  {
//    const bool SYMMETRIC = false;
//    const bool LOWERTRIANGULAR = true;
//    const bool IJONLY = true;
//    {
//      std::string filename = DATADIR + std::string( "ca-AstroPh.mtx" );
//      n = 18772;
//      hmlp::Data<T> *X = NULL;
//      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)198110 );
//      K.readmtx<LOWERTRIANGULAR, false, IJONLY>( filename );
//      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//        ( X, K, NN, n, m, k, s, stol, nrhs );
//    }
//    {
//      std::string filename = DATADIR + std::string( "email-Enron.mtx" );
//      n = 36692;
//      hmlp::Data<T> *X = NULL;
//      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)183831 );
//      K.readmtx<LOWERTRIANGULAR, false, IJONLY>( filename );
//      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//        ( X, K, NN, n, m, k, s, stol, nrhs );
//    }
//    {
//      std::string filename = DATADIR + std::string( "as-Skitter.mtx" );
//      n = 1696415;
//      hmlp::Data<T> *X = NULL;
//      hmlp::CSC<SYMMETRIC, T> K( n, n, (size_t)11095298 );
//      K.readmtx<LOWERTRIANGULAR, false, IJONLY>( filename );
//      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::spdaskit::SparsePattern<true, true, T>( n, k, K );
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//        ( X, K, NN, n, m, k, s, stol, nrhs );
//    }
//  }

//  if ( OOCTESTSUIT )
//  {
//    n = 4096;
//    for ( size_t id = 1; id < 14; id ++ )
//    {
//      hmlp::Data<T> *X = NULL;
//      hmlp::Data<std::pair<T, std::size_t>> NN;
//      std::ostringstream id_stream;
//      id_stream << id;
//      std::string filename = DATADIR + std::string( "K" ) + id_stream.str()
//        + std::string( ".dat" );
//      hmlp::OOC<T> K( n, n, filename );
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( X, K, NN, n, m, k, s, stol, nrhs );
//    }
//  }

  if ( KERNELTESTSUIT )
  {
    double h = 0.1;
    {
      std::string filename = DATADIR + std::string( "covtype.100k.trn.X.bin" );
      n = 100000;
      d = 54;
      hmlp::Data<T> X( d, n, filename );
      kernel_s<T> kernel;
      kernel.type = KS_GAUSSIAN;
      kernel.scal = -0.5 / ( h * h );
      hmlp::Kernel<T> K( n, n, d, kernel, X );
      hmlp::Data<std::pair<T, std::size_t>> NN;
      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
        ( &X, K, NN, n, m, k, s, stol, nrhs );
    }
//    {
//      std::string filename = DATADIR + std::string( "aloi.n108000.d128.trn.X.bin" );
//      n = 108000;
//      d = 128;
//      hmlp::Data<T> X( d, n, filename );
//      kernel_s<T> kernel;
//      kernel.type = KS_GAUSSIAN;
//      kernel.scal = -0.5 / ( h * h );
//      hmlp::Kernel<T> K( n, n, d, kernel, X );
//      hmlp::Data<std::pair<T, std::size_t>> NN;
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//        ( &X, K, NN, n, m, k, s, stol, nrhs );
//    }
//    {
//      std::string filename = DATADIR + std::string( "higgs.0.5M.28d.tst.X.bin" );
//      n = 500000;
//      d = 28;
//      hmlp::Data<T> X( d, n, filename );
//      kernel_s<T> kernel;
//      kernel.type = KS_GAUSSIAN;
//      kernel.scal = -0.5 / ( h * h );
//      hmlp::Kernel<T> K( n, n, d, kernel, X );
//      hmlp::Data<std::pair<T, std::size_t>> NN;
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//        ( &X, K, NN, n, m, k, s, stol, nrhs );
//    }
//    {
//      std::string filename = DATADIR + std::string( "mnist2m.d784.n1600000.trn.X.bin" );
//      n = 1600000;
//      d = 784;
//      hmlp::Data<T> X( d, n, filename );
//      kernel_s<T> kernel;
//      kernel.type = KS_GAUSSIAN;
//      kernel.scal = -0.5 / ( 0.3 * 0.3 );
//      hmlp::Kernel<T> K( n, n, d, kernel, X );
//      hmlp::Data<std::pair<T, std::size_t>> NN;
//      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//        ( &X, K, NN, n, m, k, s, stol, nrhs );
//    }
  }

    //{
    //  std::string filename = std::string( "K01N15838.bin" );
    //  n = 15838;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = std::string( "K02N15575.bin" );
    //  n = 15575;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}




  if ( SCALETESTSUIT )
  {
    std::string SUBDIR;
    using T = float;

    SUBDIR = DATADIR;
    //{
    //  std::string filename = SUBDIR + std::string( "K02N147456.bin" );
    //  n = 147456;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}


    /** data_to_use_kernels_without_points */
    printf( "\ndata_to_use_kernels_without_points\n\n" );
    SUBDIR = DATADIR + std::string( "data_to_use_kernels_without_points/" );
    //{
    //  std::string filename = SUBDIR + std::string( "K04N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K05N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K06N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K07N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K08N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K09N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K10N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}





    ///** data_to_use_65K */
    //printf( "\ndata_to_use_65K\n\n" );
    SUBDIR = DATADIR + std::string( "data_to_use_65K/" );
    //{
    //  std::string filename = SUBDIR + std::string( "K01N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    {
      std::string filename = SUBDIR + std::string( "K02N65536.bin" );
      n = 65536;
      hmlp::Data<T> *X = NULL;
      //hmlp::spdaskit::SPDMatrix<T> K;
      //K.resize( n, n );
      //K.read( n, n, filename );
      hmlp::OOC<T> K( n, n, filename );
      hmlp::Data<std::pair<T, std::size_t>> NN;
      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
      ( X, K, NN, n, m, k, s, stol, nrhs );
    }
    //{
    //  std::string filename = SUBDIR + std::string( "K03N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K04N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K05N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K06N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K07N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K08N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K09N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K10N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K11N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
     
    /** PDEs */
    SUBDIR = DATADIR + std::string( "data_to_use_65K/" );
    //{
    //  std::string filename = SUBDIR + std::string( "K13N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K14N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    {
      std::string filename = SUBDIR + std::string( "K15N65536.bin" );
      n = 65536;
      hmlp::Data<T> *X = NULL;
      hmlp::spdaskit::SPDMatrix<T> K;
      K.resize( n, n );
      K.read( n, n, filename );
      hmlp::Data<std::pair<T, std::size_t>> NN;
      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
      ( X, K, NN, n, m, k, s, stol, nrhs );
    }
    {
      std::string filename = SUBDIR + std::string( "K16N65536.bin" );
      n = 65536;
      hmlp::Data<T> *X = NULL;
      hmlp::spdaskit::SPDMatrix<T> K;
      K.resize( n, n );
      K.read( n, n, filename );
      hmlp::Data<std::pair<T, std::size_t>> NN;
      test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
      ( X, K, NN, n, m, k, s, stol, nrhs );
    }

    /** data_to_use_graphs */
    printf( "\ndata_to_use_graphs\n\n" );
    SUBDIR = DATADIR + std::string( "data_to_use_graphs/" );
    //{
    //  std::string filename = SUBDIR + std::string( "K19N21132.bin" );
    //  n = 21132;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K20N65536.bin" );
    //  n = 65536;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
    //{
    //  std::string filename = SUBDIR + std::string( "K21N89400.bin" );
    //  n = 89400;
    //  hmlp::Data<T> *X = NULL;
    //  hmlp::spdaskit::SPDMatrix<T> K;
    //  K.resize( n, n );
    //  K.read( n, n, filename );
    //  hmlp::Data<std::pair<T, std::size_t>> NN;
    //  test_spdaskit_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    //  ( X, K, NN, n, m, k, s, stol, nrhs );
    //}
  }


  //{
  //  hmlp::Data<T> u( nrhs, n );
  //  hmlp::Data<T> w( nrhs, n );
  //  double beg = omp_get_wtime();
  //  hmlp::xgemm
  //  (
  //    "N", "T",
  //    u.row(), u.col(), u.col(),
  //    1.0, w.data(), w.row(),
  //         K.data(), K.row(),
  //    0.0, u.data(), u.row()
  //  );
  //  double gemm_time = omp_get_wtime() - beg;
  //  printf( "exact gemm time %5.3lfs\n", gemm_time );
  //}

  hmlp_finalize();

  return 0;
};
