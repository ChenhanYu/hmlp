/** TODO: this header needs a precompiled region */
#ifdef HMLP_USE_MPI
#include <mpi.h>
#endif

#include <tuple>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <iomanip>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <limits>

#ifdef HMLP_AVX512
/** this is for hbw_malloc() and hnw_free */
#include <hbwmalloc.h>
/** we need hbw::allocator<T> to replace std::allocator<T> */
#include <hbw_allocator.h>
/** MKL headers */
#include <mkl.h>
#endif


#include <gofmm/gofmm.hpp>
/** use an implicit kernel matrix (only coordinates are stored) */
#include <containers/kernel.hpp>


#ifdef HMLP_USE_CUDA
#include <hmlp_gpu.hpp>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

/** by default, we use binary tree */
#define N_CHILDREN 2

using namespace hmlp::tree;
using namespace hmlp::gofmm;


template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  SplitScheme SPLIT,
  typename    SPLITTER, 
  typename    RKDTSPLITTER, 
  typename    T, 
  typename    SPDMATRIX>
void test_gofmm
( 
  hmlp::Data<T> *X,
  SPDMATRIX &K, 
  hmlp::Data<std::pair<T, std::size_t>> &NN,
  SPLITTER splitter, 
  RKDTSPLITTER rkdtsplitter,
  size_t n, size_t m, size_t k, size_t s, 
  double stol, double budget, size_t nrhs 
)
{
  /** instantiation for the Spd-Askit tree */
  using SETUP = hmlp::gofmm::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA  = hmlp::gofmm::Data<T>;
  using NODE  = Node<SETUP, N_CHILDREN, DATA, T>;
 
  /** all timers */
  double beg, dynamic_time, omptask45_time, omptask_time, ref_time;
  double ann_time, tree_time, overhead_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

  const bool CACHE = true;

  /** compress K */
  auto tree = Compress<ADAPTIVE, LEVELRESTRICTION, SPLIT, SPLITTER, RKDTSPLITTER, T>
  ( X, K, NN, splitter, rkdtsplitter, n, m, k, s, stol, budget );

//#endif


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

  /** Evaluate u ~ K * w */
  hmlp::Data<T> w( nrhs, n ); w.rand();
  auto u = Evaluate<true, false, true, true, CACHE>( tree, w );


#ifdef HMLP_AVX512
  mkl_set_dynamic( 1 );
  mkl_set_num_threads( omp_get_max_threads() );
#else
  //mkl_set_dynamic( 1 );
  //mkl_set_num_threads( omp_get_max_threads() );
#endif


//  /** omp level-by-level */
//  beg = omp_get_wtime();
//  if ( OMPLEVEL ) 
//  {
//    printf( "ComputeAll (Level-By-Level) ..." ); fflush( stdout );
//    u = hmlp::gofmm::ComputeAll<false, false, true, true, CACHE, NODE>( tree, w );
//    printf( "Done.\n" ); fflush( stdout );
//  }
//  ref_time = omp_get_wtime() - beg;
//  printf( "Done.\n" ); fflush( stdout );
//
//  /** omp recu task */
//  beg = omp_get_wtime();
//  omptask_time = omp_get_wtime() - beg;
//
//  /** omp recu task depend */
//  beg = omp_get_wtime();
//  if ( OMPDAGTASK )
//  {
//    u = hmlp::gofmm::ComputeAll<false, true, true, true, CACHE, NODE>( tree, w );
//  }
//  omptask45_time = omp_get_wtime() - beg;
//
//  printf( "Exact ratio %5.2lf Runtime %5.2lfs level-by-level %5.2lfs OMP task %5.2lfs OMP-4.5 %5.2lfs\n", 
//      exact_ratio, dynamic_time, ref_time, omptask_time, omptask45_time ); fflush( stdout );
//  // ------------------------------------------------------------------------


  /** examine accuracy with 3 setups, ASKIT, HODLR, and GOFMM */
  std::size_t ntest = 100;
  T nnerr_avg = 0.0;
  T nonnerr_avg = 0.0;
  T fmmerr_avg = 0.0;
  printf( "========================================================\n");
  printf( "Accuracy report\n" );
  printf( "========================================================\n");
  for ( size_t i = 0; i < ntest; i ++ )
  {
    hmlp::Data<T> potentials;
    /** ASKIT treecode with NN pruning */
    Evaluate<false, true>( tree, i, potentials );
    auto nnerr = ComputeError( tree, i, potentials );
    /** ASKIT treecode without NN pruning */
    Evaluate<false, false>( tree, i, potentials );
    auto nonnerr = ComputeError( tree, i, potentials );
    /** get results from GOFMM */
    for ( size_t p = 0; p < potentials.col(); p ++ )
    {
      potentials[ p ] = u( p, i );
    }
    auto fmmerr = ComputeError( tree, i, potentials );

    /** only print 10 values. */
    if ( i < 10 )
    {
#ifdef DUMP_ANALYSIS_DATA
      printf( "@DATA\n" );
      printf( "%5lu, %E, %E\n", i, nnerr, nonnerr );
#endif
      printf( "gid %6lu, ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
          i, nnerr, nonnerr, fmmerr );
    }
    nnerr_avg += nnerr;
    nonnerr_avg += nonnerr;
    fmmerr_avg += fmmerr;
  }
  printf( "========================================================\n");
  printf( "            ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
      nnerr_avg / ntest , nonnerr_avg / ntest, fmmerr_avg / ntest );
  printf( "========================================================\n");
  // ------------------------------------------------------------------------


  /** Factorization */
  const bool LU = true;
  T lambda = 1.0;
  if ( lambda < 10.0 * fmmerr_avg )
    printf( "Warning! lambda %lf may be too small for accuracy %3.1E\n",
        lambda, fmmerr_avg );
  hmlp::hfamily::Factorize<LU, NODE, T>( tree, lambda ); 

  /** compute error */
  hmlp::hfamily::ComputeError<LU, NODE>( tree, lambda, w, u );

  //#ifdef DUMP_ANALYSIS_DATA
  hmlp::gofmm::Summary<NODE> summary;
  tree.Summary( summary );
  summary.Print();



}; /** end test_gofmm() */














template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  SplitScheme SPLIT,
  typename    T, 
  typename    SPDMATRIX>
void test_gofmm_setup
( 
  hmlp::Data<T> *X,
  SPDMATRIX &K, 
  hmlp::Data<std::pair<T, std::size_t>> &NN,
  size_t n, size_t m, size_t k, size_t s, 
  double stol, double budget, size_t nrhs 
)
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
      test_gofmm<ADAPTIVE, LEVELRESTRICTION, SPLIT, SPLITTER, RKDTSPLITTER, T>
      ( X, K, NN, splitter, rkdtsplitter, n, m, k, s, stol, budget, nrhs );
      break;
    }
    case SPLIT_KERNEL_DISTANCE:
    case SPLIT_ANGLE:
    {
      using SPLITTER     = hmlp::gofmm::centersplit<SPDMATRIX, N_CHILDREN, T, SPLIT>;
      using RKDTSPLITTER = hmlp::gofmm::randomsplit<SPDMATRIX, N_CHILDREN, T, SPLIT>;
      SPLITTER splitter;
      splitter.Kptr = &K;
      RKDTSPLITTER rkdtsplitter;
      rkdtsplitter.Kptr = &K;
      test_gofmm<ADAPTIVE, LEVELRESTRICTION, SPLIT, SPLITTER, RKDTSPLITTER, T>
      ( X, K, NN, splitter, rkdtsplitter, n, m, k, s, stol, budget, nrhs );
      break;
    }
    default:
    {
      exit( 1 );
    }
  }
}; /** end test_gofmm_setup() */



/**
 *
 */ 
int main( int argc, char *argv[] )
{
  /** default adaptive scheme */
  const bool ADAPTIVE = true;
  const bool LEVELRESTRICTION = false;

  /** default geometric-oblivious scheme */
  const SplitScheme SPLIT = SPLIT_ANGLE;
  // const SplitScheme SPLIT = SPLIT_POINT_DISTANCE;
  // const SplitScheme SPLIT = SPLIT_KERNEL_DISTANCE;

  /** test suit options */
  const bool SIMPLE = true;
  const bool RANDOMMATRIX = false;
  const bool USE_LOWRANK = true;
  const bool DENSETESTSUIT = false;
  const bool SPARSETESTSUIT = false;
  const bool KERNELTESTSUIT = false;

  /** default data directory */
  std::string DATADIR( "/" );

  /** default precision */
  using T = double;

  /** read all parameters */
  size_t n, m, d, k, s, nrhs;
  double stol, budget;
  size_t nnz; /** optional */
  std::string user_matrix_filename;
  std::string user_points_filename;

  /** number of columns and rows, i.e. problem size */
  sscanf( argv[ 1 ], "%lu", &n );

  /** on-diagonal block size, such that the tree has log(n/m) levels */
  sscanf( argv[ 2 ], "%lu", &m );

  /** number of neighbors to use */
  sscanf( argv[ 3 ], "%lu", &k );

  /** maximum off-diagonal ranks */
  sscanf( argv[ 4 ], "%lu", &s );

  /** number of right hand sides */
  sscanf( argv[ 5 ], "%lu", &nrhs );

  /** desired approximation accuracy */
  sscanf( argv[ 6 ], "%lf", &stol );

  /** the maximum percentage of direct matrix-multiplication */
  sscanf( argv[ 7 ], "%lf", &budget );

  /** optional provide the path to the matrix file */
  if ( argc > 8 ) user_matrix_filename = argv[ 8 ];
    

  /** optional provide the path to the data file */
  if ( argc > 9 ) 
  {
    user_points_filename = argv[ 9 ];
    sscanf( argv[ 10 ], "%lu", &d );
  }

#ifdef HMLP_USE_MPI
  /** Message Passing Interface */
  int size = -1, rank = -1;
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  printf( "size %d rank %d\n", size, rank );
#endif

  /** HMLP API call to initialize the runtime */
  hmlp_init();


  /** run the matrix file provided by users */
  //if ( user_matrix_filename.size() )
  //{
  //  using T = float;
  //  {
  //    /** dense spd matrix format */
  //    hmlp::gofmm::SPDMatrix<T> K;
  //    K.resize( n, n );
  //    K.read( n, n, user_matrix_filename );

  //    //for ( size_t i = 0; i < n; i ++ ) K( i, i ) += 10.0;

  //    /** (optional) provide neighbors, leave uninitialized otherwise */
  //    hmlp::Data<std::pair<T, std::size_t>> NN;
  //    if ( user_points_filename.size() )
  //    {
  //      hmlp::Data<T> X( d, n, user_points_filename );
  //      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT_POINT_DISTANCE, T>
  //      ( &X, K, NN, n, m, k, s, stol, budget, nrhs );
  //    }
  //    else
  //    {
  //      hmlp::Data<T> *X = NULL;
  //      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
  //      ( X, K, NN, n, m, k, s, stol, budget, nrhs );
  //    }
  //  }
  //}

  /** test simple interface */
  if ( SIMPLE )
  {
    hmlp::gofmm::SPDMatrix<T> K;
    K.resize( n, n );
    K.randspd<USE_LOWRANK>( 0.0, 1.0 );
    auto tree = hmlp::gofmm::Compress<T>( K, stol, budget );
    size_t nrhs = 1;
    hmlp::Data<T> w( nrhs, n ); w.rand();
    auto u = hmlp::gofmm::Evaluate( tree, w );
    size_t ntest = 10;
    for ( size_t i = 0; i < ntest; i ++ )
    {
      hmlp::Data<T> potentials( 1, nrhs );
      for ( size_t p = 0; p < potentials.col(); p ++ )
        potentials[ p ] = u( p, i );
      auto fmmerr = ComputeError( tree, i, potentials );
      printf( "fmmerr %3.1E\n", fmmerr );
    }
  }


  /** create a random spd matrix, which is diagonal-dominant */
  if ( RANDOMMATRIX )
  {
    /** no geometric coordinates provided */
    hmlp::Data<T> *X = NULL;
    /** dense spd matrix format */
    hmlp::gofmm::SPDMatrix<T> K;
    K.resize( n, n );
    /** random spd initialization */
    K.randspd<USE_LOWRANK>( 0.0, 1.0 );
    /** (optional) provide neighbors, leave uninitialized otherwise */
    hmlp::Data<std::pair<T, std::size_t>> NN;
    /** routine */
    test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
    ( X, K, NN, n, m, k, s, stol, budget, nrhs );
  }

  /** generate a Gaussian kernel matrix from the coordinates */
//  if ( KERNELTESTSUIT )
//  {
//    /** set the Gaussian kernel bandwidth */
//    T h = 0.1;
//    {
//      /** filename, number of points, dimension */
//      std::string filename = DATADIR + std::string( "covtype.100k.trn.X.bin" );
//      n = 100000;
//      d = 54;
//      /** read the coordinates from the file */
//      hmlp::Data<T> X( d, n, filename );
//      /** setup the kernel object as Gaussian */
//      kernel_s<T> kernel;
//      kernel.type = KS_GAUSSIAN;
//      kernel.scal = -0.5 / ( h * h );
//      /** spd kernel matrix format (implicitly create) */
//      hmlp::Kernel<T> K( n, n, d, kernel, X );
//      /** (optional) provide neighbors, leave uninitialized otherwise */
//      hmlp::Data<std::pair<T, std::size_t>> NN;
//      /** routine */
//      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( &X, K, NN, n, m, k, s, stol, budget, nrhs );
//    }
//  }

//  /** generate (read) a CSC sparse matrix */
//  if ( SPARSETESTSUIT )
//  {
//    const bool SYMMETRIC = false;
//    const bool LOWERTRIANGULAR = true;
//    {
//      /** no geometric coordinates provided */
//      hmlp::Data<T> *X = NULL;
//      /** filename, problem size, nnz */
//      std::string filename = DATADIR + std::string( "bcsstk10.mtx" );
//      n = 1086;
//      nnz = 11578;
//      /** CSC format */
//      hmlp::CSC<SYMMETRIC, T> K( n, n, nnz );
//      K.readmtx<LOWERTRIANGULAR, false>( filename );
//      /** use non-zero pattern as neighbors */
//      hmlp::Data<std::pair<T, std::size_t>> NN = hmlp::gofmm::SparsePattern<true, true, T>( n, k, K );
//      /** routine */
//      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( X, K, NN, n, m, k, s, stol, budget, nrhs );
//    }
//  }
//
//  /** generate (read) dense spd matrix */
//  if ( DENSETESTSUIT )
//  {
//    using T = float;
//    {
//      /** no geometric coordinates provided */
//      hmlp::Data<T> *X = NULL;
//      /** filename, problem size */
//      std::string filename = DATADIR + std::string( "K04N65536.bin" );
//      n = 65536;
//      /** dense spd matrix format */
//      hmlp::gofmm::SPDMatrix<T> K;
//      K.resize( n, n );
//      K.read( n, n, filename );
//      /** (optional) provide neighbors, leave uninitialized otherwise */
//      hmlp::Data<std::pair<T, std::size_t>> NN;
//      /** routine */
//      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, SPLIT, T>
//      ( X, K, NN, n, m, k, s, stol, budget, nrhs );
//    }
//  }


  /** HMLP API call to terminate the runtime */
  hmlp_finalize();

#ifdef HMLP_USE_MPI
  /** Message Passing Interface */
  MPI_Finalize();
#endif

  return 0;

}; /** end main() */
