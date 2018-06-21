/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  



/** MPI support */
#include <hmlp_mpi.hpp>

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


/** Use GOFMM templates. */
#include <gofmm/gofmm.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use implicit matrices. */
#include <containers/VirtualMatrix.hpp>
/** Use implicit Gauss-Newton (multilevel perceptron) matrices. */
#include <containers/MLPGaussNewton.hpp>
/** Use OOC covariance matrices. */
#include <containers/OOCCovMatrix.hpp>


#ifdef HMLP_USE_CUDA
#include <hmlp_gpu.hpp>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

/** by default, we use binary tree */
#define N_CHILDREN 2

using namespace std;
//using namespace hmlp::tree;
//using namespace hmlp::gofmm;
using namespace hmlp;



template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  typename    SPLITTER, 
  typename    RKDTSPLITTER, 
  typename    T, 
  typename    SPDMATRIX>
void test_gofmm
( 
  Data<T> *X,
  SPDMATRIX &K, 
  Data<pair<T, size_t>> &NN,
  DistanceMetric metric,
  SPLITTER splitter, 
  RKDTSPLITTER rkdtsplitter,
  size_t n, size_t m, size_t k, size_t s, 
  double stol, double budget, size_t nrhs 
)
{
  /** instantiation for the Spd-Askit tree */
  using SETUP = gofmm::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA  = gofmm::NodeData<T>;
  using NODE  = tree::Node<SETUP, DATA>;
 
  /** all timers */
  double beg, dynamic_time, omptask45_time, omptask_time, ref_time;
  double ann_time, tree_time, overhead_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

  const bool CACHE = true;

	/** creatgin configuration for all user-define arguments */
  gofmm::Configuration<T> config( metric, n, m, k, s, stol, budget );

  /** compress K */
  auto *tree_ptr = gofmm::Compress<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>
  ( X, K, NN, //metric, 
		splitter, rkdtsplitter, //n, m, k, s, stol, budget, 
	  config );
	auto &tree = *tree_ptr;


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
  Data<T> w( n, nrhs ); w.rand();
  auto u = gofmm::Evaluate<true, false, true, CACHE>( tree, w );


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
  size_t ntest = 100;
  T nnerr_avg = 0.0;
  T nonnerr_avg = 0.0;
  T fmmerr_avg = 0.0;
  printf( "========================================================\n");
  printf( "Accuracy report\n" );
  printf( "========================================================\n");
  for ( size_t i = 0; i < ntest; i ++ )
  {
    Data<T> potentials;
    /** ASKIT treecode with NN pruning */
    gofmm::Evaluate<false, true>( tree, i, potentials );
    auto nnerr = gofmm::ComputeError( tree, i, potentials );
    /** ASKIT treecode without NN pruning */
    gofmm::Evaluate<false, false>( tree, i, potentials );
    auto nonnerr = gofmm::ComputeError( tree, i, potentials );
    /** get results from GOFMM */
    for ( size_t p = 0; p < potentials.col(); p ++ )
    {
      potentials[ p ] = u( i, p );
    }
    auto fmmerr = gofmm::ComputeError( tree, i, potentials );

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
  T lambda = 5.0;
  if ( lambda < 10.0 * ( fmmerr_avg / ntest ) )
    printf( "Warning! lambda %lf may be too small for accuracy %3.1E\n",
        lambda, fmmerr_avg / ntest );
  gofmm::Factorize( tree, lambda ); 

  /** compute error */
  gofmm::ComputeError( tree, lambda, w, u );

  //#ifdef DUMP_ANALYSIS_DATA
  gofmm::Summary<NODE> summary;
  tree.Summary( summary );
  summary.Print();

	/** delete tree_ptr */
  delete tree_ptr;

}; /** end test_gofmm() */











/**
 *  @brief Instantiate the splitters here.
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename T, typename SPDMATRIX>
void test_gofmm_setup
( 
  Data<T> *X,
  SPDMATRIX &K, 
  Data<pair<T, size_t>> &NN,
  DistanceMetric metric,
  size_t n, size_t m, size_t k, size_t s, 
  double stol, double budget, size_t nrhs
)
{
  switch ( metric )
  {
    case GEOMETRY_DISTANCE:
    {
      assert( X );
			/** using geometric splitters from hmlp::tree */
      using SPLITTER     = tree::centersplit<N_CHILDREN, T>;
      using RKDTSPLITTER = tree::randomsplit<N_CHILDREN, T>;
			/** GOFMM tree splitter */
      SPLITTER splitter;
      splitter.Coordinate = X;
			/** randomized tree splitter */
      RKDTSPLITTER rkdtsplitter;
      rkdtsplitter.Coordinate = X;
      test_gofmm<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>
      ( X, K, NN, metric, splitter, rkdtsplitter, n, m, k, s, stol, budget, nrhs );
      break;
    }
    case KERNEL_DISTANCE:
    case ANGLE_DISTANCE:
    {
			/** using geometric-oblivious splitters from hmlp::gofmm */
      using SPLITTER     = gofmm::centersplit<SPDMATRIX, N_CHILDREN, T>;
      using RKDTSPLITTER = gofmm::randomsplit<SPDMATRIX, N_CHILDREN, T>;
			/** GOFMM tree splitter */
      SPLITTER splitter;
      splitter.Kptr = &K;
			splitter.metric = metric;
			/** randomized tree splitter */
      RKDTSPLITTER rkdtsplitter;
      rkdtsplitter.Kptr = &K;
			rkdtsplitter.metric = metric;
      test_gofmm<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>
      ( X, K, NN, metric, splitter, rkdtsplitter, n, m, k, s, stol, budget, nrhs );
      break;
    }
    default:
    {
      exit( 1 );
    }
  }
}; /** end test_gofmm_setup() */



/**
 *  @brief Top level driver that reads arguments from the command line.
 */ 
int main( int argc, char *argv[] )
{
  /** default adaptive scheme */
  const bool ADAPTIVE = true;
  const bool LEVELRESTRICTION = false;

  /** default geometric-oblivious scheme */
  DistanceMetric metric = ANGLE_DISTANCE;

  /** test suit options */
  const bool SIMPLE = true;
  const bool RANDOMMATRIX = true;
  const bool USE_LOWRANK = true;
  const bool SPARSETESTSUIT = false;

  /** default data directory */
  std::string DATADIR( "/" );

  /** default precision */
  using T = double;

  /** read all parameters */
  size_t n, m, d, k, s, nrhs;
  double stol, budget;

  /** (Optional) */
  size_t nnz, nb; 
  string distance_type;
  string spdmatrix_type;
  string kernelmatrix_type;
  string hidden_layers;
  string user_matrix_filename;
  string user_points_filename;

  /** (Optional) set the default Gaussian kernel bandwidth */
  float h = 1.0;

  /** Number of columns and rows, i.e. problem size */
  sscanf( argv[ 1 ], "%lu", &n );

  /** On-diagonal block size, such that the tree has log(n/m) levels */
  sscanf( argv[ 2 ], "%lu", &m );

  /** Number of neighbors to use */
  sscanf( argv[ 3 ], "%lu", &k );

  /** Maximum off-diagonal ranks */
  sscanf( argv[ 4 ], "%lu", &s );

  /** Number of right hand sides */
  sscanf( argv[ 5 ], "%lu", &nrhs );

  /** Desired approximation accuracy */
  sscanf( argv[ 6 ], "%lf", &stol );

  /** The maximum percentage of direct matrix-multiplication */
  sscanf( argv[ 7 ], "%lf", &budget );

  /** Specify distance type */
  distance_type = argv[ 8 ];

  if ( !distance_type.compare( "geometry" ) )
  {
    metric = GEOMETRY_DISTANCE;
  }
  else if ( !distance_type.compare( "kernel" ) )
  {
    metric = KERNEL_DISTANCE;
  }
  else if ( !distance_type.compare( "angle" ) )
  {
    metric = ANGLE_DISTANCE;
  }
  else
  {
    printf( "%s is not supported\n", argv[ 9 ] );
    exit( 1 );
  }

  /** Specify what kind of spdmatrix is used */
  spdmatrix_type = argv[ 9 ];

  if ( !spdmatrix_type.compare( "testsuit" ) )
  {
    /** NOP */
  }
  else if ( !spdmatrix_type.compare( "userdefine" ) )
  {
    /** NOP */
  }
  else if ( !spdmatrix_type.compare( "pvfmm" ) )
  {
    /** NOP */
  }
  else if ( !spdmatrix_type.compare( "dense" ) || !spdmatrix_type.compare( "ooc" ) )
  {
    /** (Optional) provide the path to the matrix file */
    user_matrix_filename = argv[ 10 ];
    if ( argc > 11 ) 
    {
      /** (Optional) provide the path to the data file */
      user_points_filename = argv[ 11 ];
      /** Dimension of the data set */
      sscanf( argv[ 12 ], "%lu", &d );
    }
  }
  else if ( !spdmatrix_type.compare( "mlp" ) )
  {
    hidden_layers = argv[ 10 ];
    user_points_filename = argv[ 11 ];
    /** Number of attributes (dimensions) */
    sscanf( argv[ 12 ], "%lu", &d );
  }
  else if ( !spdmatrix_type.compare( "cov" ) )
  {
    kernelmatrix_type = argv[ 10 ];
    user_points_filename = argv[ 11 ];
    /** Number of attributes (dimensions) */
    sscanf( argv[ 12 ], "%lu", &d );
    /** Block size (in dimensions) per file */
    sscanf( argv[ 13 ], "%lu", &nb );
  }
  else if ( !spdmatrix_type.compare( "kernel" ) )
  {
    kernelmatrix_type = argv[ 10 ];
    user_points_filename = argv[ 11 ];
    /** Number of attributes (dimensions) */
    sscanf( argv[ 12 ], "%lu", &d );
    /** (Optional) provide Gaussian kernel bandwidth */
    if ( argc > 13 ) sscanf( argv[ 13 ], "%f", &h );
  }
  else
  {
    printf( "%s is not supported\n", argv[ 9 ] );
    exit( 1 );
  }










  /** Message Passing Interface */
  int size = -1, rank = -1;
  mpi::Init( &argc, &argv );
  mpi::Comm_size( MPI_COMM_WORLD, &size );
  mpi::Comm_rank( MPI_COMM_WORLD, &rank );
  printf( "size %d rank %d\n", size, rank );

  /** HMLP API call to initialize the runtime */
  hmlp_init();

  /** run the matrix file provided by users */
  if ( !spdmatrix_type.compare( "dense" ) && user_matrix_filename.size() )
  {
    using T = float;
    {
      /** dense spd matrix format */
      SPDMatrix<T> K;
      K.resize( n, n );
      K.read( n, n, user_matrix_filename );

      /** (optional) provide neighbors, leave uninitialized otherwise */
      Data<pair<T, size_t>> NN;

			/** (optional) provide coordinates */
      if ( user_points_filename.size() )
      {
        Data<T> X( d, n, user_points_filename );
        test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( &X, K, NN, metric, n, m, k, s, stol, budget, nrhs );
      }
      else
      {
        Data<T> *X = NULL;
        test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs );
      }
    }
  }


  /** generate a Gaussian kernel matrix from the coordinates */
  if ( !spdmatrix_type.compare( "kernel" ) && user_points_filename.size() )
  {
    using T = double;
    {
      /** Read the coordinates from the file. */
      Data<T> X( d, n, user_points_filename );

      /** Set the kernel object as Gaussian. */
      kernel_s<T> kernel;
      kernel.type = KS_GAUSSIAN;
      kernel.scal = -0.5 / ( h * h );

      /** SPD kernel matrix format (implicitly create) */
      KernelMatrix<T> K( n, n, d, kernel, X );

      /** (Optional) provide neighbors, leave uninitialized otherwise */
      Data<pair<T, size_t>> NN;

      /** routine */
      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
      ( &X, K, NN, metric, n, m, k, s, stol, budget, nrhs );
    }
  }

  /** test simple interface */
//	if ( !spdmatrix_type.compare( "testsuit" ) && SIMPLE )
//  {
//		n = 5000;
//    size_t nrhs = 1;
//
//    /** dense spd matrix format */
//    hmlp::gofmm::SPDMatrix<T> K;
//    K.resize( n, n );
//    K.randspd<USE_LOWRANK>( 0.0, 1.0 );
//
//		/** */
//    auto *tree_ptr = hmlp::gofmm::Compress<T>( K, stol, budget );
//		auto &tree = *tree_ptr;
//
//    hmlp::Data<T> w( nrhs, n ); w.rand();
//    auto u = hmlp::gofmm::Evaluate( tree, w );
//    size_t ntest = 10;
//    for ( size_t i = 0; i < ntest; i ++ )
//    {
//      hmlp::Data<T> potentials( 1, nrhs );
//      for ( size_t p = 0; p < potentials.col(); p ++ )
//        potentials[ p ] = u( p, i );
//      auto fmmerr = ComputeError( tree, i, potentials );
//      printf( "fmmerr %3.1E\n", fmmerr );
//    }
//		/** delete tree_ptr */
//		delete tree_ptr;
//  }


  /** create a random spd matrix, which is diagonal-dominant */
  if ( !spdmatrix_type.compare( "testsuit" ) && RANDOMMATRIX )
  {
    using T = double;
    {
      /** no geometric coordinates provided */
      Data<T> *X = NULL;
      /** dense spd matrix format */
      SPDMatrix<T> K;
      K.resize( n, n );
      /** random spd initialization */
      K.randspd<USE_LOWRANK>( 0.0, 1.0 );
      /** (optional) provide neighbors, leave uninitialized otherwise */
      Data<std::pair<T, std::size_t>> NN;
      /** routine */
      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs );
    }
    {
      d = 4;
      /** generate coordinates from normal(0,1) distribution */
      Data<T> X( d, n ); X.randn( 0.0, 1.0 );
      /** setup the kernel object as Gaussian */
      kernel_s<T> kernel;
      kernel.type = KS_GAUSSIAN;
      kernel.scal = -0.5 / ( h * h );
      /** spd kernel matrix format (implicitly create) */
      KernelMatrix<T> K( n, n, d, kernel, X );
      /** (optional) provide neighbors, leave uninitialized otherwise */
      Data<pair<T, size_t>> NN;
      /** routine */
      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( &X, K, NN, metric, n, m, k, s, stol, budget, nrhs );
    }
  }


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
//      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, metric, T>
//      ( X, K, NN, n, m, k, s, stol, budget, nrhs );
//    }
//  }




  if ( !spdmatrix_type.compare( "mlp" ) )
  {
    using T = double;
    {
      /** Read the coordinates from the file. */
      Data<T> X( d, n, user_points_filename );
      /** Multilevel perceptron Gauss-Newton */
      MLPGaussNewton<T> K;
      /** Create an input layer */
      Layer<INPUT, T> layer_input( d, n );
      /** Create FC layers */
      Layer<FC, T> layer_fc0( 256, n, layer_input );
      Layer<FC, T> layer_fc1( 256, n, layer_fc0 );
      Layer<FC, T> layer_fc2( 256, n, layer_fc1 );
      /** Insert layers into  */
      K.AppendInputLayer( layer_input );
      K.AppendFCLayer( layer_fc0 );
      K.AppendFCLayer( layer_fc1 );
      K.AppendFCLayer( layer_fc2 );
      /** Feed forward and compute all products */
      K.Update( X );
    }
  }


  if ( !spdmatrix_type.compare( "cov" ) )
  {
    using T = float;
    {
      /** No geometric coordinates provided */
      Data<T> *X = NULL;
      OOCCovMatrix<T> K( n, d, nb, user_points_filename );
      /** (optional) provide neighbors, leave uninitialized otherwise */
      hmlp::Data<std::pair<T, std::size_t>> NN;
      /** Routine */
      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs );
    }
  }



  /** HMLP API call to terminate the runtime */
  hmlp_finalize();

  /** Message Passing Interface */
  mpi::Finalize();

  return 0;

}; /** end main() */
