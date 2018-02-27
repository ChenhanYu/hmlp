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


/** GOFMM templates */
#include <mpi/gofmm_mpi.hpp>
/** use an implicit kernel matrix (only coordinates are stored) */
#include <containers/KernelMatrix.hpp>
/** use an implicit matrix */
#include <containers/VirtualMatrix.hpp>
/** use an implicit PVFMM kernel matrix */
#include <containers/PVFMMKernelMatrix.hpp>


#include <mpi/DistData.hpp>
#include <containers/DistKernelMatrix.hpp>
#include <containers/DistKernelMatrix_ver2.hpp>
#include <containers/DistSPDMatrix.hpp>






#ifdef HMLP_USE_CUDA
#include <hmlp_gpu.hpp>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

/** by default, we use binary tree */
#define N_CHILDREN 2


using namespace std;
using namespace hmlp;
//using namespace hmlp::tree;
//using namespace hmlp::gofmm;


template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  typename    SPLITTER, 
  typename    RKDTSPLITTER, 
  typename    T, 
  typename    SPDMATRIX>
void test_gofmm
( 
  DistData<STAR, CBLK, T> *X,
  SPDMATRIX &K, 
  DistData<STAR, CBLK, pair<T, std::size_t>> &NN,
  DistanceMetric metric,
  SPLITTER splitter, 
  RKDTSPLITTER rkdtsplitter,
  size_t n, size_t m, size_t k, size_t s, 
  double stol, double budget, size_t nrhs, 
  mpi::Comm CommGOFMM
)
{
  /** MPI: Message Passing Interface */
  int size, rank;
  mpi::Comm_size( CommGOFMM, &size );
  mpi::Comm_rank( CommGOFMM, &rank );

  /** all timers */
  double beg, dynamic_time, omptask45_time, omptask_time, ref_time;
  double ann_time, tree_time, overhead_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

  const bool CACHE = true;

	/** creatgin configuration for all user-define arguments */
  gofmm::Configuration<T> config( metric, n, m, k, s, stol, budget );

  /** compress K */
  auto *tree_ptr = mpigofmm::Compress<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>
  ( 
   X, K, NN, //metric, 
   splitter, rkdtsplitter, //n, m, k, s, stol, budget, 
   config );
  auto &tree = *tree_ptr;

  /** redistribute from RBLK to RIDS */
  //DistData<RBLK, STAR, T> w_rblk( n, nrhs, MPI_COMM_WORLD );
  DistData<RIDS, STAR, T> w_rids( n, nrhs, tree.treelist[ 0 ]->gids, CommGOFMM );
  
  /** Initialization */
  w_rids.rand();
  //auto &gids = tree.treelist[ 0 ]->gids;
  //for ( int j = 0; j < nrhs; j ++ )
  //  for ( int i = 0; i < gids.size(); i ++ )
  //    w_rids( gids[ i ], j ) = gids[ i ];

  //printf( "w( %lu, 0 ) = %E\n", gids[ 0 ], w_rids( gids[ 0 ], 0 ) ); fflush( stdout );
  //printf( "w( %lu, 0 ) = %E\n", gids[ 1 ], w_rids( gids[ 1 ], 0 ) ); fflush( stdout );



  /** Evaluate u ~ K * w */
  auto u_rids = mpigofmm::Evaluate<true, false, true, true, CACHE>( tree, w_rids, CommGOFMM );


  if ( u_rids.HasIllegalValue() )
  {
    printf( "Illegal value after Evaluate\n" ); fflush( stdout );
  }

  /** redistribution */
  DistData<RBLK, STAR, T> u_rblk( n, nrhs, CommGOFMM );
  u_rblk = u_rids;


  if ( u_rblk.HasIllegalValue() )
  {
    printf( "Illegal value after alltoall\n" ); fflush( stdout );
  }



  /** examine accuracy with 3 setups, ASKIT, HODLR, and GOFMM */
  std::size_t ntest = 100;
  T nnerr_avg = 0.0;
  T nonnerr_avg = 0.0;
  T fmmerr_avg = 0.0;

  if ( rank == 0 )
  {
    printf( "========================================================\n");
    printf( "Accuracy report\n" );
    printf( "========================================================\n");
  }
  for ( size_t i = 0; i < ntest; i ++ )
  {
    //size_t gid = tree.treelist[ 0 ]->gids[ i ];

    hmlp::Data<T> potentials( (size_t)1, nrhs );
 
    if ( rank == ( i % size ) )
    {
      for ( size_t p = 0; p < potentials.col(); p ++ )
      {
        potentials[ p ] = u_rblk( i , p );
      }
    }

    /** bcast potentials to all MPI processes */
    mpi::Bcast( potentials.data(), nrhs, i % size, CommGOFMM );

    if ( potentials.HasIllegalValue() )
    {
      printf( "potential %lu has illegal value %E\n", i, potentials[ 0 ] ); fflush( stdout );
    }


    /** ASKIT treecode with NN pruning */
    //Evaluate<false, true>( tree, i, potentials );
    //auto nnerr = ComputeError( tree, i, potentials );
    T nnerr = 0.0;
    /** ASKIT treecode without NN pruning */
    //Evaluate<false, false>( tree, i, potentials );
    //auto nonnerr = ComputeError( tree, i, potentials );
    T nonnerr = 0.0;

    //auto fmmerr = ComputeError( tree, i, potentials );
    auto fmmerr = mpigofmm::ComputeError( tree, i, potentials, CommGOFMM );

    /** only print 10 values. */
    if ( i < 10 && rank == 0 )
    {
      printf( "gid %6lu, ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
          i, nnerr, nonnerr, fmmerr );
    }
    nnerr_avg += nnerr;
    nonnerr_avg += nonnerr;
    fmmerr_avg += fmmerr;
  }
  if ( rank == 0 )
  {
    printf( "========================================================\n");
    printf( "            ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
        nnerr_avg / ntest , nonnerr_avg / ntest, fmmerr_avg / ntest );
    printf( "========================================================\n");
  }
  // ------------------------------------------------------------------------


//  /** Factorization */
//  const bool do_ulv_factorization = true;
//  T lambda = 10.0;
//  if ( lambda < 10.0 * ( fmmerr_avg / ntest ) )
//    printf( "Warning! lambda %lf may be too small for accuracy %3.1E\n",
//        lambda, fmmerr_avg / ntest );
//  hmlp::hfamily::Factorize<NODE, T>( do_ulv_factorization, tree, lambda ); 
//
//  /** compute error */
//  hmlp::hfamily::ComputeError<NODE>( tree, lambda, w, u );
//
//  //#ifdef DUMP_ANALYSIS_DATA
//  hmlp::gofmm::Summary<NODE> summary;
//  tree.Summary( summary );
//  summary.Print();

	/** delete tree_ptr */
  delete tree_ptr;

}; /** end test_gofmm() */











/**
 *  @brief Instantiate the splitters here.
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename T, typename SPDMATRIX>
void test_gofmm_setup
( 
  DistData<STAR, CBLK, T> *X,
  SPDMATRIX &K, 
  DistData<STAR, CBLK, pair<T, size_t>> &NN,
  DistanceMetric metric,
  size_t n, size_t m, size_t k, size_t s, 
  double stol, double budget, size_t nrhs,
  mpi::Comm CommGOFMM
)
{
  switch ( metric )
  {
    case GEOMETRY_DISTANCE:
    //{
    //  assert( X );
		//	/** using geometric splitters from hmlp::tree */
    //  using SPLITTER     = mpitree::centersplit<N_CHILDREN, T>;
    //  using RKDTSPLITTER = mpitree::randomsplit<N_CHILDREN, T>;
		//	/** GOFMM tree splitter */
    //  SPLITTER splitter;
    //  splitter.Coordinate = X;
		//	/** randomized tree splitter */
    //  RKDTSPLITTER rkdtsplitter;
    //  rkdtsplitter.Coordinate = X;
    //  test_gofmm<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>
    //  ( X, K, NN, metric, splitter, rkdtsplitter, n, m, k, s, stol, budget, nrhs, CommGOFMM );
    //  break;
    //}
    case KERNEL_DISTANCE:
    case ANGLE_DISTANCE:
    {
			/** using geometric-oblivious splitters from hmlp::gofmm */
      using SPLITTER     = mpigofmm::centersplit<SPDMATRIX, N_CHILDREN, T>;
      using RKDTSPLITTER = mpigofmm::randomsplit<SPDMATRIX, N_CHILDREN, T>;
			/** GOFMM tree splitter */
      SPLITTER splitter;
      splitter.Kptr = &K;
			splitter.metric = metric;
			/** randomized tree splitter */
      RKDTSPLITTER rkdtsplitter;
      rkdtsplitter.Kptr = &K;
			rkdtsplitter.metric = metric;
      test_gofmm<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER, T>
      ( X, K, NN, metric, splitter, rkdtsplitter, n, m, k, s, stol, budget, nrhs, CommGOFMM );
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
  string DATADIR( "/" );

  /** default precision */
  using T = double;

  /** read all parameters */
  size_t n, m, d, k, s, nrhs;
  double stol, budget;

  /** (optional) */
  size_t nnz; 
  string distance_type;
  string spdmatrix_type;
  string kernelmatrix_type;
  string user_matrix_filename;
  string user_points_filename;

  /** (optional) set the default Gaussian kernel bandwidth */
  float h = 1.0;

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

  /** specify distance type */
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


  /** specify what kind of spdmatrix is used */
  spdmatrix_type = argv[ 9 ];

  if ( !spdmatrix_type.compare( "testsuit" ) )
  {
    /** do nothing */
  }
  else if ( !spdmatrix_type.compare( "userdefine" ) )
  {
    /** do nothing */
  }
  else if ( !spdmatrix_type.compare( "pvfmm" ) )
  {
    /** do nothing */
  }
  else if ( !spdmatrix_type.compare( "dense" ) )
  {
    /** (optional) provide the path to the matrix file */
    user_matrix_filename = argv[ 10 ];
    if ( argc > 11 ) 
    {
      /** (optional) provide the path to the data file */
      user_points_filename = argv[ 11 ];
      /** dimension of the data set */
      sscanf( argv[ 12 ], "%lu", &d );
    }
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



  /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support */
  int size, rank, provided;

	mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
	if ( provided != MPI_THREAD_MULTIPLE )
	{
		printf( "MPI_THREAD_MULTIPLE is not supported\n" ); fflush( stdout );
    exit( 1 );
	}

  /** MPI (Message Passing Interface): create a specific comm for GOFMM */
  mpi::Comm CommGOFMM;
  mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );
  mpi::Comm_size( CommGOFMM, &size );
  mpi::Comm_rank( CommGOFMM, &rank );
  printf( "size %d rank %d\n", size, rank );



  /** HMLP API call to initialize the runtime */
  hmlp_init();

  /** run the matrix file provided by users */
  if ( !spdmatrix_type.compare( "dense" ) && user_matrix_filename.size() )
  {
    using T = float;
    {
      /** dense spd matrix format */
      gofmm::SPDMatrix<T> K;
      K.resize( n, n );
      K.read( n, n, user_matrix_filename );

      /** (optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );

			/** (optional) provide coordinates */
      if ( user_points_filename.size() )
      {
        DistData<STAR, CBLK, T> X( d, n, CommGOFMM,  user_points_filename );
        test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( &X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
      }
      else
      {
        DistData<STAR, CBLK, T> *X = NULL;
        test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
      }
    }
  }


  /** Generate a kernel matrix from the coordinates */
  if ( !spdmatrix_type.compare( "kernel" ) && user_points_filename.size() )
  {
    using T = float;
    {
      /** read the coordinates from the file */
      DistData<STAR, CBLK, T> X( d, n, CommGOFMM, user_points_filename );

      /** setup the kernel object as Gaussian */
      kernel_s<T> kernel;
      kernel.type = KS_GAUSSIAN;
      if ( !kernelmatrix_type.compare( "gaussian" ) ) kernel.type = KS_GAUSSIAN;
      if ( !kernelmatrix_type.compare( "laplace" ) )  kernel.type = KS_LAPLACE;
      kernel.scal = -0.5 / ( h * h );

      /** Distributed spd kernel matrix format (implicitly create) */
      //DistKernelMatrix<T> K( n, n, d, kernel, X, CommGOFMM );
		  DistKernelMatrix_ver2<T, T> K( n, d, kernel, X, CommGOFMM );




      /** shared spd kernel matrix format (implicitly create) */
			//hmlp::KernelMatrix<T> K( n, n, d, kernel, Xtmp );

      /** (optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );

      /** routine */
      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( &X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
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
			DistData<STAR, CBLK, T> *X = NULL;
			/** dense spd matrix format */
			gofmm::SPDMatrix<T> K;
			K.resize( n, n );
			/** random spd initialization */
			K.randspd<USE_LOWRANK>( 0.0, 1.0 );
			/** broadcast K to all other rank */
			mpi::Bcast( K.data(), n * n, 0, CommGOFMM );
      /** (optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );
      /** routine */
      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
		}


    //{
    //  d = 4;
    //	/** generate coordinates from normal(0,1) distribution */
    //	hmlp::Data<T> X( d, n ); X.randn( 0.0, 1.0 );
    //  /** setup the kernel object as Gaussian */
    //  kernel_s<T> kernel;
    //  kernel.type = KS_GAUSSIAN;
    //  kernel.scal = -0.5 / ( h * h );
    //  /** spd kernel matrix format (implicitly create) */
    //  hmlp::KernelMatrix<T> K( n, n, d, kernel, X );
    //	/** (optional) provide neighbors, leave uninitialized otherwise */
    //	hmlp::Data<std::pair<T, std::size_t>> NN;
    //	/** routine */
    //  test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
    //  ( &X, K, NN, metric, n, m, k, s, stol, budget, nrhs );
		//}
  }



  /** create a random spd matrix, which is diagonal-dominant */
  if ( !spdmatrix_type.compare( "pvfmm" ) )
  {
    using T = double;
    {
      /** no geometric coordinates provided */
      DistData<STAR, CBLK, T> *X = NULL;
      /** dense spd matrix format */
      PVFMMKernelMatrix<T> K( n, n );
      /** (optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );
      /** routine */
      test_gofmm_setup<ADAPTIVE, LEVELRESTRICTION, T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
    }
	}





  /** HMLP API call to terminate the runtime */
  hmlp_finalize();

  /** Message Passing Interface */
  mpi::Finalize();

  return 0;

}; /** end main() */
