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



/** Use STL, and OpenMP. */
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

/** MPI support */
#include <hmlp_mpi.hpp>
/** GOFMM templates */
#include <gofmm_mpi.hpp>
/** INV-GOFMM templates */
#include <igofmm_mpi.hpp>

/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** use distributed implicit kernel matrices (only coordinates are stored). */
#include <containers/DistKernelMatrix.hpp>
/** use implicit matrices */
#include <containers/VirtualMatrix.hpp>
/** use implicit PVFMM kernel matrices. */
#include <containers/PVFMMKernelMatrix.hpp>
/** Use implicit Gauss-Newton (multilevel perceptron) matrices. */
#include <containers/MLPGaussNewton.hpp>
/** Use an OOC covariance matrices. */
#include <containers/OOCCovMatrix.hpp>
/** Use dense distributed matrices. */
#include <containers/DistData.hpp>






#ifdef HMLP_USE_CUDA
#include <hmlp_gpu.hpp>
#endif

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

/** By default, we use binary tree */
#define N_CHILDREN 2


using namespace std;
using namespace hmlp;


template<
  typename    SPLITTER, 
  typename    RKDTSPLITTER, 
  typename    T, 
  typename    SPDMATRIX>
void test_gofmm
( 
  DistData<STAR, CBLK, T> *X,
  SPDMATRIX &K, 
  DistData<STAR, CBLK, pair<T, size_t>> &NN,
  DistanceMetric metric,
  SPLITTER splitter, 
  RKDTSPLITTER rkdtsplitter,
  size_t n, size_t m, size_t k, size_t s, 
  double stol, double budget, size_t nrhs, 
  mpi::Comm CommGOFMM
)
{
	/** Create configuration for all user-define arguments. */
  gofmm::Configuration<T> config( metric, n, m, k, s, stol, budget );

  /** Compress matrix K. */
  auto *tree_ptr = mpigofmm::Compress( X, K, NN, splitter, rkdtsplitter, config );
  auto &tree = *tree_ptr;

  /** Examine accuracies. */
  SelfTesting( tree, 100, nrhs );

//  /** Factorization */
//  T lambda = 10.0;
//  mpigofmm::DistFactorize( tree, lambda ); 
//  mpigofmm::ComputeError( tree, lambda, w_rids, u_rids );

	/** Delete tree_ptr. */
  delete tree_ptr;
}; /** end test_gofmm() */











/**
 *  @brief Instantiate the splitters here.
 */ 
template<typename T, typename SPDMATRIX>
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
    //  test_gofmm<SPLITTER, RKDTSPLITTER, T>
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
      test_gofmm<SPLITTER, RKDTSPLITTER, T>
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
  /** Default geometric-oblivious scheme */
  DistanceMetric metric = ANGLE_DISTANCE;

  /** Test suit options */
  const bool SIMPLE = true;
  const bool RANDOMMATRIX = true;
  const bool USE_LOWRANK = true;
  const bool SPARSETESTSUIT = false;

  /** Default data directory */
  string DATADIR( "/" );

  /** Default precision */
  using T = double;

  /** Read all parameters */
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

  /** Run the matrix file provided by users */
  if ( !spdmatrix_type.compare( "dense" ) && user_matrix_filename.size() )
  {
    using T = float;
    {
      /** Dense spd matrix format */
      SPDMatrix<T> K;
      K.resize( n, n );
      K.read( n, n, user_matrix_filename );

      /** (optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );

			/** (optional) provide coordinates */
      if ( user_points_filename.size() )
      {
        DistData<STAR, CBLK, T> X( d, n, CommGOFMM,  user_points_filename );
        test_gofmm_setup<T>
        ( &X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
      }
      else
      {
        DistData<STAR, CBLK, T> *X = NULL;
        test_gofmm_setup<T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
      }
    }
  }


  /** Run the matrix file provided by users */
  if ( !spdmatrix_type.compare( "ooc" ) && user_matrix_filename.size() )
  {
    using T = float;
    {
      //OOCSPDMatrix<T> K( n, n, user_matrix_filename );
      ///** (optional) provide neighbors, leave uninitialized otherwise */
      //DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );
      //DistData<STAR, CBLK, T> *X = NULL;
      //test_gofmm_setup<T>
      //( X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
    }
  }




  /** Generate a kernel matrix from the coordinates */
  if ( !spdmatrix_type.compare( "kernel" ) && user_points_filename.size() )
  {
    using T = double;
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
		  DistKernelMatrix_ver2<T, T> K( n, d, kernel, X, CommGOFMM );




      /** shared spd kernel matrix format (implicitly create) */
			//hmlp::KernelMatrix<T> K( n, n, d, kernel, Xtmp );

      /** (optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );

      /** routine */
      test_gofmm_setup<T>
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
		  SPDMatrix<T> K;
			K.resize( n, n );
			/** random spd initialization */
			K.randspd<USE_LOWRANK>( 0.0, 1.0 );
			/** broadcast K to all other rank */
			mpi::Bcast( K.data(), n * n, 0, CommGOFMM );
      /** (optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );
      /** routine */
      test_gofmm_setup<T>
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
    //  test_gofmm_setup<T>
    //  ( &X, K, NN, metric, n, m, k, s, stol, budget, nrhs );
		//}
  }



  /** Create a random spd matrix, which is diagonal-dominant */
  if ( !spdmatrix_type.compare( "pvfmm" ) )
  {
    using T = double;
    {
      /** No geometric coordinates provided */
      DistData<STAR, CBLK, T> *X = NULL;
      /** Dense spd matrix format */
      PVFMMKernelMatrix<T> K( n, n );
      /** (Optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );
      /** Routine */
      test_gofmm_setup<T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
    }
	}


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
      Layer<FC, T> layer_fc0( 512, n, layer_input );
      Layer<FC, T> layer_fc1( 256, n, layer_fc0 );
      Layer<FC, T> layer_fc2( 100, n, layer_fc1 );
      Layer<FC, T> layer_fc3(  10, n, layer_fc2 );
      /** Insert layers into  */
      K.AppendInputLayer( layer_input );
      K.AppendFCLayer( layer_fc0 );
      K.AppendFCLayer( layer_fc1 );
      K.AppendFCLayer( layer_fc2 );
      K.AppendFCLayer( layer_fc3 );
      /** Feed forward and compute all products */
      K.Update( X );
      K.WriteJacobianToFiles( string( "/scratch/02794/ych/data/MLPMNIST10/MLPMNIST" ) );

      ///** (Optional) provide neighbors, leave uninitialized otherwise */
      //DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );
      ///** Routine */
      //test_gofmm_setup<T>
      //  ( NULL, K, NN, metric, K.col(), m, k, s, stol, budget, nrhs, CommGOFMM );
    }
  }

  if ( !spdmatrix_type.compare( "cov" ) )
  {
    using T = float;
    {
      /** No geometric coordinates provided */
      DistData<STAR, CBLK, T> *X = NULL;
      OOCCovMatrix<T> K( n, d, nb, user_points_filename );
      /** (optional) provide neighbors, leave uninitialized otherwise */
      DistData<STAR, CBLK, pair<T, size_t>> NN( 0, n, CommGOFMM );
      /** Routine */
      test_gofmm_setup<T>
        ( X, K, NN, metric, n, m, k, s, stol, budget, nrhs, CommGOFMM );
    }
  }

  /** HMLP API call to terminate the runtime */
  hmlp_finalize();
  /** Message Passing Interface */
  mpi::Finalize();

  return 0;

}; /** end main() */
