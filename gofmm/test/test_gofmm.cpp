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

#ifdef HMLP_AVX512
/** this is for hbw_malloc() and hnw_free */
#include <hbwmalloc.h>
/** we need hbw::allocator<T> to replace std::allocator<T> */
#include <hbw_allocator.h>
/** MKL headers */
#include <mkl.h>
#endif


/** Use GOFMM templates. */
#include <gofmm.hpp>
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

#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

using namespace std;
using namespace hmlp;




/** @brief Top level driver that reads arguments from the command line. */ 
int main( int argc, char *argv[] )
{
  gofmm::CommandLineHelper cmd( argc, argv );

  /** Read all parameters */
  size_t n = cmd.n;
  /** (Optional) */
  size_t d = cmd.d;
  size_t nb = cmd.nb; 
  /** (Optional) set the default Gaussian kernel bandwidth */
  float h = cmd.h;






  /** Message Passing Interface */
  int size = -1, rank = -1;
  mpi::Init( &argc, &argv );
  mpi::Comm_size( MPI_COMM_WORLD, &size );
  mpi::Comm_rank( MPI_COMM_WORLD, &rank );
  printf( "size %d rank %d\n", size, rank );

  /** HMLP API call to initialize the runtime */
  hmlp_init();

  /** run the matrix file provided by users */
  if ( !cmd.spdmatrix_type.compare( "dense" ) )
  {
    using T = float;
    /** Dense spd matrix format */
    SPDMatrix<T> K( n, n );
    K.read( n, n, cmd.user_matrix_filename );
		/** (optional) provide coordinates */
    Data<T> *X = NULL;
    if ( cmd.user_points_filename.size() ) 
      X = new Data<T>( d, n, cmd.user_points_filename );
    gofmm::LaunchHelper( X, K, cmd );
  }


  /** generate a Gaussian kernel matrix from the coordinates */
  if ( !cmd.spdmatrix_type.compare( "kernel" ) )
  {
    using T = double;
    {
      /** Read the coordinates from the file. */
      Data<T> X( d, n, cmd.user_points_filename );
      /** Set the kernel object as Gaussian. */
      kernel_s<T> kernel;
      kernel.type = KS_GAUSSIAN;
      kernel.scal = -0.5 / ( h * h );
      /** SPD kernel matrix format (implicitly create) */
      KernelMatrix<T> K( n, n, d, kernel, X );

      gofmm::LaunchHelper( &X, K, cmd );
    }
  }


  /** create a random spd matrix, which is diagonal-dominant */
  if ( !cmd.spdmatrix_type.compare( "testsuit" ) )
  {
    using T = double;
      /** no geometric coordinates provided */
      Data<T> *X = NULL;
      /** dense spd matrix format */
      SPDMatrix<T> K( n, n );
      /** random spd initialization */
      K.randspd( 0.0, 1.0 );
      gofmm::LaunchHelper( X, K, cmd );
  }


  if ( !cmd.spdmatrix_type.compare( "mlp" ) )
  {
    using T = double;
      /** Read the coordinates from the file. */
      Data<T> X( d, n, cmd.user_points_filename );
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


  if ( !cmd.spdmatrix_type.compare( "cov" ) )
  {
    using T = float;
    /** No geometric coordinates provided */
    Data<T> *X = NULL;
    OOCCovMatrix<T> K( n, d, nb, cmd.user_points_filename );
    gofmm::LaunchHelper( X, K, cmd );
  }



  /** HMLP API call to terminate the runtime */
  hmlp_finalize();
  /** Message Passing Interface */
  mpi::Finalize();

  return 0;

}; /** end main() */
