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

/** GOFMM templates. */
#include <gofmm_mpi.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** use implicit matrices */
#include <containers/VirtualMatrix.hpp>
/** use implicit PVFMM kernel matrices. */
//#include <containers/PVFMMKernelMatrix.hpp>
/** Use implicit Gauss-Newton (multilevel perceptron) matrices. */
#include <containers/MLPGaussNewton.hpp>
/** Use an OOC covariance matrices. */
#include <containers/OOCCovMatrix.hpp>
/** Use Gauss Hessian matrices provided by Chao. */
#include <containers/GNHessian.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;



/** @brief Top level driver that reads arguments from the command line. */ 
int main( int argc, char *argv[] )
{
  try
  {
    /** Parse arguments from the command line. */
    gofmm::CommandLineHelper cmd( argc, argv );

    /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support. */
    int  provided;
    mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    if ( provided != MPI_THREAD_MULTIPLE )
    {
      printf( "MPI_THREAD_MULTIPLE is not supported\n" ); fflush( stdout );
      exit( 1 );
    }

    /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
    mpi::Comm CommGOFMM;
    mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );

    /** HMLP API call to initialize the runtime. */
    HANDLE_ERROR( hmlp_init( CommGOFMM ) );

    /** Run the matrix file provided by users. */
    if ( !cmd.spdmatrix_type.compare( "dense" ) )
    {
      using T = float;
      /** Dense spd matrix format. */
      SPDMatrix<T> K( cmd.n, cmd.n, cmd.user_matrix_filename );
      /** Launch self-testing routine. */
      mpigofmm::LaunchHelper( K, cmd, CommGOFMM );
    }


    /** Run the matrix file provided by users */
    if ( !cmd.spdmatrix_type.compare( "ooc" ) )
    {
      using T = float;
      /** Dense spd matrix format. */
      OOCSPDMatrix<T> K( cmd.n, cmd.n, cmd.user_matrix_filename );
      /** Launch self-testing routine. */
      mpigofmm::LaunchHelper( K, cmd, CommGOFMM );
    }


    /** Generate a kernel matrix from the coordinates. */
    if ( !cmd.spdmatrix_type.compare( "kernel" ) )
    {
      //using T = float;
      using T = double;
      /** Read the coordinates from the file. */
      hmlp::DistData<STAR, CBLK, T> X(cmd.d, cmd.n, CommGOFMM, cmd.user_points_filename);
      /** Setup the kernel object. */
      hmlp::kernel_s<T, T> kernel;
      kernel.type = GAUSSIAN;
      if ( !cmd.kernelmatrix_type.compare( "gaussian" ) ) kernel.type = GAUSSIAN;
      if ( !cmd.kernelmatrix_type.compare(  "laplace" ) ) kernel.type = LAPLACE;
      kernel.scal = -0.5 / ( cmd.h * cmd.h );
      /** Distributed spd kernel matrix format (implicitly create). */
      hmlp::DistKernelMatrix<T, T> K( cmd.n, cmd.d, kernel, X, CommGOFMM );
      /** Launch self-testing routine. */
      hmlp::mpigofmm::LaunchHelper( K, cmd, CommGOFMM );
    }

    /** Create a random spd matrix, which is diagonal-dominant. */
    if ( !cmd.spdmatrix_type.compare( "testsuit" ) )
    {
      using T = double;
      /** dense spd matrix format */
      SPDMatrix<T> K( cmd.n, cmd.n );
      /** random spd initialization */
      K.randspd( 0.0, 1.0 );
      /** broadcast K to all other rank */
      mpi::Bcast( K.data(), cmd.n * cmd.n, 0, CommGOFMM );
      /** Launch self-testing routine. */
      mpigofmm::LaunchHelper( K, cmd, CommGOFMM );
    }


    if ( !cmd.spdmatrix_type.compare( "pvfmm" ) ) { /** NOP */ }


    if ( !cmd.spdmatrix_type.compare( "mlp" ) )
    {
      using T = double;
      {
        /** Read the coordinates from the file. */
        Data<T> X( cmd.d, cmd.n, cmd.user_points_filename );
        /** Multilevel perceptron Gauss-Newton */
        MLPGaussNewton<T> K;
        /** Create an input layer */
        Layer<INPUT, T> layer_input( cmd.d, cmd.n );
        /** Create FC layers */
        Layer<FC, T> layer_fc0( 512, cmd.n, layer_input );
        Layer<FC, T> layer_fc1( 256, cmd.n, layer_fc0 );
        Layer<FC, T> layer_fc2( 100, cmd.n, layer_fc1 );
        Layer<FC, T> layer_fc3(  10, cmd.n, layer_fc2 );
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

    if ( !cmd.spdmatrix_type.compare( "cov" ) )
    {
      using T = float;
      OOCCovMatrix<T> K( cmd.n, cmd.d, cmd.nb, cmd.user_points_filename );
      /** Launch self-testing routine. */
      mpigofmm::LaunchHelper( K, cmd, CommGOFMM );
    }

    if ( !cmd.spdmatrix_type.compare( "jacobian" ) )
    {
      using T = float;
      GNHessian<T> K;
      K.read_jacobian( cmd.user_matrix_filename );
      mpigofmm::LaunchHelper( K, cmd, CommGOFMM );
    }

    /** HMLP API call to terminate the runtime */
    HANDLE_ERROR( hmlp_finalize() );
    /** Message Passing Interface */
    mpi::Finalize();
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;

}; /** end main() */
