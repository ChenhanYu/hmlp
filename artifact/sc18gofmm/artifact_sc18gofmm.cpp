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

/** GOFMM templates */
#include <gofmm_mpi.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** use distributed implicit kernel matrices (only coordinates are stored). */
#include <containers/VirtualMatrix.hpp>
/** use implicit PVFMM kernel matrices. */
#include <containers/OOCCovMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;



/** @brief Top level driver that reads arguments from the command line. */ 
int main( int argc, char *argv[] )
{
  /** Parse arguments from the command line. */
  gofmm::CommandLineHelper cmd( argc, argv );

  /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support. */
  int size, rank, provided;
	mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
	if ( provided != MPI_THREAD_MULTIPLE )
	{
		printf( "MPI_THREAD_MULTIPLE is not supported\n" ); fflush( stdout );
    exit( 1 );
	}

  /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
  mpi::Comm CommGOFMM;
  mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );

  mpi::PrintProgress( "\n--- Artifact of GOFMM for Super Computing 2018\n", CommGOFMM );

  /** HMLP API call to initialize the runtime. */
  hmlp_init( CommGOFMM );

  /** Run the matrix file provided by users. */
  if ( !cmd.spdmatrix_type.compare( "dense" ) )
  {
    using T = float;
    /** Dense spd matrix format. */
    SPDMatrix<T> K( cmd.n, cmd.n, cmd.user_matrix_filename );
		/** (Optional) provide coordinates. */
    DistData<STAR, CBLK, T> *X = NULL;
    /** Launch self-testing routine. */
    mpigofmm::LaunchHelper( X, K, cmd, CommGOFMM );
  }

  /** Run the matrix file provided by users */
  if ( !cmd.spdmatrix_type.compare( "ooc" ) )
  {
    using T = float;
    /** Dense spd matrix format. */
    OOCSPDMatrix<T> K( cmd.n, cmd.n, cmd.user_matrix_filename );
		/** (Optional) provide coordinates. */
    DistData<STAR, CBLK, T> *X = NULL;
    /** Launch self-testing routine. */
    mpigofmm::LaunchHelper( X, K, cmd, CommGOFMM );
  }

  /** Generate a kernel matrix from the coordinates. */
  if ( !cmd.spdmatrix_type.compare( "kernel" ) )
  {
    using T = double;
    /** Read the coordinates from the file. */
    DistData<STAR, CBLK, T> X( cmd.d, cmd.n, CommGOFMM, cmd.user_points_filename );
    /** Setup the kernel object. */
    kernel_s<T> kernel;
    kernel.type = KS_GAUSSIAN;
    if ( !cmd.kernelmatrix_type.compare( "gaussian" ) ) kernel.type = KS_GAUSSIAN;
    if ( !cmd.kernelmatrix_type.compare(  "laplace" ) ) kernel.type = KS_LAPLACE;
    kernel.scal = -0.5 / ( cmd.h * cmd.h );
    /** Distributed spd kernel matrix format (implicitly create). */
    DistKernelMatrix_ver2<T, T> K( cmd.n, cmd.d, kernel, X, CommGOFMM );
    /** Launch self-testing routine. */
    mpigofmm::LaunchHelper( &X, K, cmd, CommGOFMM );
  }

  /** Create a random spd matrix, which is diagonal-dominant. */
  if ( !cmd.spdmatrix_type.compare( "testsuit" ) )
  {
    using T = double;
    /** no geometric coordinates provided */
    DistData<STAR, CBLK, T> *X = NULL;
    /** dense spd matrix format */
    SPDMatrix<T> K( cmd.n, cmd.n );
    /** random spd initialization */
    K.randspd( 0.0, 1.0 );
    /** broadcast K to all other rank */
    mpi::Bcast( K.data(), cmd.n * cmd.n, 0, CommGOFMM );
    /** Launch self-testing routine. */
    mpigofmm::LaunchHelper( X, K, cmd, CommGOFMM );
  }

  /** HMLP API call to terminate the runtime */
  hmlp_finalize();
  /** Message Passing Interface */
  mpi::Finalize();

  return 0;

}; /** end main() */
