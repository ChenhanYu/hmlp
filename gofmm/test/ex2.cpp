/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2018, The University of Texas at Austin
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

/** Use MPI-GOFMM templates. */
#include <gofmm_mpi.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;

/** 
 *  @brief In this example, we explain how you can compute
 *         approximate all-nearest neighbors (ANN) using MPIGOFMM. 
 */ 
int main( int argc, char *argv[] )
{
  /** Use float as data type. */
  using T = float;
  /** [Required] Problem size. */
  size_t n = 5000;
  /** Maximum leaf node size (not used in neighbor search). */
  size_t m = 128;
  /** [Required] Number of nearest neighbors. */
  size_t k = 64;
  /** Maximum off-diagonal rank (not used in neighbor search). */
  size_t s = 128;
  /** Approximation tolerance (not used in neighbor search). */
  T stol = 1E-5;
  /** The amount of direct evaluation (not used in neighbor search). */
  T budget = 0.01;

  /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support. */
  int  provided = 0;
	mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
	if ( provided != MPI_THREAD_MULTIPLE ) exit( 1 );
  /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
  mpi::Comm CommGOFMM;
  mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );
  /** [Step#0] HMLP API call to initialize the runtime. */
  hmlp_init( &argc, &argv, CommGOFMM );

	/** [Step#1] Create a configuration for generic SPD matrices. */
  gofmm::Configuration<T> config1( ANGLE_DISTANCE, n, m, k, s, stol, budget );
  /** [Step#2] Create a dense random SPD matrix. */
  SPDMatrix<T> K1( n, n ); 
  K1.randspd( 0.0, 1.0 );
  /** [Step#2.5] Broadcast K to all other ranks. */
  mpi::Bcast( K1.data(), n * n, 0, CommGOFMM );
  /** [Step#3] Create a distributed randomized splitter. */
  mpigofmm::randomsplit<SPDMatrix<T>, 2, T> rkdtsplitter1( K1 );
  /** [Step#4] Perform the iterative neighbor search. */
  auto neighbors1 = mpigofmm::FindNeighbors( K1, rkdtsplitter1, config1, CommGOFMM );
  /** Here neighbors1 is distributed in DistData<STAR, CBLK, T> over CommGOFMM. */
  int rank; mpi::Comm_rank( CommGOFMM, &rank );
  int size; mpi::Comm_size( CommGOFMM, &size );
  printf( " rank/size %d/%d owns %lu/%lu rows and %lu/%lu columns of neighbors1\n ", 
      rank, size, 
      neighbors1.row_owned(), neighbors1.row(),
      neighbors1.col_owned(), neighbors1.col() );
  /** To be specific, this is called Elemental distribution (cylic distribution). */
  for ( int i = 0; i < std::min( k, (size_t)10 ); i ++ )
    printf( "rank/size %d/%d [%E,%5lu]\n", rank, size,
        neighbors1( i, rank ).first, neighbors1( i, rank ).second );
  for ( int i = 0; i < std::min( k, (size_t)10 ); i ++ )
    printf( "rank/size %d/%d [%E,%5lu]\n", rank, size,
        neighbors1( i, rank + size ).first, neighbors1( i, rank + size ).second );

	/** [Step#1] Create a configuration for kernel matrices. */
	gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );
  /** [Step#2] Create a distributed Gaussian kernel matrix with random 6D data. */
  size_t d = 6;
  DistData<STAR, CBLK, T> X( d, n, CommGOFMM ); X.randn();
  DistKernelMatrix<T, T> K2( X, CommGOFMM );
  /** [Step#3] Create a distributed randomized splitter. */
  mpigofmm::randomsplit<DistKernelMatrix<T, T>, 2, T> rkdtsplitter2( K2 );
  /** [Step#4] Perform the iterative neighbor search. */
  auto neighbors2 = mpigofmm::FindNeighbors( K2, rkdtsplitter2, config2, CommGOFMM );

  /** [Step#5] HMLP API call to terminate the runtime. */
  hmlp_finalize();
  /** Finalize Message Passing Interface. */
  mpi::Finalize();

  return 0;
}; /** end main() */
