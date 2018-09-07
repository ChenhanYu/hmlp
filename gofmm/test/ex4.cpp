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
 *  @brief In this example, we explain how you can compress generic
 *         SPD matrices and kernel matrices using MPIGOFMM. 
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
  /** Number of right-hand sides. */
  size_t nrhs = 10;
  /** Regularization for the system (K+lambda*I). */
  T lambda = 1.0;

  /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support. */
  int  provided;
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
  SPDMatrix<T> K1( n, n ); K1.randspd( 0.0, 1.0 );
  /** Broadcast K to all other rank. */
  mpi::Bcast( K1.data(), n * n, 0, CommGOFMM );
  /** [Step#3] Create randomized and center splitters. */
  mpigofmm::randomsplit<SPDMatrix<T>, 2, T> rkdtsplitter1( K1 );
  mpigofmm::centersplit<SPDMatrix<T>, 2, T> splitter1( K1 );
  /** [Step#4] Perform the iterative neighbor search. */
  auto neighbors1 = mpigofmm::FindNeighbors( K1, rkdtsplitter1, config1, CommGOFMM );
  /** [Step#5] Compress the matrix with an algebraic FMM. */
  auto* tree_ptr1 = mpigofmm::Compress( K1, neighbors1, splitter1, rkdtsplitter1, config1, CommGOFMM );
  auto& tree1 = *tree_ptr1;
  /** [Step#6] Compute an approximate MATVEC. */
  DistData<RIDS, STAR, T> w1( n, nrhs, tree1.treelist[ 0 ]->gids, CommGOFMM ); w1.randn();
  auto u1 = mpigofmm::Evaluate( tree1, w1 );
  /** [Step#7] Factorization (HSS using ULV). */
  mpigofmm::DistFactorize( tree1, lambda ); 
  /** [Step#8] Solve (K+lambda*I)w = u approximately with HSS. */
  auto x1 = u1;
  mpigofmm::DistSolve( tree1, x1 ); 

	/** [Step#1] Create a configuration for kernel matrices. */
	gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );
  /** [Step#2] Create a Gaussian kernel matrix with random 6D data. */
  size_t d = 6;
  DistData<STAR, CBLK, T> X( d, n, CommGOFMM ); X.randn();
  DistKernelMatrix<T, T> K2( X, CommGOFMM );
  /** [Step#3] Create randomized and center splitters. */
  mpigofmm::randomsplit<DistKernelMatrix<T, T>, 2, T> rkdtsplitter2( K2 );
  mpigofmm::centersplit<DistKernelMatrix<T, T>, 2, T> splitter2( K2 );
  /** [Step#4] Perform the iterative neighbor search. */
  auto neighbors2 = mpigofmm::FindNeighbors( K2, rkdtsplitter2, config2, CommGOFMM );
  /** [Step#5] Compress the matrix with an algebraic FMM. */
  auto* tree_ptr2 = mpigofmm::Compress( K2, neighbors2, splitter2, rkdtsplitter2, config2, CommGOFMM );
  auto& tree2 = *tree_ptr2;
  /** [Step#6] Compute an approximate MATVEC. */
  DistData<RIDS, STAR, T> w2( n, nrhs, tree1.treelist[ 0 ]->gids, CommGOFMM ); w2.randn();
  auto u2 = mpigofmm::Evaluate( tree2, w2 );
  /** [Step#7] Factorization (HSS using ULV). */
  mpigofmm::DistFactorize( tree2, lambda ); 
  /** [Step#8] Solve (K+lambda*I)w = u approximately with HSS. */
  auto x2 = u2;
  mpigofmm::DistSolve( tree2, x2 ); 

  /** [Step#9] HMLP API call to terminate the runtime. */
  hmlp_finalize();
  /** Finalize Message Passing Interface. */
  mpi::Finalize();

  return 0;
}; /** end main() */
