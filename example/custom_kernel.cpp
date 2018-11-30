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

/** Use GOFMM templates. */
#include <gofmm.hpp>
/** Use abstracted virtual matrices. */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;


template<typename T, typename TP>
T element_relu( const void* param, const TP* x, const TP* y, size_t d )
{
	// kernel_s inner product is defined in :gofmm/frame/containers/KernelMatrix.hpp
  return std::max( T(0), kernel_s<T, TP>::innerProduct( x, y, d ) );  
};

template<typename T, typename TP>
void matrix_relu( const void* param, const TP* X, const TP* Y, size_t d, T* K, size_t m, size_t n )
{
  kernel_s<T, TP>::innerProducts( X, Y, d, K, m, n );
  #pragma omp parallel for
  for ( size_t i = 0; i < m * n; i ++ ) K[ i ] = std::max( T(0), K[ i ] );
}

/** 
 *  @brief In this example, we explain how you can compress generic
 *         SPD matrices and kernel matrices using GOFMM.
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

  /** HMLP API call to initialize the runtime. */
  hmlp_init( &argc, &argv );

	/** [Step#1] Create a configuration for kernel matrices. */
	gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );
  /** [Step#2] Create a costomized kernel matrix with random 6D data. */
  size_t d = 6;
  Data<T> X( d, n ); X.randn();
  kernel_s<T, T> kernel;
  kernel.type = USER_DEFINE;
  kernel.user_element_function = element_relu<T, T>;
  kernel.user_matrix_function = matrix_relu<T, T>;
  KernelMatrix<T> K2( n, n, d, kernel, X );
  //KernelMatrix<T> K2( X );
  /** [Step#3] Create randomized and center splitters. */
  gofmm::randomsplit<KernelMatrix<T>, 2, T> rkdtsplitter2( K2 );
  gofmm::centersplit<KernelMatrix<T>, 2, T> splitter2( K2 );
  /** [Step#4]Perform the iterative neighbor search. */
  auto neighbors2 = gofmm::FindNeighbors( K2, rkdtsplitter2, config2 );
  /** [Step#5] Compress the matrix with an algebraic FMM. */
  auto* tree_ptr2 = gofmm::Compress( K2, neighbors2, splitter2, rkdtsplitter2, config2 );
  auto& tree2 = *tree_ptr2;
  /** [Step#6] Compute an approximate MATVEC. */
  Data<T> w2( n, nrhs ); w2.randn();
  auto u2 = gofmm::Evaluate( tree2, w2 );
  /** [Step#7] Factorization (HSS using ULV). */
  gofmm::Factorize( tree2, lambda ); 
  /** [Step#8] Solve (K+lambda*I)w = u approximately with HSS. */
  auto x2 = u2;
  gofmm::Solve( tree2, x2 ); 


  /** HMLP API call to terminate the runtime. */
  hmlp_finalize();

  return 0;
}; /** end main() */
