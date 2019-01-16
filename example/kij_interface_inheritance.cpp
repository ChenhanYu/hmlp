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
#include <containers/VirtualMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;

/** [Step#0] Define a new SPD matrix type with value type T. */
template<typename T, class Allocator = std::allocator<T>>
/** [Step#1] The new class must inherit VirtualMatrix<T, Allocator>. */
class SparseSPDMatrix : public VirtualMatrix<T, Allocator>, 
                        public ReadWrite
{
  public:

    /** [Step#2] Define a constructor that inherits VirtualMatrix(). */
    SparseSPDMatrix( size_t m = 0, size_t n = 0, size_t nnz = 0, bool issymmetric = true )
      : VirtualMatrix<T>( m, n ), K( m, n, nnz, issymmetric )
    {  
    }; /** end SparseSPDMatrix() */

    /** [Step#3] (Optional) initialize the matrix with three vectors (CSC format).*/
    void fromCSC( size_t m, size_t n, size_t nnz, bool issymmetric,
        const T *val, const size_t *row_ind, const size_t *col_ptr )
    {
      K.fromCSC( m, n, nnz, issymmetric, val, row_ind, col_ptr );
    }; /** end fromCSC() */

    /** [Step#4] implement K( i, j ) operator. */
    T operator () ( size_t i, size_t j ) override { return K( i, j ); };

    /** [Step#5] implement K( I, J ) operator. */
    Data<T> operator() ( const vector<size_t> &I, const vector<size_t> &J ) override
    {
      Data<T> KIJ( I.size(), J.size() );
      for ( size_t j = 0; j < J.size(); j ++ )
        for ( size_t i = 0; i < I.size(); i ++ )
          KIJ( i, j ) = (*this)( I[ i ], J[ j ] );
      return KIJ;
    };
    
  private:

    /** Here we use a sparse data format. */
    SparseData<T> K;
    
}; /** end class SparseSPDMatrix */




/** 
 *  @brief In this example, we explain how you can compress generic
 *         SPD matrices and kernel matrices using GOFMM.
 */ 
int main( int argc, char *argv[] )
{
  try
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
    HANDLE_ERROR( hmlp_init( &argc, &argv ) );
    /** Create a configuration for generic SPD matrices. */
    gofmm::Configuration<T> config1( ANGLE_DISTANCE, n, m, k, s, stol, budget );
    /** Create a sparse diagonal matrix. */
    size_t nnz = n;
    vector<T> vals( nnz, 1.0 );
    vector<size_t> col_ptr( n + 1, 0 );
    vector<size_t> row_ind( nnz, 0 );
    for ( size_t i = 0; i < col_ptr.size(); i ++ ) col_ptr[ i ] = i;
    for ( size_t i = 0; i < row_ind.size(); i ++ ) row_ind[ i ] = i;
    SparseSPDMatrix<T> K1; 
    K1.fromCSC( n, n, nnz /** number of nonzeros */, true /** is symmetric */,
        vals.data(), col_ptr.data(), row_ind.data() );
    printf( "K( 0, 0 ) %E here1\n", K1( 0, 0 ) ); fflush( stdout );
    printf( "K( 1, 0 ) %E here1\n", K1( 1, 0 ) ); fflush( stdout );
    printf( "K( 0, 1 ) %E here1\n", K1( 0, 1 ) ); fflush( stdout );
    printf( "K( 5, 0 ) %E here1\n", K1( 5, 0 ) ); fflush( stdout );
    printf( "K( 0, 4 ) %E here1\n", K1( 0, 4 ) ); fflush( stdout );
    vector<size_t> I( 5 ), J( 5 );
    for ( size_t i = 0; i < I.size(); i ++ ) I[ i ] = i;
    for ( size_t j = 0; j < J.size(); j ++ ) J[ j ] = j;
    auto KIJ = K1( I, J );
    KIJ.Print();
    /** Create randomized and center splitters. */
    gofmm::randomsplit<SparseSPDMatrix<T>, 2, T> rkdtsplitter1( K1 );
    gofmm::centersplit<SparseSPDMatrix<T>, 2, T> splitter1( K1 );
    /** Perform the iterative neighbor search. */
    auto neighbors1 = gofmm::FindNeighbors( K1, rkdtsplitter1, config1 );

    /** HMLP API call to terminate the runtime. */
    hmlp_finalize();
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}; /** end main() */
