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

#ifndef VIRTUALMATRIX_HPP
#define VIRTUALMATRIX_HPP

/** Use hmlp::Data<T> for a concrete dense submatrix */
#include <Data.hpp>

using namespace std;


/**
 *  @breif GOFMM relies on an arbitrary distance metric that
 *         described the order between matrix element Kij.
 *         Such metric can be the 
 *         Euclidian distance i.e. "GEOMETRY_DISTANCE", or
 *         arbitrary distances defined in the Gram vector space
 *         i.e. "KERNEL_DISTANCE" or "ANGLE_DISTANCE"
 */ 
typedef enum 
{ 
  GEOMETRY_DISTANCE, 
  KERNEL_DISTANCE, 
  ANGLE_DISTANCE, 
  USER_DISTANCE 
} DistanceMetric;


namespace hmlp
{

template<typename T>
class SPDMatrixMPISupport
{
  public:
    virtual void SendIndices( vector<size_t> ids, int dest, mpi::Comm comm ) {};
    virtual void RecvIndices( int src, mpi::Comm comm, mpi::Status *status ) {};
    virtual void BcastIndices( vector<size_t> ids, int root, mpi::Comm comm ) {};
    virtual void RequestIndices( const vector<vector<size_t>>& ids ) {};
}; /** end class SPDMatrixMPISupport */



template<typename DATATYPE, class Allocator = std::allocator<DATATYPE>>
/**
 *  @brief VirtualMatrix is the abstract base class for matrix-free
 *         access and operations. Most of the public functions will
 *         be virtual. To inherit VirtualMatrix, you "must" implement
 *         the evaluation operator. Otherwise, the code won't compile.
 */ 
class VirtualMatrix : public SPDMatrixMPISupport<DATATYPE>
{
  public:

    typedef DATATYPE T;

    VirtualMatrix() {};

    VirtualMatrix( size_t m, size_t n ) { resize( m, n ); };

    virtual void resize( size_t m, size_t n ) { this->m = m; this->n = n; };

    /** ESSENTIAL: return number of coumns */
    size_t row() { return m; };

    /** ESSENTIAL: return number of rows */
    size_t col() { return n; };

    /** ESSENTIAL: this is an abstract function  */
    virtual T operator () ( size_t i, size_t j ) = 0; 

    /** ESSENTIAL: return a submatrix */
    virtual Data<T> operator() ( const vector<size_t>& I, const vector<size_t>& J )
    {
      Data<T> KIJ( I.size(), J.size() );
      for ( size_t j = 0; j < J.size(); j ++ )
        for ( size_t i = 0; i < I.size(); i ++ )
          KIJ( i, j ) = (*this)( I[ i ], J[ j ] );
      return KIJ;
    };

    Data<T> KernelDistances( const vector<size_t> &I, const vector<size_t> &J )
    {
      auto KIJ = (*this)( I, J );
      auto DII = Diagonal( I );
      auto DJJ = Diagonal( J );
      for ( size_t j = 0; j < J.size(); j ++ )
      {
        for ( size_t i = 0; i < I.size(); i ++ )
        {
          auto kij = KIJ( i, j );
          auto kii = DII[ i ];
          auto kjj = DJJ[ j ];
          KIJ( i, j ) = kii - 2.0 * kij + kjj;
        }
      }
      return KIJ;
    };

    Data<T> AngleDistances( const vector<size_t> &I, const vector<size_t> &J )
    {
      auto KIJ = (*this)( I, J );
      auto DII = Diagonal( I );
      auto DJJ = Diagonal( J );
      for ( size_t j = 0; j < J.size(); j ++ )
      {
        for ( size_t i = 0; i < I.size(); i ++ )
        {
          auto kij = KIJ( i, j );
          auto kii = DII[ i ];
          auto kjj = DJJ[ j ];
          KIJ( i, j ) = 1.0 - ( kij * kij ) / ( kii * kjj );
        }
      }
      return KIJ;
    };

    virtual Data<T> UserDistances( const vector<size_t> &I, const vector<size_t> &J )
    {
      printf( "UserDistances(): not defined.\n" ); exit( 1 );
      return Data<T>( I.size(), J.size(), 0 );
    };
    
    virtual Data<T> GeometryDistances( const vector<size_t> &I, const vector<size_t> &J )
    {
      printf( "GeometricDistances(): not defined.\n" ); exit( 1 );
      return Data<T>( I.size(), J.size(), 0 );
    };


    Data<T> Distances( DistanceMetric metric, const vector<size_t> &I, const vector<size_t> &J )
    {
      switch ( metric )
      {
        case KERNEL_DISTANCE:   return KernelDistances( I, J );
        case ANGLE_DISTANCE:    return AngleDistances( I, J );
        case GEOMETRY_DISTANCE: return GeometryDistances( I, J );
        case USER_DISTANCE:     return UserDistances( I, J );
        default:
        {
          printf( "ERROR: Unknown distance type." );
          exit( 1 );
        }
      }
    };

    virtual Data<pair<T, size_t>> NeighborSearch( 
        DistanceMetric metric, size_t kappa, 
        const vector<size_t> &Q,
        const vector<size_t> &R, pair<T, size_t> init )
    {
      Data<pair<T, size_t>> NN( kappa, Q.size(), init );

      /** Compute all pairwise distances. */
      auto DRQ = Distances( metric, R, Q );
      /** Sanity check: distance must be >= 0. */
      for ( auto &dij: DRQ ) dij = std::max( dij, T(0) );

      /** Loop over each query. */
      for ( size_t j = 0; j < Q.size(); j ++ )
      {
        vector<pair<T, size_t>> candidates( R.size() );
        for ( size_t i = 0; i < R.size(); i ++)
        {
          candidates[ i ].first  = DRQ( i, j );
          candidates[ i ].second = R[ i ];
        }
        /** Sort the query according to distances. */
        sort( candidates.begin(), candidates.end() );
        /** Fill-in the neighbor list. */
        for ( size_t i = 0; i < kappa; i ++ ) NN( i, j ) = candidates[ i ];
      }

      return NN;
    };

		virtual Data<T> Diagonal( const vector<size_t> &I )
		{
			Data<T> DII( I.size(), 1, 0.0 );
			for ( size_t i = 0; i < I.size(); i ++ ) 
				DII[ i ] = (*this)( I[ i ], I[ i ] );
			return DII;
		};

    virtual pair<T, size_t> ImportantSample( size_t j )
    {
      size_t i = std::rand() % m;
      pair<T, size_t> sample( (*this)( i, j ), i );
      return sample; 
    }; /** end ImportantSample() */

    virtual pair<T, int> ImportantSample( int j )
    {
      int i = std::rand() % m;
      pair<T, int> sample( (*this)( i, j ), i );
      return sample; 
    }; /** end ImportantSample() */

  private:

    size_t m = 0;

    size_t n = 0;

}; /** end class VirtualMatrix */



#ifdef HMLP_MIC_AVX512
template<typename T, class Allocator = hbw::allocator<T> >
#else
template<typename T, class Allocator = std::allocator<T> >
#endif
/**
 *  @brief DistVirtualMatrix is the abstract base class for matrix-free
 *         access and operations. Most of the public functions will
 *         be virtual. To inherit DistVirtualMatrix, you "must" implement
 *         the evaluation operator. Otherwise, the code won't compile.
 *
 *         Two virtual functions must be implemented:
 *
 *         T operator () ( size_t i, size_t j ), and
 *         Data<T> operator () ( vector<size_t> &I, vector<size_t> &J ).
 *
 *         These two functions can involve nonblocking MPI routines, but
 *         blocking collborative communication routines are not allowed.
 *         
 *         DistVirtualMatrix inherits mpi::MPIObject, which is initalized
 *         with the provided comm. MPIObject duplicates comm into
 *         sendcomm and recvcomm, which allows concurrent multi-threaded 
 *         nonblocking send/recv. 
 *
 *         For example, RequestKIJ( I, J, p ) sends I and J to rank-p,
 *         requesting the submatrix. MPI process rank-p has at least
 *         one thread will execute BackGroundProcess( do_terminate ),
 *         waiting for incoming requests.
 *         Rank-p then invoke K( I, J ) locally, and send the submatrix
 *         back to the clients. Overall the pattern usually looks like
 *
 *         Data<T> operator () ( vector<size_t> &I, vector<size_t> &J ).
 *         {
 *           for each submatrix KAB entirely owned by p
 *             KAB = RequestKIJ( A, B )
 *             pack KAB back to KIJ
 *
 *           return KIJ 
 *         }
 *         
 */ 
class DistVirtualMatrix : public VirtualMatrix<T, Allocator>,
                          public mpi::MPIObject
{
  public:

    /** (Default) constructor  */
    DistVirtualMatrix( size_t m, size_t n, mpi::Comm comm )
      : VirtualMatrix<T, Allocator>( m, n ), mpi::MPIObject( comm )
    {};


}; /** end class DistVirtualMatrix */


}; /** end namespace hmlp */

#endif /** define VIRTUALMATRIX_HPP */
