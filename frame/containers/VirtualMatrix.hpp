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

/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif

/** Use hmlp::Data<T> for a concrete dense submatrix */
#include <containers/data.hpp>

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

#ifdef HMLP_MIC_AVX512
template<typename T, class Allocator = hbw::allocator<T> >
#elif  HMLP_USE_CUDA
/** use pinned (page-lock) memory for NVIDIA GPUs */
template<class T, class Allocator = thrust::system::cuda::experimental::pinned_allocator<T> >
#else
template<typename T, class Allocator = std::allocator<T> >
#endif
/**
 *  @brief VirtualMatrix is the abstract base class for matrix-free
 *         access and operations. Most of the public functions will
 *         be virtual. To inherit VirtualMatrix, you "must" implement
 *         the evaluation operator. Otherwise, the code won't compile.
 */ 
class VirtualMatrix
{
  public:

    VirtualMatrix() {};

    VirtualMatrix( size_t m, size_t n ) { resize( m, n ); };

    virtual void resize( size_t m, size_t n )
    {
      this->m = m;
      this->n = n;
    };

    /** ESSENTIAL: return number of coumns */
    virtual size_t row() { return m; };

    /** ESSENTIAL: return number of rows */
    virtual size_t col() { return n; };

    /** ESSENTIAL: this is an abstract function  */
    virtual T operator()( size_t i, size_t j ) = 0; 

    /** ESSENTIAL: return a submatrix */
    virtual Data<T> operator() ( const vector<size_t> &I, 
                                 const vector<size_t> &J )
    {
      Data<T> KIJ( I.size(), J.size() );
      for ( size_t j = 0; j < J.size(); j ++ )
        for ( size_t i = 0; i < I.size(); i ++ )
          KIJ( i, j ) = (*this)( I[ i ], J[ j ] );
      return KIJ;
    };

    virtual Data<T> PairwiseDistances( const vector<size_t> &I, 
                                       const vector<size_t> &J )
    {
      return (*this)( I, J );
    };


    virtual Data<T> KernelDistances( const vector<size_t> &I, 
                                     const vector<size_t> &J )
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

    virtual Data<T> AngleDistances( const vector<size_t> &I, 
                                    const vector<size_t> &J )
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

    virtual Data<T> UserDistances( const vector<size_t> &I, 
                                   const vector<size_t> &J )
    {
      printf( "UserDistances(): not defined.\n" ); exit( 1 );
      return (*this)( I, J );
    };
    
    virtual Data<T> GeometryDistances( const vector<size_t> &I, 
                                       const vector<size_t> &J )
    {
      printf( "GeometricDistances(): not defined.\n" ); exit( 1 );
      return (*this)( I, J );
    };


    virtual Data<T> Distances( DistanceMetric metric,
                               const vector<size_t> &I, 
                               const vector<size_t> &J )
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

}; /** end namespace hmlp */

#endif /** define VIRTUALMATRIX_HPP */
