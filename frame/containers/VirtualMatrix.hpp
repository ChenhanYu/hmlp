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

/** use hmlp::Data<T> for a concrete dense submatrix */
#include <containers/data.hpp>


namespace hmlp
{

#ifdef HMLP_MIC_AVX512
template<typename T, class Allocator = hbw::allocator<T> >
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

    VirtualMatrix( std::size_t m, std::size_t n )
    {
      this->m = m;
      this->n = n;
    };

    /** ESSENTIAL: return number of coumns */
    virtual std::size_t row() { return m; };

    /** ESSENTIAL: return number of rows */
    virtual std::size_t col() { return n; };

    /** ESSENTIAL: this is an abstract function  */
    virtual T operator()( std::size_t i, std::size_t j ) = 0; 

    /** ESSENTIAL: return a submatrix */
    virtual hmlp::Data<T> operator()
		  ( std::vector<size_t> &imap, std::vector<size_t> &jmap )
    {
      hmlp::Data<T> submatrix( imap.size(), jmap.size() );
      #pragma omp parallel for
      for ( size_t j = 0; j < jmap.size(); j ++ )
        for ( size_t i = 0; i < imap.size(); i ++ )
          submatrix[ j * imap.size() + i ] = 
            (*this)( imap[ i ], jmap[ j ] );
      return submatrix;
    };

    virtual hmlp::Data<T> operator()
		  ( std::vector<int> &imap, std::vector<int> &jmap )
    {
      hmlp::Data<T> submatrix( imap.size(), jmap.size() );
      #pragma omp parallel for
      for ( size_t j = 0; j < jmap.size(); j ++ )
        for ( size_t i = 0; i < imap.size(); i ++ )
          submatrix[ j * imap.size() + i ] = 
            (*this)( imap[ i ], jmap[ j ] );
      return submatrix;
    };

    virtual std::pair<T, size_t> ImportantSample( size_t j )
    {
      size_t i = std::rand() % m;
      std::pair<T, size_t> sample( (*this)( i, j ), i );
      return sample; 
    }; /** end ImportantSample() */

    virtual std::pair<T, int> ImportantSample( int j )
    {
      int i = std::rand() % m;
      std::pair<T, int> sample( (*this)( i, j ), i );
      return sample; 
    }; /** end ImportantSample() */


    virtual void BackGroundProcess( bool *do_terminate )
		{
		};


  private:

    std::size_t m;

    std::size_t n;

}; /** end class VirtualMatrix */

}; /** end namespace hmlp */

#endif /** define VIRTUALMATRIX_HPP */
