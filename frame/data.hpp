#ifndef DATA_HPP
#define DATA_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <vector>
#include <deque>
#include <iostream>
#include <random>

#include <hmlp.h>

#include <hmlp_util.hpp>
#include <hmlp_runtime.hpp>

#define DEBUG_DATA 1


namespace hmlp
{

template<class T, class Allocator = std::allocator<T> >
class Data : public std::vector<T, Allocator>
{
  public:

    Data() : d( 0 ), n( 0 ) {};

    Data( std::size_t d, std::size_t n ) : std::vector( d * n )
    { 
      this->d = d;
      this->n = n;
    };

    //std::size_t d;

    //std::size_t n;

    enum Pattern : int { STAR = -1 };

    void resize( std::size_t d, std::size_t n )
    { 
      this->d = d;
      this->n = n;
      std::vector<T, Allocator>::resize( d * n );
    };

    void resize( std::size_t d, std::size_t n, T initT )
    {
      this->d = d;
      this->n = n;
      std::vector<T, Allocator>::resize( d * n, initT );
    };

    void reserve( std::size_t d, std::size_t n ) 
    {
      std::vector::reserve( d * n );
    };

    std::tuple<size_t, size_t> shape()
    {
      return std::make_tuple( d, n );
    };

    std::vector<T>& operator()( std::vector<int> &imap, std::vector<int> &jmap )
    {
      std::vector<T> submatrix( imap.size() * jmap.size() );

      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < imap.size(); i ++ )
        {
          submatrix[ j * imap.size() + i ] = (*this)[ d * jmap[ j ] + imap[ i ] ];
        }
      }

      return submatrix;
    }; 

    std::vector<T>& operator()( std::vector<int> &jmap )
    {
      std::vector<T> submatrix( d * jmap.size() );

      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < d; i ++ )
        {
          submatrix[ j * d + i ] = (*this)[ d * jmap[ j ] + i ];
        }
      }

      return submatrix;
    }; 


    void rand( T a, T b )
    {
      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution( a, b );
      for ( std::size_t i = 0; i < d * n; i ++ )
      {
        (*this)[ i ] =  distribution( generator );
      }
    };

    void rand()
    {
      rand( 0.0, 1.0 );
    };

    void randn( T mu, T sd )
    {
      std::default_random_engine generator;
      std::normal_distribution<T> distribution( mu, sd );
      for ( std::size_t i = 0; i < d * n; i ++ )
      {
        (*this)[ i ] =  distribution( generator );
      }
    };

    void randin()
    {
      randn( 0.0, 1.0 );
    };

    void Print()
    {
      printf( "Data in %lu * %lu\n", d, n );
      hmlp::hmlp_printmatrix( d, n, this->data(), d );
    };

  private:

    std::size_t d;

    std::size_t n;

};


}; // end namespace hmlp

#endif //define DATA_HPP
