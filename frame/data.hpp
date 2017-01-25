#ifndef DATA_HPP
#define DATA_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <random>

#include <hmlp.h>

#include <hmlp_blas_lapack.h>
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

    Data( std::size_t d, std::size_t n ) : std::vector<T, Allocator>( d * n )
    { 
      this->d = d;
      this->n = n;
    };

    Data( std::size_t d, std::size_t n, T initT ) : std::vector<T, Allocator>( d * n, initT )
    { 
      this->d = d;
      this->n = n;
    };

    Data( std::size_t d, std::size_t n, std::string &filename ) : std::vector<T, Allocator>( d * n )
    {
      this->d = d;
      this->n = n;

      std::ifstream file( filename.data(), std::ios::in|std::ios::binary|std::ios::ate );
      if ( file.is_open() )
      {
        auto size = file.tellg();
        assert( size == d * n * sizeof(T) );
        file.seekg( 0, std::ios::beg );
        file.read( (char*)this->data(), size );
        file.close();
      }
    };

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

    void read( std::size_t d, std::size_t n, std::string &filename )
    {
      assert( this->d == d );
      assert( this->n == n );
      assert( this->size() == d * n );

      std::ifstream file( filename.data(), std::ios::in|std::ios::binary|std::ios::ate );
      if ( file.is_open() )
      {
        auto size = file.tellg();
        assert( size == d * n * sizeof(T) );
        file.seekg( 0, std::ios::beg );
        file.read( (char*)this->data(), size );
        file.close();
      }
    };

    std::tuple<size_t, size_t> shape()
    {
      return std::make_tuple( d, n );
    };

    template<typename TINDEX>
    inline T operator()( TINDEX i, TINDEX j )
    {
      return (*this)[ d * j + i ];
    };


    template<typename TINDEX>
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &imap, std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( imap.size(), jmap.size() );

      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < imap.size(); i ++ )
        {
          submatrix[ j * imap.size() + i ] = (*this)[ d * jmap[ j ] + imap[ i ] ];
        }
      }

      return submatrix;
    }; 

    template<typename TINDEX>
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( d, jmap.size() );

      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < d; i ++ )
        {
          submatrix[ j * d + i ] = (*this)[ d * jmap[ j ] + i ];
        }
      }

      return submatrix;
    }; 

    //std::vector<T> operator()( std::vector<std::size_t> &jmap )
    //{
    //  std::vector<T> submatrix( d * jmap.size() );

    //  for ( int j = 0; j < jmap.size(); j ++ )
    //  {
    //    for ( int i = 0; i < d; i ++ )
    //    {
    //      submatrix[ j * d + i ] = (*this)[ d * jmap[ j ] + i ];
    //    }
    //  }

    //  return submatrix;
    //}; 



    template<bool SYMMETRIC = false>
    void rand( T a, T b )
    {
      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution( a, b );

      if ( SYMMETRIC ) assert( n == d );

      for ( std::size_t j = 0; j < n; j ++ )
      {
        for ( std::size_t i = 0; i < d; i ++ )
        {
          if ( SYMMETRIC )
          {
            if ( i > j )
              (*this)[ j * d + i ] = distribution( generator );
            else
              (*this)[ j * d + i ] = (*this)[ i * d + j ];
          }
          else
          {
            (*this)[ j * d + i ] = distribution( generator );
          }
        }
      }
    };

    template<bool SYMMETRIC = false>
    void rand()
    {
      rand<SYMMETRIC>( 0.0, 1.0 );
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

    void randn()
    {
      randn( 0.0, 1.0 );
    };

    template<bool USE_LOWRANK>
    void randspd( T a, T b )
    {
      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution( a, b );

      assert( n == d );

      if ( USE_LOWRANK )
      {
        hmlp::Data<T> X( ( std::rand() % n ) / 2 + 1, n );
        X.rand();
        xgemm
        (
          "T", "N",
          n, n, X.dim(),
          1.0, X.data(), X.dim(),
               X.data(), X.dim(),
          0.0, this->data(), this->dim()
        );
      }
      else // diagonal dominating
      {
        for ( std::size_t j = 0; j < n; j ++ )
        {
          for ( std::size_t i = 0; i < d; i ++ )
          {
            if ( i > j )
              (*this)[ j * d + i ] = distribution( generator );
            else
              (*this)[ j * d + i ] = (*this)[ i * d + j ];

            // Make sure diagonal dominated
            (*this)[ j * d + j ] += std::abs( (*this)[ j * d + i ] );
          }
        }
      }
    };

    template<bool USE_LOWRANK>
    void randspd()
    {
      randspd<USE_LOWRANK>( 0.0, 1.0 );
    };


    std::size_t dim()
    {
      return d;
    };

    std::size_t num()
    {
      return n;
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
