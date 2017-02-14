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

#ifdef HMLP_USE_CUDA
#include <host_vector.h>
#include <device_vector.h>
#endif // ifdef HMLP_USE_CUDA




#define DEBUG_DATA 1

namespace hmlp
{

#ifdef HMLP_USE_CUDA
template<class T>
class Data : public ReadWrite
{
  publuc:

  private:

}; // end class Data

#else
template<class T, class Allocator = std::allocator<T> >
class Data : public ReadWrite, public std::vector<T, Allocator>
{
  public:

    Data() : m( 0 ), n( 0 ) {};

    Data( std::size_t m, std::size_t n ) : std::vector<T, Allocator>( m * n )
    { 
      this->m = m;
      this->n = n;
    };

    Data( std::size_t m, std::size_t n, T initT ) : std::vector<T, Allocator>( m * n, initT )
    { 
      this->m = m;
      this->n = n;
    };

    Data( std::size_t m, std::size_t n, std::string &filename ) : std::vector<T, Allocator>( m * n )
    {
      this->m = m;
      this->n = n;

      std::cout << filename << std::endl;

      std::ifstream file( filename.data(), std::ios::in|std::ios::binary|std::ios::ate );
      if ( file.is_open() )
      {
        auto size = file.tellg();
        assert( size == m * n * sizeof(T) );
        file.seekg( 0, std::ios::beg );
        file.read( (char*)this->data(), size );
        file.close();
      }
    };

    enum Pattern : int { STAR = -1 };

    void resize( std::size_t m, std::size_t n )
    { 
      this->m = m;
      this->n = n;
      std::vector<T, Allocator>::resize( m * n );
    };

    void resize( std::size_t m, std::size_t n, T initT )
    {
      this->m = m;
      this->n = n;
      std::vector<T, Allocator>::resize( m * n, initT );
    };

    void reserve( std::size_t m, std::size_t n ) 
    {
      std::vector<T, Allocator>::reserve( m * n );
    };

    void read( std::size_t m, std::size_t n, std::string &filename )
    {
      assert( this->m == m );
      assert( this->n == n );
      assert( this->size() == m * n );

      std::cout << filename << std::endl;

      std::ifstream file( filename.data(), std::ios::in|std::ios::binary|std::ios::ate );
      if ( file.is_open() )
      {
        auto size = file.tellg();
        assert( size == m * n * sizeof(T) );
        file.seekg( 0, std::ios::beg );
        file.read( (char*)this->data(), size );
        file.close();
      }
    };

    std::tuple<size_t, size_t> shape()
    {
      return std::make_tuple( m, n );
    };

    template<typename TINDEX>
    inline T operator()( TINDEX i, TINDEX j )
    {
      return (*this)[ m * j + i ];
    };


    template<typename TINDEX>
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &imap, std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( imap.size(), jmap.size() );

      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < imap.size(); i ++ )
        {
          submatrix[ j * imap.size() + i ] = (*this)[ m * jmap[ j ] + imap[ i ] ];
        }
      }

      return submatrix;
    }; 

    template<typename TINDEX>
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( m, jmap.size() );

      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < m; i ++ )
        {
          submatrix[ j * m + i ] = (*this)[ m * jmap[ j ] + i ];
        }
      }

      return submatrix;
    }; 

    template<bool SYMMETRIC = false>
    void rand( T a, T b )
    {
      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution( a, b );

      if ( SYMMETRIC ) assert( m == n );

      for ( std::size_t j = 0; j < n; j ++ )
      {
        for ( std::size_t i = 0; i < m; i ++ )
        {
          if ( SYMMETRIC )
          {
            if ( i > j )
              (*this)[ j * m + i ] = distribution( generator );
            else
              (*this)[ j * m + i ] = (*this)[ i * m + j ];
          }
          else
          {
            (*this)[ j * m + i ] = distribution( generator );
          }
        }
      }
    };

    template<bool SYMMETRIC = false>
    void rand() { rand<SYMMETRIC>( 0.0, 1.0 ); };

    void randn( T mu, T sd )
    {
      std::default_random_engine generator;
      std::normal_distribution<T> distribution( mu, sd );
      for ( std::size_t i = 0; i < m * n; i ++ )
      {
        (*this)[ i ] = distribution( generator );
      }
    };

    void randn() { randn( 0.0, 1.0 ); };

    template<bool USE_LOWRANK>
    void randspd( T a, T b )
    {
      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution( a, b );

      assert( m == n );

      if ( USE_LOWRANK )
      {
        hmlp::Data<T> X( ( std::rand() % n ) / 2 + 1, n );
        X.rand();
        xgemm
        (
          "T", "N",
          n, n, X.row(),
          1.0, X.data(), X.row(),
               X.data(), X.row(),
          0.0, this->data(), this->row()
        );
      }
      else // diagonal dominating
      {
        for ( std::size_t j = 0; j < n; j ++ )
        {
          for ( std::size_t i = 0; i < m; i ++ )
          {
            if ( i > j )
              (*this)[ j * m + i ] = distribution( generator );
            else
              (*this)[ j * m + i ] = (*this)[ i * m + j ];

            // Make sure diagonal dominated
            (*this)[ j * m + j ] += std::abs( (*this)[ j * m + i ] );
          }
        }
      }
    };

    template<bool USE_LOWRANK>
    void randspd() { randspd<USE_LOWRANK>( 0.0, 1.0 ); };

    std::size_t row() { return m; };

    std::size_t col() { return n; };

    void Print()
    {
      printf( "Data in %lu * %lu\n", m, n );
      hmlp::hmlp_printmatrix( m, n, this->data(), m );
    };

  private:

    std::size_t m;

    std::size_t n;

};
#endif // ifdef HMLP_USE_CUDA


template<class T, class Allocator = std::allocator<T> >
class CSC : public ReadWrite
{
  public:

    template<typename TINDEX>
    CSC( TINDEX m, TINDEX n, TINDEX nnz )
    {
      this->m = m;
      this->n = n;
      this->nnz = nnz;
      this->val.resize( nnz, 0.0 );
      this->row_ind.resize( nnz, 0 );
      this->col_ptr.resize( n + 1, 0 );
    };

    // Construct from three arrays.
    // val[ nnz ]
    // row_ind[ nnz ]
    // col_ptr[ n + 1 ]
    template<typename TINDEX>
    CSC( TINDEX m, TINDEX n, TINDEX nnz, T *val, TINDEX *row_ind, TINDEX *col_ptr ) 
      : CSC( m, n, nnz )
    {
      assert( val ); assert( row_ind ); assert( col_ptr );
      for ( size_t i = 0; i < nnz; i ++ )
      {
        this->val[ i ] = val[ i ];
        this->row_ind[ i ] = row_ind[ i ];
      }
      for ( size_t j = 0; j < n + 1; j ++ )
      {
        this->col_ptr[ j ] = col_ptr[ j ];
      }
    };

    ~CSC() {};

    template<bool SYMMETRIC = true, typename TINDEX>
    inline T operator()( TINDEX i, TINDEX j )
    {
      if ( SYMMETRIC && ( i < j  ) ) std::swap( i, j );
      size_t row_beg = col_ptr[ j ];
      size_t row_end = col_ptr[ j + 1 ];
      auto lower = std::lower_bound
                   ( 
                     row_ind.begin() + row_beg, 
                     row_ind.begin() + row_end,
                     i
                   );
      if ( *lower == i )
      {
        return val[ std::distance( row_ind.begin(), lower ) ];
      }
      else
      {
        return 0.0;
      }
    };

    template<bool SYMMETRIC = true, typename TINDEX>
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &imap, std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( imap.size(), jmap.size() );
      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < imap.size(); i ++ )
        {
          submatrix[ j * imap.size() + i ] = (*this)( imap[ i ], jmap[ j ] );
        }
      }
      return submatrix;
    }; 

    void Print()
    {
      for ( size_t j = 0; j < n; j ++ )
      {
        printf( "%8lu ", j );
      }
      printf( "\n" );
      for ( size_t i = 0; i < m; i ++ )
      {
        for ( size_t j = 0; j < n; j ++ )
        {
          printf( "% 3.1E ", (*this)( i, j ) );
        }
        printf( "\n" );
      }
    }; // end Print()


    /** 
     *  @brief Read matrix market format (ijv) format. Only lower triangular
     *         part is stored
     */ 
    template<bool ISZEROBASE>
    void readmtx( std::string &filename )
    {
      size_t m_mtx, n_mtx, nnz_mtx;

      std::vector<size_t> col_ind( nnz );

      // Read all tuples.
      std::cout << filename << std::endl;
      std::ifstream file( filename.data() );
      std::string line;
      if ( file.is_open() )
      {
        size_t nnz_count = 0;

        while ( std::getline( file, line ) )
        {
          if ( line.size() )
          {
            if ( line[ 0 ] != '%' )
            {
              std::istringstream iss( line );
              iss >> m_mtx >> n_mtx >> nnz_mtx;
              assert( this->m == m_mtx );
              assert( this->n == n_mtx );
              assert( this->nnz == nnz_mtx );
              break;
            }
          }
        }

        while ( std::getline( file, line ) )
        {
          std::istringstream iss( line );
          if ( !( iss >> row_ind[ nnz_count ] >> col_ind[ nnz_count ] >> val[ nnz_count ] ) )
          {
            printf( "line %lu has illegle format\n", nnz_count );
            break;
          }
          nnz_count ++;
        }
      }
      // Close the file.
      file.close();

      for ( size_t j = 0; j < nnz; j ++ )
      {
        if ( ISZEROBASE )
        {
          col_ptr[ col_ind[ j ] + 1 ] += 1;
        }
        else
        {
          col_ptr[ col_ind[ j ] ] += 1;
          row_ind[ j ] -= 1;
        }
      }
      for ( size_t j = 0; j < n; j ++ )
      {
        col_ptr[ j + 1 ] += col_ptr[ j ];
      }
      printf( "finish readmatrix %s\n", filename.data() );
    };


    std::size_t row() { return m; };

    std::size_t col() { return n; };

  private:

    std::size_t m;

    std::size_t n;

    std::size_t nnz;

    std::vector<T> val;

    std::vector<std::size_t> row_ind;
   
    std::vector<std::size_t> col_ptr;


}; // end class CSC


template<class T, class Allocator = std::allocator<T> >
class OOC : public ReadWrite
{
  public:

    template<typename TINDEX>
    OOC( TINDEX m, TINDEX n, std::string filename ) :
      file( filename.data(), std::ios::in|std::ios::binary|std::ios::ate )
    {
      this->m = m;
      this->n = n;
      this->filename = filename;
      if ( file.is_open() )
      {
        auto size = file.tellg();
        assert( size == m * n * sizeof(T) );
      }
      std::cout << filename << std::endl;
    };

    ~OOC()
    {
      if ( file.is_open() ) file.close();
      printf( "finish readmatrix %s\n", filename.data() );
    };


    template<typename TINDEX>
    inline T operator()( TINDEX i, TINDEX j )
    {
      T Kij;
      if ( !read_from_cache( i, j, &Kij ) )
      {
        if ( !read_from_disk( i, j, &Kij ) )
        {
          printf( "Accessing disk fail\n" );
          exit( 1 );
        }
      }
      return Kij;
    };

    template<typename TINDEX>
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &imap, std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( imap.size(), jmap.size() );
      for ( int j = 0; j < jmap.size(); j ++ )
      {
        for ( int i = 0; i < imap.size(); i ++ )
        {
          submatrix[ j * imap.size() + i ] = (*this)( imap[ i ], jmap[ j ] );
        }
      }
      return submatrix;
    }; 

    std::size_t row() { return m; };

    std::size_t col() { return n; };

  private:

    template<typename TINDEX>
    bool read_from_cache( TINDEX i, TINDEX j, T *Kij )
    {
      // Need some method to search in the caahe.
      return false;
    };

    /**
     *  @brief We need a lock here.
     */ 
    template<typename TINDEX>
    bool read_from_disk( TINDEX i, TINDEX j, T *Kij )
    {
      if ( file.is_open() )
      {
        //printf( "try %4lu, %4lu ", i, j );
        #pragma omp critical
        {
          file.seekg( ( j * m + i ) * sizeof(T) );
          file.read( (char*)Kij, sizeof(T) );
        }
        // printf( "read %4lu, %4lu, %E\n", i, j, *Kij );
        // TODO: Need some method to cache the data.
        return true;
      }
      else
      {
        return false;
      }
    };

    std::size_t m;

    std::size_t n;

    std::string filename;

    std::ifstream file;


}; // end class OOC

}; // end namespace hmlp

#endif //define DATA_HPP
