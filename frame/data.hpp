#ifndef DATA_HPP
#define DATA_HPP

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <deque>
#include <map>
#include <iostream>
#include <fstream>
#include <random>

#include <hmlp.h>

#include <hmlp_blas_lapack.h>
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>

#ifdef HMLP_USE_CUDA
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif // ifdef HMLP_USE_CUDA




#define DEBUG_DATA 1

namespace hmlp
{
/*
#ifdef HMLP_USE_CUDA
template<class T>
class Data : public ReadWrite, public host_vector<T>
{
  publuc:

    Data() : m( 0 ), n( 0 ) {};

    Data( std::size_t m, std::size_t n ) : host_vector<T>( m * n )
    { 
      this->m = m;
      this->n = n;
    };

    Data( std::size_t m, std::size_t n, T initT ) : host_vector<T>( m * n, initT )
    { 
      this->m = m;
      this->n = n;
    };

    Data( std::size_t m, std::size_t n, std::string &filename ) : host_vector<T>( m * n )
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

    void resize( std::size_t m, std::size_t n )
    { 
      this->m = m;
      this->n = n;
      host_vector<T>::resize( m * n );
    };

    void resize( std::size_t m, std::size_t n, T initT )
    {
      this->m = m;
      this->n = n;
      host_vector<T>::resize( m * n, initT );
    };

    void reserve( std::size_t m, std::size_t n ) 
    {
      host_vector<T>::reserve( m * n );
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

    thrust::tuple<size_t, size_t> shape()
    {
      return thrust::make_tuple( m, n );
    };

    template<typename TINDEX>
    __host__ inline T operator()( TINDEX i, TINDEX j )
    {
      return (*this)[ m * j + i ];
    };

    template<typename TINDEX>
    __host__ inline hmlp::Data<T> operator()( host_vector<TINDEX> &imap, host_vector<TINDEX> &jmap )
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



  private:

    std::size_t m;

    std::size_t n;

    std::map<hmlp::Device*, T*> distribution;

}; // end class Data

#else
*/


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

    //enum Pattern : int { STAR = -1 };

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
	std::pair<T, TINDEX> ImportantSample( TINDEX j )
	{
	  TINDEX i = std::rand() % m;
	  std::pair<T, TINDEX> sample( (*this)( i, j ), i );
	  return sample; 
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


    void prefetch( hmlp::Device *dev )
    {
      if ( !distribution.count( dev ) )
      {
        auto it = device_map.find( dev );
        if ( it != device_map.end() )
        {
          //dev->prefetch( data(), *it, m * n * sizeof(T) );
        }
        else
        {
          //T *device_ptr = dev->allocate( m * n * sizeof(T) );
          //device_map[ dev ] = device_ptr; 
          //dev->prefetch( data(), device_ptr, m * n * sizeof(T) );
        }
      }
      else /** the device has the latest copy */
      {
        assert( device_map.find( dev ) != device_map.end() );
      }
    };

    void wait( hmlp::Device *dev )
    {
      //dev->wait();
    };

    void fetch( hmlp::Device *dev )
    {
      prefetch( dev );
      wait( dev );
    };

    T* device_data( hmlp::Device *dev )
    {
      auto it = device_map.find( dev );
      if ( it != device_map.end() )
      {
        return *it;
      }
      else
      {
        printf( "no device pointer for the target device\n" );
      }
    };


  private:

    std::size_t m;

    std::size_t n;

    /** map a device to its data pointer */
    std::map<hmlp::Device*, T*> device_map;
   
    /** distribution */
    std::set<hmlp::Device*> distribution;
};
//#endif // ifdef HMLP_USE_CUDA


template<bool SYMMETRIC, typename T, class Allocator = std::allocator<T> >
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
	// TODO: setup full_row_ind
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

    template<typename TINDEX>
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

    template<typename TINDEX>
	std::size_t ColPtr( TINDEX j )
	{
	  return col_ptr[ j ];
	}

    template<typename TINDEX>
	std::size_t RowInd( TINDEX offset )
	{
	  return row_ind[ offset ];
	};

    template<typename TINDEX>
	T Value( TINDEX offset )
	{
	  return val[ offset ];
	};

	template<typename TINDEX>
	std::pair<T, TINDEX> ImportantSample( TINDEX j )
	{
      size_t offset = col_ptr[ j ] + rand() % ( col_ptr[ j + 1 ] - col_ptr[ j ] );
	  std::pair<T, TINDEX> sample( val[ offset ], row_ind[ offset ] );
	  return sample; 
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
    template<bool LOWERTRIANGULAR, bool ISZEROBASE, bool IJONLY = false>
    void readmtx( std::string &filename )
    {
      size_t m_mtx, n_mtx, nnz_mtx;

	  std::vector<std::deque<size_t>> full_row_ind( n );
	  std::vector<std::deque<T>> full_val( n );

      // Read all tuples.
	  printf( "%s ", filename.data() ); fflush( stdout );
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
		  if ( nnz_count % ( nnz / 10 ) == 0 )
		  {
			printf( "%lu%% ", ( nnz_count * 100 ) / nnz ); fflush( stdout );
		  }

          std::istringstream iss( line );

		  size_t i, j;
		  T v;

		  if ( IJONLY )
		  {
            if ( !( iss >> i >> j ) )
            {
              printf( "line %lu has illegle format\n", nnz_count );
              break;
            }
			v = 1;
		  }
		  else
		  {
            if ( !( iss >> i >> j >> v ) )
            {
              printf( "line %lu has illegle format\n", nnz_count );
              break;
            }
		  }

		  if ( !ISZEROBASE )
		  {
			i -= 1;
			j -= 1;
		  }

		  if ( v != 0.0 )
		  {
            full_row_ind[ j ].push_back( i );
	        full_val[ j ].push_back( v );

		    if ( !SYMMETRIC && LOWERTRIANGULAR && i > j  )
		    {
			  full_row_ind[ i ].push_back( j );
			  full_val[ i ].push_back( v );
		    }
		  }
          nnz_count ++;
        }
		assert( nnz_count == nnz );
      }
	  printf( "Done.\n" ); fflush( stdout );
      // Close the file.
      file.close();

	  //printf( "Here nnz %lu\n", nnz );

	  // Recount nnz for the full storage.
	  size_t full_nnz = 0;
	  for ( size_t j = 0; j < n; j ++ )
	  {
		col_ptr[ j ] = full_nnz;
		full_nnz += full_row_ind[ j ].size();
	  }
	  nnz = full_nnz;
	  col_ptr[ n ] = full_nnz;
	  row_ind.resize( full_nnz );
	  val.resize( full_nnz );

	  //printf( "Here nnz %lu\n", nnz );

      //full_nnz = 0;
      //for ( size_t j = 0; j < n; j ++ )
      //{
	  //  for ( size_t i = 0; i < full_row_ind[ j ].size(); i ++ )
	  //  {
      //    row_ind[ full_nnz ] = full_row_ind[ j ][ i ];
      //    val[ full_nnz ] = full_val[ j ][ i ];
	  //    full_nnz ++;
	  //  }
      //}

	  //printf( "Close the file. Reformat.\n" );

      #pragma omp parallel for
      for ( size_t j = 0; j < n; j ++ )
      {
		for ( size_t i = 0; i < full_row_ind[ j ].size(); i ++ )
		{
          row_ind[ col_ptr[ j ] + i ] = full_row_ind[ j ][ i ];
          val[ col_ptr[ j ] + i ] = full_val[ j ][ i ];
		}
      }

      printf( "finish readmatrix %s\n", filename.data() ); fflush( stdout );
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

      //if ( file.is_open() )
      //{
      //  auto size = file.tellg();
      //  assert( size == m * n * sizeof(T) );
      //}

	  fd = open( filename.data(), O_RDONLY, 0 );
	  assert( fd != -1 );
	  mmappedData = (T*)mmap( NULL, m * n * sizeof(T), PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0 );
      assert( mmappedData != MAP_FAILED );

      std::cout << filename << std::endl;
    };

    ~OOC()
    {
      //if ( file.is_open() ) file.close();

	  int rc = munmap( mmappedData, m * n * sizeof(T) );
      assert( rc == 0 );
	  close( fd );

      printf( "finish readmatrix %s\n", filename.data() );
    };


    template<typename TINDEX>
    inline T operator()( TINDEX i, TINDEX j )
    {
      T Kij;

      //if ( !read_from_cache( i, j, &Kij ) )
      //{
      //  if ( !read_from_disk( i, j, &Kij ) )
      //  {
      //    printf( "Accessing disk fail\n" );
      //    exit( 1 );
      //  }
      //}

	  Kij = mmappedData[ j * m + i ];

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

	template<typename TINDEX>
	std::pair<T, TINDEX> ImportantSample( TINDEX j )
	{
	  TINDEX i = std::rand() % m;
	  std::pair<T, TINDEX> sample( (*this)( i, j ), i );
	  return sample; 
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

	// Use mmp
	T *mmappedData;

	int fd;

}; // end class OOC


template<bool SYMMETRIC, typename T, class Allocator = std::allocator<T> >
class Kernel : public ReadWrite
{
  public:

    template<typename TINDEX>
    Kernel( TINDEX m, TINDEX n, TINDEX d, kernel_s<T> &kernel )
    {
      this->m = m;
      this->n = n;
      this->d = d;
      this->kernel = kernel;

      if ( SYMMETRIC ) assert( m == n );
    };

    ~Kernel() {};

    void read( std::size_t n, std::size_t d, std::string &filename )
    {
      assert( SYMMETRIC );
      assert( this->n == n );
      assert( this->d == d );

      sources.resize( d, n );
      sources.read( d, n, filename );
    }

    void read( std::size_t m, std::size_t n, std::size_t d,
               std::string &sourcefilename, std::string &targetfilename )
    {
      assert( !SYMMETRIC );
      assert( this->m == m );
      assert( this->n == n );
      assert( this->d == d )

      sources.resize( d, n );
      sources.read( d, n, sourcefilename );

      targets.resize( d, m );
      targets.read( d, m, targetfilename );
    }

    template<typename TINDEX>
    inline T operator()( TINDEX i, TINDEX j )
    {
      T Kij = 0;

      switch ( this->kernel.type )
      {
        case KS_GAUSSIAN:
          {
            for ( TINDEX k = 0; k < this->d; k++ )
            {
              if ( SYMMETRIC )
              {
                Kij += sources[ i * this->d + k] * sources[ j * this->d + k ];
              }
              else
              {
                Kij += targets[ i * this->d + k] * sources[ j * this->d + k ];
              }
            }
            Kij = exp( this->kernel.scal * Kij );
            break;
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

    template<typename TINDEX>
    std::pair<T, TINDEX> ImportantSample( TINDEX j )
    {
      TINDEX i = std::rand() % m;
      std::pair<T, TINDEX> sample( (*this)( i, j ), i );
      return sample; 
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

    std::size_t row() { return m; };

    std::size_t col() { return n; };

    std::size_t dim() { return d; };


  private:

    std::size_t m;

    std::size_t n;

    std::size_t d;

    Data<T> sources;

    Data<T> targets;

    kernel_s<T> kernel;

}; // end class Kernel

}; // end namespace hmlp

#endif //define DATA_HPP
