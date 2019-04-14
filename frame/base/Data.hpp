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

#ifndef DATA_HPP
#define DATA_HPP

/** Use mmap as simple out-of-core solution. */
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>

/** Use STL and inherit vector<T>. */
#include <cassert>
#include <typeinfo>
#include <algorithm>
#include <random>
#include <chrono>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <vector>
#include <deque>
#include <map>
#include <string>

/** std::istringstream */
#include <iostream>
#include <fstream>
#include <sstream>

/** Use HMLP support (device, read/write). */
#include <base/device.hpp>
#include <base/runtime.hpp>
#include <base/util.hpp>


/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif // ifdef HMLP}_MIC_AVX512

/** gpu related */
#ifdef HMLP_USE_CUDA
#include <hmlp_gpu.hpp>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

template<class T>
class managed_allocator
{
  public:

    T* allocate( size_t n )
    {
      T* result = nullptr;

      cudaError_t error = cudaMallocManaged( 
          &result, n * sizeof(T), cudaMemAttachGlobal );

      if ( error != cudaSuccess )
      {
        throw thrust::system_error( error, thrust::cuda_category(), 
            "managed_allocator::allocate(): cudaMallocManaged" );
      }

      return result;
    }; /** end allocate() */

    void deallocate( T* ptr, size_t )
    {
      cudaError_t error = cudaFree( ptr );

      if ( error != cudaSuccess )
      {
        throw thrust::system_error( error, thrust::cuda_category(), 
            "managed_allocator::deallocate(): cudaFree" );
      }
    }; /** end deallocate() */
};
#endif // ifdef HMLP_USE_CUDA





/** debug flag */
#define DEBUG_DATA 1


using namespace std;




namespace hmlp
{

template<typename T, class Allocator = std::allocator<T>>
class Tensor : public ReadWrite, public std::vector<T, Allocator>
{
  public:

    Tensor() : 
      std::vector<T, Allocator>() 
    {};

    Tensor( int32_t num_modes, const uint64_t extent[], const int64_t stride[] ) : 
        num_modes_( num_modes ), 
        extent_( extent, extent + num_modes ), 
        stride_( stride, stride + num_modes )
    {
      try
      {
        allocate_();
      }
      catch ( const std::exception & e )
      {
        HANDLE_EXCEPTION( e );
      }
    };

    Tensor( 
        int32_t num_modes, 
        const std::vector<uint64_t> & extent,
        const std::vector<int64_t> & stride ) :
      num_modes( num_modes ), 
      extent_( extent ), 
      stride_( stride )
    {
      try
      {
        allocate_();
      }
      catch ( const std::exception & e )
      {
        HANDLE_EXCEPTION( e );
      }
    };

    int32_t getNumModes() const noexcept 
    {
      return num_modes_;
    };

    uint64_t getExtent( int32_t mode_index ) const noexcept
    {
      if ( mode_index >= getNumModes() )
      {
        throw std::exception( "Too many modes in getExtent" );
      }
      return extent_[ mode_index ];
    }

    uint64_t getStride( int32_t mode_index ) const noexcept
    {
      if ( mode_index >= getNumModes() )
      {
        throw std::exception( "Too many modes in getStride" );
      }
      return stride_[ mode_index ];
    }

    template<typename... Args>
    T & operator() ( const Args & ... args )
    {
      auto offset = recuGetOffset_( 0, args );
      return (*this).at( offset );
    };

    template<typename... Args>
    T operator() ( const Args & ... args ) const
    {
      auto offset = recuGetOffset_( 0, args );
      return (*this).at( offset );
    };




  protected:

    int32_t num_modes_ = 0;

    std::vector<uint64_t> extent_;

    std::vector<uint64_t> stride_;

    std::vector<int8_t> mode_;

    void allocate_() 
    {
      uint64_t num_elements = 0;

      for ( int32_t mode_order = 0; mode_order < getNumModes(); mode_order ++ )
      {
        auto mode_extent = getExtent( mode_order );
        auto max_mode_index = ( mode_extent > 0 ) ? mode_extent - 1 : 0;
        num_elements += max_mode_index * getStride( mode_order );
      }

      (*this).resize( num_elements );
    }

    uint64_t recuGetOffset_( int32_t mode_index, uint64_t i ) const
    {
      return i * getStride( mode_index );
    }

    template<typename... Args>
    uint64_t recuGetOffset_( int32_t mode_index, uint64_t i, const Args & ... args ) const
    {
      return i * getStride( mode_index ) + recuGetOffset_( mode_index + 1, args ); 
    }
    

}; /* end class Tensor */


template<typename T, class Allocator = std::allocator<T>>
class Matrix : public hmlp::Tensor<T, Allocator>
{
  public:

    Matrix() :
      Tensor<T, Allocator>()
    {};

    Matrix( 
        const uint64_t height, 
        const uint64_t width ) :
      Tensor( 2, { height, width }, { (const uint64_t)1, height } ) 
    {};

    Matrix( 
        const uint64_t height, 
        const uint64_t width,
        const uint64_t row_stride,
        const uint64_t column_stride
        ) :
      Tensor( 2, { height, width }, { row_stride, column_stride } ) 
    {};

    uint64_t getHieght() const
    {
      return getExtent( 0 );
    }

    uint64_t getWidth() const 
    {
      return getExtent( 1 );
    }

    uint64_t getRowStride() const
    {
      return getStride( 0 );
    }

    uint64_t getColumnStride() const
    {
      return getStride( 1 );
    }

  protected:

}; /* end class Matrix */


#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#elif  HMLP_USE_CUDA
/** use pinned (page-lock) memory for NVIDIA GPUs */
template<class T, class Allocator = thrust::system::cuda::experimental::pinned_allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
class Data : public ReadWrite, public vector<T, Allocator>
#ifdef HMLP_USE_CUDA
/** inheritate the interface fot the host-device model. */
, public DeviceMemory<T>
#endif
{
  public:

    /** Default empty constructor. */
    Data() : vector<T, Allocator>() {};

    /** Copy constructor for hmlp::Data. */
    Data( const Data<T>& other_data ) : vector<T, Allocator>( other_data )
    {
      resize_( other_data.row(), other_data.col() );
    }

    /** TODO: Copy constructor for std::vector. */
    Data( size_t m, size_t n, const vector<T>& other_vector ) : vector<T, Allocator>( other_vector )
    {
      assert( other_vector.size() == m * n );
      resize( m, n );
    };

    Data( size_t m, size_t n ) 
    { 
      resize( m, n ); 
    };

    Data( size_t m, size_t n, const T initT ) 
    { 
      resize( m, n, initT ); 
    };

    Data( size_t m, size_t n, const std::string & filename ) : Data( m, n )
    {
      try 
      {
        HANDLE_ERROR( this->readBinaryFile( m, n, filename ) );
      }
      catch ( const std::exception & e )
      {
        HANDLE_EXCEPTION( e );
      }
    };

    void resize( size_t m, size_t n ) 
    { 
      resize_( m, n ); 
    };

    void resize( size_t m, size_t n, T initT ) 
    { 
      resize_( m, n, initT ); 
    };

    void reserve( size_t m, size_t n ) { reserve_( m, n ); };

    /**
     *  \brief Make it a 0-by-0 matrix and also call vector::clear() 
     *         and vector::shrink_to_fit().
     */ 
    void clear() 
    { 
      clear_(); 
    }; 

    hmlpError_t readBinaryFile( uint64_t m, uint64_t n, const std::string & filename )
    {
      if ( this->m != m || this->n != n || this->size() != m * n )
      {
        std::cerr << "ERROR: mismatching height or width" << std::endl;
        return HMLP_ERROR_INVALID_VALUE;
      }
      if ( filename.size() == 0 )
      {
        std::cerr << "ERROR: the filename cannot be empty" << std::endl;
        return HMLP_ERROR_INVALID_VALUE;
      }
      else
      {
        /* Print out filename. */
        std::cout << filename << endl;
      }

      std::ifstream file( filename.data(), std::ios::in|std::ios::binary|std::ios::ate );

      if ( !file.is_open() )
      {
        std::cerr << "ERROR: fail to open " << filename << std::endl;
        return HMLP_ERROR_INVALID_VALUE;
      }

      auto size = file.tellg();
      if ( size != m * n * sizeof(T) )
      {
        std::cerr << "ERROR: only " << file.tellg() << " bytes, expecting " << m * n * sizeof(T) << std::endl;
        file.close();
        return HMLP_ERROR_INVALID_VALUE;
      }

      file.seekg( 0, std::ios::beg );
      if ( !file.read( (char*)this->data(), size ) )
      {
        std::cerr << "ERROR: only " << file.gcount() << " bytes, expecting " << m * n * sizeof(T) << std::endl;
      }
      file.close();

      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t writeBinaryFile( const std::string & filename )
    {
      if ( filename.size() == 0 )
      {
        std::cerr << "ERROR: the filename cannot be empty" << std::endl;
        return HMLP_ERROR_INVALID_VALUE;
      }
      else
      {
        /* Print out filename. */
        std::cout << filename << endl;
      }
      /* Create an ofstream. */
      std::ofstream file( filename.data(), std::ios::out | std::ios::binary );
      if ( !file.is_open() )
      {
        std::cerr << "ERROR: fail to open " << filename << std::endl;
        return HMLP_ERROR_INVALID_VALUE;
      }
      file.write( (char*)(this->data()), this->size() * sizeof(T) );
      
      return HMLP_ERROR_SUCCESS;
    };

    template<int SKIP_ATTRIBUTES = 0, bool TRANS = false>
    hmlpError_t readmtx( uint64_t height, uint64_t width, const std::string & filename,
        uint64_t num_skipped_lines = 0, bool transpose = false )
    {
      if ( this->getHeight() != height || this->getWidth() != width )
      {
        std::cerr << "ERROR: mismatching height or width" << std::endl;
        return HMLP_ERROR_INVALID_VALUE;
      }
      if ( filename.size() == 0 )
      {
        std::cerr << "ERROR: the filename cannot be empty" << std::endl;
        return HMLP_ERROR_INVALID_VALUE;
      }
      else
      {
        /* Print out filename. */
        std::cout << filename << endl;
      }

      std::ifstream file( filename.data() );
      std::string line;

      if ( !file.is_open() )
      {
        std::cerr << "ERROR: fail to open " << filename << std::endl;
        return HMLP_ERROR_INVALID_VALUE;
      }

      /* Line counter starting from zero. */
      uint64_t num_lines = 0;
      uint64_t expected_num_lines = num_skipped_lines + ( transpose ? width : height );
      /* Loop over all lines. */
      while ( getline( file, line ) )
      {
        if ( num_lines >= expected_num_lines )
        {
          break;
        }
        if ( num_lines < num_skipped_lines )
        {
          continue;
        }
        /* Replace all ',' and ';' with '\n' */
        std::replace( line.begin(), line.end(), ',', '\n' );
        std::replace( line.begin(), line.end(), ';', '\n' );

        std::istringstream iss( line );

        for ( uint64_t i = 0; i < ( transpose ? height : width ); i ++ )
        {
          T element;
          if ( !( iss >> element ) )
          {
            file.close();
            std::cerr << "ERROR: line " << num_lines << " does not have enough elements " << i << std::endl; 
            return HMLP_ERROR_INVALID_VALUE;
          }
          if ( transpose )
          {
              (*this)( i, num_lines - num_skipped_lines ) = element;
          }
          else
          {
              (*this)( num_lines - num_skipped_lines, i ) = element;
          }
        }
        num_lines ++;
      }
      file.close();
      if ( num_lines < expected_num_lines )
      {
        std::cerr << "ERROR: does not have enough lines " << num_lines << std::endl; 
        return HMLP_ERROR_INVALID_VALUE;
      }
      return HMLP_ERROR_SUCCESS;
    };



    tuple<size_t, size_t> shape() { return make_tuple( m, n ); };

   
    T* rowdata( size_t i ) 
    {
      assert( i < m );
      return ( this->data() + i );
    };

    T* columndata( size_t j )
    {
      assert( j < n );
      return ( this->data() + j * m );
    };

    T getvalue( size_t i ) { return (*this)[ i ]; };

    void setvalue( size_t i, T v ) { (*this)[ i ] = v; };

    T getvalue( size_t i, size_t j ) { return (*this)( i, j ); };

    void setvalue( size_t i, size_t j, T v ) { (*this)( i, j ) = v; };

    /** ESSENTIAL: return number of coumns */
    size_t row() const noexcept { return m; };
    uint64_t getHeight() const noexcept 
    {
      return m;
    };

    /** ESSENTIAL: return number of rows */
    size_t col() const noexcept { return n; };
    uint64_t getWidth() const noexcept 
    {
      return n;
    };


    Data<T> operator + ( const Data<T>& b ) const
    {
      if ( b.row() != this->row() || b.col != this->col() )
      {
        throw std::invalid_argument( "Operator Data::operator + must have the same shape" );
      }
      auto a = (*this);
      for ( size_t i = 0; i < a.size(); i ++ )
      {
        a[ i ] += b[ i ];
      }
      return a;
    }

    Data<T> operator - ( const Data<T>& b ) const
    {
      if ( b.row() != this->row() || b.col() != this->col() )
      {
        throw std::invalid_argument( "Operator Data::operator - must have the same shape" );
      }
      auto a = (*this);
      for ( size_t i = 0; i < a.size(); i ++ )
      {
        a[ i ] -= b[ i ];
      }
      return a;
    }

    double squaredFrobeniusNorm() const noexcept
    {
      double sqfnrm = 0;
      for ( auto v : *this ) 
      {
        sqfnrm += v * v;
      }
      return sqfnrm;
    }

    double sumOfSquaredError( const Data<T>& b ) const
    {
      auto c = (*this) - b;
      return c.squaredFrobeniusNorm();
    }

    double relativeError( const Data<T>& b )
    {
      auto c = (*this) - b;
      double c_nrm = c.squaredFrobeniusNorm();
      double b_nrm = b.squaredFrobeniusNorm();

      if ( b_nrm )
      {
        return std::sqrt( c_nrm ) / std::sqrt( b_nrm );
      }
      else
      {
        if ( c_nrm ) 
        {
          return 1;
        }
        else
        {
          return 0;
        }
      }
    };

    /** ESSENTIAL: return an element */
    T& operator()( size_t i, size_t j )       { return (*this)[ m * j + i ]; };
    T  operator()( size_t i, size_t j ) const { return (*this)[ m * j + i ]; };


    /** ESSENTIAL: return a submatrix */
    Data<T> operator()( const vector<size_t>& I, const vector<size_t>& J ) const
    {
      Data<T> KIJ( I.size(), J.size() );
      for ( int j = 0; j < J.size(); j ++ )
      {
        for ( int i = 0; i < I.size(); i ++ )
        {
          KIJ[ j * I.size() + i ] = (*this)[ m * J[ j ] + I[ i ] ];
        }
      }
      return KIJ;
    }; 

    /** ESSENTIAL: */
    pair<T, size_t> ImportantSample( size_t j )
    {
      size_t i = std::rand() % m;
      pair<T, size_t> sample( (*this)( i, j ), i );
      return sample; 
    };


    Data<T> operator()( const vector<size_t> &jmap )
    {
      Data<T> submatrix( m, jmap.size() );
      #pragma omp parallel for
      for ( int j = 0; j < jmap.size(); j ++ )
        for ( int i = 0; i < m; i ++ )
          submatrix[ j * m + i ] = (*this)[ m * jmap[ j ] + i ];
      return submatrix;
    };

    template<typename TINDEX>
    void GatherColumns( bool TRANS, vector<TINDEX> &jmap, Data<T> &submatrix )
    {
      if ( TRANS )
      {
        submatrix.resize( jmap.size(), m );
        for ( int j = 0; j < jmap.size(); j ++ )
          for ( int i = 0; i < m; i ++ )
            submatrix[ i * jmap.size() + j ] = (*this)[ m * jmap[ j ] + i ];
      }
      else
      {
        submatrix.resize( m, jmap.size() );
        for ( int j = 0; j < jmap.size(); j ++ )
          for ( int i = 0; i < m; i ++ )
            submatrix[ j * m + i ] = (*this)[ m * jmap[ j ] + i ];
      }
    }; 

    void setvalue( T value )
    {
      for ( auto it = this->begin(); it != this->end(); it ++ )
        (*it) = value;
    };

    template<bool SYMMETRIC = false>
    void rand( T a, T b )
    {
      default_random_engine generator;
      uniform_real_distribution<T> distribution( a, b );

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
    void rand() 
    { 
      rand<SYMMETRIC>( 0.0, 1.0 ); 
    };

    void randn( T mu, T sd )
    {
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator( seed );
      std::normal_distribution<T> distribution( mu, sd );
      for ( std::size_t i = 0; i < m * n; i ++ )
      {
        (*this)[ i ] = distribution( generator );
      }
    };

    void randn() { randn( 0.0, 1.0 ); };

    template<bool USE_LOWRANK=true>
    void randspd( T a, T b )
    {
      std::default_random_engine generator;
      std::uniform_real_distribution<T> distribution( a, b );

      assert( m == n );

      if ( USE_LOWRANK )
      {
        hmlp::Data<T> X( ( std::rand() % n ) / 10 + 1, n );
        X.randn( a, b );
        xgemm
        (
          "T", "N",
          n, n, X.row(),
          1.0, X.data(), X.row(),
               X.data(), X.row(),
          0.0, this->data(), this->row()
        );
      }
      else /** diagonal dominating */
      {
        for ( std::size_t j = 0; j < n; j ++ )
        {
          for ( std::size_t i = 0; i < m; i ++ )
          {
            if ( i > j )
              (*this)[ j * m + i ] = distribution( generator );
            else
              (*this)[ j * m + i ] = (*this)[ i * m + j ];

            /** Make sure diagonal dominated */
            (*this)[ j * m + j ] += std::abs( (*this)[ j * m + i ] );
          }
        }
      }
    };

    void randspd() 
    { 
      randspd<true>( 0.0, 1.0 ); 
    };

    bool HasIllegalValue()
    {
      for ( auto it = this->begin(); it != this->end(); it ++ )
      {
        if ( std::isinf( *it ) ) return true;
        if ( std::isnan( *it ) ) return true;
      }
      return false;
    };

    void Print()
    {
      printf( "Data in %lu * %lu\n", m, n );
      if ( m < 11 && n < 7 )
      {
        hmlp_printmatrix( m, n, this->data(), m );
      }
      else
      {
        hmlp_printmatrix( 10, 6, this->data(), m );
      }
    };

    void WriteFile( char *name )
    {
      FILE * pFile;
      pFile = fopen ( name, "w" );
      fprintf( pFile, "%s=[\n", name );
      for ( size_t i = 0; i < m; i ++ )
      {
        for ( size_t j = 0; j < n; j ++ )
        {
          fprintf( pFile, "%lf,", (*this)( i, j ) );
        }
        fprintf( pFile, ";\n" );
      }
      fprintf( pFile, "];\n" );
      fclose( pFile );
    };

    template<typename TINDEX>
    double flops( TINDEX na, TINDEX nb ) { return 0.0; };

#ifdef HMLP_USE_CUDA

    void CacheD( hmlp::Device *dev )
    {
      DeviceMemory<T>::CacheD( dev, this->size() * sizeof(T) );
    };

    void AllocateD( hmlp::Device *dev )
    {
      double beg = omp_get_wtime();
      DeviceMemory<T>::AllocateD( dev, m * n * sizeof(T) );
      double alloc_time = omp_get_wtime() - beg;
      printf( "AllocateD %5.3lf\n", alloc_time );
    };

    void FreeD( Device *dev )
    {
      DeviceMemory<T>::FreeD( dev, this->size() * sizeof(T) );
    };

    void PrefetchH2D( Device *dev, int stream_id )
    {
      double beg = omp_get_wtime();
      DeviceMemory<T>::PrefetchH2D
        ( dev, stream_id, m * n * sizeof(T), this->data() );
      double alloc_time = omp_get_wtime() - beg;
      //printf( "PrefetchH2D %5.3lf\n", alloc_time );
    };

    void PrefetchD2H( Device *dev, int stream_id )
    {
      DeviceMemory<T>::PrefetchD2H
        ( dev, stream_id, m * n * sizeof(T), this->data() );
    };

    void WaitPrefetch( Device *dev, int stream_id )
    {
      DeviceMemory<T>::Wait( dev, stream_id );
    };

    void FetchH2D( Device *dev )
    {
      DeviceMemory<T>::FetchH2D( dev, m * n * sizeof(T), this->data() );
    };

    void FetchD2H( Device *dev )
    {
      DeviceMemory<T>::FetchD2H( dev, m * n * sizeof(T), this->data() );
    };
#endif

  private:

    void resize_( size_t m, size_t n )
    {
      vector<T, Allocator>::resize( m * n );
      this->m = m; 
      this->n = n; 
    };

    void resize_( size_t m, size_t n, T initT )
    {
      vector<T, Allocator>::resize( m * n, initT );
      this->m = m; this->n = n; 
    };

    void reserve_( size_t m, size_t n )
    {
      vector<T, Allocator>::reserve( m * n );
    }

    void clear_()
    {
      vector<T, Allocator>::clear();
      vector<T, Allocator>::shrink_to_fit();
      this->m = 0; 
      this->n = 0;
    }

    size_t m = 0;

    size_t n = 0;

}; /** end class Data */






template<typename T, class Allocator = std::allocator<T> >
class SparseData : public ReadWrite
{
  public:

    /** (Default) constructor. */
    SparseData( size_t m = 0, size_t n = 0, size_t nnz = 0, bool issymmetric = true )
    {
      Resize( m, n, nnz, issymmetric );
    };

    /** Adjust the storage size. */
    void Resize( size_t m, size_t n, size_t nnz, bool issymmetric )
    {
      this->m = m;
      this->n = n;
      this->nnz = nnz;
      this->val.resize( nnz, 0.0 );
      this->row_ind.resize( nnz, 0 );
      this->col_ptr.resize( n + 1, 0 );
      this->issymmetric = issymmetric;
    }; /** end Resize() */

    /** Construct from three arrays: val[ nnz ],row_ind[ nnz ], and col_ptr[ n + 1 ]. */
    void fromCSC( size_t m, size_t n, size_t nnz, bool issymmetric,
        const T *val, const size_t *row_ind, const size_t *col_ptr ) 
    {
      Resize( m, n, nnz, issymmetric );
      for ( size_t i = 0; i < nnz; i ++ )
      {
        this->val[ i ] = val[ i ];
        this->row_ind[ i ] = row_ind[ i ];
      }
      for ( size_t j = 0; j < n + 1; j ++ )
      {
        this->col_ptr[ j ] = col_ptr[ j ];
      }
    }; /** end fromCSC()*/

    /** Retrive an element K( i, j ).  */
    T operator () ( size_t i, size_t j ) const
    {
      if ( issymmetric && i < j ) std::swap( i, j );
      auto row_beg = col_ptr[ j ];
      auto row_end = col_ptr[ j + 1 ];
      /** Early return if there is no nonzero entry in this column. */
      if ( row_beg == row_end ) return 0;
      /** Search (BST) for row indices. */
      auto lower = find( row_ind.begin() + row_beg, row_ind.begin() + row_end - 1, i );
      //if ( lower != row_ind.end() ) printf( "lower %lu, i %lu, j %lu, row_beg %lu, row_end %lu\n", 
      //    *lower, i, j, row_beg, row_end ); fflush( stdout );
      /** If the lower bound matches, then return the value. */
      if ( *lower == i ) return val[ distance( row_ind.begin(), lower ) ];
      /** Otherwise, return 0. */
      return 0;
    }; /** end operator () */

    /** Retrive a subblock K( I, J ).*/
    Data<T> operator()( const vector<size_t> &I, const vector<size_t> &J ) const
    {
      Data<T> KIJ( I.size(), J.size() );
      /** Evaluate Kij element by element. */
      for ( size_t j = 0; j < J.size(); j ++ )
        for ( size_t i = 0; i < I.size(); i ++ )
          KIJ( i, j ) = (*this)( I[ i ], J[ j ] );
      /** Return submatrix KIJ. */
      return KIJ;
    }; /** end operator () */ 

    size_t ColPtr( size_t j ) { return col_ptr[ j ]; };

    size_t RowInd( size_t offset ) { return row_ind[ offset ]; };

    T Value( size_t offset ) { return val[ offset ]; };

    pair<T, size_t> ImportantSample( size_t j )
    {
      size_t offset = col_ptr[ j ] + rand() % ( col_ptr[ j + 1 ] - col_ptr[ j ] );
      pair<T, size_t> sample( val[ offset ], row_ind[ offset ] );
      return sample; 
    };

    void Print()
    {
      for ( size_t j = 0; j < n; j ++ ) printf( "%8lu ", j );
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
    void readmtx( string &filename )
    {
      size_t m_mtx, n_mtx, nnz_mtx;

      vector<deque<size_t>> full_row_ind( n );
      vector<deque<T>> full_val( n );

      // Read all tuples.
      printf( "%s ", filename.data() ); fflush( stdout );
      ifstream file( filename.data() );
      string line;
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

            if ( LOWERTRIANGULAR && i > j  )
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


    size_t row() { return m; };

    size_t col() { return n; };
    
    template<typename TINDEX>
    double flops( TINDEX na, TINDEX nb ) { return 0.0; };

  private:

    size_t m = 0;

    size_t n = 0;

    size_t nnz = 0;

    bool issymmetric = false;

    vector<T, Allocator> val;

    vector<size_t> row_ind;
   
    vector<size_t> col_ptr;

}; /** end class CSC */



template<class T, class Allocator = std::allocator<T> >
class OOCData : public ReadWrite
{
  public:

    OOCData() {};

    OOCData( size_t m, size_t n, string filename ) 
    { 
      HANDLE_ERROR( initFromFile( m, n, filename ) ); 
    };

    ~OOCData()
    {
      /** Unmap */
      int rc = munmap( mmappedData, m * n * sizeof(T) );
      assert( rc == 0 );
      close( fd );
      printf( "finish readmatrix %s\n", filename.data() );
    };

    hmlpError_t initFromFile( size_t m, size_t n, std::string filename )
    {
      this->m = m;
      this->n = n;
      this->filename = filename;

      /* Get the file size in bytes. */
      uint64_t file_size = (uint64_t)m * (uint64_t)n * sizeof(T);

      /* Open the file */
      fd = open( filename.data(), O_RDONLY, 0 ); 
      /* Return error if fail to open the file. */
      if ( fd == -1 )
      {
        fprintf( stderr, "fail to open %s\n", filename.data() );
        return HMLP_ERROR_INVALID_VALUE;
      }
#ifdef __APPLE__
      mmappedData = (T*)mmap( nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0 );
#else /* Assume Linux */
      mmappedData = (T*)mmap( nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0 );
#endif
      /* Return error if fail to map to the file. */
      if ( mmappedData != MAP_FAILED )
      {
        fprintf( stderr, "mmap %s with %lu bytes failure\n", filename.data(), file_size );
        return HMLP_ERROR_ALLOC_FAILED;
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    };

    T operator()( size_t i, size_t j ) const 
    {
      assert( i < m && j < n );
      return mmappedData[ j * m + i ];
    };

    Data<T> operator()( const vector<size_t>& I, const vector<size_t>& J ) const 
    {
      Data<T> KIJ( I.size(), J.size() );
      for ( int j = 0; j < J.size(); j ++ )
        for ( int i = 0; i < I.size(); i ++ )
          KIJ[ j * I.size() + i ] = (*this)( I[ i ], J[ j ] );
      return KIJ;
    }; 

    template<typename TINDEX>
    pair<T, TINDEX> ImportantSample( TINDEX j )
    {
      TINDEX i = std::rand() % m;
      pair<T, TINDEX> sample( (*this)( i, j ), i );
      return sample; 
    };

    size_t row() const noexcept 
    { 
      return m; 
    };

    size_t col() const noexcept { return n; };

    template<typename TINDEX>
    double flops( TINDEX na, TINDEX nb ) { return 0.0; };

  protected:

    /** Whether the data has been initialized? */
    bool is_init_ = false;

    size_t m = 0;

    size_t n = 0;

    std::string filename;

    /** Use mmap */
    T *mmappedData = nullptr;

    int fd = -1;

}; /** end class OOCData */







}; /** end namespace hmlp */

#endif /** define DATA_HPP */
