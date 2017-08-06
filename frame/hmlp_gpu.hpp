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


#ifndef HMLP_GPU_HPP
#define HMLP_GPU_HPP

#include <cassert>
#include <set>
#include <map>

#include <hmlp_runtime.hpp>

/** CUDA header files */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#define NUM_CUBLAS_HANDLE 8
#define NUM_CUDA_STREAM 10

namespace hmlp
{
namespace gpu
{


class Nvidia : public hmlp::Device
{
  public:

    Nvidia( int device_id )
    {
      printf( "Setup device %d\n", device_id );
      if ( cudaSetDevice( device_id ) )
      {
        int device_count = 0;
        cudaGetDeviceCount( &device_count );
        printf( "cudaSetDevice(), fail to set device %d / %d\n", 
            device_id, device_count );
        exit( 1 );
      }
      else
      {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, device_id );
        this->device_id = device_id;
        this->devicetype = hmlp::DeviceType::NVIDIA_GPU;
        this->name = std::string( prop.name );
        this->total_memory = prop.totalGlobalMem;
        this->memory_left = prop.totalGlobalMem;

        for ( int i = 0; i < NUM_CUDA_STREAM; i ++ )
        {
          if ( cudaStreamCreate( &(stream[ i ] ) ) )
            printf( "cudaStreamCreate(), fail on device %d\n", device_id );
        }

        for ( int i = 0; i < NUM_CUBLAS_HANDLE; i ++ )
        {
          if ( cublasCreate( &handle[ i ] ) )
            printf( "cublasCreate(), fail on device %d\n", device_id );
          if ( cublasSetStream( handle[ i ], stream[ i ] ) )
            printf( "cublasSetStream(), fail on device %d\n", device_id );
        }
        std::cout << name << ", " << this->total_memory / 1E+9 << "GB" << std::endl;

        /** allocate workspace */
        work_d = (char*)malloc( work_size );

        /** cache */
        cache.Setup( this );

      }
    };

    ~Nvidia()
    {
      for ( int i = 0; i < NUM_CUBLAS_HANDLE; i ++ )
      {
        if ( cublasDestroy( handle[ i ] ) )
          printf( "cublasDestroy(), fail on device %d\n", device_id );
      }
    };


    void prefetchd2h( void *ptr_h, void *ptr_d, size_t size, int stream_id )
    {
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }
      if ( cudaMemcpyAsync( ptr_h, ptr_d, size, cudaMemcpyDeviceToHost, stream[ stream_id ] ) )
      {
        exit( 1 );
      }
    };

    void prefetchh2d( void *ptr_d, void *ptr_h, size_t size, int stream_id )
    {
      if ( cudaSetDevice( device_id ) )
      {
        printf( "cudaSetDevice(), fail to set device %d\n", device_id );
        exit( 1 );
      }

      struct cudaPointerAttributes attribute;

      if ( cudaPointerGetAttributes ( &attribute, ptr_h ) )
      {
        printf( "cudaPointerGetAttributes(), fail on device %d\n", device_id );
        exit( 1 );
      }

      if ( attribute.isManaged )
      {
        printf( "ptr_h is managed\n" );
        if ( cudaMemPrefetchAsync( ptr_d, size, device_id, stream[ stream_id ] ) )
        {
          printf( "cudaMemPrefetchAsync(), fail on device %d\n", device_id );
        }
      }



      if ( cudaMemcpyAsync( ptr_d, ptr_h, size, cudaMemcpyHostToDevice, stream[ stream_id ] ) )
      //if ( cudaMemcpy( ptr_d, ptr_h, size, cudaMemcpyHostToDevice ) )
      {
        printf( "cudaMemcpyAsync(), %lu bytes fail to device %d\n", size, device_id );
        exit( 1 );
      }
    };

    void waitexecute()
    {
      for ( int stream_id = 0; stream_id < NUM_CUDA_STREAM; stream_id ++ )
      {
        wait( stream_id );
      }
    };

    void wait( int stream_id )
    {
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }
      if ( cudaStreamSynchronize( stream[ stream_id ] ) )
      {
        exit( 1 );
      }
    };

    size_t get_memory_left()
    {
      return memory_left;
    };

    void* malloc( size_t size )
    {
      void *ptr_d = NULL;
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }
      if ( size + 268435456 < memory_left )
      {
        memory_left -= size;
        if ( cudaMalloc( (void**)&ptr_d, size ) )
        {
          printf( "cudaMalloc() error\n");
          exit( 1 );
        }
        cudaMemset( ptr_d, 0, size );
      }
      else
      {
        printf( "not allocated, only %5.2lf GB left\n", memory_left / 1E+9 );
      }
      return ptr_d;
    };

    void malloc( void *ptr_d, size_t size )
    {
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }

      if ( size + 1073741824 < memory_left )
      {
        memory_left -= size;
        if ( cudaMalloc( (void**)&ptr_d, size ) )
        {
          exit( 1 );
        }
      }
      else
      {
        printf( "not allocated, only %5.2lf GB left\n", memory_left / 1E+9 );
      }
    };

    char *workspace()
    {
      return work_d;
    };

    void free( void *ptr_d, size_t size )
    {
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }
      if ( ptr_d ) 
      {
        if ( cudaFree( ptr_d ) )
        {
          exit( 1 );
        }
        memory_left += size;
      }
      else
      {
        printf( "try to free a null device pointer\n" );
      }
      printf( "free %lu memory_left %5.2lfGB\n", size, memory_left / 1E+9 );
    };

    cudaStream_t &getstream( int stream_id )
    {
      return stream[ stream_id ];
    }

    cublasHandle_t &gethandle( int stream_id )
    {
      return handle[ stream_id ];
    };

    //Cache<8> cache;

  private:

    int device_id;

    /** use 10 device streams for asynchronous execution */
    cudaStream_t stream[ NUM_CUDA_STREAM ];

    /** use 8 cublas handles */
    cublasHandle_t handle[ NUM_CUBLAS_HANDLE ];

    char *work_d = NULL;

    size_t work_size = 1073741824;

    size_t total_memory = 0;

    size_t memory_left = 0;

};



}; // end namespace gpu
}; // end namespace hmlp

#endif // define HMLP_GPU_HPP
