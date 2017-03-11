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

        for ( int i = 0; i < 10; i ++ )
        {
          if ( cudaStreamCreate( &(stream[ i ] ) ) )
            printf( "cudaStreamCreate(), fail on device %d\n", device_id );
        }

        for ( int i = 0; i < 8; i ++ )
        {
          if ( cublasCreate( &handle[ i ] ) )
            printf( "cublasCreate(), fail on device %d\n", device_id );
          if ( cublasSetStream( handle[ i ], stream[ i ] ) )
            printf( "cublasSetStream(), fail on device %d\n", device_id );
        }
        std::cout << name << std::endl;
      }
    };

    ~Nvidia()
    {
      for ( int i = 0; i < 8; i ++ )
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
      if ( cudaMemcpyAsync( ptr_d, ptr_h, size, cudaMemcpyHostToDevice, stream[ stream_id ] ) )
      //if ( cudaMemcpy( ptr_d, ptr_h, size, cudaMemcpyHostToDevice ) )
      {
        printf( "cudaMemcpyAsync(), %lu bytes fail to device %d\n", size, device_id );
        exit( 1 );
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

    template<typename T>
    T* malloc( size_t size )
    {
      T *ptr_d = NULL;
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }
      if ( cudaMalloc( (void**)&ptr_d, size ) )
      {
        exit( 1 );
      }
      return ptr_d;
    };

    template<typename T>
    void malloc( T *ptr_d, size_t size )
    {
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }
      if ( cudaMalloc( (void**)&ptr_d, size ) )
      {
        exit( 1 );
      }
    };


    template<typename T>
    void free( T *ptr_d )
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
      }
    };
  
    cublasHandle_t &gethandle( int stream_id )
    {
      return handle[ stream_id ];
    };

  private:

    int device_id;

    /** use 10 device streams for asynchronous execution */
    cudaStream_t stream[ 10 ];

    /** use 8 cublas handles */
    cublasHandle_t handle[ 8 ];
};

/**
 *  @brief 
 *
 */ 
template<class T>
class DeviceMemory
{
  public:

    DeviceMemory() 
    {
      host = &(hmlp_get_runtime_handle()->host);
      distribution.insert( host );
    };

    void AllocateD( hmlp::Device *dev, size_t size )
    {
      size *= sizeof(T);
      
      if ( !device_map.count( dev ) )
      {
        T *ptr_d = NULL;
        cudaMalloc( (void**)&ptr_d, size );
        device_map[ dev ] = ptr_d; 
      }
    };

    /** */
    void PrefetchH2D( hmlp::Device *dev, int stream_id, size_t size, T* ptr_h )
    {
      size *= sizeof(T);

      if ( !distribution.count( dev ) )
      {
        //printf( "target device does not have the latest copy.\n" );
        if ( !device_map.count( dev ) )
        {
          //printf( "allocate %lu bytes on %s\n", size, dev->name.data() );
          T *ptr_d;
          cudaMalloc( (void**)&ptr_d, size );
          //dev->malloc( ptr_d, size );
          device_map[ dev ] = ptr_d; 
        }
        //printf( "memcpy H2D\n" );
        dev->prefetchh2d( device_map[ dev ], ptr_h, size, stream_id );
        /** TODO: maybe update the distribution here? */
        //printf( "redistribute\n" );
        Redistribute<false>( dev );
      }
      else /** the device has the latest copy */
      {
        assert( device_map.find( dev ) != device_map.end() );
      }
    };

    /** if host does not have the latest copy */
    void PrefetchD2H( hmlp::Device *dev, int stream_id, size_t size, T* ptr_h )
    {
      size *= sizeof(T);

      if ( !distribution.count( host ) )
      {
        //printf( "host does not have the latest copy.\n" );
        assert( device_map.count( dev ) );
        //printf( "memcpy D2H\n" );
        dev->prefetchd2h( ptr_h, device_map[ dev ], size, stream_id );
        //printf( "redistribute\n" );
        Redistribute<false>( host );
      }
      else /** the host has the latest copy */
      {
        assert( device_map.count( host ) );
      }
    };


    /** */
    void Wait( hmlp::Device *dev, int stream_id )
    {
      dev->wait( stream_id );
    };

    /** */
    template<bool OVERWRITE>
    void Redistribute( hmlp::Device *dev )
    {
      assert( dev );
      if ( OVERWRITE ) distribution.clear();
      distribution.insert( dev );
    };

    bool is_up_to_date( hmlp::Device *dev )
    {
      return distribution.count( dev );
    };

    T* device_data( hmlp::Device *dev )
    {
      auto it = device_map.find( dev );
      if ( it == device_map.end() )
      {
        printf( "no device pointer for the target device\n" );
        return NULL;
      }
      return device_map[ dev ];
    };

  private:

    hmlp::Device *host = NULL;

    /** map a device to its data pointer */
    std::map<hmlp::Device*, T*> device_map;
   
    /** distribution */
    std::set<hmlp::Device*> distribution;

}; // end class DeviceMemory

}; // end namespace gpu
}; // end namespace hmlp

#endif // define HMLP_GPU_HPP
