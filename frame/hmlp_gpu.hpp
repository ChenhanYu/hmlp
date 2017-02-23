#ifndef HMLP_GPU_HPP
#define HMLP_GPU_HPP

#include <cassert>

/** CUDA header files */
#include <cuda_runtime.h>
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
        if ( cudaStreamCreate( &(stream[ 0 ]) ) )
        {
          printf( "cudaStreamCreate(), fail on device %d\n", device_id );
        }
        if ( cudaStreamCreate( &(stream[ 1 ]) ) )
        {
          printf( "cudaStreamCreate(), fail on device %d\n", device_id );
        }
        std::cout << name << std::endl;
      }
    };

    ~Nvidia()
    {
    };


    void prefetchd2h( void *ptr_h, void *ptr_d, size_t size )
    {
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }
      if ( cudaMemcpyAsync( ptr_h, ptr_d, size, cudaMemcpyDeviceToHost, stream[ 0 ] ) )
      {
        exit( 1 );
      }
    };

    void prefetchh2d( void *ptr_d, void *ptr_h, size_t size )
    {
      if ( cudaSetDevice( device_id ) )
      {
        exit( 1 );
      }
      if ( cudaMemcpyAsync( ptr_d, ptr_h, size, cudaMemcpyHostToDevice, stream[ 0 ] ) )
      {
        exit( 1 );
      }
    };

    void wait()
    {
      if ( cudaStreamSynchronize( stream[ 0 ] ) )
      {
        exit( 1 );
      }
    };

    template<typename T>
    T* malloc( size_t size )
    {
      T *ptr_d;
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
   
  private:

    int device_id;

    /** use 2 device stream for asynchronous execution */
    cudaStream_t stream[ 2 ];
};

/**
 *  @brief 
 *
 */ 
template<class T>
class DeviceMemory
{
  public:

    DeviceMemory() {};

    /** */
    void prefetchh2d( hmlp::Device *dev, size_t size, T* ptr_h )
    {
      if ( !distribution.count( dev ) )
      {
        auto it = device_map.find( dev );
        if ( it == device_map.end() )
        {
          T *ptr_d = reinterpret_cast<T*>( dev->malloc( size ) );
          device_map[ dev ] = ptr_d; 
        }
        dev->prefetchh2d( device_map[ dev ], ptr_h, size );
      }
      else /** the device has the latest copy */
      {
        assert( device_map.find( dev ) != device_map.end() );
      }
    };

    /** */
    void wait( hmlp::Device *dev )
    {
      dev->wait();
    };

    void fetch( hmlp::Device *dev )
    {
      prefetch( dev );
      wait( dev );
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

    hmlp::Device host;

    /** map a device to its data pointer */
    std::map<hmlp::Device*, T*> device_map;
   
    /** distribution */
    std::set<hmlp::Device*> distribution;

}; // end class DeviceMemory

}; // end namespace gpu
}; // end namespace hmlp

#endif // define HMLP_GPU_HPP
