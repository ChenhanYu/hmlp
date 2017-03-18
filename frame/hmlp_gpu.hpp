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


/**
 *  @brief 
 *
 */ 
//template<class T>
//class DeviceMemory
//{
//  /** allow other instance to access device_map */
//  friend class DeviceMemory;
//
//  public:
//
//    DeviceMemory() 
//    {
//      host = &(hmlp_get_runtime_handle()->host);
//      distribution.insert( host );
//    };
//
//    void Cache( hmlp::Device *dev, size_t size )
//    {
//      if ( cache )
//      {
//        /** has been cached before, check availability */
//        if ( cache->isCached( this ) ) return;
//      }
//      /** request a new cache location on the device */
//       
//    };
//
//    /** this will be the cache of target */
//    void asCache( DeviceMemory *target )
//    {
//      cache = target;
//    };
//
//    void AllocateD( hmlp::Device *dev, size_t size )
//    {
//      size *= sizeof(T);
//      
//      if ( !device_map.count( dev ) )
//      {
//        T *ptr_d = NULL;
//        //cudaMalloc( (void**)&ptr_d, size );
//        //dev->malloc( ptr_d, size );
//        ptr_d = (T*)dev->malloc( size );
//        if ( !ptr_d ) return;
//        device_map[ dev ] = ptr_d; 
//      }
//    };
//
//    /** free device memory, remove from the map and the distribution */
//    void FreeD( hmlp::Device *dev, size_t size )
//    {
//      if ( device_map.count( dev ) )
//      {
//        if ( !device_map[ dev ] )
//        {
//          printf( "NULL device ptr in device_map\n" ); fflush( stdout );
//        }
//
//        dev->free( device_map[ dev ], size );
//        device_map.erase( dev );
//        distribution.erase( dev );
//      }
//    };
//
//
//    /** */
//    void PrefetchH2D( hmlp::Device *dev, int stream_id, size_t size, T* ptr_h )
//    {
//      size *= sizeof(T);
//
//      if ( !distribution.count( dev ) )
//      {
//        //printf( "target device does not have the latest copy.\n" );
//        if ( !device_map.count( dev ) )
//        {
//          //printf( "allocate %lu bytes on %s\n", size, dev->name.data() );
//          T *ptr_d;
//          //cudaMalloc( (void**)&ptr_d, size );
//          //dev->malloc( ptr_d, size );
//          ptr_d = (T*)dev->malloc( size );
//          if ( !ptr_d ) return;
//          device_map[ dev ] = ptr_d; 
//        }
//        //printf( "memcpy H2D\n" );
//        dev->prefetchh2d( device_map[ dev ], ptr_h, size, stream_id );
//        /** TODO: maybe update the distribution here? */
//        //printf( "redistribute\n" );
//        Redistribute<false>( dev );
//      }
//      else /** the device has the latest copy */
//      {
//        assert( device_map.find( dev ) != device_map.end() );
//      }
//    };
//
//    /** if host does not have the latest copy */
//    void PrefetchD2H( hmlp::Device *dev, int stream_id, size_t size, T* ptr_h )
//    {
//      size *= sizeof(T);
//
//      if ( !distribution.count( host ) )
//      {
//        //printf( "host does not have the latest copy.\n" );
//        assert( device_map.count( dev ) );
//        //printf( "memcpy D2H\n" );
//        dev->prefetchd2h( ptr_h, device_map[ dev ], size, stream_id );
//        //printf( "redistribute\n" );
//        Redistribute<false>( host );
//      }
//      else /** the host has the latest copy */
//      {
//        assert( device_map.count( host ) );
//      }
//    };
//
//
//    /** */
//    void Wait( hmlp::Device *dev, int stream_id )
//    {
//      dev->wait( stream_id );
//    };
//
//    /** */
//    template<bool OVERWRITE>
//    void Redistribute( hmlp::Device *dev )
//    {
//      assert( dev );
//      if ( OVERWRITE ) distribution.clear();
//      distribution.insert( dev );
//    };
//
//    bool is_up_to_date( hmlp::Device *dev )
//    {
//      return distribution.count( dev );
//    };
//
//    T* device_data( hmlp::Device *dev )
//    {
//      if ( cache )
//      {
//        cache->device_map[ dev ];
//      }
//      else
//      {
//        auto it = device_map.find( dev );
//        if ( it == device_map.end() )
//        {
//          printf( "no device pointer for the target device\n" );
//          return NULL;
//        }
//      }
//      return device_map[ dev ];
//    };
//
//  private:
//
//    hmlp::Device *host = NULL;
//
//    /** map a device to its data pointer */
//    std::map<hmlp::Device*, T*> device_map;
//   
//    /** distribution */
//    std::set<hmlp::Device*> distribution;
//
//    DeviceMemory *cache = NULL;
//
//    bool isCached( DeviceMemory *target )
//    {
//      return ( cache == target );
//    };
//
//}; // end class DeviceMemory
//









typedef enum
{
  CACHE_CLEAN, 
  CACHE_DIRTY
} CacheStatus;

class CacheLine : public DeviceMemory<char>
{
  public:

    CacheLine() {};

    void Setup( hmlp::Device *device, size_t line_size )
    {
      this->status = CACHE_CLEAN;
      this->line_size = line_size;
      AllocateD( device, line_size );
    };

    bool isClean()
    {
      return ( status == CACHE_CLEAN );
    };

    //void Read( DeviceMemory *target )
    //{
    //  asCache( target );
    //};

    //void Write()
    //{
    //};

  private:

    CacheStatus status;

    size_t line_size;

};

template<int NUM_LINE>
class Cache
{
  public:

    Cache() {};
      
    void Setup( hmlp::Device *device )
    {
      for ( int line_id = 0; line_id < NUM_LINE; line_id ++ )
      {
        line[ line_id ].Setup( device, 4096 * 4096 * 8 );
      }
    };

    //void Read( hmlp::Device *device, DeviceMemory *data )
    //{
    //   size_t line_id = fifo % NUM_LINE;
    //   if ( !cache[ line_id ]->isClean() )
    //   {
    //   }
    //   cache[ line_id ]->Read( data );
    //};

    //char *Write( hmlp::Device *device, size_t line_id )
    //{
    //  return (char*)line[ line_id ].device_data( device );
    //};

  private:

    size_t fifo = 0;

    CacheLine line[ NUM_LINE ];
};

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

        for ( int i = 0; i < 10; i ++ )
        {
          if ( cudaStreamCreate( &(stream[ i ] ) ) )
            printf( "cudaStreamCreate(), fail on device %d\n", device_id );
        }

        for ( int i = 0; i < 1; i ++ )
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
        //cache.Setup( this );
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
    cudaStream_t stream[ 10 ];

    /** use 1 cublas handles */
    cublasHandle_t handle[ 1 ];

    char *work_d = NULL;

    size_t work_size = 800000000;

    size_t total_memory = 0;

    size_t memory_left = 0;

};



}; // end namespace gpu
}; // end namespace hmlp

#endif // define HMLP_GPU_HPP
