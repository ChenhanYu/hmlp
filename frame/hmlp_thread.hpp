#ifndef HMLP_THREAD_HPP
#define HMLP_THREAD_HPP

#include <string>
#include <stdio.h>
#include <iostream>
#include <cstddef>
#include <cassert>
#include <map>
#include <set>
#include <omp.h>

#include <hmlp_device.hpp>


namespace hmlp
{


//typedef enum 
//{
//  HOST,
//  NVIDIA_GPU,
//  OTHER_GPU,
//  TI_DSP
//} DeviceType;


class thread_communicator 
{
	public:
    
    thread_communicator();

	  thread_communicator( int jc_nt, int pc_nt, int ic_nt, int jr_nt );

    void Create( int level, int num_threads, int *config );

    void Barrier();

    void Print();

    int GetNumThreads();

    int GetNumGroups();

    friend std::ostream& operator<<( std::ostream& os, const thread_communicator& obj );

    thread_communicator *kids;

    std::string name;

	private:

	  void          *sent_object;

    int           comm_id;

	  int           n_threads;

    int           n_groups;

	  volatile bool barrier_sense;

	  int           barrier_threads_arrived;
};


/**
 *  @brief This class describes devices or accelerators that require
 *         a master thread to control. A device can accept tasks from
 *         multiple workers. All received tasks are expected to be
 *         executed independently in a time-sharing fashion.
 *         Whether these tasks are executed in parallel, sequential
 *         or with some built-in context switching scheme does not
 *         matter.
 *
 */ 
//class Device
//{
//  public:
//
//    Device();
//
//    virtual void prefetchd2h( void *ptr_h, void *ptr_d, size_t size, int stream_id );
//
//    virtual void prefetchh2d( void *ptr_d, void *ptr_h, size_t size, int stream_id );
//
//    virtual void wait( int stream_id );
//
//    virtual void *malloc( size_t size );
//
//    virtual void malloc( void *ptr_d, size_t size );
//
//    virtual size_t get_memory_left();
//
//    virtual void free( void *ptr_d, size_t size );
//
//    DeviceType devicetype;
//
//    std::string name;
//
//  private:
//
//    class Worker *workers;
//};


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
//        T *ptr_d = (T*)dev->malloc( size );
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
//        assert( device_map.count( dev ) );
//        dev->prefetchd2h( ptr_h, device_map[ dev ], size, stream_id );
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














class Worker 
{
  public:

    Worker();

	  //worker( int jc_nt, int pc_nt, int ic_nt, int jr_nt );
   
    Worker( thread_communicator *my_comm );

    void Communicator( thread_communicator *comm );

    void SetDevice( class Device *device );

    class Device *GetDevice();

    bool Execute( class Task *task );

    void WaitExecute();

    float EstimateCost( class Task* task );

    class Scheduler *scheduler;

#ifdef USE_PTHREAD_RUNTIME
    pthread_t pthreadid;
#endif

    int tid;

    int jc_id;

    int pc_id;

    int ic_id;

    int jr_id;

    int ic_jr;

    int jc_nt;

    int pc_nt;

    int ic_nt;

    int jr_nt;

    thread_communicator *my_comm;

    thread_communicator *jc_comm;

    thread_communicator *pc_comm;

    thread_communicator *ic_comm;

  private:

    class Task *current_task;

    class Device *device;
};







}; // end namespace hmlp



#endif // end define HMLP_THREAD_HPP
