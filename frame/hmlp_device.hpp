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


#ifndef HMLP_DEVICE_HPP
#define HMLP_DEVICE_HPP

#include <string>
#include <stdio.h>
#include <iostream>
#include <cstddef>
#include <cassert>
#include <map>
#include <set>
#include <omp.h>

//#include <hmlp_runtime.hpp>

#define MAX_LINE 16

namespace hmlp
{

class Device *hmlp_get_device_host();



typedef enum
{
  CACHE_CLEAN, 
  CACHE_DIRTY
} CacheStatus;

class CacheLine
{
  public:

    CacheLine();

    void Setup( hmlp::Device *device, size_t line_size );

    bool isClean();
     
    void Bind( void *ptr_h );    

    bool isCache( void *ptr_h, size_t size );

    char *device_data();

  private:

    void *ptr_h = NULL;

    char *ptr_d = NULL;

    CacheStatus status;

    size_t line_size;

};


class Cache
{
  public:

    Cache();
      
    void Setup( hmlp::Device *device );

    CacheLine *Read( size_t size );

  private:

    size_t fifo = 0;

    class CacheLine line[ MAX_LINE ];
};








typedef enum 
{
  HOST,
  NVIDIA_GPU,
  OTHER_GPU,
  TI_DSP
} DeviceType;


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
class Device
{
  public:

    Device();

    virtual class CacheLine *getline( size_t size );

    virtual void prefetchd2h( void *ptr_h, void *ptr_d, size_t size, int stream_id );

    virtual void prefetchh2d( void *ptr_d, void *ptr_h, size_t size, int stream_id );

    virtual void waitexecute();

    virtual void wait( int stream_id );

    virtual void *malloc( size_t size );

    virtual void malloc( void *ptr_d, size_t size );

    virtual size_t get_memory_left();

    virtual void free( void *ptr_d, size_t size );

    DeviceType devicetype;

    std::string name;

    class Cache cache;

  private:

    class Worker *workers;
};


template<class T>
class DeviceMemory
{
  /** allow other instance to access device_map */
  //friend class DeviceMemory;

  public:

    DeviceMemory() 
    {
      this->host = hmlp_get_device_host();
      distribution.insert( host );
    };

    void CacheD( hmlp::Device *dev, size_t size )
    {
      if ( !isCached( size ) )
      {
        /** request a new cache location on the device */
        cache = dev->getline( size );
        cache->Bind( this ); 
        Redistribute<true>( host );
      }
    };

    ///** this will be the cache of target */
    //void asCache( DeviceMemory *target )
    //{
    //  cache = target;
    //};

    void AllocateD( hmlp::Device *dev, size_t size )
    {
      if ( !device_map.count( dev ) )
      {
        T *ptr_d = (T*)dev->malloc( size );
        if ( !ptr_d ) return;
        device_map[ dev ] = ptr_d; 
      }
    };

    /** free device memory, remove from the map and the distribution */
    void FreeD( hmlp::Device *dev, size_t size )
    {
      if ( device_map.count( dev ) )
      {
        if ( !device_map[ dev ] )
        {
          printf( "NULL device ptr in device_map\n" ); fflush( stdout );
        }
        dev->free( device_map[ dev ], size );
        device_map.erase( dev );
        distribution.erase( dev );
      }
    };

    /** */
    void PrefetchH2D( hmlp::Device *dev, int stream_id, size_t size, T* ptr_h )
    {
      this->stream_id = stream_id;

      if ( cache )
      {
        //printf( "Is cached\n" ); fflush( stdout );
        CacheD( dev, size );
        if ( !distribution.count( dev ) )
        {
          dev->prefetchh2d( cache->device_data(), ptr_h, size, stream_id );
          /** TODO need to be careful about the definition here. */
          Redistribute<false>( dev );
        }
      }
      else
      {
        //printf( "Not cached\n" ); fflush( stdout );
        if ( !distribution.count( dev ) )
        {
          //printf( "PrefetchH2D: target device does not have the latest copy.\n" );
          AllocateD( dev, size );
          //if ( !device_map.count( dev ) )
          //{
          //  //printf( "allocate %lu bytes on %s\n", size, dev->name.data() );
          //  T *ptr_d = (T*)dev->malloc( size );
          //  if ( !ptr_d ) return;
          //  device_map[ dev ] = ptr_d; 
          //}
          //printf( "memcpy H2D\n" );
          dev->prefetchh2d( device_map[ dev ], ptr_h, size, stream_id );
          /** TODO: maybe update the distribution here? */
          //printf( "redistribute\n" );
          Redistribute<false>( dev );
        }
        else /** the device has the latest copy */
        {
          //printf( "PrefetchH2D: target device has the latest copy\n" );
          assert( device_map.find( dev ) != device_map.end() );
        }
      }
    };

    /** if host does not have the latest copy */
    void PrefetchD2H( hmlp::Device *dev, int stream_id, size_t size, T* ptr_h )
    {
      this->stream_id = stream_id;

      if ( cache )
      {
        CacheD( dev, size );
        if ( !distribution.count( host ) )
        {
          dev->prefetchd2h( ptr_h, cache->device_data(), size, stream_id );
          Redistribute<false>( host );
        }
      }
      else
      {
        if ( !distribution.count( host ) )
        {
          //printf( "PrefetchD2H: host does not have the latest copy.\n" );
          assert( device_map.count( dev ) );
          dev->prefetchd2h( ptr_h, device_map[ dev ], size, stream_id );
          Redistribute<false>( host );
        }
        else /** the host has the latest copy */
        {
          //printf( "PrefetchD2H: host has the latest copy\n" );
          assert( device_map.count( host ) );
        }
      }
    };

    void FetchH2D( hmlp::Device *dev, size_t size, T* ptr_h )
    {
      PrefetchH2D( dev, stream_id, size, ptr_h );
      dev->wait( stream_id );
    };

    void FetchD2H( hmlp::Device *dev, size_t size, T* ptr_h )
    {
      PrefetchD2H( dev, stream_id, size, ptr_h );
      dev->wait( stream_id );
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
      if ( cache )
      {
        return (T*)cache->device_data();
      }
      else
      {
        auto it = device_map.find( dev );
        if ( it == device_map.end() )
        {
          printf( "no device pointer for the target device\n" );
          return NULL;
        }
        return device_map[ dev ];
      }
    };


  private:

    int stream_id = 0;

    hmlp::Device *host = NULL;

    /** map a device to its data pointer */
    std::map<hmlp::Device*, T*> device_map;
   
    /** distribution */
    std::set<hmlp::Device*> distribution;

    CacheLine *cache = NULL;

    bool isCached( size_t size )
    {
      bool iscached = false;
      if ( cache )
      {
        iscached = cache->isCache( this, size );
      }    
      return iscached;
    };

}; // end class DeviceMemory

}; // end namespace hmlp

#endif //#define HMLP_DEVICE_HPP
