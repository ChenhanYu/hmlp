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


#include <hmlp_device.hpp>
#include <hmlp_runtime.hpp>

using namespace std;

namespace hmlp
{

CacheLine::CacheLine() {};

void CacheLine::Setup( hmlp::Device *device, size_t line_size )
{
  this->status = CACHE_CLEAN;
  this->line_size = line_size;
  this->ptr_d = (char*)device->malloc( line_size );
};

bool CacheLine::isClean()
{
  return ( status == CACHE_CLEAN );
};

void CacheLine::Bind( void *ptr_h )
{
  this->ptr_h = ptr_h;
};


bool CacheLine::isCache( void *ptr_h, size_t size )
{
  if ( size > line_size )
  {
    printf( "Cache line %lu request %lu\n", line_size, size );
    return false;
  }
  else
  {
    return (this->ptr_h == ptr_h);
  }
};

char *CacheLine::device_data()
{
  return ptr_d;
};



Cache::Cache() {};

void Cache::Setup( hmlp::Device *device )
{
  for ( int line_id = 0; line_id < MAX_LINE; line_id ++ )
  {
    line[ line_id ].Setup( device, 2048 * 2048 * 8 );
  }
};

CacheLine *Cache::Read( size_t size )
{
  int line_id = fifo;
  fifo = ( fifo + 1 ) % MAX_LINE;
  if ( !line[ line_id ].isClean() )
  {
    printf( "The cache line is not clean\n" );
    exit( 1 );
  }
  else
  {
    //printf( "line_id %d\n", line_id );
  }
  return &(line[ line_id ]);
};






/**
 *  @brief Device implementation
 */
Device::Device()
{
  name = std::string( "Host CPU" );
  devicetype = hmlp::DeviceType::HOST;
};


CacheLine *Device::getline( size_t size ) 
{
  return cache.Read( size );
};

void Device::prefetchd2h( void *ptr_h, void *ptr_d, size_t size, int stream_id ) {};

void Device::prefetchh2d( void *ptr_d, void *ptr_h, size_t size, int stream_id ) {};

void Device::waitexecute() {};

void Device::wait( int stream_id ) {};

void *Device::malloc( size_t size ) { return NULL; };

void Device::malloc( void *ptr_d, size_t size ) {};

size_t Device::get_memory_left() { return 0; };

void Device::free( void *ptr_d, size_t size ) {};


}; // end namespace hmlp
