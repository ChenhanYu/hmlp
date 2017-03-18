#include <hmlp_device.hpp>
#include <hmlp_runtime.hpp>

namespace hmlp
{


/**
 *  @brief Device implementation
 */
Device::Device()
{
  name = std::string( "Host CPU" );
  devicetype = hmlp::DeviceType::HOST;
};

void Device::prefetchd2h( void *ptr_h, void *ptr_d, size_t size, int stream_id ) {};

void Device::prefetchh2d( void *ptr_d, void *ptr_h, size_t size, int stream_id ) {};

void Device::wait( int stream_id ) {};

void *Device::malloc( size_t size ) { return NULL; };

void Device::malloc( void *ptr_d, size_t size ) {};

size_t Device::get_memory_left() { return 0; };

void Device::free( void *ptr_d, size_t size ) {};


}; // end namespace hmlp
