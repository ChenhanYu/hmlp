#ifndef HMLP_UTIL_HXX
#define HMLP_UTIL_HXX

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

namespace hmlp
{

template<int SIMD_ALIGN_SIZE, typename T>
T *hmlp_malloc( int m, int n, int size )
{
  T *ptr;
  int err;

  err = posix_memalign( (void**)&ptr, (size_t)SIMD_ALIGN_SIZE, size * m * n );

  if ( err ) 
  {
    printf( "hmlp_malloc(): posix_memalign() failures" );
    exit( 1 );
  }

  return ptr;
}

}; // end namespace hmlp

#endif // define HMLP_UTIL_HXX
