#ifndef HMLP_UTIL_HPP
#define HMLP_UTIL_HPP

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

namespace hmlp
{

template<int ALIGN_SIZE, typename T>
T *hmlp_malloc( int m, int n, int size )
{
  T *ptr;
  int err;
#ifdef HMLP_MIC_AVX512
  err = hbw_posix_memalign( (void**)&ptr, (size_t)ALIGN_SIZE, size * m * n );
#else
  err = posix_memalign( (void**)&ptr, (size_t)ALIGN_SIZE, size * m * n );
#endif

  if ( err ) 
  {
    printf( "hmlp_malloc(): posix_memalign() failures\n" );
    exit( 1 );
  }

  return ptr;
}

template<typename T>
void hmlp_free( T *ptr )
{
#ifdef HMLP_MIC_AVX512
  if ( ptr ) hbw_free( ptr );
  ptr = NULL;
#else
  if ( ptr ) free( ptr );
  ptr = NULL;
#endif
}

}; // end namespace hmlp

#endif // define HMLP_UTIL_HPP
