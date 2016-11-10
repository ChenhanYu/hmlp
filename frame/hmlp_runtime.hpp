#ifndef HMLP_RUNTIME_HPP
#define HMLP_RUNTIME_HPP

#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstddef>
#include <omp.h>

namespace hmlp
{

typedef enum 
{
  HMLP_SCHEDULE_DEFAULT,
  HMLP_SCHEDULE_ROUND_ROBIN,
  HMLP_SCHEDULE_UNIFORM,
  HMLP_SCHEDULE_HEFT
} hmlpSchedule_t;

class range
{
  public:

    range( int beg, int end, int inc );

    int beg();

    int end();

    int inc();

  private:

    std::tuple<int, int, int > info;

};

range GetRange
( 
  hmlpSchedule_t strategy, 
  int beg, int end, int nb, 
  int tid, int nparts 
);

range GetRange
( 
  int beg, int end, int nb, 
  int tid, int nparts 
);

range GetRange
( 
  int beg, int end, int nb 
);

class hmlp_runtime
{
  public:

    hmlp_runtime();

    void init();

    void finalize();

    void pool_init();

    void acquire_memory();

    void release_memory( void* ptr );

  private:
    
    bool is_init = false;

    std::size_t pool_size_in_bytes;

    void *pool;
};

// Create the global runtime object.
static hmlp_runtime rt;

}; // end namespace hmlp

#endif // define HMLP_RUNTIME_HPP
