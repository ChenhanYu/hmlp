#ifndef HMLP_RUNTIME_HPP
#define HMLP_RUNTIME_HPP


#include <stdio.h>
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

  private:
};


}; // end namespace hmlp

#endif // define HMLP_RUNTIME_HPP
