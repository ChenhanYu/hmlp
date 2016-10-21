#ifndef HMLP_THREAD_INFO
#define HMLP_THREAD_INFO

#include <stdio.h>
#include <iostream>
#include <cstddef>
#include <omp.h>
#include <hmlp_thread_communicator.hpp>

namespace hmlp
{

class worker 
{
  public:

    //worker();

	  //worker( int jc_nt, int pc_nt, int ic_nt, int jr_nt );
   
    worker( thread_communicator *my_comm );

    int tid;

    int jc_id;

    int pc_id;

    int ic_id;

    int jr_id;

    thread_communicator *my_comm;

    thread_communicator *jc_comm;

    thread_communicator *pc_comm;

    thread_communicator *ic_comm;
};


};

#endif
