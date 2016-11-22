#ifndef HMLP_THREAD_COMMUNICATOR_HPP
#define HMLP_THREAD_COMMUNICATOR_HPP

#include <string>
#include <stdio.h>
#include <iostream>
#include <cstddef>
#include <omp.h>

namespace hmlp
{





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


class Worker 
{
  public:

    Worker();

	  //worker( int jc_nt, int pc_nt, int ic_nt, int jr_nt );
   
    Worker( thread_communicator *my_comm );

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
};







}; // end namespace hmlp



#endif // end define HMLP_THREAD_COMMUNICATOR_HPP
