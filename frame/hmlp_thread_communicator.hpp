#ifndef HMLP_THREAD_COMMUNICATOR_HPP
#define HMLP_THREAD_COMMUNICATOR_HPP

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

    char *name;

	private:

	  void          *sent_object;

    int           comm_id;

	  int           n_threads;

    int           n_groups;

	  volatile bool barrier_sense;

	  int           barrier_threads_arrived;
};



}; // end namespace hmlp



#endif // end define HMLP_THREAD_COMMUNICATOR_HPP
