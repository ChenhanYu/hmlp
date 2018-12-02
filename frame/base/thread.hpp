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






#ifndef HMLP_THREAD_HPP
#define HMLP_THREAD_HPP

#include <string>
#include <stdio.h>
//#include <iostream>
#include <cstddef>
#include <cassert>
#include <map>
#include <set>
#include <omp.h>


#include <base/tci.hpp>
#include <base/device.hpp>



using namespace std;


namespace hmlp
{


//typedef enum 
//{
//  HMLP_SCHEDULE_DEFAULT,
//  HMLP_SCHEDULE_ROUND_ROBIN,
//  HMLP_SCHEDULE_UNIFORM,
//  HMLP_SCHEDULE_HEFT
//} SchedulePolicy;



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
  SchedulePolicy strategy, 
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











class thread_communicator 
{
	public:
    
    thread_communicator();

	  thread_communicator( int jc_nt, int pc_nt, int ic_nt, int jr_nt );

    void Create( int level, int num_threads, int *config );

    //void Initialize( int )


    void Barrier();

    void Send( void** buffer );

    void Recv( void** buffer );

    void Print();

    int GetNumThreads();

    int GetNumGroups();

    friend ostream& operator<<( ostream& os, const thread_communicator& obj );

    thread_communicator *kids;

    string name;






	private:

	  void          *sent_object;

    int           comm_id;

	  int           n_threads = 1;

    int           n_groups = 1;

	  volatile bool barrier_sense;

	  int           barrier_threads_arrived;

}; /** end class thread_communicator */




/**
 *
 *
 **/ 
class Worker 
{
  public:

    Worker();

	  //worker( int jc_nt, int pc_nt, int ic_nt, int jr_nt );
   
    Worker( thread_communicator *my_comm );

    void Communicator( thread_communicator *comm );

//    void SetDevice( class Device *device );
//
//    class Device *GetDevice();
//
//    bool Execute( class Task *task );
//
//    void WaitExecute();
//
//    float EstimateCost( class Task* task );
//
//    class Scheduler *scheduler;
//
//#ifdef USE_PTHREAD_RUNTIME
//    pthread_t pthreadid;
//#endif

    int tid = 0;

    int gid = 0;

    int child_gid = 0;

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

    bool Master();
  
    void Barrier();

    void InitWithCommunicator( thread_communicator* comm, size_t tid, size_t gid );

    Worker Split();

    thread_communicator *comm = NULL;

    template<typename Arg>
    void Bcast( Arg& buffer )
    {
      if ( Master() ) comm->Send( (void**)&buffer );
      Barrier();
      if ( !Master() ) comm->Recv( (void**)&buffer );
    }

    template<int ALIGN_SIZE, typename T>
    T *AllocateSharedMemory( size_t count )
    {
      T* ptr = NULL;
      if ( Master() ) ptr = hmlp_malloc<ALIGN_SIZE, T>( count );
      Bcast( ptr );
      return ptr;
    };

    template<typename T>
    void FreeSharedMemory( T *ptr )
    {
      if ( Master() ) hmlp_free( ptr );
    };

    size_t BalanceOver1DGangs( size_t n, size_t default_size, size_t nb );

    tuple<size_t, size_t, size_t> DistributeOver1DGangs(
      size_t beg, size_t end, size_t nb );

    tuple<size_t, size_t, size_t> DistributeOver1DThreads(
      size_t beg, size_t end, size_t nb );















    void SetDevice( class Device *device );

    class Device *GetDevice();

    //bool Execute( vector<hmlp::Task*> &batch );

    bool Execute( class Task *task );

    void WaitExecute();

    float EstimateCost( class Task* task );

    class Scheduler *scheduler;

#ifdef USE_PTHREAD_RUNTIME
    pthread_t pthreadid;
#endif

  private:

    class Task *current_task = NULL;

    class Device *device = NULL;


}; /** end class Worker */

}; /** end namespace hmlp */

#endif /** end define HMLP_THREAD_HPP */
