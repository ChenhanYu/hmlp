#ifndef HMLP_THREAD_HPP
#define HMLP_THREAD_HPP

#include <string>
#include <stdio.h>
#include <iostream>
#include <cstddef>
#include <omp.h>


namespace hmlp
{


typedef enum 
{
  HOST,
  NVIDIA_GPU,
  OTHER_GPU,
  TI_DSP
} DeviceType;


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

/**
 *  @brief This class describes devices or accelerators that require
 *         a master thread to control. A device can accept tasks from
 *         multiple workers. All received tasks are expected to be
 *         executed independently in a time-sharing fashion.
 *         Whether these tasks are executed in parallel, sequential
 *         or with some built-in context switching scheme does not
 *         matter.
 *
 */ 
class Device
{
  public:

    Device();

    virtual void prefetchd2h( void *ptr_h, void *ptr_d, size_t size );

    virtual void prefetchh2d( void *ptr_d, void *ptr_h, size_t size );

    virtual void wait();

    virtual void* malloc( size_t size );

    virtual void free( void *ptr_d );

    DeviceType devicetype;

    std::string name;

  private:

    class Worker *workers;
};


class Worker 
{
  public:

    Worker();

	  //worker( int jc_nt, int pc_nt, int ic_nt, int jr_nt );
   
    Worker( thread_communicator *my_comm );

    bool Execute( class Task *task );

    void WaitExecute();

    float EstimateCost( class Task* task );

    class Scheduler *scheduler;

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

    class Device *device;
};







}; // end namespace hmlp



#endif // end define HMLP_THREAD_HPP
