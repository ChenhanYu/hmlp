#ifndef HMLP_RUNTIME_HPP
#define HMLP_RUNTIME_HPP

#include <tuple>
#include <deque>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstddef>
#include <omp.h>

#ifdef USE_PTHREAD_RUNTIME
#include <pthread.h>
#endif

#include <hmlp_thread.hpp>

#define MAX_WORKER 20

namespace hmlp
{

typedef enum 
{
  HMLP_SCHEDULE_DEFAULT,
  HMLP_SCHEDULE_ROUND_ROBIN,
  HMLP_SCHEDULE_UNIFORM,
  HMLP_SCHEDULE_HEFT
} SchedulePolicy;

typedef enum { ALLOCATED, NOTREADY, QUEUED, RUNNING, DONE, CANCELLED } TaskStatus;

typedef enum { R, W, RW } ReadWriteType;


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

/**
 * @brief Lock
 */ 
class Lock
{
  public:

    Lock();

    ~Lock();

    void Acquire();

    void Release();

  private:
#ifdef USE_PTHREAD_RUNTIME
    pthread_mutex_t lock;
#else
    omp_lock_t lock;
#endif
};


class Task
{
  Worker *worker;

  std::string name;

  std::string label;

  int taskid;

  Lock task_lock;
  
  float cost;

  // priority

  /* function ptr */
  void (*function)(void*);

  volatile TaskStatus status;

  volatile int n_dependencies_remaining;

  /* dependency */
  std::deque<Task*> in;

  std::deque<Task*> out;

  // argument list
  // arg

};













class Scheduler
{
  public:

    Scheduler();
    
    ~Scheduler();

    void Init( int n_worker );

  private:

    void EntryPoint();

    std::deque<Task*> ready_queue[ MAX_WORKER ];

    float time_remaining[ MAX_WORKER ];

    int ntask;

    // lock
    Lock run_lock[ MAX_WORKER ];

    Lock ready_queue_lock[ MAX_WORKER ];

    Lock ntask_lock;

    Lock pci_lock;

    Lock gpu_lock;
};


class RunTime
{
  public:

    RunTime();

    //~RunTime();

    void Init();

    void Finalize();

    //void pool_init();

    //void acquire_memory();

    //void release_memory( void* ptr );

  private:
    
    bool is_init = false;

    //std::size_t pool_size_in_bytes;

    //void *pool;

    int n_worker;

    Worker workers[ MAX_WORKER ];

    static Scheduler scheduler;

};

// Create the global runtime object.
static RunTime rt;

}; // end namespace hmlp



#endif // define HMLP_RUNTIME_HPP
