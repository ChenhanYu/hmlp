#ifndef HMLP_RUNTIME_HPP
#define HMLP_RUNTIME_HPP

#include <tuple>
#include <deque>
#include <cstdint>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstddef>
#include <omp.h>

#ifdef USE_PTHREAD_RUNTIME
#include <pthread.h>
#endif

#include <hmlp_thread.hpp>

#define MAX_WORKER 24

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
  public:

    Task();

    ~Task();

    Worker *worker;

    std::string name;

    std::string label;

    int taskid;

    float cost;

    // priority

    TaskStatus GetStatus();

    void SetStatus( TaskStatus status );

    void Submit();

    virtual void Set( std::string user_name, void (*user_function)(Task*), void *user_arg );

    void Enqueue();

    virtual void Execute( Worker* );

    /* function ptr */
    void (*function)(Task*);

    /* function context */
    void *arg;

    volatile int n_dependencies_remaining;

    /* dependency */
    void DependenciesUpdate();

    std::deque<Task*> in;

    std::deque<Task*> out;

    // argument list
    // arg

    // Task lock
    Lock task_lock;

  private:

    volatile TaskStatus status;

};


class Scheduler
{
  public:

    Scheduler();
    
    ~Scheduler();

    void Init( int n_worker );

    void Finalize();

    int n_worker;

    int n_task;

    std::deque<Task*> ready_queue[ MAX_WORKER ];

    std::deque<Task*> tasklist;

    float time_remaining[ MAX_WORKER ];

    // Manually describe the dependencies.
    static void DependencyAdd( Task *source, Task *target );

    void NewTask( Task *task );

    Lock ready_queue_lock[ MAX_WORKER ];

    Lock n_task_lock;

  private:

    static void* EntryPoint( void* );

    Lock run_lock[ MAX_WORKER ];

    Lock pci_lock;

    Lock gpu_lock;

    Lock tasklist_lock;

};


class RunTime
{
  public:

    RunTime();

    ~RunTime();

    void Init();

    void Run();

    void Finalize();

    //void pool_init();

    //void acquire_memory();

    //void release_memory( void* ptr );

    int n_worker;

    Worker workers[ MAX_WORKER ];

    Scheduler *scheduler;

  private:
    
    bool is_init = false;

    //std::size_t pool_size_in_bytes;

    //void *pool;
};

}; // end namespace hmlp


#endif // define HMLP_RUNTIME_HPP
