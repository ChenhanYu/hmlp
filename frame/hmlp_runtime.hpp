#ifndef HMLP_RUNTIME_HPP
#define HMLP_RUNTIME_HPP

#include <time.h>
#include <tuple>
#include <algorithm>
#include <vector>
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

#include <hmlp_device.hpp>
#include <hmlp_thread.hpp>

#define MAX_WORKER 68

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


class Event
{
  public:

    Event(); 

    //Event( float, float );
    
    //~Event();

    void Set( std::string, double, double );

    void AddFlopsMops( double, double );

    void Begin( size_t );

    double GetBegin();

    double GetEnd();

    double GetDuration();

    double GetFlops();

    double GetMops();

    double GflopsPerSecond();

    void Normalize( double shift );

    void Terminate();

    void Print();

    void Timeline( bool isbeg, size_t tag );

    void MatlabTimeline( FILE *pFile );

  private:

    size_t tid;

	std::string label;

    double flops;

    double mops;

    double beg;

    double end;

    double sec;

}; // end class Event


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

    bool priority = false;

    Event event;

    TaskStatus GetStatus();

    void SetStatus( TaskStatus status );

    void Submit();

    virtual void Set( std::string user_name, void (*user_function)(Task*), void *user_arg );

    virtual void Prefetch( Worker* );

    void Enqueue();

    void Enqueue( size_t tid );

    void TryEnqueue();

    void ForceEnqueue( size_t tid );

    void CallBackWhileWaiting();

    virtual void Execute( Worker* );

    virtual void GetEventRecord();

    virtual void DependencyAnalysis();

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

    /** task lock */
    Lock task_lock;

    /** the next task in the batch job */
    Task *next = NULL;

  private:

    volatile TaskStatus status;

    /**  */
    bool is_created_in_epoch_session = false;

};



class ReadWrite
{
  public:

    ReadWrite();

    // Tracking the read set of the object.
    std::deque<Task*> read;

    // Tracking the write set of the object.
    std::deque<Task*> write;

    void DependencyAnalysis( ReadWriteType type, Task *task );

    void DependencyCleanUp();

  private:

}; /** end class ReadWrite */



class MatrixReadWrite
{
  public:

    MatrixReadWrite();

    void Setup( size_t m, size_t n );

    bool HasBeenSetup();

    void DependencyAnalysis( size_t i, size_t j, ReadWriteType type, Task *task );

    void DependencyCleanUp();

  private:

    bool has_been_setup = false;

    size_t m = 0;

    size_t n = 0;

    std::vector<std::vector<ReadWrite> > Submatrices;

}; /** end class MatrixReadWrite */



class Scheduler
{
  public:

    Scheduler();
    
    ~Scheduler();

    void Init( int n_worker, int n_nested_worker );

    void Finalize();

    int n_worker;

    int n_task;

    size_t timeline_tag;

    double timeline_beg;

    std::deque<Task*> ready_queue[ MAX_WORKER ];

    /** the read queue for nested tasks */
    std::deque<Task*> nested_queue;

    std::deque<Task*> tasklist;

    std::deque<Task*> nested_tasklist;

    float time_remaining[ MAX_WORKER ];

    void ReportRemainingTime();

    /** manually describe the dependencies */
    static void DependencyAdd( Task *source, Task *target );

    void NewTask( Task *task );

    Task *TryDispatchFromNestedQueue();

    void Summary();

    Lock ready_queue_lock[ MAX_WORKER ];

    Lock nested_queue_lock;

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

    /** whether the runtime is in a epoch session */
    bool IsInEpochSession();

    /** consuming nested tasks while in a epoch session */
    void ExecuteNestedTasksWhileWaiting( Task *waiting_task );

    //void pool_init();

    //void acquire_memory();

    //void release_memory( void* ptr );

    int n_worker;

    int n_max_worker;

    int n_nested_worker = 1;

    thread_communicator *mycomm;

    Worker workers[ MAX_WORKER ];

    Device host;

    Device* device[ 1 ];

    Scheduler *scheduler;

  private:
   
    bool is_init = false;

    bool is_in_epoch_session = false;

}; /** end class Runtime */

}; // end namespace hmlp

hmlp::RunTime *hmlp_get_runtime_handle();

hmlp::Device *hmlp_get_device( int i );

bool hmlp_is_in_epoch_session();


#endif // define HMLP_RUNTIME_HPP
