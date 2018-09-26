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

#include <hmlp_runtime.hpp>

#ifdef HMLP_USE_CUDA
#include <hmlp_gpu.hpp>
#endif

#ifdef HMLP_USE_MAGMA
#include <magma_v2.h>
#include <magma_lapack.h>
#endif

#define REPORT_RUNTIME_STATUS 1
// #define DEBUG_RUNTIME 1
// #define DEBUG_SCHEDULER 1

using namespace std;
using namespace hmlp;


struct 
{
  bool operator()( const tuple<bool, double, size_t> &a, 
                   const tuple<bool, double, size_t> &b )
  {   
    return get<1>( a ) < get<1>( b );
  }   
} EventLess;



namespace hmlp
{

/** IMPORTANT: we allocate a static runtime system per (MPI) process */
static RunTime rt;

/** 
 *  class Lock
 */ 

///** @brief Shared-memory lock that calls either pthread or omp mutex.. */ 
//Lock::Lock()
//{
//#ifdef USE_PTHREAD_RUNTIME
//  if ( pthread_mutex_init( &lock, NULL ) )
//  {
//    printf( "pthread_mutex_init(): cannot initialize locks properly\n" );
//  }
//#else
//  omp_init_lock( &lock );
//#endif
//}; /** end Lock::Lock() */
//
//Lock::~Lock()
//{
//#ifdef USE_PTHREAD_RUNTIME
//  if ( pthread_mutex_destroy( &lock ) )
//  {
//    printf( "pthread_mutex_destroy(): cannot destroy locks properly\n" );
//  }
//#else
//  omp_destroy_lock( &lock );
//#endif
//}; /** end Lock::~Lock() */
//
//void Lock::Acquire()
//{
//#ifdef USE_PTHREAD_RUNTIME
//  if ( pthread_mutex_lock( &lock ) )
//  {
//    printf( "pthread_mutex_lock(): cannot acquire locks properly\n" );
//  }
//#else
//  omp_set_lock( &lock );
//#endif
//};
//
//void Lock::Release()
//{
//#ifdef USE_PTHREAD_RUNTIME
//  if ( pthread_mutex_unlock( &lock ) )
//  {
//    printf( "pthread_mutex_lock(): cannot release locks properly\n" );
//  }
//#else
//  omp_unset_lock( &lock );
//#endif
//};
//

/** 
 *  class Event
 */ 

/** @brief (Default) Event constructor. */
Event::Event() {};

/** @brief Set the label, flops, and mops. */
void Event::Set( string _label, double _flops, double _mops )
{
  flops = _flops;
  mops  = _mops;
  label = _label;
};

void Event::AddFlopsMops( double _flops, double _mops )
{
  flops += _flops; //Possible concurrent write
  mops += _mops;
};


void Event::Begin( size_t _tid )
{
  tid = _tid;
  beg = omp_get_wtime();
};

void Event::Normalize( double shift )
{
  beg -= shift;
  end -= shift;
};

void Event::Terminate()
{
  end = omp_get_wtime();
  sec = end - beg;
};

double Event::GetBegin() { return beg; };

double Event::GetEnd() { return end; };

double Event::GetDuration() { return sec; };

double Event::GetFlops() { return flops; };

double Event::GetMops() { return mops; };

double Event::GflopsPerSecond() { return ( flops / sec ) / 1E+9; };


void Event::Print()
{
  printf( "beg %5.3lf end %5.3lf sec %5.3lf flops %E mops %E\n",
      beg, end, sec, flops, mops );
};

void Event::Timeline( bool isbeg, size_t tag )
{
  double gflops_peak = 45.0;
  double flops_efficiency = flops / ( gflops_peak * sec * 1E+9 );
  if ( isbeg )
  {
    //printf( "@TIMELINE\n" );
    //printf( "worker%lu, %lu, %E, %lf\n", tid, 2 * tag + 0, beg, (double)tid + 0.0 );
    //printf( "@TIMELINE\n" );
    //printf( "worker%lu, %lu, %E, %lf\n", tid, 2 * tag + 1, beg, (double)tid + flops_efficiency );
    printf( "@TIMELINE\n" );
    printf( "worker%lu, %lu, %E, %lf\n", tid, tag, beg, (double)tid + flops_efficiency );
  }
  else
  {
    //printf( "@TIMELINE\n" );
    //printf( "worker%lu, %lu, %E, %lf\n", tid, 2 * tag + 0, beg, (double)tid + flops_efficiency );
    //printf( "@TIMELINE\n" );
    //printf( "worker%lu, %lu, %E, %lf\n", tid, 2 * tag + 1, beg, (double)tid + 0.0 );
    printf( "@TIMELINE\n" );
    printf( "worker%lu, %lu, %E, %lf\n", tid, tag, end, (double)tid + 0.0 );
  }

};

void Event::MatlabTimeline( FILE *pFile )
{
  /** TODO: this needs to change according to the arch */
  double gflops_peak = 45.0;
  double flops_efficiency = 0.0;
  if ( sec * 1E+9 > 0.1 )
  {
    flops_efficiency = flops / ( gflops_peak * sec * 1E+9 );
    if ( flops_efficiency > 1.0 ) flops_efficiency = 1.0;
  }
  fprintf( pFile, "rectangle('position',[%lf %lu %lf %d],'facecolor',[1.0,%lf,%lf]);\n",
      beg, tid, ( end - beg ), 1, 
      flops_efficiency, flops_efficiency );
  fprintf( pFile, "text( %lf,%lf,'%s');\n", beg, (double)tid + 0.5, label.data() );
};




/** 
 *  class Task
 */ 

/** @brief (Default) Task constructor. */ 
Task::Task()
{
  /** Change status to allocated. */
  SetStatus( ALLOCATED );
  /** Whether this is a nested task? */
  is_created_in_epoch_session = rt.IsInEpochSession();
  /** Which thread creates me? */
  created_by = omp_get_thread_num();
  /** Move forward to next status "NOTREADY". */
  SetStatus( NOTREADY );
}; /** end Task::Task() */

/** @brief (Default) Task destructor. */ 
Task::~Task() {};

/** @brief (Default) MessageTask constructor. */ 
MessageTask::MessageTask( int src, int tar, int key )
{
  this->src = src;
  this->tar = tar;
  this->key = key;
}; /** end MessageTask::MessageTask() */

/** @brief (Default) ListenerTask constructor. */ 
ListenerTask::ListenerTask( int src, int tar, int key ) : MessageTask( src, tar, key ) {};

/** @brief Status is a private member. */
TaskStatus Task::GetStatus() { return status; };

/** @brief Move foward to the next status. */
void Task::SetStatus( TaskStatus next_status ) { this->status = next_status; };

/** Change the status of all tasks in the batch. */
void Task::SetBatchStatus( TaskStatus next_status )
{
  auto *task = this;
  while ( task )
  {
    task->status = next_status;
    /** Move to the next task in the batch */
    task = task->next;
  }
};

/** @brief Ask the runtime to create an normal task in file. */
void Task::Submit() { rt.scheduler->NewTask( this ); };

/** @brief Ask the runtime to create an message task in file. */
void MessageTask::Submit() { rt.scheduler->NewMessageTask( this ); };

/** @brief Ask the runtime to create a listener task in file. */
void ListenerTask::Submit() { rt.scheduler->NewListenerTask( this ); };

/** @brief This is only for virtual function pointer. */
void Task::Set( string user_name, void (*user_function)(Task*), void *user_arg )
{
  name = user_name;
  function = user_function;
  arg = user_arg;
  status = NOTREADY;
}; /** end Task::Set() */

/** @brief Update the my outgoing and children's incoming edges. */
void Task::DependenciesUpdate()
{
  /** Loop over each out-going edge. */
  while ( out.size() )
  {
    Task *child = out.front();
    /** There should be at least "one" remaining dependency to satisfy. */
    assert( child->n_dependencies_remaining > 0 && child->GetStatus() == NOTREADY );
    /** Acquire execlusive right to modify the task. */
    //assert( child->task_lock );
    child->Acquire();
    {
      child->n_dependencies_remaining --;
      /** If there is no dependency left, enqueue the task. */
      if ( !child->n_dependencies_remaining )
      {
        /** Nested tasks may not carry the worker pointer. */
        if ( worker ) child->Enqueue( worker->tid );
        else          child->Enqueue();
      }
    }
    child->Release();
    /** Remove this out-going edge. */
    out.pop_front();
  }
  /** Move forward to the last status "DONE". */
  SetStatus( DONE );
}; /** end Task::DependenciesUpdate() */


void Task::Acquire() 
{ 
  if ( !task_lock ) 
  {
    cout << name << " not submitted" << endl;
    assert( task_lock );
  }
  task_lock->Acquire(); 
};

void Task::Release() 
{ 
  if ( !task_lock ) 
  {
    cout << name << " not submitted" << endl;
    assert( task_lock );
  }
  task_lock->Release(); 
};


/** All virtual functions. */
void Task::GetEventRecord() {};
void Task::Prefetch( Worker *user_worker ) {};
void Task::DependencyAnalysis() {};

/** @brief Try to dispatch the task if there is no dependency left. */
bool Task::TryEnqueue()
{
  if ( GetStatus() == NOTREADY && !n_dependencies_remaining ) 
  {
    Enqueue();
    return true;
  }
  else return false;
};

void Task::Enqueue() { Enqueue( 0 ); };

void Task::ForceEnqueue( size_t tid )
{
  int assignment = tid;

  rt.scheduler->ready_queue_lock[ assignment ].Acquire();
  {
    float cost = rt.workers[ assignment ].EstimateCost( this );
    /** Move forward to next status "QUEUED". */
    SetStatus( QUEUED );
    if ( priority )
      rt.scheduler->ready_queue[ assignment ].push_front( this );
    else
      rt.scheduler->ready_queue[ assignment ].push_back( this );

    /** update the remaining time */
    rt.scheduler->time_remaining[ assignment ] += cost; 
  }
  rt.scheduler->ready_queue_lock[ assignment ].Release();
}; /** end Task::ForceEnqueue() */


/** @brief */ 
void Task::Enqueue( size_t tid )
{
  float cost = 0.0;
  float earliest_t = -1.0;
  int assignment = -1;

  /** Dispatch to nested queue if created in the epoch session. */
  if ( is_created_in_epoch_session )
  {
    assert( created_by < rt.n_worker );
    rt.scheduler->nested_queue_lock[ created_by ].Acquire();
    {
      /** Move forward to next status "QUEUED". */
      SetStatus( QUEUED );
      if ( priority )
        rt.scheduler->nested_queue[ created_by ].push_front( this );
      else
        rt.scheduler->nested_queue[ created_by ].push_back( this );
    }
    rt.scheduler->nested_queue_lock[ created_by ].Release();
    /** Finish and return without further going down. */
    return;
  };

  /** Determine which worker the task should go to using HEFT policy. */
  for ( int p = 0; p < rt.n_worker; p ++ )
  {
    int i = ( tid + p ) % rt.n_worker;
    float cost = rt.workers[ i ].EstimateCost( this );
    float terminate_t = rt.scheduler->time_remaining[ i ];
    if ( earliest_t == -1.0 || terminate_t + cost < earliest_t )
    {
      earliest_t = terminate_t + cost;
      assignment = i;
    }
  }

  /** Dispatch to normal ready queue. */
  ForceEnqueue( assignment );

}; /** end Task::Enqueue() */


/** @brief This is the callback function for the owner of thenested task.*/ 
void Task::CallBackWhileWaiting()
{
  rt.ExecuteNestedTasksWhileWaiting( this );
}; /** end CallBackWhileWaiting() */

/** @brief */
bool Task::IsNested() { return is_created_in_epoch_session; };








/** @breief (Default) ReadWrite constructor. */ 
ReadWrite::ReadWrite() {};

/** Clean both read and write sets. */
void ReadWrite::DependencyCleanUp()
{
  read.clear();
  write.clear();
}; /** end DependencyCleanUp() */

/** @brief This is the key function that encode the dependency. **/ 
void ReadWrite::DependencyAnalysis( ReadWriteType type, Task *task )
{
  if ( type == R || type == RW )
  {
    /** Update the read set. */
    read.push_back( task );
    /** Read-After-Write (RAW) data dependencies. */
    for ( auto it : write ) Scheduler::DependencyAdd( it, task );
  }
  if ( type == W || type == RW )
  {
    /** Write-After-Read (WAR) anti-dependencies. */
    for ( auto it : read ) Scheduler::DependencyAdd( it, task );
    /** Clean up both read and write sets. */
    DependencyCleanUp();
    /** Update the write set. */
    write.push_back( task );
  }
}; /** end ReadWrite::DependencyAnalysis() */








/**
 *  class MatrixReadWrite
 */ 

/** @brief (Default) MatrixReadWrite constructor. */ 
MatrixReadWrite::MatrixReadWrite() {};

/** @brief */
void MatrixReadWrite::Setup( size_t m, size_t n )
{
  //printf( "%lu %lu setup\n", m, n  );
  this->has_been_setup = true;
  this->m = m;
  this->n = n;
  Submatrices.resize( m );
  for ( size_t i = 0; i < m; i ++ ) Submatrices[ i ].resize( n );
}; /** end MatrixReadWrite::MatrixReadWrite() */


/** @brief */
bool MatrixReadWrite::HasBeenSetup() { return has_been_setup; };

/** @brief */
void MatrixReadWrite::DependencyAnalysis( 
    size_t i, size_t j, ReadWriteType type, Task *task )
{
  //printf( "%lu %lu analysis\n", i, j  ); fflush( stdout );
  assert( i < m && j < n );
  Submatrices[ i ][ j ]. DependencyAnalysis( type, task );
}; /** end MatrixReadWrite::DependencyAnalysis() */

/** @brief */
void MatrixReadWrite::DependencyCleanUp()
{
  for ( size_t i = 0; i < m; i ++ )
    for ( size_t j = 0; j < n; j ++ )
      Submatrices[ i ][ j ]. DependencyCleanUp();
}; /** end MatrixReadWrite::DependencyCleanUp() */





/**
 *  class Scheduler
 */ 

/**  @brief (Default) Scheduler constructor. */ 
Scheduler::Scheduler( mpi::Comm user_comm ) 
  : mpi::MPIObject( user_comm ), timeline_tag( 500 )
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler()\n" );
#endif
  listener_tasklist.resize( this->GetCommSize() );
  /** Set now as the begining of the time table. */
  timeline_beg = omp_get_wtime();
};


/** @brief  */
Scheduler::~Scheduler()
{
#ifdef DEBUG_SCHEDULER
  printf( "~Scheduler()\n" );
#endif
};


/** @brief  */
void Scheduler::Init( int user_n_worker )
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::Init()\n" );
#endif

  /** Adjust the number of active works. */
  n_worker = user_n_worker;
  /** Reset normal and nested task counter. */
  n_task_completed = 0;
  n_nested_task_completed = 0;
	/** Reset async distributed consensus variables. */
	do_terminate = false;
  has_ibarrier = false;
  ibarrier_consensus = 0;

#ifdef USE_PTHREAD_RUNTIME
  for ( int i = 0; i < n_worker; i ++ )
  {
    rt.workers[ i ].tid = i;
    rt.workers[ i ].scheduler = this;
    pthread_create
    ( 
      &(rt.workers[ i ].pthreadid), NULL,
      EntryPoint, (void*)&(rt.workers[ i ])
    );
  }
  /** Now the master thread will enter the EntryPoint. */
  EntryPoint( (void*)&(rt.workers[ 0 ]) );
#else
  #pragma omp parallel for num_threads( n_worker )
  for ( int i = 0; i < n_worker; i ++ )
  {
    assert( omp_get_thread_num() == i );
    rt.workers[ i ].tid = i;
    rt.workers[ i ].scheduler = this;
    EntryPoint( (void*)&(rt.workers[ i ]) );
  } /** end pragma omp parallel for */
#endif
}; /** end Scheduler::Init() */


/** @brief  */
void Scheduler::MessageDependencyAnalysis( 
    int key, int p, ReadWriteType type, Task *task )
{
  if ( msg_dependencies.find( key ) == msg_dependencies.end() )
  {
    msg_dependencies[ key ] = vector<ReadWrite>( this->GetCommSize() );
  }
  msg_dependencies[ key ][ p ].DependencyAnalysis( type, task );
}; /** end Scheduler::MessageDependencyAnalysis() */


/** @brief This function is called by RunTime::Submit() to record a new task. */
void Scheduler::NewTask( Task *task )
{
  if ( !task ) return;
  /** Acquire the exclusive right to access the tasklist. */
  tasklist_lock.Acquire();
  {
    if ( rt.IsInEpochSession() ) 
    {
      task->task_lock = &(task_lock[ nested_tasklist.size() % ( 2 * MAX_WORKER ) ]);
      nested_tasklist.push_back( task );
    }
    else 
    {
      task->task_lock = &(task_lock[ tasklist.size() % ( 2 * MAX_WORKER ) ]);
      tasklist.push_back( task );
    }
  }
  tasklist_lock.Release();
}; /** end Scheduler::NewTask()  */


/** @brief  */
void Scheduler::NewMessageTask( MessageTask *task )
{
  tasklist_lock.Acquire();
  {
    /** We use a duplicated (private) communicator to handle message tasks. */
    task->comm = this->GetPrivateComm();
    task->task_lock = &(task_lock[ tasklist.size() % ( 2 * MAX_WORKER ) ]);
    /** Counted toward termination criteria. */
    tasklist.push_back( task );
    //printf( "NewMessageTask src %d tar %d key %d tasklist.size() %lu\n",
    //    task->src, task->tar, task->key, tasklist.size() );
  }
  tasklist_lock.Release();
};


/** @brief  */
void Scheduler::NewListenerTask( ListenerTask *task )
{
  tasklist_lock.Acquire();
  {
    /** We use a duplicated (private) communicator to handle message tasks. */
    task->comm = this->GetPrivateComm();
    task->task_lock = &(task_lock[ tasklist.size() % ( 2 * MAX_WORKER ) ]);
    listener_tasklist[ task->src ][ task->key ] = task;
    /** Counted toward termination criteria. */
    tasklist.push_back( task );
    //printf( "NewListenerTask src %d tar %d key %d listener_tasklist[].size() %lu\n",
    //    task->src, task->tar, task->key, listener_tasklist[ task->src ].size() );
  }
  tasklist_lock.Release();
}; /** ebd Scheduler::NewListenerTask() */



/** @brief  */
void Scheduler::Finalize()
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::Finalize()\n" );
#endif
#ifdef USE_PTHREAD_RUNTIME
  for ( int i = 0; i < rt.n_worker; i ++ )
  {
    pthread_join( rt.workers[ i ].pthreadid, NULL );
  }
#else
#endif

  /** Print out statistics of this epoch */
  if ( REPORT_RUNTIME_STATUS ) Summary();

  /** Reset remaining time. */
  for ( int i = 0; i < n_worker; i ++ ) time_remaining[ i ] = 0.0;
  /** Free all normal tasks and reset tasklist. */
  try
  {
    for ( auto task : tasklist ) delete task; 
    tasklist.clear();
  }
  catch ( exception & e ) { cout << e.what() << endl; };
  /** Free all nested tasks and reset nested_tasklist. */
  try
  {
    for ( auto task : nested_tasklist ) delete task; 
    nested_tasklist.clear();
  }
  catch ( exception & e ) { cout << e.what() << endl; };
  //printf( "Begin Scheduler::Finalize() [cleanup listener_tasklist]\n" );
  /** Reset listener_tasklist  */
  try
  {
    for ( auto & plist : listener_tasklist ) plist.clear();
  }
  catch ( exception & e ) { cout << e.what() << endl; };
  //printf( "End   Scheduler::Finalize() [cleanup listener_tasklist]\n" );
  
  /** Clean up all message dependencies. */
  msg_dependencies.clear();

}; /** end Scheduler::Finalize() */


/** @brief  */
void Scheduler::ReportRemainingTime()
{
  printf( "ReportRemainingTime:" ); fflush( stdout );
  printf( "--------------------\n" ); fflush( stdout );
  for ( int i = 0; i < rt.n_worker; i ++ )
  {
    printf( "worker %2d --> %7.2lf (%4lu jobs)\n", 
        i, rt.scheduler->time_remaining[ i ], 
           rt.scheduler->ready_queue[ i ].size() ); fflush( stdout );
  }
  printf( "--------------------\n" ); fflush( stdout );
}; /** end Scheduler::ReportRemainingTime() */


/** @brief Add an direct edge (dependency) from source to target. */ 
void Scheduler::DependencyAdd( Task *source, Task *target )
{
  /** Avoid self-loop. */
  if ( source == target ) return;
  /** Update the source out-going edges. */
  source->Acquire();
  {
    source->out.push_back( target );
  }
  source->Release();
  /** Update the target incoming edges. */
  target->Acquire();
  {
    target->in.push_back( source );
    /** Only increase the dependency count for incompleted tasks. */
    if ( source->GetStatus() != DONE ) target->n_dependencies_remaining ++;
  }
  target->Release();
}; /** end Scheduler::DependencyAdd() */


Task *Scheduler::StealFromQueue( size_t target )
{
  Task *target_task = NULL;

  /** get the lock of the target ready queue */
  ready_queue_lock[ target ].Acquire();
  {
    if ( ready_queue[ target ].size() ) 
    {
      target_task = ready_queue[ target ].back();
      assert( target_task );
      if ( target_task->stealable )
      {
        ready_queue[ target ].pop_back();
        time_remaining[ target ] -= target_task->cost;
      }
      else target_task = NULL;
    }
  }
  ready_queue_lock[ target ].Release();

  return target_task;
}; /** end Scheduler::TryStealFromQueue() */


/** @brief */
vector<Task*> Scheduler::StealFromOther()
{
  int max_remaining_tasks = 0;
  int max_remaining_nested_tasks = 0;
  int target = 0;
  /** Decide which target's normal queue to steal. */
  for ( int p = 0; p < n_worker; p ++ )
  {
    if ( ready_queue[ p ].size() > max_remaining_tasks )
    {
      max_remaining_tasks = ready_queue[ p ].size();
      target = p;
    }
  }
  /** Try to steal from target's ready queue.  */
  auto batch = DispatchFromNormalQueue( target );
  /** Return if batch is not empty. */
  if ( batch.size() ) return batch;
  /** Decide which target's nested queue to steal. */
  for ( int p = 0; p < n_worker; p ++ )
  {
    if ( nested_queue[ p ].size() > max_remaining_nested_tasks )
    {
      max_remaining_nested_tasks = nested_queue[ p ].size();
      target = p;
    }
  }
  /** Try to steal from target's nested queue.  */
  batch = DispatchFromNestedQueue( target );
  /** Return regardless if batch is empty or not. */
  return batch;
}; /** end Scheduler::StealFromOther() */



/** @brief Dispatch a nested task from tid's nested_queue. */
vector<Task*> Scheduler::DispatchFromNestedQueue( int tid )
{
  vector<Task*> batch;
  /** Dispatch a nested task from tid's nested_queue. */
  nested_queue_lock[ tid ].Acquire();
  {
    if ( nested_queue[ tid ].size() )
    {
      auto *target_task = nested_queue[ tid ].front();
      /** Check if I can dispatch this task? */
      if ( tid == omp_get_thread_num() || target_task->stealable )
      {
        /** Fetch the first task in the nested queue. */
        batch.push_back( target_task );
        /** Remove the task from the nested queue. */
        nested_queue[ tid ].pop_front();
      }
    }
  }
  nested_queue_lock[ tid ].Release();
  /** Notice that this can be an empty vector. */
  return batch;
}; /** end Scheduler::DispatchFromNestedQueue() */


/** @brief */
bool Scheduler::IsTimeToExit( int tid )
{
  /** In the case that do_terminate has been set, return "true". */
  if ( do_terminate ) return true;
  /** Both normal and nested tasks should all be executed. */
  if ( n_task_completed >= tasklist.size() )
  {
    if ( n_nested_task_completed >= nested_tasklist.size() )
    {
      /** My ready_queue and nested_queue should all be empty. */
      assert( !ready_queue[ tid ].size() );
      assert( !nested_queue[ tid ].size() );
		  /** Set the termination flag to true. */
	    do_terminate = true;
      /** Now there should be no tasks left locally. Return "true". */
      return true;
    }
    else printf( "normal %d/%lu nested %d/%lu\n",
        n_task_completed, tasklist.size(),
        n_nested_task_completed, nested_tasklist.size() );
  }
  /** Otherwise, it is not yet to terminate. */
  return false;
}; /** end Scheduler::IsTimeToExit() */


/** @brief */
vector<Task*> Scheduler::DispatchFromNormalQueue( int tid )
{
  size_t maximum_batch_size = 1;
  vector<Task*> batch;
  /** Dispatch normal tasks from tid's ready queue. */
  ready_queue_lock[ tid ].Acquire();
  {
    for ( int it = 0; it < maximum_batch_size; it ++ )
    {
      if ( ready_queue[ tid ].size() )
      {
        auto *target_task = ready_queue[ tid ].front();
        /** The target ready_queue is not my queue. */
        if ( tid != omp_get_thread_num() )
        {
          maximum_batch_size = 1;
          /** If this task cannot be stole, then break. */
          if ( !target_task->stealable ) break;
          else time_remaining[ tid ] -= target_task->cost;
        }
        /** Dequeue a task and push into this batch. */
        batch.push_back( target_task );
        ready_queue[ tid ].pop_front();
      }
      else
      {
        /** Reset my workload counter. */
        time_remaining[ tid ] = 0.0;
      }
    }
  }
  ready_queue_lock[ tid ].Release();
  /** Notice that this can be an empty vector. */
  return batch;
}; /** end Scheduler::DispatchFromNormalQueue() */


/** @brief */
bool Scheduler::ConsumeTasks( Worker *me, vector<Task*> &batch )
{
  /** Early return if there is no task to execute. */
  if ( !batch.size() ) return false;
  /** For each task, move forward to the next status "RUNNING". */
  for ( auto task : batch ) task->SetStatus( RUNNING );
  /** Now the worker will execute all tasks in the batch. */
  for ( auto task : batch ) me->Execute( task );
  /** For each task, update dependencies and my remining time. */
  for ( auto task : batch )
  {
    task->DependenciesUpdate();
    /** Update my remaining time and n_task_completed. */
    if ( !task->IsNested() )
    {
      ready_queue_lock[ me->tid ].Acquire();
      {
        time_remaining[ me->tid ] -= task->cost;
        if ( time_remaining[ me->tid ] < 0.0 )
          time_remaining[ me->tid ] = 0.0;
      }
      ready_queue_lock[ me->tid ].Release();
      n_task_lock.Acquire();
      {
        n_task_completed ++;
      }
      n_task_lock.Release();
    }
    else
    {
      n_task_lock.Acquire();
      {
        n_nested_task_completed ++;
      }
      n_task_lock.Release();
    }
  }
  /** Return "true" if at least one task was executed. */
  return true;
}; /** end Scheduler::ConsumeTasks() */


/** @brief */
void Scheduler::ExecuteNestedTasksWhileWaiting( Worker *me, Task *waiting_task )
{
  assert( me->tid == omp_get_thread_num() );
  while ( waiting_task->GetStatus() != DONE )
  {
    /** Try to get a nested task. */
    auto nested_batch = DispatchFromNestedQueue( me->tid );
    ConsumeTasks( me, nested_batch );
  }
}; /** end ExcuteNestedTasksWhileWaiting() */


/** @brief */
bool Scheduler::ConsumeTasks( Worker *me, Task *batch, bool is_nested )
{
  /** Early return */
  if ( !batch ) return false;

  /** Update status */
  batch->SetBatchStatus( RUNNING );

  if ( me->Execute( batch ) )
  {
    Task *task = batch;
    while ( task )
    {
      task->DependenciesUpdate();
      if ( !is_nested )
      {
        ready_queue_lock[ me->tid ].Acquire();
        {
          time_remaining[ me->tid ] -= task->cost;
          if ( time_remaining[ me->tid ] < 0.0 )
            time_remaining[ me->tid ] = 0.0;
        }
        ready_queue_lock[ me->tid ].Release();
        n_task_lock.Acquire();
        {
          n_task_completed ++;
        }
        n_task_lock.Release();
      }
      else
      {
        n_task_lock.Acquire();
        {  
          n_nested_task_completed ++;
        }
        n_task_lock.Release();
      }
      /**  Move to the next task in te batch */
      task = task->next;
    }
  }
  return true;
}; /** end Scheduler::ConsumeTasks() */













/** @brief This is the main body that each worker will go through. */
void* Scheduler::EntryPoint( void* arg )
{
  /** First cast arg back to Worker*. */
  Worker *me = reinterpret_cast<Worker*>( arg );
  /** Get the callback point of scheduler. */
  Scheduler *scheduler = me->scheduler;
  /** This counter measures the idle iteration. */
  size_t idle = 0;

#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::EntryPoint()\n" );
  printf( "pthreadid %d\n", me->tid );
#endif

  /** Prepare listeners (half of total workers). */
  if ( ( me->tid % 2 ) && true )
  {
    /** Update my termination time to infinite. */
    scheduler->ready_queue_lock[ me->tid ].Acquire();
    {
      scheduler->time_remaining[ me->tid ] = numeric_limits<float>::max();
    }
    scheduler->ready_queue_lock[ me->tid ].Release();
    /** Enter listening mode. */
    scheduler->Listen( me );
  }
  /** Start to consume all tasks in this epoch session. */
  while ( 1 )
  {
    /** Try to get a normal task from my own ready queue. */
    auto normal_batch = scheduler->DispatchFromNormalQueue( me->tid );
    /** If there is some jobs to execute, then reset the counter. */
    if ( scheduler->ConsumeTasks( me, normal_batch ) )
    {
      /** Reset the idle counter. */
      idle = 0;
    }
    else /** No task in my ready_queue. Try nested_queue. */
    {
      /** Increase the idle counter. */
      idle ++;
      /** Try to get a nested task. */
      auto nested_batch = scheduler->DispatchFromNestedQueue( me->tid );
      /** Reset the idle counter if there is executable nested tasks. */
      if ( scheduler->ConsumeTasks( me, nested_batch ) ) idle = 0;
    }
    /** Try to steal from others. */
    if ( idle > 10 )
    {
      /** Try to steal a (normal or nested) task. */
      auto stolen_batch = scheduler->StealFromOther();
      /** Reset the idle counter if there is executable stolen tasks. */
      if ( scheduler->ConsumeTasks( me, stolen_batch ) ) idle = 0;
    } /** end if ( idle > 10 ) */

    /** Check if is time to terminate. */
    if ( scheduler->IsTimeToExit( me->tid ) ) break;
  }
  /** Return "NULL". */
  return NULL;
}; /** end Scheduler::EntryPoint() */


/** @brief Listen for asynchronous incoming MPI messages. */
void Scheduler::Listen( Worker *me )
{
  /** We use a duplicated (private) communicator to handle message tasks. */
  auto comm = this->GetPrivateComm();
  auto size = this->GetCommSize();
  auto rank = this->GetCommRank();

  /** Iprobe flag and recv status */
  int probe_flag = 0;
  mpi::Status status;

  /** Keep probing for incoming messages. */
  while ( 1 ) 
  {
    ListenerTask *task = NULL;
    /** Only one thread will probe and recv message at a time. */
    #pragma omp critical
    {
      /** Asynchronously probe form incoming messages. */
      mpi::Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &probe_flag, &status );
      /** If receive any message, then handle it. */
      if ( probe_flag )
      {
        /** Info from mpi::Status */
        int recv_src = status.MPI_SOURCE;
        int recv_tag = status.MPI_TAG;
        int recv_key = 3 * ( recv_tag / 3 );
   
        if ( recv_key < 300 )
        {
          printf( "rank %d Iprobe src %d tag %d\n", rank, recv_src, recv_tag ); 
          fflush( stdout );
        }

        /** Check if there is a corresponding task. */
        auto it = listener_tasklist[ recv_src ].find( recv_key );
        if ( it != listener_tasklist[ recv_src ].end() )
        {
          task = it->second;
          if ( task->GetStatus() == NOTREADY )
          {
            //printf( "rank %d Find a task src %d tag %d\n", rank, recv_src, recv_tag ); 
            //fflush( stdout );
            task->SetStatus( QUEUED );
            task->Listen();
          }
          else
          {
            printf( "rank %d Find a QUEUED task src %d tag %d\n", rank, recv_src, recv_tag ); 
            fflush( stdout );
            task = NULL;
            probe_flag = false;
          }
        }
        else 
        {
          //printf( "rank %d not match task src %d tag %d\n", rank, recv_src, recv_tag ); 
          //fflush( stdout );
          probe_flag = false;
        }
      }
    }

    if ( probe_flag )
    {
      /** Execute tasks and update dependencies */
      ConsumeTasks( me, task, false );
    }
    else
    {
      /** Steal a (normal or nested) task from other. */
      auto stolen_batch = StealFromOther();
      ConsumeTasks( me, stolen_batch );
    }

    /** Nonblocking consensus for termination. */
    if ( do_terminate ) 
    {
      /** We use an ibarrier to make sure global concensus. */
      #pragma omp critical
      {
        /** Only the first worker will issue an Ibarrier. */
        if ( !has_ibarrier ) 
        {
          mpi::Ibarrier( comm, &ibarrier_request );
          has_ibarrier = true;
        }
        /** Test global consensus on "terminate_request". */
        if ( !ibarrier_consensus )
        {
          mpi::Test( &ibarrier_request, &ibarrier_consensus, 
              MPI_STATUS_IGNORE );
        }
      }
      /** If terminate_request has been tested = true, then exit! */
      if ( ibarrier_consensus ) break;
    }
  }
}; /** end Scheduler::Listen() */


/** @brief */
void Scheduler::Summary()
{
  int total_normal_tasks = tasklist.size();
  int total_nested_tasks = nested_tasklist.size();
  int total_listen_tasks = 0;
  for ( auto list : listener_tasklist ) total_listen_tasks += list.size();
  double total_flops = 0.0, total_mops = 0.0;
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[ 80 ];

  time( &rawtime );
  timeinfo = localtime( &rawtime );
  strftime( buffer, 80, "%T.", timeinfo );

  //printf( "%s\n", buffer );

  for ( size_t i = 0; i < tasklist.size(); i ++ )
  {
    total_flops += tasklist[ i ]->event.GetFlops();
    total_mops  += tasklist[ i ]->event.GetMops();
  }

#ifdef HMLP_USE_MPI
	/** In the MPI environment, reduce all flops and mops. */
  int global_normal_tasks = 0, global_listen_tasks = 0, global_nested_tasks = 0;
	double global_flops = 0.0, global_mops = 0.0;
	mpi::Reduce( &total_flops, &global_flops, 1, MPI_SUM, 0, this->GetPrivateComm() );
	mpi::Reduce( &total_mops,  &global_mops,  1, MPI_SUM, 0, this->GetPrivateComm() );
	mpi::Reduce( &total_normal_tasks, &global_normal_tasks, 1, MPI_SUM, 0, this->GetPrivateComm() );
	mpi::Reduce( &total_listen_tasks, &global_listen_tasks, 1, MPI_SUM, 0, this->GetPrivateComm() );
	mpi::Reduce( &total_nested_tasks, &global_nested_tasks, 1, MPI_SUM, 0, this->GetPrivateComm() );
	if ( this->GetCommRank() == 0 ) 
  {
    printf( "[ RT] %5d [normal] %5d [listen]  %5d [nested] %5.3E flops %5.3E mops\n", 
      global_normal_tasks, global_listen_tasks, global_nested_tasks, global_flops, global_mops );
  }
#else
  printf( "[ RT] %5d [normal] %5d [nested] %5.3E flops %5.3E mops\n", 
      total_normal_tasks, total_nested_tasks, total_flops, total_mops );
#endif


#ifdef DUMP_ANALYSIS_DATA
  deque<tuple<bool, double, size_t>> timeline;

  if ( tasklist.size() )
  {
    string filename = string( "timeline" ) + 
	  to_string( tasklist.size() ) + string( "_rank") + 
    to_string( rank) + string( ".m" ); 
    FILE *pFile = fopen( filename.data(), "w" );

    fprintf( pFile, "figure('Position',[100,100,800,800]);" );
    fprintf( pFile, "hold on;" );

    for ( size_t i = 0; i < tasklist.size(); i ++ )
    {
      tasklist[ i ]->event.Normalize( timeline_beg );
    }

    for ( size_t i = 0; i < tasklist.size(); i ++ )
    {
      auto &event = tasklist[ i ]->event;
      timeline.push_back( make_tuple(  true, event.GetBegin(), i ) );
      timeline.push_back( make_tuple( false, event.GetEnd(),   i ) );
    }

    sort( timeline.begin(), timeline.end(), EventLess );

    for ( size_t i = 0; i < timeline.size(); i ++ )
    {
      auto &data = timeline[ i ];
      auto &event = tasklist[ get<2>( data ) ]->event;  
      //event.Timeline( get<0>( data ), i + timeline_tag );
      event.MatlabTimeline( pFile );
    }

    fclose( pFile );

    timeline_tag += timeline.size();
  }
#endif

}; /** end Scheduler::Summary() */





/**
 *  class RunTime
 */ 

/** @brief */
RunTime::RunTime() {};

/** @brief */
RunTime::~RunTime() {};

/** @brief */
void RunTime::Init( mpi::Comm comm = MPI_COMM_WORLD )
{
  #pragma omp critical (init)
  {
    if ( !is_init )
    {
      n_worker = omp_get_max_threads();
      n_max_worker = n_worker;
      /** Check whether MPI has been initialized? */
      int is_mpi_init = false;
      mpi::Initialized( &is_mpi_init );
      if ( !is_mpi_init )
      {

      }

      scheduler = new Scheduler( comm );
#ifdef HMLP_USE_CUDA
      /** TODO: detect devices */
      device[ 0 ] = new hmlp::gpu::Nvidia( 0 );
      if ( n_worker )
      {
        workers[ 0 ].SetDevice( device[ 0 ] );
      }
#endif
#ifdef HMLP_USE_MAGMA
        magma_init();
#endif
      /** Set the flag such that this is only executed once. */
      is_init = true;
    }
  } /** end pragma omp critical */
}; /** end RunTime::Init() */


/** @brief */
void RunTime::Init( int* argc, char*** argv, mpi::Comm comm = MPI_COMM_WORLD )
{
  #pragma omp critical
  {
    if ( !is_init )
    {
      /** Set argument count. */
      this->argc = argc;
      /** Set argument values. */
      this->argv = argv;
      /** Acquire the number of (maximum) workers from OpenMP. */
      n_worker = omp_get_max_threads();
      n_max_worker = n_worker;
      /** Check whether MPI has been initialized? */
      int is_mpi_init = false;
      mpi::Initialized( &is_mpi_init );
      /** Initialize MPI inside HMLP otherwise. */
      if ( !is_mpi_init )
      {
        int provided;
	      mpi::Init_thread( argc, argv, MPI_THREAD_MULTIPLE, &provided );
	      if ( provided != MPI_THREAD_MULTIPLE ) 
          ExitWithError( string( "MPI_THTREAD_MULTIPLE is not supported" ) );
        /** Flag that MPI is initialized by HMLP. */
        is_mpi_init_by_hmlp = true;
      }
      /** Initialize the scheduler. */
      scheduler = new Scheduler( comm );
#ifdef HMLP_USE_CUDA
      /** TODO: detect devices */
      device[ 0 ] = new hmlp::gpu::Nvidia( 0 );
      if ( n_worker )
      {
        workers[ 0 ].SetDevice( device[ 0 ] );
      }
#endif
#ifdef HMLP_USE_MAGMA
        magma_init();
#endif
      /** Set the flag such that this is only executed once. */
      is_init = true;
    }
  } /** end pragma omp critical */
}; /** end RunTime::Init() */




/** @brief **/
void RunTime::Run()
{
  if ( is_in_epoch_session )
  {
    printf( "Fatal Error: more than one concurrent epoch session!\n" );
    exit( 1 );
  }
  if ( !is_init ) Init();
  /** begin this epoch session */
  is_in_epoch_session = true;
  /** schedule jobs to n workers */
  scheduler->Init( n_worker );
  /** clean up */
  scheduler->Finalize();
  /** finish this epoch session */
  is_in_epoch_session = false;
}; /** end RunTime::Run() */

/** @brief */
void RunTime::Finalize()
{
  #pragma omp critical (init)
  {
    if ( is_init )
    {
      /** Finalize the scheduler and delete it. */
      scheduler->Finalize();
      delete scheduler;
      /** Set the initialized flag to false. */
      is_init = false;
      /** Finalize MPI if it was initialized by HMLP. */
      if ( is_mpi_init_by_hmlp ) mpi::Finalize();
    }
  }
}; /** end RunTime::Finalize() */

/** @brief */
bool RunTime::IsInEpochSession() { return is_in_epoch_session; };

/** @brief */
void RunTime::ExecuteNestedTasksWhileWaiting( Task *waiting_task )
{
  /** Use omp_get_thread_num() to acquire tid. */
  auto *me = &(workers[ omp_get_thread_num() ]);
  /** If I am not in a epoch session, then return. */
  if ( IsInEpochSession() )
  {
    scheduler->ExecuteNestedTasksWhileWaiting( me, waiting_task );
  }
}; /** end RunTime::ExecuteNestedTasksWhileWaiting() */


void RunTime::Print( string msg )
{
  cout << "[RT ] " << msg << endl; fflush( stdout );
}; /** end RunTime::Print() */


void RunTime::ExitWithError( string msg )
{
  cout << "[RT ] (Error) " << msg << endl; fflush( stdout );
  exit( 1 );
}; /** end RunTime::ExitWithError() */


//void hmlp_runtime::pool_init()
//{
//
//};
//
//void hmlp_runtime::acquire_memory()
//{
//
//};
//
//void hmlp_runtime::release_memory( void *ptr )
//{
//
//};


/** @brief */
hmlp::Device *hmlp_get_device_host() { return &(hmlp::rt.host); };

}; /** end namespace hmlp */


/** @brief */
void hmlp_init() { hmlp::rt.Init(); };
void hmlp_init( mpi::Comm comm = MPI_COMM_WORLD ) 
{ 
  hmlp::rt.Init( comm );
};

void hmlp_init( int *argc, char ***argv )
{
  hmlp::rt.Init( argc, argv );
};

void hmlp_init( int *argc, char ***argv, mpi::Comm comm )
{
  hmlp::rt.Init( argc, argv, comm );
};

/** @brief */
void hmlp_set_num_workers( int n_worker ) { hmlp::rt.n_worker = n_worker; };



void hmlp_run() { hmlp::rt.Run(); };

void hmlp_finalize() { hmlp::rt.Finalize(); };

hmlp::RunTime *hmlp_get_runtime_handle() { return &hmlp::rt; };

hmlp::Device *hmlp_get_device( int i ) { return hmlp::rt.device[ i ]; };

bool hmlp_is_in_epoch_session() { return hmlp::rt.IsInEpochSession(); };

int hmlp_get_mpi_rank() { return hmlp::rt.scheduler->GetCommRank(); };

int hmlp_get_mpi_size() { return hmlp::rt.scheduler->GetCommSize(); };

void hmlp_msg_dependency_analysis( int key, int p, ReadWriteType type, Task *task )
{
  hmlp::rt.scheduler->MessageDependencyAnalysis( key, p, type, task );
};

