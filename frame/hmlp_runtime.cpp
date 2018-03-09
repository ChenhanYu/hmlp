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

#define MAX_BATCH_SIZE 4

#define REPORT_RUNTIME_STATUS 1
// #define DEBUG_RUNTIME 1
// #define DEBUG_SCHEDULER 1

using namespace std;



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

//range::range( int beg, int end, int inc )
//{
//  info = std::make_tuple( beg, end, inc );
//};
//
//int range::beg()
//{
//  return std::get<0>( info );
//};
//
//int range::end()
//{
//  return std::get<1>( info );
//};
//
//int range::inc()
//{
//  return std::get<2>( info );
//};
//
//range GetRange
//( 
//  SchedulePolicy strategy, 
//  int beg, int end, int nb, int tid, int nparts 
//)
//{
//  switch ( strategy )
//  {
//    case HMLP_SCHEDULE_DEFAULT:
//      {
//        auto tid_beg = beg + tid * nb;
//        auto tid_inc = nparts * nb;
//        return range( tid_beg, end, tid_inc );
//      }
//    case HMLP_SCHEDULE_ROUND_ROBIN:
//      {
//        auto tid_beg = beg + tid * nb;
//        auto tid_inc = nparts * nb;
//        return range( tid_beg, end, tid_inc );
//      }
//    case HMLP_SCHEDULE_UNIFORM:
//      printf( "GetRange(): HMLP_SCHEDULE_UNIFORM not yet implemented yet.\n" );
//      exit( 1 );
//    case HMLP_SCHEDULE_HEFT:
//      {
//        assert( nparts == 4 );
//        int len = end - beg - 1;
//        int big = ( len * 30 ) / 100 + 1;
//        int sma = ( len * 20 ) / 100 + 1;
//
//        int tid_beg, tid_end;
//
//        if ( tid == 0 )
//        {
//          tid_beg = beg;
//          tid_end = beg + big;
//        }
//        beg += big;
//
//        if ( tid == 1 )
//        {
//          tid_beg = beg;
//          tid_end = beg + sma;
//        }
//        beg += sma;
//
//        if ( tid == 2 )
//        {
//          tid_beg = beg;
//          tid_end = beg + sma;
//        }
//        beg += sma;
//
//        if ( tid == 3 )
//        {
//          tid_beg = beg;
//          tid_end = beg + big;
//        }
//        beg += big;
//
//        if ( tid_end > end ) tid_end = end;
//        return range( tid_beg, tid_end, nb );
//      }
//    default:
//      printf( "GetRange(): not a legal scheduling strategy.\n" );
//      exit( 1 );
//  }
//};
//
//range GetRange( int beg, int end, int nb, int tid, int nparts )
//{
//  return GetRange( HMLP_SCHEDULE_DEFAULT, beg, end, nb, tid, nparts );
//};
//
//range GetRange( int beg, int end, int nb )
//{
//  return GetRange( HMLP_SCHEDULE_DEFAULT, beg, end, nb, 0, 1 );
//};

/**
 *  @brief Lock
 */ 
Lock::Lock()
{
#ifdef USE_PTHREAD_RUNTIME
  if ( pthread_mutex_init( &lock, NULL ) )
  {
    printf( "pthread_mutex_init(): cannot initialize locks properly\n" );
  }
#else
  omp_init_lock( &lock );
#endif
};

Lock::~Lock()
{
#ifdef USE_PTHREAD_RUNTIME
  if ( pthread_mutex_destroy( &lock ) )
  {
    printf( "pthread_mutex_destroy(): cannot destroy locks properly\n" );
  }
#else
  omp_destroy_lock( &lock );
#endif
};

void Lock::Acquire()
{
#ifdef USE_PTHREAD_RUNTIME
  if ( pthread_mutex_lock( &lock ) )
  {
    printf( "pthread_mutex_lock(): cannot acquire locks properly\n" );
  }
#else
  omp_set_lock( &lock );
#endif
};

void Lock::Release()
{
#ifdef USE_PTHREAD_RUNTIME
  if ( pthread_mutex_unlock( &lock ) )
  {
    printf( "pthread_mutex_lock(): cannot release locks properly\n" );
  }
#else
  omp_unset_lock( &lock );
#endif
};


Event::Event() : flops( 0.0 ), mops( 0.0 ), beg( 0.0 ), end( 0.0 ), sec( 0.0 ) {};

//Event::Event( float _flops, float _mops ) : beg( 0.0 ), end( 0.0 ), sec( 0.0 )
//{
//  flops = _flops;
//  mops = _mops;
//};


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

double Event::GetBegin()
{
  return beg;
};

double Event::GetEnd()
{
  return end;
};

double Event::GetDuration()
{
  return sec;
};

double Event::GetFlops()
{
  return flops;
};

double Event::GetMops()
{
  return mops;
};

double Event::GflopsPerSecond()
{
  return ( flops / sec ) / 1E+9;
};


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
 *  @brief Task
 */ 
Task::Task()
{
  /** Whether this is a nested task? */
  is_created_in_epoch_session = rt.IsInEpochSession();
  status = ALLOCATED;
  status = NOTREADY;
};

Task::~Task()
{};

TaskStatus Task::GetStatus()
{
  return status;
};

void Task::SetStatus( TaskStatus next_status )
{
  this->status = next_status;
};

/** Change the status of all tasks in the batch */
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

void Task::Submit()
{
  rt.scheduler->NewTask( this );
};

void MessageTask::Submit()
{
  rt.scheduler->NewMessageTask( this );
};

void ListenerTask::Submit()
{
  rt.scheduler->NewListenerTask( this );
};

void Task::SetAsBackGround()
{
	rt.scheduler->SetBackGroundTask( this );
};

/** virtual function */
void Task::Set( string user_name, void (*user_function)(Task*), void *user_arg )
{
  name = user_name;
  function = user_function;
  arg = user_arg;
  status = NOTREADY;
};

/** virtual function */
void Task::Prefetch( Worker *user_worker ) {};

void Task::DependenciesUpdate()
{
  while ( out.size() )
  {
    Task *child = out.front();

    child->task_lock.Acquire();
    {
      child->n_dependencies_remaining --;

      //std::cout << child->n_dependencies_remaining << std::endl;

      if ( !child->n_dependencies_remaining && child->status == NOTREADY )
      {
        /** Nested tasks may not carry the worker pointer */
        if ( worker ) child->Enqueue( worker->tid );
        else          child->Enqueue();
      }
    }
    child->task_lock.Release();
    out.pop_front();
  }
  status = DONE;
};


//void Task::Execute( Worker *user_worker )
//{
//  function( this );
//};


bool Task::ContextSwitchToNextTask( Worker *user_worker )
{
  auto *task = next;
  /** Find an unexecuted task */
  while ( task && task->GetStatus() != RUNNING ) task = task->next;
  /** If found, then execute it and change its status to "EXECUTED". */
  if ( task ) 
  {
    task->worker = user_worker;
    task->event.Begin( user_worker->tid );
    task->Execute( user_worker );
    task->SetStatus( EXECUTED );
  }
  return (bool)task;
};


/** virtual function */
void Task::GetEventRecord() {};

/** virtual function */
void Task::DependencyAnalysis() {};

/** try to dispatch the task if there is no dependency left */
void Task::TryEnqueue()
{
  if ( status == NOTREADY && !n_dependencies_remaining ) Enqueue();
};

void Task::Enqueue()
{
  Enqueue( 0 );
};

void Task::ForceEnqueue( size_t tid )
{
  int assignment = tid;

  rt.scheduler->ready_queue_lock[ assignment ].Acquire();
  {
    float cost = rt.workers[ assignment ].EstimateCost( this );
    status = QUEUED;
    if ( priority )
      rt.scheduler->ready_queue[ assignment ].push_front( this );
    else
      rt.scheduler->ready_queue[ assignment ].push_back( this );

    /** update the remaining time */
    rt.scheduler->time_remaining[ assignment ] += cost; 
  }
  rt.scheduler->ready_queue_lock[ assignment ].Release();
};


/**
 *  @brief 
 */ 
void Task::Enqueue( size_t tid )
{
  float cost = 0.0;
  float earliest_t = -1.0;
  int assignment = -1;

  /** dispatch to mpi queue */
  if ( has_mpi_routines )
  {
    rt.scheduler->mpi_queue_lock.Acquire();
    {
      /** change status */
      status = QUEUED;
      if ( priority )
        rt.scheduler->mpi_queue.push_front( this );
      else
        rt.scheduler->mpi_queue.push_back( this );
    }
    rt.scheduler->mpi_queue_lock.Release();

    /** finish and return without further going down */
    return;
  }

  /** dispatch to nested queue if in the epoch session */
  if ( is_created_in_epoch_session )
  {
    rt.scheduler->nested_queue_lock.Acquire();
    {
      /** change status */
      status = QUEUED;
      if ( priority )
        rt.scheduler->nested_queue.push_front( this );
      else
        rt.scheduler->nested_queue.push_back( this );
    }
    rt.scheduler->nested_queue_lock.Release();

    /** finish and return without further going down */
    return;
  };

  /** determine which work the task should go to using HEFT policy */
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

  /** dispatch to normal ready queue */
  rt.scheduler->ready_queue_lock[ assignment ].Acquire();
  {
    float cost = rt.workers[ assignment ].EstimateCost( this );
    status = QUEUED;
    if ( priority )
      rt.scheduler->ready_queue[ assignment ].push_front( this );
    else
      rt.scheduler->ready_queue[ assignment ].push_back( this );

    /** update the remaining time */
    rt.scheduler->time_remaining[ assignment ] += cost; 
  }
  rt.scheduler->ready_queue_lock[ assignment ].Release();

}; /** end Task::Enqueue() */


/**
 *  @brief 
 **/ 
void Task::CallBackWhileWaiting()
{
  rt.ExecuteNestedTasksWhileWaiting( this );
}; /** end CallBackWhileWaiting() */


/**
 *  @breief ReadWrite
 */ 
ReadWrite::ReadWrite() {};

/**
 *  @brief 
 **/ 
void ReadWrite::DependencyAnalysis( ReadWriteType type, Task *task )
{
  if ( type == R || type == RW )
  {
    read.push_back( task );
    /** read after write (RAW) data dependencies */
    for ( auto it = write.begin(); it != write.end(); it ++ )
    {
      Scheduler::DependencyAdd( (*it), task );
#ifdef DEBUG_RUNTIME
      printf( "RAW %s (%s) --> %s (%s)\n", 
          (*it)->name.data(), (*it)->label.data(), 
          task->name.data(), task->label.data() );
#endif
    }
  }

  if ( type == W || type == RW )
  {
    /** write after read (WAR) anti-dependencies */
    for ( auto it = read.begin(); it != read.end(); it ++ )
    {
      Scheduler::DependencyAdd( (*it), task );
#ifdef DEBUG_RUNTIME
      printf( "WAR %s (%s) --> %s (%s)\n", 
          (*it)->name.data(), (*it)->label.data(), 
          task->name.data(), task->label.data() );
#endif
    }
    write.clear();
    write.push_back( task );
    read.clear();
  }

}; /** end ReadWrite::DependencyAnalysis() */


/**
 *
 *
 */ 
void ReadWrite::DependencyCleanUp()
{
  read.clear();
  write.clear();
}; /** end DependencyCleanUp() */


/**
 *  @breief MatrixReadWrite
 */ 
MatrixReadWrite::MatrixReadWrite() {};


void MatrixReadWrite::Setup( size_t m, size_t n )
{
  //printf( "%lu %lu setup\n", m, n  );
  this->has_been_setup = true;
  this->m = m;
  this->n = n;
  Submatrices.resize( m );
  for ( size_t i = 0; i < m; i ++ ) Submatrices[ i ].resize( n );
};


bool MatrixReadWrite::HasBeenSetup()
{
  return has_been_setup;
}

void MatrixReadWrite::DependencyAnalysis( 
    size_t i, size_t j, ReadWriteType type, Task *task )
{
  //printf( "%lu %lu analysis\n", i, j  ); fflush( stdout );
  assert( i < m && j < n );
  Submatrices[ i ][ j ]. DependencyAnalysis( type, task );
};

void MatrixReadWrite::DependencyCleanUp()
{
  for ( size_t i = 0; i < m; i ++ )
    for ( size_t j = 0; j < n; j ++ )
      Submatrices[ i ][ j ]. DependencyCleanUp();
};



/**
 *  @brief Scheduler
 */ 
Scheduler::Scheduler() : timeline_tag( 500 )
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler()\n" );
#endif
  /** MPI support (use a private COMM_WORLD) */
  mpi::Comm_dup( MPI_COMM_WORLD, &comm );
  int size; mpi::Comm_size( comm, &size );
  listener_tasklist.resize( size );
  /** Set now as the begining of the time table */
  timeline_beg = omp_get_wtime();
};

Scheduler::~Scheduler()
{
#ifdef DEBUG_SCHEDULER
  printf( "~Scheduler()\n" );
#endif
};


void Scheduler::Init( int user_n_worker, int user_n_nested_worker )
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::Init()\n" );
#endif

  /** Adjust the number of active works. */
  n_worker = user_n_worker;
  /** Reset task counter. */
  n_task = 0;
	/** Reset async distributed consensus variables. */
	do_terminate = false;
  has_ibarrier = false;
  ibarrier_consensus = 0;

#ifdef USE_PTHREAD_RUNTIME
  if ( user_n_nested_worker > 1 )
  {
    printf( "pthread runtime does not support nested parallism\n" );
    exit( 1 );
  }

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
  /** now the master thread */
  EntryPoint( (void*)&(rt.workers[ 0 ]) );
#else

  /** nested setup */
  if ( user_n_nested_worker > 1 )
  {
    omp_set_dynamic( 0 );
    omp_set_nested( 1 );
    omp_set_max_active_levels( 2 );
#ifdef USE_INTEL
    mkl_set_dynamic( 0 );
    //mkl_set_num_threads( omp_get_max_threads() - n_worker );
    mkl_set_num_threads( user_n_nested_worker );
#endif
  }

  //printf( "mkl_get_max_threads %d\n", mkl_get_max_threads() );
  //printf( "before omp workers\n" ); fflush( stdout );

  #pragma omp parallel for num_threads( n_worker )
  for ( int i = 0; i < n_worker; i ++ )
  {
    /** setup nested thread number */
    omp_set_num_threads( user_n_nested_worker );

    rt.workers[ i ].tid = i;
    rt.workers[ i ].scheduler = this;
    EntryPoint( (void*)&(rt.workers[ i ]) );
  }

  if ( user_n_nested_worker > 1 )
  {
    omp_set_dynamic( 1 );
    omp_set_nested( 0 );
    omp_set_max_active_levels( 1 );
    omp_set_num_threads( omp_get_max_threads() );
#ifdef USE_INTEL
    mkl_set_dynamic( 1 );
    mkl_set_num_threads( omp_get_max_threads() );
#endif
  }
#endif
};

void Scheduler::MessageDependencyAnalysis( 
    int key, int p, ReadWriteType type, Task *task )
{
  int size; mpi::Comm_size( comm, &size );

  auto it = msg_dependencies.find( key );

  if ( it == msg_dependencies.end() )
  {
    //printf( "Create msg ReadWrite object %d\n", key ); fflush( stdout );
    msg_dependencies[ key ] = vector<ReadWrite>( size );
  }
  //printf( "Add rank %d to object %d\n", p, key ); fflush( stdout );
  msg_dependencies[ key ][ p ].DependencyAnalysis( type, task );
};


void Scheduler::NewTask( Task *task )
{
  //printf( "tasklist.size() %lu beg\n", tasklist.size() ); fflush( stdout );
  tasklist_lock.Acquire();
  {
    if ( rt.IsInEpochSession() ) 
    {
      nested_tasklist.push_back( task );
    }
    else tasklist.push_back( task );
  }
  tasklist_lock.Release();
  //printf( "tasklist.size() %lu end\n", tasklist.size() ); fflush( stdout );
};

void Scheduler::NewMessageTask( MessageTask *task )
{
  tasklist_lock.Acquire();
  {
    task->comm = comm;
    /** Counted toward termination criteria. */
    tasklist.push_back( task );
    //printf( "NewMessageTask src %d tar %d key %d tasklist.size() %lu\n",
    //    task->src, task->tar, task->key, tasklist.size() );
  }
  tasklist_lock.Release();
};

void Scheduler::NewListenerTask( ListenerTask *task )
{
  tasklist_lock.Acquire();
  {
    task->comm = comm;
    listener_tasklist[ task->src ][ task->key ] = task;
    /** Counted toward termination criteria. */
    tasklist.push_back( task );
    //printf( "NewListenerTask src %d tar %d key %d listener_tasklist[].size() %lu\n",
    //    task->src, task->tar, task->key, listener_tasklist[ task->src ].size() );
  }
  tasklist_lock.Release();
};



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

  /** Reset remaining time */
  for ( int i = 0; i < n_worker; i ++ ) time_remaining[ i ] = 0.0;
  /** Reset tasklist */
  for ( auto task : tasklist ) delete task; 
  tasklist.clear();
  /** Reset nested_tasklist */
  for ( auto task : nested_tasklist ) delete task; 
  nested_tasklist.clear();
  //printf( "Begin Scheduler::Finalize() [cleanup listener_tasklist]\n" );
  /** Reset listener_tasklist  */
  for ( auto p : listener_tasklist ) p.clear();
  //printf( "End   Scheduler::Finalize() [cleanup listener_tasklist]\n" );
  
  /** Clean up all message dependencies. */
  msg_dependencies.clear();

}; /** end Scheduler::Finalize() */

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
};


/**
 *  @brief Add an direct edge (dependency) from source to target. 
 *         That is to say, target depends on source.
 *
 */ 
void Scheduler::DependencyAdd( Task *source, Task *target )
{
  /** Avoid self-loop */
  if ( source == target ) return;

  /** Update the source list */
  source->task_lock.Acquire();
  {
    source->out.push_back( target );
  }
  source->task_lock.Release();

  /** Update the target list */
  target->task_lock.Acquire();
  {
    target->in.push_back( source );
    if ( source->GetStatus() != DONE )
    {
      target->n_dependencies_remaining ++;
    }
  }
  target->task_lock.Release();

}; /** end Scheduler::DependencyAdd() */


Task *Scheduler::TryStealFromQueue( size_t target )
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



/**
 *  @brief  Try to dispatch a task from the nested queue. This routine
 *          is only invoked by the runtime while nested tasks were
 *          created during the epoch session.
 **/ 
Task *Scheduler::TryDispatchFromNestedQueue()
{
  Task *nested_task = NULL;

  /** Early return if no available task. */
  if ( !nested_queue.size() ) return nested_task;

  nested_queue_lock.Acquire();
  { /** begin critical session "nested_queue" */
    if ( nested_queue.size() )
    {
      /** Fetch the first task in the queue */
      nested_task = nested_queue.front();
      /** Remove the task from the queue */
      nested_queue.pop_front();
    }
  } /** end critical session "nested_queue" */
  nested_queue_lock.Release();

  /** Notice that this can be a NULL pointer */
  return nested_task;

}; /** end Scheduler::TryDispatchFromNestedQueue() */



void Scheduler::SetBackGroundTask( Task *task )
{
  this->UnsetBackGroundTask();	
	this->bgtask = task;
};

Task *Scheduler::GetBackGroundTask()
{
	return bgtask;
};

void Scheduler::UnsetBackGroundTask()
{
	if ( this->bgtask ) delete this->bgtask;
	this->bgtask = NULL;
};

bool Scheduler::IsTimeToExit( int tid )
{
  /** In the case that do_terminate has been set */
  if ( do_terminate ) return true;
  //printf( "worker %d has do_terminate = true, scheduler->n_task = %d, tasklist.size() %lu\n", 
	//		me->tid, scheduler->n_task, scheduler->tasklist.size() ); fflush( stdout );

  if ( n_task >= tasklist.size() )
  {
		/** Set the termination flag to true */
	  do_terminate = true;
    /** Sanity check: no nested tasks should left */
    if ( nested_queue.size() )
    {
      printf( "bug nested_queue.size() %lu\n", nested_queue.size() ); 
      fflush( stdout );
    }
    /** Sanity check: no task should left */
    if ( ready_queue[ tid ].size() )
    {
      auto *task = ready_queue[ tid ].front();
      printf( "taskid %d, %s, tasklist.size() %lu  left\n", 
          task->taskid, task->name.data(), tasklist.size() ); fflush( stdout );
    }
    return true;
  }

  return false;
};


/**
 *  @brief Different from DependencyAdd, which asks programmer to manually
 *         add the dependency between two tasks. DependencyAnalysis track
 *         the read/write dependencies if the data extends hmlp::Object.
 *         
 *         There are 3 different data dependencies may happen: 
 *         RAW: read after write a.k.a data-dependency;
 *         WAR: write after read a.k.a anti-dependency, and
 *         WAW: write after write (we don't care).
 *
 *         Take a task that read/write the object, then this task depends
 *         on the previous write task. Since it also overwrites the object,
 *         this task must wait for all other tasks that want to read the
 *         previous state of the object.
 *
 */ 
//template<bool READ, bool WRITE>
//void Scheduler::DependencyAnalysis( Task *task, Object &object )
//{
//  // read and write are std::deque<Task*>.
//  auto &read = object.read;
//  auto &write = object.read;
//
//  if ( READ )
//  {
//    read.push_back( task );
//    // Read after write (RAW) data dependencies.
//    for ( auto it = write.begin(); it != write.end(); it ++ )
//    {
//      DependencyAdd( (*it), task );
//    }
//  }
//
//  if ( WRITE )
//  {
//    // Write after read (WAR) anti-dependencies.
//    for ( auto it = read.begin(); it != read.end(); it ++ )
//    {
//      DependencyAdd( (*it), task );
//    }
//    write.clear();
//    write.push_back( task );
//    read.clear();
//  }
//
//}; // end DependencyAnalysis()

Task *Scheduler::TryDispatch( int tid )
{
  size_t batch_size = 0;
  Task *batch = NULL;
  /** Enter critical section */
  ready_queue_lock[ tid ].Acquire();
  {
    if ( ready_queue[ tid ].size() )
    {
      /** Pop the front task */
      batch = ready_queue[ tid ].front();
      ready_queue[ tid ].pop_front();
      batch_size ++;

      /** Create a batched job if there is not enough flops */
      Task *task = batch;
      while ( ready_queue[ tid ].size() && batch_size < MAX_BATCH_SIZE + 1 )
      {
        task->next = ready_queue[ tid ].front();
        ready_queue[ tid ].pop_front();
        batch_size ++;
        task = task->next;
      }
    }
    else
    {
      /** Reset my workload counter */
      time_remaining[ tid ] = 0.0;
    }

    /** Try to prefetch the next task */
    //if ( ready_queue[ tid ].size() ) nexttask = ready_queue[ tid ].front();
  }
  ready_queue_lock[ tid ].Release();

  return batch;
};

Task *Scheduler::TryDispatchFromMPIQueue()
{
  Task *batch = NULL;
  mpi_queue_lock.Acquire();
  {
    if ( mpi_queue.size() )
    {
      batch = mpi_queue.front();
      mpi_queue.pop_front();
    }
  }
  mpi_queue_lock.Release();
  return batch;
};

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
          n_task ++;
        }
        n_task_lock.Release();
      }
      /**  Move to the next task in te batch */
      task = task->next;
    }
  }
  return true;
};


void* Scheduler::EntryPoint( void* arg )
{
  Worker *me = reinterpret_cast<Worker*>( arg );
  Scheduler *scheduler = me->scheduler;
  size_t idle = 0;

#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::EntryPoint()\n" );
  printf( "pthreadid %d\n", me->tid );
#endif

	/** if there is a background task, tid = 1 is assigned to it */
  //if ( me->tid < rt.n_background_worker )
	//{
	//	auto *bgtask = scheduler->GetBackGroundTask();
	//	if ( bgtask ) 
  //  {
	//		/** only use 1 thread */
  //    omp_set_num_threads( 1 );

  //    /** update my termination time to infinite */
  //    scheduler->ready_queue_lock[ me->tid ].Acquire();
  //    {
  //      scheduler->time_remaining[ me->tid ] = 999999.9;
  //    }
  //    scheduler->ready_queue_lock[ me->tid ].Release();

  //    //printf( "Enter background task\n" ); fflush( stdout );
  //    me->Execute( bgtask );
  //    //printf( "Exit  background task\n" ); fflush( stdout );
  //  }
	//}


  /** Prepare listeners */
  if ( me->tid == 1 )
  {
    /** Update my termination time to infinite */
    scheduler->ready_queue_lock[ me->tid ].Acquire();
    {
      scheduler->time_remaining[ me->tid ] = numeric_limits<float>::max();
    }
    scheduler->ready_queue_lock[ me->tid ].Release();
    /** Enter listening mode */
    scheduler->Listen( me );
  }








  /** Start to consume all tasks in this epoch session */
  while ( 1 )
  {
    Task *batch = NULL;
    Task *nexttask = NULL;

    if ( me->tid == 0 && scheduler->mpi_queue.size() )
    {
      batch = scheduler->TryDispatchFromMPIQueue();
    }
    else
    {
      batch = scheduler->TryDispatch( me->tid );
    }

    if ( nexttask ) nexttask->Prefetch( me );


    /** If there is some jobs to do */
    if ( scheduler->ConsumeTasks( me, batch, false /** !is_nested */ ) )
    {
      /** Reset the idle counter */
      idle = 0;
    }
    else /** No task in my ready_queue. Steal from others. */
    {
      /** Increase the idle counter */
      idle ++;
      /** Try to get a nested task; (can be a NULL pointer) */
      Task *nested_task = scheduler->TryDispatchFromNestedQueue();
      /** Reset the idle counter if there is executable nested tasks */
      if ( scheduler->ConsumeTasks( me, nested_task, true ) ) idle = 0;
    }

    /** Try to steal from others */
    if ( idle > 10 )
    {
      int max_remaining_task = 0;
      float max_remaining_time = 0.0;
      int target = -1;

      /** Decide which target to steal */
      for ( int p = 0; p < scheduler->n_worker; p ++ )
      {
        /** Update the target if it has the maximum amount of remaining tasks. */ 
        if ( scheduler->ready_queue[ p ].size() > max_remaining_task )
        {
          max_remaining_task = scheduler->ready_queue[ p ].size();
          target = p;
        }
      }

      /** If there is a target to steal, and it is not myself */ 
      if ( target >= 0 && target != me->tid )
      {
        Task *target_task = scheduler->TryStealFromQueue( target );
        /** Reset the idle counter if there is executable nested tasks */
        if ( scheduler->ConsumeTasks( me, target_task, false ) ) idle = 0;
      }
    } /** end if ( idle > 10 ) */

    /** Check if is time to terminate. */
    if ( scheduler->IsTimeToExit( me->tid ) ) break;
  }

  return NULL;
};


/** Listen asynchronous incoming MPI messages */
void Scheduler::Listen( Worker *worker )
{
  int size; mpi::Comm_size( comm, &size );
  int rank; mpi::Comm_rank( comm, &rank );

  /** Buffer for recv_sizes, recv_skels, recv_buffs */
  vector<size_t> recv_sizes;
  vector<size_t> recv_skels;

  /** Iprobe flag and recv status */
  int probe_flag = 0;
  mpi::Status status;

  /** Keep probing for messages */
  while ( 1 ) 
  {
    ListenerTask *task = NULL;

    /** Only one thread will probe and recv message at a time */
    #pragma omp critical
    {
      mpi::Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &probe_flag, &status );

      /** If receive any message, then handle it */
      if ( probe_flag )
      {
        /** Info from mpi::Status */
        int recv_src = status.MPI_SOURCE;
        int recv_tag = status.MPI_TAG;
        int recv_key = 3 * ( recv_tag / 3 );
        int recv_cnt;
   
        if ( recv_key < 300 )
        {
          printf( "rank %d Iprobe src %d tag %d\n", rank, recv_src, recv_tag ); 
          fflush( stdout );
        }

        /** Check if there is a corresponding task */
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

    /** Execute tasks and update dependencies */
    ConsumeTasks( worker, task, false );

    /** Check if is time to terminate */

    /** Nonblocking consensus for termination */
    if ( do_terminate ) 
    {
      /** 
       *  While do_terminate = true, there is at lease one worker
       *  can terminate. We use an ibarrier to make sure global concensus.
       */
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

};



void Scheduler::Summary()
{
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
	/** in the MPI environment, reduce all flops and mops */
	int rank = 0;
	double global_flops = 0.0, global_mops = 0.0;
	mpi::Comm_rank( MPI_COMM_WORLD, &rank );
	mpi::Reduce( &total_flops, &global_flops, 1, MPI_SUM, 0, MPI_COMM_WORLD );
	mpi::Reduce( &total_mops,  &global_mops,  1, MPI_SUM, 0, MPI_COMM_WORLD );
	if ( rank == 0 ) printf( "Epoch summary: flops %E mops %E\n", global_flops, global_mops );
#else
  printf( "Epoch summary: flops %E mops %E\n", total_flops, total_mops );
#endif


#ifdef DUMP_ANALYSIS_DATA
  std::deque<std::tuple<bool, double, size_t>> timeline;

  if ( tasklist.size() )
  {
    std::string filename = std::string( "timeline" ) + 
	  std::to_string( tasklist.size() ) + std::string( ".m" ); 
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
      timeline.push_back( std::make_tuple( true,  event.GetBegin(), i ) );
      timeline.push_back( std::make_tuple( false, event.GetEnd(),   i ) );
    }

    std::sort( timeline.begin(), timeline.end(), EventLess );

    for ( size_t i = 0; i < timeline.size(); i ++ )
    {
      auto &data = timeline[ i ];
      auto &event = tasklist[ std::get<2>( data ) ]->event;  
      event.Timeline( std::get<0>( data ), i + timeline_tag );
      event.MatlabTimeline( pFile );
    }

    fclose( pFile );

    timeline_tag += timeline.size();
  }
#endif

}; // end void Schediler::Summary()




RunTime::RunTime() :
  n_worker( 0 )
{
#ifdef DEBUG_RUNTIME
  printf( "Runtime()\n" );
#endif
};

RunTime::~RunTime()
{
#ifdef DEBUG_RUNTIME
  printf( "~Runtime()\n" );
#endif
};

void RunTime::Init()
{
  #pragma omp critical (init)
  {
    if ( !is_init )
    {
      n_worker = omp_get_max_threads();
      n_max_worker = n_worker;
      n_nested_worker = 1;
      scheduler = new Scheduler();

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
      is_init = true;
    }
  }
};

/**
 *  @brief
 **/
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
  scheduler->Init( n_worker, n_nested_worker );
  /** clean up */
  scheduler->Finalize();
  /** finish this epoch session */
  is_in_epoch_session = false;
};

void RunTime::Finalize()
{
  #pragma omp critical (init)
  {
    if ( is_init )
    {
      scheduler->Finalize();
      delete scheduler;
      is_init = false;
    }
  }
};

bool RunTime::IsInEpochSession()
{
  return is_in_epoch_session;
};

void RunTime::ExecuteNestedTasksWhileWaiting( Task *waiting_task )
{
  if ( IsInEpochSession() )
  {
    while ( waiting_task->GetStatus() != DONE )
    {
      /** first try to consume tasks in the nested queue */
      if ( this->scheduler->nested_queue.size() )
      {
        /** try to get a nested task; (can be a NULL pointer) */
        Task *nested_task = this->scheduler->TryDispatchFromNestedQueue();

        if ( nested_task )
        {
          nested_task->SetStatus( RUNNING );
          nested_task->Execute( NULL );
          nested_task->DependenciesUpdate();
        }
      }
    }
  }
};

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


hmlp::Device *hmlp_get_device_host()
{
  return &(hmlp::rt.host);
};


}; // end namespace hmlp

void hmlp_init()
{
  hmlp::rt.Init();
};

void hmlp_set_num_workers( int n_worker )
{
  if ( n_worker != hmlp::rt.n_worker )
  {
    hmlp::rt.n_nested_worker = hmlp::rt.n_max_worker / n_worker;
    hmlp::rt.n_worker = n_worker;
  }
  printf( "%2d/%2d (n_worker/n_nested_worker)\n",
      hmlp::rt.n_worker, hmlp::rt.n_nested_worker ); fflush( stdout );
};

void hmlp_redistribute_workers( 
		int n_worker, 
		int n_background_worker,
		int n_nested_worker )
{
	hmlp::rt.n_worker = n_worker;
	hmlp::rt.n_background_worker = n_background_worker;
	hmlp::rt.n_nested_worker = n_nested_worker;



	int rank = 0;
	hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &rank );
	if ( rank == 0 )
  printf( "%2d/%2d/%2d/%2d (n_worker/n_background_worker/n_nested_worker/max_worker)\n",
      hmlp::rt.n_worker, 
			hmlp::rt.n_background_worker, 
			hmlp::rt.n_nested_worker, 
			omp_get_max_threads() ); fflush( stdout );
};

void hmlp_run()
{
  hmlp::rt.Run();
};

void hmlp_finalize()
{
  hmlp::rt.Finalize();
};

hmlp::RunTime *hmlp_get_runtime_handle()
{
  return &hmlp::rt;
};

hmlp::Device *hmlp_get_device( int i )
{
  return hmlp::rt.device[ i ];
};

bool hmlp_is_in_epoch_session()
{
  return hmlp::rt.IsInEpochSession();
};

void hmlp_msg_dependency_analysis( 
    int key, int p, hmlp::ReadWriteType type, hmlp::Task *task )
{
  hmlp::rt.scheduler->MessageDependencyAnalysis(
      key, p, type, task );
};

void hmlp_set_num_background_worker( int n_background_worker )
{
	int rank = 0;
	hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &rank );
  if ( n_background_worker <= 0 )
  {
    if ( rank == 0 ) printf( "(WARNING!) no background worker left\n" ); fflush( stdout );
    n_background_worker = 0;
  }
  hmlp::rt.n_background_worker = n_background_worker;
};
