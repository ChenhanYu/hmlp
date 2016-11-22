#include <hmlp_runtime.hpp>

// #define DEBUG_RUNTIME 1
// #define DEBUG_SCHEDULER 1


namespace hmlp
{

static RunTime rt;

range::range( int beg, int end, int inc )
{
  info = std::make_tuple( beg, end, inc );
};

int range::beg()
{
  return std::get<0>( info );
};

int range::end()
{
  return std::get<1>( info );
};

int range::inc()
{
  return std::get<2>( info );
};

range GetRange
( 
  SchedulePolicy strategy, 
  int beg, int end, int nb, int tid, int nparts 
)
{
  switch ( strategy )
  {
    case HMLP_SCHEDULE_DEFAULT:
      {
        auto tid_beg = beg + tid * nb;
        auto tid_inc = nparts * nb;
        return range( tid_beg, end, tid_inc );
      }
    case HMLP_SCHEDULE_ROUND_ROBIN:
      {
        auto tid_beg = beg + tid * nb;
        auto tid_inc = nparts * nb;
        return range( tid_beg, end, tid_inc );
      }
    case HMLP_SCHEDULE_UNIFORM:
      printf( "GetRange(): HMLP_SCHEDULE_UNIFORM not yet implemented yet.\n" );
      exit( 1 );
    case HMLP_SCHEDULE_HEFT:
      printf( "GetRange(): HMLP_SCHEDULE_HEFT not yet implemented yet.\n" );
      exit( 1 );
    default:
      printf( "GetRange(): not a legal scheduling strategy.\n" );
      exit( 1 );
  }
};

range GetRange( int beg, int end, int nb, int tid, int nparts )
{
  return GetRange( HMLP_SCHEDULE_DEFAULT, beg, end, nb, tid, nparts );
};

range GetRange( int beg, int end, int nb )
{
  return GetRange( HMLP_SCHEDULE_DEFAULT, beg, end, nb, 0, 1 );
};

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


/**
 *  @brief Task
 */ 
Task::Task()
{};

Task::~Task()
{};

void Task::DependenciesUpdate()
{
  while ( out.size() )
  {
    Task *child = out.front();

    child->task_lock.Acquire();
    {
      child->n_dependencies_remaining --;
      if ( !child->n_dependencies_remaining && child->status == NOTREADY )
      {
        child->Enqueue();
      }
    }
    child->task_lock.Release();
    out.pop_front();
  }
  status = DONE;
};

void Task::Enqueue()
{
  float cost = 0.0;
  float earliest_t = -1.0;
  int assignment = -1;

  for ( int i = 0; i < rt.n_worker; i ++ )
  {
    float terminate_t = rt.scheduler.time_remaining[ i ];
    float cost = rt.workers[ i ].EstimateCost( this );
    if ( earliest_t == -1.0 || terminate_t + cost < earliest_t )
    {
      earliest_t = terminate_t + cost;
      assignment = i;
    }
  }

  rt.scheduler.ready_queue_lock[ assignment ].Acquire();
  {
    status = QUEUED;
    rt.scheduler.time_remaining[ assignment ] = earliest_t;
    rt.scheduler.ready_queue[ assignment ].push_back( this );
  }
  rt.scheduler.ready_queue_lock[ assignment ].Release();
};




/**
 *  @brief Scheduler
 */ 
Scheduler::Scheduler()
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler()\n" );
#endif
};

Scheduler::~Scheduler()
{
#ifdef DEBUG_SCHEDULER
  printf( "~Scheduler()\n" );
#endif
};


void Scheduler::Init( int n_worker )
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::Init()\n" );
#endif
  n_task = 0;
#ifdef USE_PTHREAD_RUNTIME
  for ( int i = 0; i < rt.n_worker; i ++ )
  {
    time_remaining[ i ] = 0.0;
    rt.workers[ i ].tid = i;
    rt.workers[ i ].scheduler = this;
    pthread_create
    ( 
      &(rt.workers[ i ].pthreadid), NULL,
      EntryPoint, (void*)&(rt.workers[i])
    );
  }
#else
  #pragma omp parallel num_threads( n_worker )
  {
    EntryPoint( NULL );
  }
#endif
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
};

void* Scheduler::EntryPoint( void* arg )
{
  Worker *me = reinterpret_cast<Worker*>( arg );
  Scheduler *scheduler = me->scheduler;

#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::EntryPoint()\n" );
  printf( "pthreadid %d\n", me->tid );
#endif

  while ( 1 )
  {
    Task *task = NULL;

    scheduler->ready_queue_lock[ me->tid ].Acquire();
    {
      if ( scheduler->ready_queue[ me->tid ].size() )
      {
        task = scheduler->ready_queue[ me->tid ].front();
        scheduler->ready_queue[ me->tid ].pop_front();
      }
    }
    scheduler->ready_queue_lock[ me->tid ].Release();

    if ( task )
    {
      bool committed = me->Execute( task );
      
      if ( committed )
      {
        task->DependenciesUpdate();
        scheduler->n_task_lock.Acquire();
        {
          scheduler->n_task ++;
        }
        scheduler->n_task_lock.Release();
      }
    }

    if ( 1 )
    {
      break;
    }
  }

  return NULL;
};


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
      //pool_init();
    }
  }
  n_worker = 4;
  scheduler.Init( n_worker );
};

void RunTime::Finalize()
{
  #pragma omp critical (init)
  {
    //printf( "hmlp_finalize()\n" );
  }
  scheduler.Finalize();
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

}; // end namespace hmlp

void hmlp_init()
{
  hmlp::rt.Init();
};

void hmlp_finalize()
{
  hmlp::rt.Finalize();
};

