#include <hmlp_runtime.hpp>

#ifdef HMLP_USE_CUDA
#include <hmlp_gpu.hpp>
#endif

#ifdef HMLP_USE_MAGMA
#include <magma_v2.h>
#include <magma_lapack.h>
#endif

#define MAX_BATCH_SIZE 4

// #define DEBUG_RUNTIME 1
// #define DEBUG_SCHEDULER 1


struct 
{
  bool operator()( std::tuple<bool, double, size_t> a, std::tuple< bool , double, size_t> b )
  {   
    return std::get<1>( a ) < std::get<1>( b );
  }   
} EventLess;



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
      {
        assert( nparts == 4 );
        int len = end - beg - 1;
        int big = ( len * 30 ) / 100 + 1;
        int sma = ( len * 20 ) / 100 + 1;

        int tid_beg, tid_end;

        if ( tid == 0 )
        {
          tid_beg = beg;
          tid_end = beg + big;
        }
        beg += big;

        if ( tid == 1 )
        {
          tid_beg = beg;
          tid_end = beg + sma;
        }
        beg += sma;

        if ( tid == 2 )
        {
          tid_beg = beg;
          tid_end = beg + sma;
        }
        beg += sma;

        if ( tid == 3 )
        {
          tid_beg = beg;
          tid_end = beg + big;
        }
        beg += big;

        if ( tid_end > end ) tid_end = end;
        return range( tid_beg, tid_end, nb );
      }
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


Event::Event() : flops( 0.0 ), mops( 0.0 ), beg( 0.0 ), end( 0.0 ), sec( 0.0 ) {};

//Event::Event( float _flops, float _mops ) : beg( 0.0 ), end( 0.0 ), sec( 0.0 )
//{
//  flops = _flops;
//  mops = _mops;
//};


void Event::Set( std::string _label, double _flops, double _mops )
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
  status = ALLOCATED;
  //rt.scheduler->NewTask( this );
  status = NOTREADY;
};

Task::~Task()
{};

TaskStatus Task::GetStatus()
{
  return status;
};

/** change the status of all tasks in the batch */
void Task::SetStatus( TaskStatus next_status )
{
  auto *task = this;
  while ( task )
  {
    task->status = next_status;
    /** move to the next task in the batch */
    task = task->next;
  }
};

void Task::Submit()
{
  rt.scheduler->NewTask( this );
};

void Task::Set( std::string user_name, void (*user_function)(Task*), void *user_arg )
{
  name = user_name;
  function = user_function;
  arg = user_arg;
  status = NOTREADY;
};

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
        child->Enqueue( worker->tid );
      }
    }
    child->task_lock.Release();
    out.pop_front();
  }
  status = DONE;
};


void Task::Execute( Worker *user_worker )
{
  function( this );
};

void Task::GetEventRecord() {};

void Task::DependencyAnalysis() {};


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

  // Determine which work the task should go to using HEFT policy.
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

//  if ( assignment == 0 )
//  {
//    rt.scheduler->ReportRemainingTime();
//  }
//
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
 *  @breief ReadWrite
 */ 
ReadWrite::ReadWrite() {};

void ReadWrite::DependencyAnalysis( ReadWriteType type, Task *task )
{
  if ( type == R || type == RW )
  {
    read.push_back( task );
    // Read after write (RAW) data dependencies.
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
    // Write after read (WAR) anti-dependencies.
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

}; // end DependencyAnalysis()



/**
 *  @brief Scheduler
 */ 
Scheduler::Scheduler() : timeline_tag( 500 )
{
#ifdef DEBUG_SCHEDULER
  printf( "Scheduler()\n" );
#endif
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

  /** adjust the number of active works */
  n_worker = user_n_worker;

  /** reset task counter */
  n_task = 0;

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
  }

  //printf( "mkl_get_max_threads %d\n", mkl_get_max_threads() );

  printf( "before omp workers\n" ); fflush( stdout );

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
  }
#endif
};


void Scheduler::NewTask( Task *task )
{
  tasklist_lock.Acquire();
  {
    tasklist.push_back( task );
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
  Summary();

  /** reset remaining time */
  for ( int i = 0; i < n_worker; i ++ )
  {
    time_remaining[ i ] = 0.0;
  }

  /** reset tasklist */
  for ( auto it = tasklist.begin(); it != tasklist.end(); it ++ )
  {
    delete *it; 
  }
  tasklist.clear();
};

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
  // Avoid self-loop
  if ( source == target ) return;

  // Update the source list.
  source->task_lock.Acquire();
  {
    source->out.push_back( target );
  }
  source->task_lock.Release();

  // Update the target list.
  target->task_lock.Acquire();
  {
    target->in.push_back( source );
    if ( source->GetStatus() != DONE )
    {
      target->n_dependencies_remaining ++;
    }
  }
  target->task_lock.Release();
}; // end DependencyAdd()


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


void* Scheduler::EntryPoint( void* arg )
{
  Worker *me = reinterpret_cast<Worker*>( arg );
  Scheduler *scheduler = me->scheduler;
  size_t idle = 0;

#ifdef DEBUG_SCHEDULER
  printf( "Scheduler::EntryPoint()\n" );
  printf( "pthreadid %d\n", me->tid );
#endif

  while ( 1 )
  {
    size_t batch_size = 0;
    Task *batch = NULL;
    Task *nexttask = NULL;

    scheduler->ready_queue_lock[ me->tid ].Acquire();
    {
      if ( scheduler->ready_queue[ me->tid ].size() )
      {
        /** pop the front task */
        batch = scheduler->ready_queue[ me->tid ].front();
        scheduler->ready_queue[ me->tid ].pop_front();
        batch_size ++;

        /** create a batched job if there is not enough flops */
        if ( me->GetDevice() && batch->cost < 0.5 )
        {
          Task *task = batch;
          while ( scheduler->ready_queue[ me->tid ].size() && 
                  batch_size < MAX_BATCH_SIZE + 1 )
          {
            task->next = scheduler->ready_queue[ me->tid ].front();
            scheduler->ready_queue[ me->tid ].pop_front();
            batch_size ++;
            task = task->next;
          }
        }
      }
      else
      {
        /** reset my workload counter */
        scheduler->time_remaining[ me->tid ] = 0.0;
      }

      if ( scheduler->ready_queue[ me->tid ].size() )
      {
        /** try to prefetch the next task */
        nexttask = scheduler->ready_queue[ me->tid ].front();
      }
    }
    scheduler->ready_queue_lock[ me->tid ].Release();

    if ( nexttask ) nexttask->Prefetch( me );


    if ( batch )
    {
      idle = 0;
      batch->SetStatus( RUNNING );

      if ( me->Execute( batch ) )
      {
        Task *task = batch;
        while ( task )
        {
          scheduler->ready_queue_lock[ me->tid ].Acquire();
          {
            scheduler->time_remaining[ me->tid ] -= task->cost;
            if ( scheduler->time_remaining[ me->tid ] < 0.0 )
              scheduler->time_remaining[ me->tid ] = 0.0;
          }
          scheduler->ready_queue_lock[ me->tid ].Release();

          task->DependenciesUpdate();
          scheduler->n_task_lock.Acquire();
          {
            scheduler->n_task ++;
          }
          scheduler->n_task_lock.Release();
          /** move to the next task in te batch */
          task = task->next;
        }
      }
    }
    else // No task in my ready_queue. Steal from others.
    {
      idle ++;

      if ( idle > 10 )
      {
        int max_remaining_task = 0;
        float max_remaining_time = 0.0;
        int target = -1;

        /** only steal jobs within the numa node */
        //int numa_grp = 2;
        //if ( scheduler->n_worker > 31 ) numa_grp = 4;
        //int numa_beg = ( me->tid / numa_grp ) * ( scheduler->n_worker / numa_grp );
        //int numa_end = numa_beg + ( scheduler->n_worker / numa_grp );

        //for ( int p = numa_beg; p < numa_end; p ++ )
        /** TODO: do not steal job from 0 (with GPU) */
        for ( int p = 0; p < scheduler->n_worker; p ++ )
        {
          //printf( "worker %d try to steal from worker %d\n", me->tid, p );  
          //if ( scheduler->time_remaining[ p ] > max_remaining_time )
          //{
          //  max_remaining_time = scheduler->time_remaining[ p ];
          //  target = p;
          //}

          if ( scheduler->ready_queue[ p ].size() > max_remaining_task )
          {
            max_remaining_task = scheduler->ready_queue[ p ].size();
            target = p;
          }
        }

        if ( target >= 0 && target != me->tid )
        {
          Task *target_task = NULL;
          scheduler->ready_queue_lock[ target ].Acquire();
          {
            if ( scheduler->ready_queue[ target ].size() ) 
            {
              target_task = scheduler->ready_queue[ target ].back();
              if ( target_task ) scheduler->ready_queue[ target ].pop_back();
              scheduler->time_remaining[ target ] -= target_task->cost;
            }
          }
          scheduler->ready_queue_lock[ target ].Release();
          if ( target_task )
          {
            //scheduler->ready_queue_lock[ me->tid ].Acquire();
            //{
            //  scheduler->ready_queue[ me->tid ].push_back( target_task );
            //  scheduler->time_remaining[ me->tid ] += target_task->cost;
            //}
            //scheduler->ready_queue_lock[ me->tid ].Release();

            idle = 0;
            target_task->SetStatus( RUNNING );
            if ( me->Execute( target_task ) )
            {
              target_task->DependenciesUpdate();
              scheduler->n_task_lock.Acquire();
              {
                scheduler->n_task ++;
              }
              scheduler->n_task_lock.Release();
            }
          }
        }
      }
    }

    if ( scheduler->n_task >= scheduler->tasklist.size() )
    {
      /** sanity check: no task should left */
      if ( scheduler->ready_queue[ me->tid ].size() == 0 )
      {
        break;
      }
      else
      {
        auto *task = scheduler->ready_queue[ me->tid ].front();
        printf( "taskid %d, %s, tasklist.size() %lu  left\n", 
            task->taskid, task->name.data(),
            scheduler->tasklist.size() ); fflush( stdout );
      }
    }
    else
    {
      //printf( "worker %d\n", me->tid ); fflush( stdout );
      //#pragma omp barrier
      //if ( me->tid  == 0 ) printf( "\nnext\n" ); fflush( stdout );
    }
  }

  return NULL;
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
  printf( "flops %E mops %E\n", total_flops, total_mops );


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

void RunTime::Run()
{
  if ( !is_init ) 
  {
    Init();
  }
  scheduler->Init( n_worker, n_nested_worker );
  scheduler->Finalize();
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
