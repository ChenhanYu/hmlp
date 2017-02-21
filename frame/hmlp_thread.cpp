#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>

namespace hmlp
{

std::ostream& operator<<( std::ostream& os, const thread_communicator& obj )
{
  os << obj.name << obj.comm_id;
  return os;
};

/**
 *  @brief Recursively create tree base communicators.
 */ 
void thread_communicator::Create( int level, int num_threads, int *config )
{
  if ( level < 2 ) 
  {
    kids = NULL;
  }
  else 
  {
    n_threads = num_threads;
    n_groups = config[ level ]; 

    if      ( level == 4 ) name = std::string( "jc_comm" );
    else if ( level == 3 ) name = std::string( "pc_comm" );
    else if ( level == 2 ) name = std::string( "ic_comm" );
    else if ( level == 1 ) name = std::string( "jr_comm" );
    else                   name = std::string( "na_comm" );

    // std::cout << name << ", " << n_threads << ", " << n_groups << "\n";

    kids = new thread_communicator[ n_groups ]();
    for ( int i = 0; i < n_groups; i ++ ) 
    {
      kids[ i ].Create( level - 1, n_threads / n_groups, config );
    }
  }
};

thread_communicator::thread_communicator() :
  sent_object( NULL ), 
  comm_id( 0 ),
  n_threads( 0 ), 
  n_groups( 0 ),
  barrier_sense( false ),
  barrier_threads_arrived( 0 )
{};

/**
 *  @brief Default constructor takes 4 partitioning numbers.
 */ 
thread_communicator::thread_communicator( int jc_nt, int pc_nt, int ic_nt, int jr_nt ) :
  sent_object( NULL ), 
  comm_id( 0 ),
  n_threads( 0 ), 
  n_groups( 0 ),
  barrier_sense( false ),
  barrier_threads_arrived( 0 )
{
  int config[ 6 ] = { 0, 0, jr_nt, ic_nt, pc_nt, jc_nt };
  n_threads = jc_nt * pc_nt * ic_nt * jr_nt;
  n_groups  = jc_nt;
  name = std::string( "my_comm" );
  kids = new thread_communicator[ n_groups ]();

  for ( int i = 0; i < n_groups; i ++ ) 
  {
    kids[ i ].Create( 4, n_threads / n_groups, config );
  }
};


int thread_communicator::GetNumThreads() 
{
  return n_threads;
};

int thread_communicator::GetNumGroups() 
{
  return n_groups;
};

/**
 *  @brief OpenMP thread barrier from BLIS.
 */  
void thread_communicator::Barrier()
{
  if ( n_threads < 2 ) return;

  bool my_sense = barrier_sense;
  int my_threads_arrived;

  #pragma omp atomic capture
  my_threads_arrived = ++ barrier_threads_arrived;

  if ( my_threads_arrived == n_threads )
  {
    barrier_threads_arrived = 0;
    barrier_sense = !barrier_sense;
  }
  else
  {
    volatile bool *listener = &barrier_sense;
    while ( *listener == my_sense ) {}
  }
};

void thread_communicator::Print()
{
  Barrier();
};


/**
 *  @brief Device implementation
 */
Device::Device()
{};


/**
 *  @brief Worker implementation
 */ 
Worker::Worker() :
  device( NULL )
{};

Worker::Worker( thread_communicator *comm ) :
  tid( 0 ), 
  jc_id( 0 ), 
  pc_id( 0 ), 
  ic_id( 0 ), 
  jr_id( 0 ),
  device( NULL )
{
  int tmp;

  tid   = omp_get_thread_num();
  tmp   = tid;

  my_comm = comm;

  jc_nt = my_comm->GetNumGroups();
  jc_id = tmp / ( my_comm->GetNumThreads() / my_comm->GetNumGroups() );
  tmp   = tmp % ( my_comm->GetNumThreads() / my_comm->GetNumGroups() );

  jc_comm = &(my_comm->kids[ jc_id ]);

  pc_nt = jc_comm->GetNumGroups();
  pc_id = tmp / ( jc_comm->GetNumThreads() / jc_comm->GetNumGroups() );
  tmp   = tmp % ( jc_comm->GetNumThreads() / jc_comm->GetNumGroups() );

  pc_comm = &(jc_comm->kids[ pc_id ]);

  ic_jr = tmp; // for parallel packB
  ic_nt = pc_comm->GetNumGroups();
  ic_id = tmp / ( pc_comm->GetNumThreads() / pc_comm->GetNumGroups() );
  jr_id = tmp % ( pc_comm->GetNumThreads() / pc_comm->GetNumGroups() );

  ic_comm = &(pc_comm->kids[ ic_id ]);
  jr_nt = ic_comm->GetNumGroups();

  //printf( "tid %2d jc_id %2d pc_id %2d ic_id %2d jr_id %2d, ic_jr %2d\n",
  //    tid, jc_id, pc_id, ic_id, jr_id, ic_jr );
};

/**
 *  @brief The work executes the task in the runtime system. I left some
 *         code commented out because there is no GPU support now.
 *         With GPUs (or some distributed devices), we need to first 
 *         gather data before the execution.
 *
 *  @param *task The current task pointer.
 *
 */ 
bool Worker::Execute( Task *task )
{
  current_task = task;
  task->worker = this;

  // Fetching data from GPU memory or from other processes.
  // Fetch( task );
  // Prefetch( task );

//#ifdef DUMP_ANALYSIS_DATA
  task->event.Begin( this->tid );
//#endif
  task->Execute( this );
  WaitExecute();
//#ifdef DUMP_ANALYSIS_DATA
  task->event.Terminate();
  task->GetEventRecord();
//#endif

  // WaitPrefetch

  current_task = NULL;

  return true;

}; // end bool Worker::Execute()



/**
 *  @brief Pose a barrier if the device owned by this worker
 *         is performing asybchronous execution.
 */ 
void Worker::WaitExecute()
{
  // Synchroize device
};

float Worker::EstimateCost( class Task * task )
{
  return task->cost;
};





}; // end namespace hmlp
