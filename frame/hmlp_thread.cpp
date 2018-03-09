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


#include <hmlp_thread.hpp>

#ifdef HMLP_HAVE_RUNTIME
#include <hmlp_runtime.hpp>
#else
#warning HMLP runtime system is not compiled (-HMLP_HAVE_RUNTIME=false)
#endif /** ifdef HMLP_HAVE_RUNTIME */


using namespace std;


namespace hmlp
{

std::ostream& operator<<( std::ostream& os, const thread_communicator& obj )
{
  os << obj.name << obj.comm_id;
  return os;
};



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
 *  @brief Recursively create tree base communicators.
 */ 
void thread_communicator::Create( int level, int num_threads, int *config )
{
  if ( level < 1 ) 
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

    //std::cout << name << ", " << n_threads << ", " << n_groups << "\n";

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
  n_threads( 1 ), 
  n_groups( 1 ),
  barrier_sense( false ),
  barrier_threads_arrived( 0 )
{};

/**
 *  @brief Default constructor takes 4 partitioning numbers.
 */ 
thread_communicator::thread_communicator( int jc_nt, int pc_nt, int ic_nt, int jr_nt ) :
  sent_object( NULL ), 
  comm_id( 0 ),
  n_threads( 1 ), 
  n_groups( 1 ),
  barrier_sense( false ),
  barrier_threads_arrived( 0 )
{
  int config[ 6 ] = { 1, 1, jr_nt, ic_nt, pc_nt, jc_nt };
  n_threads = jc_nt * pc_nt * ic_nt * jr_nt;
  //n_groups  = jc_nt;
  n_groups  = 1;
  name = std::string( "my_comm" );
  kids = new thread_communicator[ n_groups ]();

  for ( int i = 0; i < n_groups; i ++ ) 
  {
    kids[ i ].Create( 5, n_threads / n_groups, config );
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

void thread_communicator::Send( void** buffer )
{
  sent_object = *buffer;
};

void thread_communicator::Recv( void** buffer )
{
  *buffer = sent_object;
};

/**
 *  @brief Device implementation
 */
//Device::Device()
//{
//  name = std::string( "Host CPU" );
//  devicetype = hmlp::DeviceType::HOST;
//};
//
//void Device::prefetchd2h( void *ptr_h, void *ptr_d, size_t size, int stream_id ) {};
//
//void Device::prefetchh2d( void *ptr_d, void *ptr_h, size_t size, int stream_id ) {};
//
//void Device::wait( int stream_id ) {};
//
//void *Device::malloc( size_t size ) { return NULL; };
//
//void Device::malloc( void *ptr_d, size_t size ) {};
//
//size_t Device::get_memory_left() { return 0; };
//
//
//void Device::free( void *ptr_d, size_t size ) {};




/**
 *  @brief Worker implementation
 */ 
Worker::Worker() {};

Worker::Worker( thread_communicator *comm ) :
  tid( 0 ), 
  jc_id( 0 ), 
  pc_id( 0 ), 
  ic_id( 0 ), 
  jr_id( 0 )
{
  int tmp;

  tid   = omp_get_thread_num();
  tmp   = tid;

  //my_comm = comm;
  my_comm = &(comm->kids[ 0 ]);

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

void Worker::Communicator( thread_communicator *comm )
{
  int tmp;

  tid   = omp_get_thread_num();
  tmp   = tid;

  //my_comm = comm;
  my_comm = &(comm->kids[ 0 ]);

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
};

bool Worker::Master() { return tid == 0; };

void Worker::Barrier() 
{
  assert( comm );
  comm->Barrier();
};


//void Worker::Bcast( void** buffer )
//{
//  if ( Master() ) comm->Send( buffer );
//  Barrier();
//  if ( !Master() ) comm->Recv( buffer );
//};



void Worker::InitWithCommunicator( thread_communicator* comm, size_t tid, size_t gid )
{
  this->comm = comm;
  this->tid = tid;
  this->gid = gid;
};

Worker Worker::Split()
{
  Worker child;

  //printf( "Before Split %s\n", comm->name.data() ); fflush( stdout );

  size_t n_splits  = comm->GetNumGroups();
  size_t n_threads = comm->GetNumThreads();

  //printf( "Split %s n_split %lu n_thread %lu\n", 
  //    comm->name.data(), n_splits, n_threads ); fflush( stdout );

  if ( n_splits == 1 )
  {
    child.InitWithCommunicator( &(comm->kids[ 0 ]), tid, 0 );
    return child;
  }

  /** 
   *  By default, we split threads evenly usine "close" affinity.
   *  Threads with the same color will be in the same subcomm.
   *
   *  example: (n_splits=2)
   *
   *  tid        0  1  2  3  4  5  6  7  
   *  color      0  0  0  0  1  1  1  1  (gang id)
   *  first      0  0  0  0  4  4  4  4 
   *  last       4  4  4  4  8  8  8  8
   *  child_tid  0  1  2  3  0  1  2  3
   *             4  4  4  4  4  4  4  4  (child_n_threads)
   */
  size_t color = ( n_splits * tid ) / n_threads;
  size_t first = ( ( color + 0 ) * n_threads ) / n_splits;
  size_t last  = ( ( color + 1 ) * n_threads ) / n_splits;
  size_t child_tid = tid - first;
  size_t child_n_threads = last - first;

  //printf( "Split %s n_split %lu n_thread %lu color %lu\n", 
  //    comm->name.data(), n_splits, n_threads, color ); fflush( stdout );

  gid = color;

  /** make sure all threads have the new communicator */
  child.InitWithCommunicator( &(comm->kids[ gid ]), child_tid, 0 );

  return child;

};


size_t Worker::BalanceOver1DGangs( size_t n, size_t default_size, size_t nb )
{
  size_t nparts = comm->GetNumGroups();
  if ( nparts > 1 ) return ( ( n - 1 ) / ( nb * nparts ) + 1 ) * nb;
  return default_size;
};


tuple<size_t, size_t, size_t> Worker::DistributeOver1DGangs(
    size_t beg, size_t end, size_t nb )
{
  size_t nparts = comm->GetNumGroups();

  SchedulePolicy strategy = HMLP_SCHEDULE_DEFAULT;

  switch ( strategy )
  {
    case HMLP_SCHEDULE_DEFAULT:
    {
      size_t gid_beg = beg + gid * nb;
      size_t gid_inc = nparts * nb;
      return make_tuple( gid_beg, end, gid_inc );
    }
    case HMLP_SCHEDULE_ROUND_ROBIN:
    {
      auto gid_beg = beg + gid * nb;
      auto gid_inc = nparts * nb;
      return make_tuple( gid_beg, end, gid_inc );
    }
    case HMLP_SCHEDULE_UNIFORM:
    {
      printf( "GetRange(): HMLP_SCHEDULE_UNIFORM not yet implemented yet.\n" );
      exit( 1 );
    }
    case HMLP_SCHEDULE_HEFT:
    {
      printf( "GetRange(): HMLP_SCHEDULE_HEFT not yet implemented yet.\n" );
      exit( 1 );
    }
    default:
    {
      exit( 1 );
    }
  }
};




tuple<size_t, size_t, size_t> Worker::DistributeOver1DThreads(
    size_t beg, size_t end, size_t nb )
{
  size_t nparts = comm->GetNumThreads();

  SchedulePolicy strategy = HMLP_SCHEDULE_DEFAULT;

  switch ( strategy )
  {
    case HMLP_SCHEDULE_DEFAULT:
    {
      size_t tid_beg = beg + tid * nb;
      size_t tid_inc = nparts * nb;
      return make_tuple( tid_beg, end, tid_inc );
    }
    case HMLP_SCHEDULE_ROUND_ROBIN:
    {
      auto tid_beg = beg + tid * nb;
      auto tid_inc = nparts * nb;
      return make_tuple( tid_beg, end, tid_inc );
    }
    case HMLP_SCHEDULE_UNIFORM:
    {
      printf( "GetRange(): HMLP_SCHEDULE_UNIFORM not yet implemented yet.\n" );
      exit( 1 );
    }
    case HMLP_SCHEDULE_HEFT:
    {
      printf( "GetRange(): HMLP_SCHEDULE_HEFT not yet implemented yet.\n" );
      exit( 1 );
    }
    default:
    {
      exit( 1 );
    }
  }
};



















#ifdef HMLP_HAVE_RUNTIME

void Worker::SetDevice( class Device *device )
{
  this->device = device;
};

class Device* Worker::GetDevice()
{
  return device;
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
bool Worker::Execute( Task *batch )
{
  current_task = batch;
  Task *task = batch;

  while ( task )
  {
    /** Some tasks may be in "EXECUTED" status. */
    if ( task->GetStatus() == RUNNING )
    {
      task->worker = this;
      task->event.Begin( this->tid );
      task->Execute( this );
    }
    /** Move to the next task in the batch */
    task = task->next;
  }

  /** Wait for all tasks in the batch to terminate */
  WaitExecute();

  task = batch;
  while ( task )
  {
    task->event.Terminate();
    task->GetEventRecord();
    /** Move to the next task in the batch */
    task = task->next;
  }

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
  if ( device ) device->waitexecute();
};

float Worker::EstimateCost( class Task * task )
{
  return task->cost;
};

#endif /** ifdef HMLP_HAVE_RUNTIME */




}; // end namespace hmlp
