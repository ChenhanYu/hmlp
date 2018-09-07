#include <hmlp_tci.hpp>

namespace hmlp
{

  
  
Range::Range( int beg, int end, int inc )
{
  info = make_tuple( beg, end, inc );
};

int Range::beg() { return get<0>( info ); };

int Range::end() { return get<1>( info ); };

int Range::inc() { return get<2>( info ); };
 
void Range::Print( int prefix ) 
{ 
  printf( "%2d %5d %5d %5d\n", prefix, beg(), end(), inc() ); 
  fflush( stdout ); 
};




/** @brief Shared-memory lock that calls either pthread or omp mutex.. */ 
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
}; /** end Lock::Lock() */

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
}; /** end Lock::~Lock() */

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




namespace tci
{

void Context::Barrier( int size )
{
  /** Early return if there is only one thread. */
  if ( size < 2 ) return;

  //printf( "%2d size %2d Barrier( BEG ) %lu\n", omp_get_thread_num(), size, this ); 
  //fflush( stdout );


  /** Get my barrier sense. */
  bool my_sense = barrier_sense;
  /** Check how many threads in the communicator have arrived. */
  int my_threads_arrived;
  #pragma omp atomic capture
  my_threads_arrived = ++ barrier_threads_arrived;

  //printf( "%2d my_threads_arrived %d\n", omp_get_thread_num(), 
  //    my_threads_arrived ); fflush( stdout );

  /** If I am the last thread to arrive, then reset. */
  if ( my_threads_arrived == size )
  {
    barrier_threads_arrived = 0;
    barrier_sense = !barrier_sense;
  }
  /** Otherwise, wait until barrier_sense is changed. */
  else
  {
    volatile bool *listener = &barrier_sense;
    while ( *listener == my_sense ) {}
  }

  //printf( "%2d size %2d Barrier( END )\n", omp_get_thread_num(), size ); 
  //fflush( stdout );

}; /** end Context::Barrier() */



/** (Default) within OpenMP parallel construct (all threads). */
Comm::Comm() 
{
  /** Assign all threads to the communicator.  */
  size = omp_get_num_threads();
  /** Assign my rank (tid) in the communicator. */
  rank = omp_get_thread_num();
}; /** end Comm::Comm() */

Comm::Comm( Context* context ) : Comm::Comm()
{
  /** Assign the shared context. */
  this->context = context;
}; /** end Comm::Comm() */

Comm::Comm( Comm* parent, Context* context, int assigned_size, int assigned_rank )
{
  /** Use the assigned size as my size. */
  size = assigned_size;
  /** Use the assigned rank as my rank. */
  rank = assigned_rank;
  /** Assign the shared context. */
  this->context = context;
  /** Assign the parent communicator pointer. */
  this->parent = parent;
}; /** end Comm::Comm() */

Comm Comm::Split( int num_splits )
{
  /** Early return if possible. */
  if ( num_splits == 1 || size <= 1 ) return Comm( this, context, size, rank );
  /** Prepare to create gang_size subcommunicators. */
  gang_size = num_splits;
  /** 
   *  By default, we split threads evenly using "close" affinity.
   *  Threads with the same color will be in the same subcomm.
   *
   *  example: (num_splits=2)
   *
   *  rank       0  1  2  3  4  5  6  7  
   *  color      0  0  0  0  1  1  1  1  (gang_rank)
   *  first      0  0  0  0  4  4  4  4 
   *  last       4  4  4  4  8  8  8  8
   *  child_rank 0  1  2  3  0  1  2  3
   *             4  4  4  4  4  4  4  4  (child_size)
   */
  size_t color = ( num_splits * rank ) / size;
  size_t first = ( ( color + 0 ) * size ) / num_splits;
  size_t last  = ( ( color + 1 ) * size ) / num_splits;
  size_t child_rank = rank - first;
  size_t child_size = last - first;
  /** Use color to be the gang_rank. */
  gang_rank = color;
  /** Create new contexts. */
  Context** child_contexts;
  if ( Master() ) child_contexts = new Context*[ num_splits ];
  /** Master bcast its buffer. */
  Bcast( child_contexts, 0 );
  /** The master of each gang will allocate the new context. */
  if ( child_rank == 0 ) child_contexts[ color ] = new Context();
  Barrier();
  /** Create and return the subcommunicator. */
  Comm child_comm( this, child_contexts[ color ], child_size, child_rank );
  Barrier();
  if ( Master() ) delete child_contexts;
  return child_comm;
}; /** end Comm::Split() */


bool Comm::Master() { return rank == 0; };

void Comm::Barrier() { context->Barrier( size ); };

void Comm::Send( void** sent_object ) { context->buffer = *sent_object; };

void Comm::Recv( void** recv_object ) { *recv_object = context->buffer; };

void Comm::Create1DLocks( int n )
{
  if ( Master() ) 
  {
    lock1d = new vector<Lock>( n );
  }
  Bcast( lock1d, 0 );
};

void Comm::Destroy1DLocks()
{
  Barrier();
  if ( Master() ) delete lock1d;
};

void Comm::Create2DLocks( int m, int n )
{
  if ( Master() ) 
  {
    lock2d = new vector<vector<Lock>>( m );
    for ( int i = 0; i < m; i ++ ) (*lock2d)[ i ].resize( n );
  }
  Bcast( lock2d, 0 );
};


void Comm::Destroy2DLocks()
{
  Barrier();
  if ( Master() ) delete lock2d;
};

void Comm::Acquire1DLocks( int j )
{
  if ( lock1d ) 
  {
    auto n = lock1d->size();
    (*lock1d)[ j % n ].Acquire();
  }
  else if ( parent ) parent->Acquire1DLocks( j );
};

void Comm::Release1DLocks( int j )
{
  if ( lock1d ) 
  {
    auto n = lock1d->size();
    (*lock1d)[ j % n ].Release();
  }
  else if ( parent ) parent->Release1DLocks( j );
};

void Comm::Acquire2DLocks( int i, int j )
{
  if ( lock2d ) 
  {
    auto m = (*lock2d).size();
    auto n = (*lock2d)[ 0 ].size();
    (*lock2d)[ i % m ][ j % n ].Acquire();
  }
  else if ( parent ) parent->Acquire2DLocks( i, j );
};

void Comm::Release2DLocks( int i, int j )
{
  if ( lock2d ) 
  {
    auto m = (*lock2d).size();
    auto n = (*lock2d)[ 0 ].size();
    (*lock2d)[ i % m ][ j % n ].Release();
  }
  else if ( parent ) parent->Release2DLocks( i, j );
};

int Comm::GetCommSize() { return size; };

int Comm::GetCommRank() { return rank; };

int Comm::GetGangSize() { return gang_size; };

int Comm::GetGangRank() { return gang_rank; };

void Comm::Print( int prefix )
{
  printf( "%2d size %2d rank %2d gang_size %2d gang_rank %2d\n",
      prefix, size, rank, gang_size, gang_rank );
  fflush( stdout );
};


int Comm::BalanceOver1DGangs( int n, int default_size, int nb )
{
  size_t nparts = gang_size;
  if ( nparts > 1 ) return ( ( n - 1 ) / ( nb * nparts ) + 1 ) * nb;
  return default_size;
};

Range Comm::DistributeOver1DThreads( int beg, int end, int nb )
{
  size_t nparts = size;
  /** Select the proper partitioning policy. */
  SchedulePolicy strategy = HMLP_SCHEDULE_DEFAULT;
  //SchedulePolicy strategy = HMLP_SCHEDULE_UNIFORM;
  /** Return the tuple accordingly. */
  switch ( strategy )
  {
    case HMLP_SCHEDULE_DEFAULT:
    {
      /** Default is Round Robin. */
    }
    case HMLP_SCHEDULE_ROUND_ROBIN:
    {
      return Range( beg + rank * nb, end, nparts * nb );
    }
    case HMLP_SCHEDULE_UNIFORM:
    {
      int len = ( ( end - beg - 1 ) / ( nparts * nb ) + 1 ) * nb;
      beg = beg + rank * len;
      end = std::min( end, beg + len );
      return Range( beg, end, nb );
      //printf( "GetRange(): HMLP_SCHEDULE_UNIFORM not yet implemented yet.\n" );
      //exit( 1 );
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
}; /** end Comm::DistributeOver1DThreads() */


Range Comm::DistributeOver1DGangs( int beg, int end, int nb )
{
  size_t nparts = gang_size;
  /** Select the proper partitioning policy. */
  SchedulePolicy strategy = HMLP_SCHEDULE_DEFAULT;
  //SchedulePolicy strategy = HMLP_SCHEDULE_UNIFORM;
  /** Return the tuple accordingly. */
  switch ( strategy )
  {
    case HMLP_SCHEDULE_DEFAULT:
    {
      /** Default is Round Robin. */
    }
    case HMLP_SCHEDULE_ROUND_ROBIN:
    {
      return Range( beg + gang_rank * nb, end, nparts * nb );
    }
    case HMLP_SCHEDULE_UNIFORM:
    {
      int len = ( ( end - beg - 1 ) / ( nparts * nb ) + 1 ) * nb;
      beg = beg + gang_rank * len;
      end = std::min( end, beg + len );
      return Range( beg, end, nb );
      //printf( "GetRange(): HMLP_SCHEDULE_UNIFORM not yet implemented yet.\n" );
      //exit( 1 );
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
}; /** end Comm::DistributeOver1DGangs() */



}; /** end namespace tci */
}; /** end namespace hmlp */
