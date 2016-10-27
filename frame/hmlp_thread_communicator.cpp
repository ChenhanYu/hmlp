#include <hmlp_thread_communicator.hpp>

namespace hmlp
{

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

std::ostream& operator<<( std::ostream& os, const thread_communicator& obj )
{
  os << obj.name << obj.comm_id;
  return os;
};

range thread_communicator::GetRange( int beg, int end, int nb, int tid )
{
  {
    auto tid_beg = beg + tid * nb;
    auto tid_inc = n_threads * nb;
    return range( tid_beg, end, tid_inc );
  }
};

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

    if      ( level == 4 ) name = "jc_comm";
    else if ( level == 3 ) name = "pc_comm";
    else if ( level == 2 ) name = "ic_comm";
    else if ( level == 1 ) name = "jr_comm";
    else                   name = "na_comm";

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
  name = "my_comm";
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
}

// OpenMP thread barrier from BLIS
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
}


}; // end namespace hmlp
