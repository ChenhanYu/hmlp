#include <hmlp_runtime.hpp>

void hmlp_init()
{
  hmlp::rt.init();
};

void hmlp_finalize()
{
  hmlp::rt.finalize();
};


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

range GetRange
( 
  hmlpSchedule_t strategy, 
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


hmlp_runtime::hmlp_runtime()
{};

void hmlp_runtime::init()
{
  #pragma omp critical (init)
  {
    if ( !is_init )
    {
      printf( "hmlp_init()\n" );

      pool_init();
    }
  }
};

void hmlp_runtime::finalize()
{
  #pragma omp critical (init)
  {
    printf( "hmlp_finalize()\n" );
  }
};

void hmlp_runtime::pool_init()
{

};

void hmlp_runtime::acquire_memory()
{

};

void hmlp_runtime::release_memory( void *ptr )
{

};

}; // end namespace hmlp
