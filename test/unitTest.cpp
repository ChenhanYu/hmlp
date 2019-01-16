#include <gtest/gtest.h>
#ifdef HMLP_USE_MPI
#include "external/gtest-mpi-listener.hpp"
#endif 

/* [INTERNAL] Runtime */
#include "unit/runtime.hpp"

int main( int argc, char **argv ) 
{
  ::testing::InitGoogleTest( &argc, argv );
#ifdef HMLP_USE_MPI
  int  provided = 0;
  /* Initial MPI with MPI_THREAD_MULTIPLE. */
  hmlp::mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
	if ( provided != MPI_THREAD_MULTIPLE )
	{
		fprintf( stderr, "MPI_THREAD_MULTIPLE is not supported\n" );
    return -1;
	}
  /* Add object that will finalize MPI on exit; Google Test owns this pointer. */
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  /* Get the event listener list. */
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  /* Remove default listener */
  delete listeners.Release( listeners.default_result_printer() );
  /* Adds MPI listener; Google Test owns this pointer. */
  listeners.Append(new MPIMinimalistPrinter);
#endif
  return RUN_ALL_TESTS();
};
