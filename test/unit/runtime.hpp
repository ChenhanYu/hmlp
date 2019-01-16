#ifndef HMLP_TEST_RUNTIME_HPP
#define HMLP_TEST_RUNTIME_HPP

/* Public headers. */
#include <hmlp.h>
/* Internal headers. */
#include <base/hmlp_mpi.hpp>

namespace hmlp
{
namespace test
{

}; /* end namespace test */
}; /* end namespace hmlp */

TEST(runtime, hmlp_init)
{
  EXPECT_EQ( hmlp_init(),
      HMLP_ERROR_SUCCESS );
}

TEST(runtime, hmlp_init_with_arguments)
{
  int argc = 1; 
  char **argv = nullptr;
  EXPECT_EQ( hmlp_init( &argc, &argv ),
      HMLP_ERROR_SUCCESS );
}

TEST(runtime, hmlp_set_num_workers)
{
  EXPECT_EQ( hmlp_init(),
      HMLP_ERROR_SUCCESS );
  /* Positive test. */
  EXPECT_EQ( hmlp_set_num_workers( 1 ),
      HMLP_ERROR_SUCCESS );
  /* Negative test. */
  EXPECT_EQ( hmlp_set_num_workers( 0 ),
      HMLP_ERROR_INVALID_VALUE );
  EXPECT_EQ( hmlp_set_num_workers( -1 ),
      HMLP_ERROR_INVALID_VALUE );
}




/* Put all tests involving MPI here. */
#ifdef HMLP_USE_MPI
TEST(runtime, hmlp_init_with_comm)
{
  EXPECT_EQ( hmlp_init( MPI_COMM_WORLD ),
      HMLP_ERROR_SUCCESS );
}

TEST(runtime, hmlp_init_with_arguments_and_comm)
{
  int argc = 1; 
  char **argv = nullptr;
  EXPECT_EQ( hmlp_init( &argc, &argv, MPI_COMM_WORLD ),
      HMLP_ERROR_SUCCESS );
}
#endif /* ifdef HMLP_USE_MPI */

#endif /* define HMLP_TEST_RUNTIME_HPP */
