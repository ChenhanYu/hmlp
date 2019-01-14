#ifndef HMLP_TEST_RUNTIME_HPP
#define HMLP_TEST_RUNTIME_HPP

#include <hmlp.h>

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

#endif /* define HMLP_TEST_RUNTIME_HPP */
