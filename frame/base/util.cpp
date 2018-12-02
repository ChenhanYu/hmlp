#include <util.hpp>

namespace hmlp
{

const char* getErrorString( hmlpError_t error )  
{
  switch ( error )
  {
    case HMLP_ERROR_SUCCESS:
      return "SUCCESS";
    case HMLP_ERROR_ALLOC_FAILED:
      return "ALLOC_FAILED";
    case HMLP_ERROR_INVALID_VALUE:
      return "INVALID_VALUE";
    case HMLP_ERROR_EXECUTION_FAILED:
      return "EXECUTION_FAILED";
    case HMLP_ERROR_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case HMLP_ERROR_NOT_SUPPORTED:
      return "NOT_SUPPORTED";
  }
  return "<unknown>";
};


/** Handling runtime error with information. */
void handleError( hmlpError_t error, const char* file, int line )
{
  if ( error == HMLP_ERROR_SUCCESS ) return;
  /** Otherwise, handle the error and provide information. */
  printf( "Error: %s in %s at line %d\n", getErrorString( error ), file, line );
  exit( -1 );
};

}; /** end namespace hmlp */
