#include <util.hpp>

namespace hmlp
{

/** 
 *  \breif Translate hmlpError_t to error string.
 */
const char* getErrorString( hmlpError_t error )  
{
  switch ( error )
  {
    case HMLP_ERROR_SUCCESS:
      return "HMLP_ERROR_SUCCESS";
    case HMLP_ERROR_ALLOC_FAILED:
      return "HMLP_ERROR_ALLOC_FAILED";
    case HMLP_ERROR_INVALID_VALUE:
      return "HMLP_ERROR_INVALID_VALUE";
    case HMLP_ERROR_EXECUTION_FAILED:
      return "HMLP_ERROR_EXECUTION_FAILED";
    case HMLP_ERROR_NOT_SUPPORTED:
      return "HMLP_ERROR_NOT_SUPPORTED";
    case HMLP_ERROR_INTERNAL_ERROR:
      return "HMLP_ERROR_INTERNAL_ERROR";
  }
  return "<unknown>";
};


/** 
 *  \breif Handling runtime error with information. 
 */
void handleError( hmlpError_t error, const char* file, int line )
{
  if ( error == HMLP_ERROR_SUCCESS ) 
  {
    return;
  }
  if ( error < HMLP_ERROR_MPI_ERROR )
  {
    /** Otherwise, handle the error and provide information. */
    fprintf( stderr, "Error: HMLP_ERROR_MPI_ERROR %d in %s at line %d\n",
        error, file, line );
  }
  else
  {
    /** Otherwise, handle the error and provide information. */
    fprintf( stderr, "Error: %s in %s at line %d\n",
        getErrorString( error ), file, line );
  }
  throw std::invalid_argument( "Program encounters hmlp error." );
};


hmlpError_t returnIfError( hmlpError_t error, const char* file, int line )
{
  if ( error == HMLP_ERROR_SUCCESS ) 
  {
    return error;
  }
  if ( error < HMLP_ERROR_MPI_ERROR )
  {
    /** Otherwise, handle the error and provide information. */
    fprintf( stderr, "Error: HMLP_ERROR_MPI_ERROR %d in %s at line %d\n",
        error, file, line );
  }
  else
  {
    /** Otherwise, handle the error and provide information. */
    fprintf( stderr, "Error: %s in %s at line %d\n",
        getErrorString( error ), file, line );
  }
  return error;
};

}; /* end namespace hmlp */
