#ifndef HMLP_RUNTIME_MPI_HPP
#define HMLP_RUNTIME_MPI_HPP

#ifdef HMLP_USE_MPI
#include <mpi.h>
#else
#warning MPI routines are disable (-HMLP_USE_MPI=false)
/** define MPI macros */
#define MPI_COMM_WORLD              0
#define MPI_SUCCESS                 0
#define MPI_BYTE                    2
#define MPI_INT                    18
#define MPI_LONG                   18
#define MPI_FLOAT                  19
#define MPI_DOUBLE                 20
#define MPI_FLOAT_INT              29
#define MPI_DOUBLE_INT             30
#define MPI_2FLOAT                 35
#define MPI_2DOUBLE                36
#define MPI_2INT                   37
#define MPI_UNDEFINED              -1
#define MPI_ERR_IN_STATUS          67
#define MPI_ANY_SOURCE             -1
#define MPI_ANY_TAG                -1
#define MPI_REQUEST_NULL           -1
#define MPI_COMM_SELF               1
#define MPI_COMM_NULL              -1
#define MPI_PROC_NULL              -3
#define MPI_ERRORS_RETURN           0
#define MPI_ERRORS_ARE_FATAL        1
#define MPI_MAX_PROCESSOR_NAME     24
#endif


#include <type_traits>
#include <cstdint>

#include <hmlp_runtime.hpp>


namespace hmlp
{
/** MPI compatable interface */
namespace mpi
{

typedef struct 
{
  int source;
  int tag;
  int error;
  int len;
  int type;
} Status;

typedef int Comm;

typedef int Datatype;

typedef int Op;





int Init( int *argc, char ***argv );

int Finalize( void );

int Send( const void *buf, int count, Datatype datatype, 
    int dest, int tag, Comm comm );

int Recv( void *buf, int count, Datatype datatype, 
    int source, int tag, Comm comm, Status *status );

int Sendrecv( void *sendbuf, int sendcount, Datatype sendtype,
    int dest, int sendtag, void *recvbuf, int recvcount,
    Datatype recvtype, int source, int recvtag,
    Comm comm, Status *status );

int Comm_size( Comm comm, int *size );

int Comm_rank( Comm comm, int *rank );

int Comm_dup( Comm comm, Comm *newcomm );

int Comm_split( Comm comm, int color, int key, Comm *newcomm );

int Bcast( void *buffer, int count, Datatype datatype,
    int root, Comm comm );

int Barrier( Comm comm );

int Reduce( void *sendbuf, void *recvbuf, int count,
    Datatype datatype, Op op, int root, Comm comm );

int Scan( void *sendbuf, void *recvbuf, int count,
    Datatype datatype, Op op, Comm comm );

int Allreduce( void* sendbuf, void* recvbuf, int count,
    Datatype datatype, Op op, Comm comm );


template<typename T>
struct NumberIntPair { T val; int key; };

template<typename T>
int Allreduce( T* sendbuf, T* recvbuf, int count,
    Op op, Comm comm )
{
  Datatype datatype;
  if ( std::is_same<T, int>::value ) 
    datatype = MPI_INT;
  else if ( std::is_same<T, float>::value )
    datatype = MPI_FLOAT;
  else if ( std::is_same<T, double>::value )
    datatype = MPI_DOUBLE;
  else if ( std::is_same<T, NumberIntPair<float> >::value )
    datatype = MPI_FLOAT_INT;
  else if ( std::is_same<T, NumberIntPair<double> >::value )
  {
    printf( "MPI_DOUBLE_INT\n" );
    datatype = MPI_DOUBLE_INT;
  }
  else
  {
    printf( "request datatype is not supported in the simplified interface\n" );
    exit( 1 );
  }
  return Allreduce( sendbuf, recvbuf, count, datatype, op, comm );
};



}; /** end namespace mpi */






//class MPITask : public Task
//{
//
//  public:
//
//  private:
//
//}; /** end class MPITask */



}; /** end namespace hmlp */

#endif /** define HMLP_RUNTIME_MPI_HPP */


