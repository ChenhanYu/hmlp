#ifndef HMLP_RUNTIME_MPI_HPP
#define HMLP_RUNTIME_MPI_HPP

#include <hmlp_runtime.hpp>


#define HMLP_MPI_COMM_WORLD              0
#define HMLP_MPI_SUCCESS                 0
#define HMLP_MPI_BYTE                    2
#define HMLP_MPI_INT                    18
#define HMLP_MPI_LONG                   18
#define HMLP_MPI_FLOAT                  19
#define HMLP_MPI_DOUBLE                 20
#define HMLP_MPI_FLOAT_INT              29
#define HMLP_MPI_DOUBLE_INT             30
#define HMLP_MPI_2FLOAT                 35
#define HMLP_MPI_2DOUBLE                36
#define HMLP_MPI_2INT                   37
#define HMLP_MPI_UNDEFINED              -1
#define HMLP_MPI_ERR_IN_STATUS          67
#define HMLP_MPI_ANY_SOURCE             -1
#define HMLP_MPI_ANY_TAG                -1
#define HMLP_MPI_REQUEST_NULL           -1
#define HMLP_MPI_COMM_SELF               1
#define HMLP_MPI_COMM_NULL              -1
#define HMLP_MPI_PROC_NULL              -3
#define HMLP_MPI_ERRORS_RETURN           0
#define HMLP_MPI_ERRORS_ARE_FATAL        1
#define HMLP_MPI_MAX_PROCESSOR_NAME     24

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


