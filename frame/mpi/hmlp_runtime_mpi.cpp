#include <hmlp_runtime_mpi.hpp>

namespace hmlp
{
namespace mpi
{

int Init( int *argc, char ***argv )
{
#ifdef HMLP_USE_MPI
  return MPI_Init( argc, argv );
#else
  return 0;
#endif
};

int Finalize( void )
{
#ifdef HMLP_USE_MPI
  return MPI_Finalize();
#else
  return 0;
#endif
};

int Send( const void *buf, int count, Datatype datatype, 
    int dest, int tag, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Send( buf, count, datatype, 
      dest, tag, comm );
#else
  return 0;
#endif
};

int Recv( void *buf, int count, Datatype datatype, 
    int source, int tag, Comm comm, Status *status )
{
#ifdef HMLP_USE_MPI
  return MPI_Recv( buf, count, datatype, 
      source, tag, comm, (MPI_Status*)status );
#else
  return 0;
#endif
};

int Sendrecv( void *sendbuf, int sendcount, Datatype sendtype,
    int dest, int sendtag, void *recvbuf, int recvcount,
    Datatype recvtype, int source, int recvtag,
    Comm comm, Status *status )
{
#ifdef HMLP_USE_MPI
  return MPI_Sendrecv( sendbuf, sendcount, sendtype, 
      dest, sendtag, recvbuf, recvcount,
      recvtype, source, recvtag,
      comm, (MPI_Status*)status );
#else
  return 0;
#endif
};

int Comm_size( Comm comm, int *size )
{
#ifdef HMLP_USE_MPI
  return MPI_Comm_size( comm, size );
#else
  /** only one process with rank 0 */
  *size = 1;
  return 0;
#endif
};

int Comm_rank( Comm comm, int *rank )
{
#ifdef HMLP_USE_MPI
  return MPI_Comm_rank( comm, rank );
#else
  /** only one process with rank 0 */
  *rank = 0;
  return 0;
#endif
};

int Comm_dup( Comm comm, Comm *newcomm )
{
#ifdef HMLP_USE_MPI
  return MPI_Comm_dup( comm, newcomm );
#else
  return 0;
#endif
};

int Comm_split( Comm comm, int color, int key, Comm *newcomm )
{
#ifdef HMLP_USE_MPI
  return MPI_Comm_split( comm, color, key, newcomm );
#else
  return 0;
#endif
};

int Bcast( void *buffer, int count, Datatype datatype,
    int root, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Bcast( buffer, count, datatype, root, comm );
#else
  return 0;
#endif
};

int Barrier( Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Barrier( comm );
#else
  return 0;
#endif
};


int Reduce( void *sendbuf, void *recvbuf, int count,
    Datatype datatype, Op op, int root, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Reduce( sendbuf, recvbuf, count, 
      datatype, op, root, comm );
#else
  return 0;
#endif
};

int Scan( void *sendbuf, void *recvbuf, int count,
    Datatype datatype, Op op, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Scan( sendbuf, recvbuf, count, 
      datatype, op, comm );
#else
  return 0;
#endif
};


int Allreduce( void *sendbuf, void *recvbuf, int count,
    Datatype datatype, Op op, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Allreduce( sendbuf, recvbuf, count, 
      datatype, op, comm );
#else
  return 0;
#endif
};





}; /** end namespace mpi */
}; /** end namespace hmlp */


