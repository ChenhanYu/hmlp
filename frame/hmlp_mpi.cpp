#include <hmlp_mpi.hpp>

namespace hmlp
{
namespace mpi
{

/**
 *  MPI 1.0 functionality
 */ 


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
  return MPI_Send( buf, count, datatype, dest, tag, comm );
#else
  return 0;
#endif
};



int Isend( const void *buf, int count, Datatype datatype, 
		int dest, int tag, Comm comm, Request *request )
{
#ifdef HMLP_USE_MPI
  return MPI_Isend( buf, count, datatype, dest, tag, comm, request );
#else
	return 0;
#endif
};


int Recv( void *buf, int count, Datatype datatype, 
    int source, int tag, Comm comm, Status *status )
{
#ifdef HMLP_USE_MPI
  return MPI_Recv( buf, count, datatype, source, tag, comm, status );
#else
  return 0;
#endif
};


int Irecv( void *buf, int count, Datatype datatype, 
		int source, int tag, Comm comm, Request *request )
{
#ifdef HMLP_USE_MPI
  return Irecv( buf, count, datatype, source, tag, comm, request );
#else
	return 0;
#endif
};


int Sendrecv( 
    void *sendbuf, int sendcount, Datatype sendtype, int   dest, int sendtag, 
    void *recvbuf, int recvcount, Datatype recvtype, int source, int recvtag,
    Comm comm, Status *status )
{
#ifdef HMLP_USE_MPI
  return MPI_Sendrecv( 
      sendbuf, sendcount, sendtype,   dest, sendtag, 
      recvbuf, recvcount, recvtype, source, recvtag,
      comm, status );
#else
  return 0;
#endif
};


int Get_count( Status *status, Datatype datatype, int *count )
{
#ifdef HMLP_USE_MPI
  return MPI_Get_count( status, datatype, count );
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



int Type_contiguous( int count, Datatype oldtype, Datatype *newtype )
{
#ifdef HMLP_USE_MPI
  return MPI_Type_contiguous( count, oldtype, newtype );
#else
  return 0;
#endif
};

int Type_commit( Datatype *datatype )
{
#ifdef HMLP_USE_MPI
  return MPI_Type_commit( datatype );
#else
  return 0;
#endif
};




int Test( Request *request, int *flag, Status *status )
{
#ifdef HMLP_USE_MPI
	return MPI_Test( request, flag, status );
#else
  /** Set *flag = 1 to prevent blocking. */
  *flag = 1;
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


int Ibarrier( Comm comm, Request *request )
{
#ifdef HMLP_USE_MPI
	return MPI_Ibarrier( comm, request );
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



int Gather( const void *sendbuf, int sendcount, Datatype sendtype,
    void *recvbuf, int recvcount, Datatype recvtype,
    int root, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Gather( sendbuf, sendcount, sendtype,
      recvbuf, recvcount, recvtype, root, comm );
#else
  return 0;
#endif
};



int Gatherv( void *sendbuf, int sendcount, Datatype sendtype,
             void *recvbuf, const int *recvcounts, const int *displs,
             Datatype recvtype, int root, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Gatherv( sendbuf, sendcount, sendtype,
      recvbuf, recvcounts, displs, recvtype, root, comm );
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


int Allgather(
    void *sendbuf, int sendcount, Datatype sendtype, 
    void *recvbuf, int recvcount, Datatype recvtype, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Allgather( 
      sendbuf, sendcount, sendtype,
      recvbuf, recvcount, recvtype, comm );
#else
  return 0;
#endif
};

int Allgatherv( 
    void *sendbuf, int sendcount, Datatype sendtype,
    void *recvbuf, int *recvcounts, int *displs, Datatype recvtype, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Allgatherv( 
      sendbuf, sendcount, sendtype,
      recvbuf, recvcounts, displs, recvtype, comm );
#else
  return 0;
#endif
};







int Alltoall( void *sendbuf, int sendcount, Datatype sendtype,
    void *recvbuf, int recvcount, Datatype recvtype, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Alltoall( sendbuf, sendcount, sendtype,
      recvbuf, recvcount, recvtype, comm ); 
#else
  return 0;
#endif
};


int Alltoallv( void *sendbuf, int *sendcounts, int *sdispls, Datatype sendtype, 
    void *recvbuf, int *recvcounts, int *rdispls, Datatype recvtype, Comm comm )
{
#ifdef HMLP_USE_MPI
  return MPI_Alltoallv( sendbuf, sendcounts, sdispls, sendtype,
      recvbuf, recvcounts, rdispls, recvtype, comm ); 
#else
  return 0;
#endif
};



/**
 *  MPI 2.0 and 3.0 functionality
 */ 

int Init_thread( int *argc, char ***argv, int required, int *provided )
{
#ifdef HMLP_USE_MPI
	return MPI_Init_thread( argc, argv, required, provided );
#else
	return 0;
#endif
};


int Probe( int source, int tag, Comm comm, Status *status )
{
#ifdef HMLP_USE_MPI
  return MPI_Probe( source, tag, comm, status );
#else
  return 0;
#endif
}; /** end Probe() */

int Iprobe( int source, int tag, Comm comm, int *flag, Status *status )
{
#ifdef HMLP_USE_MPI
  return MPI_Iprobe( source, tag, comm, flag, status );
#else
  return 0;
#endif
}; /** end Iprobe() */


/** HMLP MPI extension */
void PrintProgress( const char *s, mpi::Comm comm )
{
  int rank; mpi::Comm_rank( comm, &rank );
  if ( !rank ) 
  {
    printf( "%s\n", s ); fflush( stdout );
  }
  mpi::Barrier( comm );
};




}; /** end namespace mpi */
}; /** end namespace hmlp */


