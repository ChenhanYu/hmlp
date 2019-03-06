#ifndef HMLP_MPI_HPP
#define HMLP_MPI_HPP
#include <external/mpi_prototypes.h>

#include <vector>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <type_traits>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>

#if SIZE_MAX == UCHAR_MAX
#define HMLP_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define HMLP_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define HMLP_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define HMLP_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define HMLP_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif


/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif // ifdef HMLP}_MIC_AVX512


using namespace std;

namespace hmlp
{
/** MPI compatable interface */
namespace mpi
{
typedef MPI_Status Status;
typedef MPI_Request Request;
typedef MPI_Comm Comm;
typedef MPI_Datatype Datatype;
typedef MPI_Op Op;

/** MPI 1.0 functionality */ 
int Init( int *argc, char ***argv );
int Initialized( int *flag );
int Finalize( void );
int Finalized( int *flag );
int Send( const void *buf, int count, Datatype datatype, int dest, int tag, Comm comm );
int Isend( const void *buf, int count, Datatype datatype, int dest, int tag, Comm comm, Request *request );
int Recv( void *buf, int count, Datatype datatype, int source, int tag, Comm comm, Status *status );
int Irecv( void *buf, int count, Datatype datatype, int source, int tag, Comm comm, Request *request );
int Sendrecv( void *sendbuf, int sendcount, Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, Datatype recvtype, int source, int recvtag, Comm comm, Status *status );
int Get_count( Status *status, Datatype datatype, int *count );
int Comm_size( Comm comm, int *size );
int Comm_rank( Comm comm, int *rank );
int Comm_dup( Comm comm, Comm *newcomm );
int Comm_split( Comm comm, int color, int key, Comm *newcomm );
int Type_contiguous( int count, Datatype oldtype, Datatype *newtype );
int Type_commit( Datatype *datatype );
int Test( Request *request, int *flag, Status *status );
int Barrier( Comm comm );
int Ibarrier( Comm comm, Request *request );
int Bcast( void *buffer, int count, Datatype datatype, int root, Comm comm );
int Reduce( void *sendbuf, void *recvbuf, int count, Datatype datatype, Op op, int root, Comm comm );
int Gather( const void *sendbuf, int sendcount, Datatype sendtype, void *recvbuf, int recvcount, Datatype recvtype, int root, Comm comm );
int Gatherv( void *sendbuf, int sendcount, Datatype sendtype, void *recvbuf, const int *recvcounts, const int *displs, Datatype recvtype, int root, Comm comm );
int Scan( void *sendbuf, void *recvbuf, int count, Datatype datatype, Op op, Comm comm );
int Allreduce( void* sendbuf, void* recvbuf, int count, Datatype datatype, Op op, Comm comm );
int Allgather( void *sendbuf, int sendcount, Datatype sendtype, void *recvbuf, int recvcount, Datatype recvtype, Comm comm );
int Allgatherv( void *sendbuf, int sendcount, Datatype sendtype, void *recvbuf, int *recvcounts, int *displs, Datatype recvtype, Comm comm );
int Alltoall( void *sendbuf, int sendcount, Datatype sendtype, void *recvbuf, int recvcount, Datatype recvtype, Comm comm );
int Alltoallv( void *sendbuf, int *sendcounts, int *sdispls, Datatype sendtype, void *recvbuf, int *recvcounts, int *rdispls, Datatype recvtype, Comm comm );

/** MPI 2.0 and 3.0 functionality */ 
int Init_thread( int *argc, char ***argv, int required, int *provided );
int Probe( int source, int tag, Comm comm, Status *status );
int Iprobe( int source, int tag, Comm comm, int *flag, Status *status );

/** HMLP MPI extension */ 
void PrintProgress( const char *s, mpi::Comm comm );

class MPIObject
{
  public:

    MPIObject() {};

    MPIObject( mpi::Comm comm ) 
    { 
      AssignCommunicator( comm ); 
    };

    void AssignCommunicator( mpi::Comm& comm )
    {
      this->comm_ = comm;
      mpi::Comm_dup( comm, &private_comm_ );
      mpi::Comm_size( comm, &comm_size_ );
      mpi::Comm_rank( comm, &comm_rank_ );
    };

    mpi::Comm GetComm() 
    { 
      return comm_; 
    };

    mpi::Comm GetPrivateComm() 
    { 
      return private_comm_; 
    };

    int GetCommSize() const noexcept 
    { 
      return comm_size_; 
    };

    int GetCommRank() const noexcept 
    { 
      return comm_rank_; 
    };

    int Comm_size()
    {
      int size;
      mpi::Comm_size( comm_, &size );
      return size;
    };

    int Comm_rank()
    {
      int rank;
      mpi::Comm_rank( comm_, &rank );
      return rank;
    };

    int Barrier() 
    { 
      return mpi::Barrier( comm_ ); 
    };

    int PrivateBarrier() 
    { 
      return mpi::Barrier( private_comm_ ); 
    };

  private:

    /** This is the assigned communicator. */
    mpi::Comm comm_ = MPI_COMM_WORLD;
    /** This communicator is duplicated from comm */
    mpi::Comm private_comm_;
    /** (Default) there is only one MPI process. */
    int comm_size_ = 1;
    /** (Default) the only MPI process has rank 0. */
    int comm_rank_ = 0;

}; /* end class MPIObject */






/** Type insensitive interface */ 

template<typename T>
struct NumberIntPair { T val; int key; };


template<typename T>
Datatype GetMPIDatatype()
{
  Datatype datatype;
  if ( std::is_same<T, int>::value ) 
    datatype = MPI_INT;
  else if ( std::is_same<T, float>::value )
    datatype = MPI_FLOAT;
  else if ( std::is_same<T, double>::value )
    datatype = MPI_DOUBLE;
  else if ( std::is_same<T, size_t>::value )
    datatype = HMLP_MPI_SIZE_T;
  else if ( std::is_same<T, NumberIntPair<float> >::value )
    datatype = MPI_FLOAT_INT;
  else if ( std::is_same<T, NumberIntPair<double> >::value )
  {
    datatype = MPI_DOUBLE_INT;
  }
  else
  {
    Type_contiguous( sizeof( T ), MPI_BYTE, &datatype );
    Type_commit( &datatype );

    //printf( "request datatype is not supported in the simplified interface\n" );
    //exit( 1 );
  }
  return datatype;

}; /** end GetMPIDatatype() */



template<typename TSEND>
int Send( TSEND *buf, int count, 
    int dest, int tag, Comm comm )
{
  Datatype datatype = GetMPIDatatype<TSEND>();
  return Send( buf, count, datatype, dest, tag, comm );
}; /** end Send() */


template<typename TSEND>
int Isend( TSEND *buf, int count,
		int dest, int tag, Comm comm, Request *request )
{
  Datatype datatype = GetMPIDatatype<TSEND>();
  return Isend( buf, count, datatype, dest, tag, comm, request );
}; /** end Isend() */



template<typename TRECV>
int Recv( TRECV *buf, int count, 
    int source, int tag, Comm comm, Status *status )
{
  Datatype datatype = GetMPIDatatype<TRECV>();
  return Recv( buf, count, datatype, source, tag, comm, status );
}; /** end Recv() */

template<typename T>
int Bcast( T *buffer, int count, int root, Comm comm )
{
  Datatype datatype = GetMPIDatatype<T>();
  return Bcast( buffer, count, datatype, root, comm );
}; /** end Bcast() */


template<typename TSEND, typename TRECV>
int Sendrecv( 
    TSEND *sendbuf, int sendcount, int dest,   int sendtag, 
    TRECV *recvbuf, int recvcount, int source, int recvtag,
    Comm comm, Status *status )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Sendrecv( 
      sendbuf, sendcount, sendtype,   dest, sendtag, 
      recvbuf, recvcount, recvtype, source, recvtag,
      comm, status );

}; /** end Sendrecv() */


template<typename T>
int Reduce( T *sendbuf, T *recvbuf, int count,
    Op op, int root, Comm comm )
{
  Datatype datatype = GetMPIDatatype<T>();
	return Reduce( sendbuf, recvbuf, count, datatype, op, root, comm );
}; /** end Reduce() */



template<typename TSEND, typename TRECV>
int Gather( const TSEND *sendbuf, int sendcount,
    TRECV *recvbuf, int recvcount,
    int root, Comm comm )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Gather( sendbuf, sendcount, sendtype,
      recvbuf, recvcount, recvtype, root, comm );
};


template<typename TSEND, typename TRECV>
int Gatherv( 
    TSEND *sendbuf, int sendcount,
    TRECV *recvbuf, const int *recvcounts, const int *displs,
    int root, Comm comm )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Gatherv( sendbuf, sendcount, sendtype,
      recvbuf, recvcounts, displs, recvtype, root, comm );
}; /** end Gatherv() */


template<typename T>
int Allreduce( T* sendbuf, T* recvbuf, int count, Op op, Comm comm )
{
  Datatype datatype = GetMPIDatatype<T>();
  return Allreduce( sendbuf, recvbuf, count, datatype, op, comm );
}; /** end Allreduce() */


template<typename TSEND, typename TRECV>
int Allgather(
    TSEND *sendbuf, int sendcount, 
    TRECV *recvbuf, int recvcount, Comm comm )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Allgather( 
      sendbuf, sendcount, sendtype,
      recvbuf, recvcount, recvtype, comm );
}; /** end Allgather() */


template<typename TSEND, typename TRECV>
int Allgatherv( 
    TSEND *sendbuf, int sendcount,
    TRECV *recvbuf, int *recvcounts, int *displs, Comm comm )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Allgatherv( 
      sendbuf, sendcount, sendtype,
      recvbuf, recvcounts, displs, recvtype, comm );
}; /** end Allgatherv() */


template<typename TSEND, typename TRECV>
int Alltoall( 
    TSEND *sendbuf, int sendcount,
    TRECV *recvbuf, int recvcount, Comm comm )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Alltoall( 
      sendbuf, sendcount, sendtype, 
      recvbuf, recvcount, recvtype, comm );
}; /** end Alltoall() */


template<typename TSEND, typename TRECV>
int Alltoallv( 
    TSEND *sendbuf, int *sendcounts, int *sdispls,
    TRECV *recvbuf, int *recvcounts, int *rdispls, Comm comm )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Alltoallv( 
      sendbuf, sendcounts, sdispls, sendtype, 
      recvbuf, recvcounts, rdispls, recvtype, comm );
}; /** end Alltoallv() */


/**
 *  @brief This is a short hand for sending a vector, which
 *         involves two MPI_Send() calls.
 */
#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
int SendVector( 
    std::vector<T, Allocator> &bufvector, int dest, int tag, Comm comm )
{
  Datatype datatype = GetMPIDatatype<T>();
  size_t count = bufvector.size();

  /** send the count size first */
  //printf( "beg send count %lu to %d\n", count, dest );
  Send( &count, 1, dest, tag, comm );
  //printf( "end send count %lu to %d\n", count, dest );

  /** now send the vector itself */
  return Send( bufvector.data(), count, dest, tag, comm );

}; /** end SendVector() */



/**
 *  @brief This is a short hand for receving a vector, which
 *         involves two MPI_Recv() calls.
 */
#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
int RecvVector(
    std::vector<T, Allocator> &bufvector, int source, int tag, Comm comm, Status *status )
{
  Datatype datatype = GetMPIDatatype<T>();
  size_t count = 0;

  /** recv the count size first */
  //printf( "beg recv count %lu from %d\n", count, source );
  Recv( &count, 1, source, tag, comm, status );
  //printf( "end recv count %lu from %d\n", count, source );

  /** resize receiving vector */
  bufvector.resize( count );

  /** now recv the vector itself */
  return Recv( bufvector.data(), count, source, tag, comm, status );

}; /** end RecvVector() */



#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
int ExchangeVector(
    vector<T, Allocator> &sendvector,   int dest, int sendtag,
    vector<T, Allocator> &recvvector, int source, int recvtag,
    Comm comm, Status *status )
{
  Datatype datatype = GetMPIDatatype<T>();
  size_t send_size = sendvector.size();
  size_t recv_size = 0;

  /** exechange size */
  Sendrecv(
      &send_size, 1,   dest, sendtag,
      &recv_size, 1, source, recvtag, 
      comm, status );
  /** resize receiving vector */
  recvvector.resize( recv_size );
  /** exechange buffer */
  return Sendrecv(
      sendvector.data(), send_size,   dest, sendtag,
      recvvector.data(), recv_size, source, recvtag, 
      comm, status );

}; /** end ExchangeVector() */


template<typename T>
int GatherVector( vector<T> &sendvector, vector<T> &recvvector, int root, Comm comm )
{
  int size = 0; Comm_size( comm, &size );
  int rank = 0; Comm_rank( comm, &rank );

  Datatype datatype = GetMPIDatatype<T>();
  int send_size = sendvector.size();
  vector<int> recv_sizes( size, 0 );
  vector<int> rdispls( size + 1, 0 );

  /** Use typeless Gather() for counting. */
  Gather( &send_size, 1, recv_sizes.data(), 1, root, comm );
  /** Accumulate received displs. */
  for ( int i = 1; i < size + 1; i ++ ) 
    rdispls[ i ] = rdispls[ i - 1 ] + recv_sizes[ i - 1 ];
  /** Resize recvvector to fit. */
  recvvector.resize( rdispls[ size ] );
  /** Gather vectors. */
  return Gatherv( sendvector.data(), send_size,
      recvvector.data(), recv_sizes.data(), rdispls.data(), root, comm );
}; /** end GatherVector() */





#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
int AlltoallVector(
    const vector<vector<T, Allocator>>& sendvector, 
    vector<vector<T, Allocator>>& recvvector, Comm comm )
{
  int size = 0; Comm_size( comm, &size );
  int rank = 0; Comm_rank( comm, &rank );

  assert( sendvector.size() == size );
  assert( recvvector.size() == size );

  vector<T> sendbuf;
  vector<T> recvbuf;
  vector<int> sendcounts( size, 0 );
  vector<int> recvcounts( size, 0 );
  vector<int> sdispls( size + 1, 0 );
  vector<int> rdispls( size + 1, 0 );

  //printf( "rank %d ", rank );
  for ( size_t p = 0; p < size; p ++ )
  {
    sendcounts[ p ] = sendvector[ p ].size();
    sdispls[ p + 1 ] = sdispls[ p ] + sendcounts[ p ];
    sendbuf.insert( sendbuf.end(), 
        sendvector[ p ].begin(), 
        sendvector[ p ].end() );
    //printf( "%d ", sendcounts[ j ] );
  }
  //printf( "\n" );

  
  //printf( "before Alltoall (mpi size = %d)\n", size ); fflush( stdout );
  //Barrier( comm );

  /** exchange sendcount */
  Alltoall( sendcounts.data(), 1, recvcounts.data(), 1, comm );


  //printf( "after Alltoall\n" ); fflush( stdout );
  //Barrier( comm );

  //printf( "rank %d ", rank );
  size_t total_recvcount = 0;
  for ( size_t p = 0; p < size; p ++ )
  {
    rdispls[ p + 1 ] = rdispls[ p ] + recvcounts[ p ];
    total_recvcount += recvcounts[ p ];
    //printf( "%d ", recvcounts[ j ] );
  }
  //printf( "\n" );

  /** resize receving buffer */
  recvbuf.resize( total_recvcount );


  //printf( "before Alltoall2 recvbuf.size() %lu\n", recvbuf.size() ); fflush( stdout );
  //Barrier( comm );


  Alltoallv(
      sendbuf.data(), sendcounts.data(), sdispls.data(), 
      recvbuf.data(), recvcounts.data(), rdispls.data(), comm );


  //printf( "after Alltoall2\n" ); fflush( stdout );
  //Barrier( comm );


  recvvector.resize( size );
  for ( size_t p = 0; p < size; p ++ )
  {
    recvvector[ p ].insert( recvvector[ p ].end(), 
        recvbuf.begin() + rdispls[ p ], 
        recvbuf.begin() + rdispls[ p + 1 ] );
  }

  return 0;

}; /** end AlltoallVector() */



/**
 *  This interface supports hmlp::Data. [ m, n ] will be set
 *  after the routine.
 *
 */ 
//template<typename T>
//int AlltoallVector(
//    std::vector<hmlp::Data<T>> &sendvector, 
//    std::vector<hmlp::Data<T>> &recvvector, Comm comm )
//{
//  int size = 0;
//  int rank = 0;
//  Comm_size( comm, &size );
//  Comm_rank( comm, &rank );
//
//  assert( sendvector.size() == size );
//  assert( recvvector.size() == size );
//
//  std::vector<T> sendbuf;
//  std::vector<T> recvbuf;
//  std::vector<int> sendcounts( size, 0 );
//  std::vector<int> recvcounts( size, 0 );
//  std::vector<int> sdispls( size + 1, 0 );
//  std::vector<int> rdispls( size + 1, 0 );
//
//
//  for ( size_t p = 0; p < size; p ++ )
//  {
//    sendcounts[ p ] = sendvector[ p ].size();
//    sdispls[ p + 1 ] = sdispls[ p ] + sendcounts[ p ];
//    sendbuf.insert( sendbuf.end(), 
//        sendvector[ p ].begin(), 
//        sendvector[ p ].end() );
//  }
//
//  /** exchange sendcount */
//  Alltoall( sendcounts.data(), 1, recvcounts.data(), 1, comm );
//
//  /** compute total receiving count */
//  size_t total_recvcount = 0;
//  for ( size_t p = 0; p < size; p ++ )
//  {
//    rdispls[ p + 1 ] = rdispls[ p ] + recvcounts[ p ];
//    total_recvcount += recvcounts[ p ];
//  }
//
//  /** resize receving buffer */
//  recvbuf.resize( total_recvcount );
//
//
//
//
//
//
//}; /** end AlltoallVector() */




}; /** end namespace mpi */
}; /** end namespace hmlp */

#endif /** define HMLP_MPI_HPP */


