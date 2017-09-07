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
#include <stdint.h>
#include <limits.h>
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

#include <hmlp_runtime.hpp>




namespace hmlp
{
/** MPI compatable interface */
namespace mpi
{

#ifdef HMLP_USE_MPI
typedef MPI_Status Status;
typedef MPI_Comm Comm;
typedef MPI_Datatype Datatype;
typedef MPI_Op Op;
#else
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
#endif





int Init( int *argc, char ***argv );

int Finalize( void );

int Send( const void *buf, int count, Datatype datatype, 
    int dest, int tag, Comm comm );

int Recv( void *buf, int count, Datatype datatype, 
    int source, int tag, Comm comm, Status *status );

int Sendrecv( 
    void *sendbuf, int sendcount, Datatype sendtype, int dest, int sendtag, 
    void *recvbuf, int recvcount, Datatype recvtype, int source, int recvtag,
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

int Alltoall( void *sendbuf, int sendcount, Datatype sendtype,
    void *recvbuf, int recvcount, Datatype recvtype, Comm comm );

int Alltoallv( void *sendbuf, int *sendcounts, int *sdispls, Datatype sendtype, 
    void *recvbuf, int *recvcounts, int *rdispls, Datatype recvtype, Comm comm );




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
    printf( "request datatype is not supported in the simplified interface\n" );
    exit( 1 );
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
int Allreduce( T* sendbuf, T* recvbuf, int count, Op op, Comm comm )
{
  Datatype datatype = GetMPIDatatype<T>();
  return Allreduce( sendbuf, recvbuf, count, datatype, op, comm );
}; /** end Allreduce() */


template<typename TSEND, typename TRECV>
int Alltoall( TSEND *sendbuf, int sendcount,
    TRECV *recvbuf, int recvcount, Comm comm )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Alltoall( sendbuf, sendcount, sendtype, 
      recvbuf, recvcount, recvtype, comm );
}; /** end Alltoall() */


template<typename TSEND, typename TRECV>
int Alltoallv( TSEND *sendbuf, int *sendcounts, int *sdispls,
    TRECV *recvbuf, int *recvcounts, int *rdispls, Comm comm )
{
  Datatype sendtype = GetMPIDatatype<TSEND>();
  Datatype recvtype = GetMPIDatatype<TRECV>();
  return Alltoallv( sendbuf, sendcount, sdispls, sendtype, 
      recvbuf, recvcount, rdispls, recvtype, comm );
}; /** end Alltoallv() */


/**
 *  @brief This is a short hand for sending a vector, which
 *         involves two MPI_Send() calls.
 */
template<typename T>
int SendVector( 
    std::vector<T> &bufvector, int dest, int tag, Comm comm )
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
template<typename T>
int RecvVector(
    std::vector<T> &bufvector, int source, int tag, Comm comm, Status *status )
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



template<typename T>
int ExchangeVector(
    std::vector<T> &sendvector,   int dest, int sendtag,
    std::vector<T> &recvvector, int source, int recvtag,
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
int All2allVector(
    std::vector<std::vector<T>> &sendvector, 
    std::vector<std::vector<T>> &recvvector, Comm comm )
{
  int size = 0;
  Comm_size( comm, &size );

  assert( sendvector.size() == size )

  std::vector<T> sendbuf;
  std::vector<T> recvbuf;
  std::vector<int> sendcounts( size, 0 );
  std::vector<int> recvcounts( size, 0 );
  std::vector<int> sdispls( size + 1, 0 );
  std::vector<int> rdispls( size + 1, 0 );

  for ( size_t j = 0; j < size; j ++ )
  {
    sendcount[ j ] = sendvector[ j ].size();
    sdispls[ j + 1 ] = sdispls[ j ] + sendcounts[ j ];
    sendbuf.insert( sendbuf.end(), sendvector[ j ].beg(), sendvector[ j ].end() );
  }

  /** exchange sendcount */
  Alltoall(
      sendcounts.data(), size, 
      recvcounts.data(), size, comm );

  size_t total_recvcount = 0;
  for ( size_t j = 0; j < size; j ++ )
  {
    rdispls[ j + 1 ] = rdispls[ j ] + recvcounts[ j ];
    total_recvcount += recvcounts[ j ];
  }

  /** resize receving buffer */
  recvbuf.resize( total_recvcount );

  Alltoallv(
      sendbuf.data(), sendcounts.data(), sdispls.data(), 
      recvbuf.data(), recvcounts.data(), rdispls.data(), comm );

  recvvector.resize( size );
  for ( size_t j = 0; j < size; j ++ )
  {
    recvvector[ j ].insert( recvvector[ j ].end(), 
        recvbuf.beg() + rdispls[ j ], recvcounts[ j ] );
  }

}; /** end All2allVector() */




}; /** end namespace mpi */
}; /** end namespace hmlp */

#endif /** define HMLP_RUNTIME_MPI_HPP */


