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

#define MPI_UNSIGNED_CHAR           2
#define MPI_UNSIGNED_SHORT          5
#define MPI_UNSIGNED                7 
#define MPI_UNSIGNED_LONG           9
#define MPI_UNSIGNED_LONG_LONG     13

//#define MPI_FLOAT_INT              17
//#define MPI_DOUBLE_INT             18
//#define MPI_LONG_INT               19
//#define MPI_SHORT_INT              20
//#define MPI_2INT                   21
//#define MPI_LONG_DOUBLE_INT        22


typedef int MPI_Op;

#define MPI_MAX    (MPI_Op)(100)
#define MPI_MIN    (MPI_Op)(101)
#define MPI_SUM    (MPI_Op)(102)
#define MPI_PROD   (MPI_Op)(103)
#define MPI_LAND   (MPI_Op)(104)
#define MPI_BAND   (MPI_Op)(105)
#define MPI_LOR    (MPI_Op)(106)
#define MPI_BOR    (MPI_Op)(107)
#define MPI_LXOR   (MPI_Op)(108)
#define MPI_BXOR   (MPI_Op)(109)
#define MPI_MINLOC (MPI_Op)(110)
#define MPI_MAXLOC (MPI_Op)(111)



#endif

#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <type_traits>
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

//#include <hmlp_runtime.hpp>




//#include <containers/data.hpp>


namespace hmlp
{
/** MPI compatable interface */
namespace mpi
{

/**
 *  MPI 1.0 functionality
 */ 


#ifdef HMLP_USE_MPI
typedef MPI_Status Status;
typedef MPI_Request Request;
typedef MPI_Comm Comm;
typedef MPI_Datatype Datatype;
typedef MPI_Op Op;
#else
/** if there is no generic MPI support, use the following definition */
typedef struct 
{
  int count;
  int cancelled;
  int MPI_SOURCE;
  int MPI_TAG;
  int MPI_ERROR;
} Status;
typedef int Request;
typedef int Comm;
typedef int Datatype;
typedef int Op;
#endif /** ifdef HMLP_USE_MPI */


int Init( int *argc, char ***argv );

int Finalize( void );

int Send( const void *buf, int count, Datatype datatype, 
    int dest, int tag, Comm comm );

int Isend( const void *buf, int count, Datatype datatype, 
		int dest, int tag, Comm comm, Request *request );

int Recv( void *buf, int count, Datatype datatype, 
    int source, int tag, Comm comm, Status *status );

int Irecv( void *buf, int count, Datatype datatype, 
		int source, int tag, Comm comm, Request *request );

int Sendrecv( 
    void *sendbuf, int sendcount, Datatype sendtype, int dest, int sendtag, 
    void *recvbuf, int recvcount, Datatype recvtype, int source, int recvtag,
    Comm comm, Status *status );

int Get_count( Status *status, Datatype datatype, int *count );

int Comm_size( Comm comm, int *size );

int Comm_rank( Comm comm, int *rank );

int Comm_dup( Comm comm, Comm *newcomm );

int Comm_split( Comm comm, int color, int key, Comm *newcomm );


int Test( Request *request, int *flag, Status *status );



int Barrier( Comm comm );

int Ibarrier( Comm comm, Request *request );

int Bcast( void *buffer, int count, Datatype datatype,
    int root, Comm comm );




int Reduce( void *sendbuf, void *recvbuf, int count,
    Datatype datatype, Op op, int root, Comm comm );

int Scan( void *sendbuf, void *recvbuf, int count,
    Datatype datatype, Op op, Comm comm );

int Allreduce( void* sendbuf, void* recvbuf, int count,
    Datatype datatype, Op op, Comm comm );

int Alltoall( 
    void *sendbuf, int sendcount, Datatype sendtype,
    void *recvbuf, int recvcount, Datatype recvtype, Comm comm );

int Alltoallv( 
    void *sendbuf, int *sendcounts, int *sdispls, Datatype sendtype, 
    void *recvbuf, int *recvcounts, int *rdispls, Datatype recvtype, Comm comm );


/**
 *  MPI 2.0 and 3.0 functionality
 */ 

int Init_thread( int *argc, char ***argv, int required, int *provided );

int Probe( int source, int tag, Comm comm, Status *status );
int Iprobe( int source, int tag, Comm comm, int *flag, Status *status );





/** 
 *  HMLP MPI extension
 */ 

class MPIObject
{
  public:

    MPIObject( mpi::Comm comm )
    {
      this->comm = comm;
			mpi::Comm_dup( comm, &recvcomm );
			mpi::Comm_dup( comm, &sendcomm );
    };

    mpi::Comm GetComm()
    {
      return comm;
    };

		mpi::Comm GetRecvComm()
		{
			return recvcomm;
		};

		mpi::Comm GetSendComm()
		{
			return sendcomm;
		};

    int Comm_size()
    {
      int size;
      mpi::Comm_size( comm, &size );
      return size;
    };

    int Comm_rank()
    {
      int rank;
      mpi::Comm_rank( comm, &rank );
      return rank;
    };

    int Barrier()
    {
      return mpi::Barrier( comm );
    };

  private:

    mpi::Comm comm = MPI_COMM_WORLD;

		/** this communicator is duplicated from comm */
		mpi::Comm recvcomm;

		/** this communicator is duplicated from comm */
		mpi::Comm sendcomm;

}; /** end class MPIObject */






/**
 *  Type insensitive interface
 */ 

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
int AlltoallVector(
    std::vector<std::vector<T>> &sendvector, 
    std::vector<std::vector<T>> &recvvector, Comm comm )
{
  int size = 0;
  int rank = 0;
  Comm_size( comm, &size );
  Comm_rank( comm, &rank );

  assert( sendvector.size() == size );
  assert( recvvector.size() == size );

  std::vector<T> sendbuf;
  std::vector<T> recvbuf;
  std::vector<int> sendcounts( size, 0 );
  std::vector<int> recvcounts( size, 0 );
  std::vector<int> sdispls( size + 1, 0 );
  std::vector<int> rdispls( size + 1, 0 );

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

#endif /** define HMLP_RUNTIME_MPI_HPP */


