#ifndef DISTDATA_HPP
#define DISTDATA_HPP

#include <containers/data.hpp>
#include <hmlp_mpi.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{
namespace mpi
{

template<typename T>
int SendData( Data<T> &senddata, int dest, mpi::Comm comm )
{
  int m = senddata.row();
  int n = senddata.col();
  Send( &m, 1, dest, 0, comm );
  Send( &n, 1, dest, 0, comm );
  Send( senddata.data(), m * n, dest, 0, comm );
  return 0;
}; /** end SendData() */

template<typename T>
int RecvData( Data<T> &recvdata, int source, mpi::Comm comm )
{
  int m, n;
  mpi::Status status;
  Recv( &m, 1, source, 0, comm, &status );
  Recv( &n, 1, source, 0, comm, &status );
  recvdata.resize( m, n );
  Recv( recvdata.data(), m * n, source, 0, comm, &status );
  return 0;
}; /** end RecvData() */


  
/**
 *  This interface supports hmlp::Data. [ m, n ] will be set
 *  after the routine.
 *
 */ 
template<typename T>
int AlltoallData( size_t m,
    vector<Data<T>> &sendvector, 
    vector<Data<T>> &recvvector, mpi::Comm comm )
{
  int size = 0; Comm_size( comm, &size );
  int rank = 0; Comm_rank( comm, &rank );

  assert( sendvector.size() == size );
  assert( recvvector.size() == size );


  /** determine the concatenated dimension */
  //size_t m = 0;

  ///** all data need to have the same m */
  //for ( size_t p = 0; p < size; p ++ )
  //{
  //  if ( sendvector[ p ].row() )
  //  {
  //    if ( m ) assert( m == sendvector[ p ].row() );
  //    else     m = sendvector[ p ].row();
  //  }
  //}

  //printf( "after all assertion m %lu\n", m ); fflush( stdout );



  vector<T, ALLOCATOR> sendbuf;
  vector<T, ALLOCATOR> recvbuf;
  vector<int> sendcounts( size, 0 );
  vector<int> recvcounts( size, 0 );
  vector<int> sdispls( size + 1, 0 );
  vector<int> rdispls( size + 1, 0 );

  for ( size_t p = 0; p < size; p ++ )
  {
    sendcounts[ p ] = sendvector[ p ].size();
    sdispls[ p + 1 ] = sdispls[ p ] + sendcounts[ p ];
    sendbuf.insert( sendbuf.end(), 
        sendvector[ p ].begin(), 
        sendvector[ p ].end() );
  }

  /** Exchange sendcount. */
  Alltoall( sendcounts.data(), 1, recvcounts.data(), 1, comm );

  /** Compute total receiving count. */
  size_t total_recvcount = 0;
  for ( size_t p = 0; p < size; p ++ )
  {
    rdispls[ p + 1 ] = rdispls[ p ] + recvcounts[ p ];
    total_recvcount += recvcounts[ p ];
  }

  /** Resize receving buffer. */
  recvbuf.resize( total_recvcount );

  Alltoallv( sendbuf.data(), sendcounts.data(), sdispls.data(), 
             recvbuf.data(), recvcounts.data(), rdispls.data(), comm );

  //printf( "after Alltoallv\n" ); fflush( stdout );

  recvvector.resize( size );
  for ( size_t p = 0; p < size; p ++ )
  {
    recvvector[ p ].insert( recvvector[ p ].end(), 
        recvbuf.begin() + rdispls[ p ], 
        recvbuf.begin() + rdispls[ p + 1 ] );

    if ( recvvector[ p ].size() && m )
    {
      recvvector[ p ].resize( m, recvvector[ p ].size() / m );
    }
  }

  //printf( "after resize\n" ); fflush( stdout );
  //mpi::Barrier( comm );

  return 0;

}; /** end AlltoallVector() */
}; /** end namespace mpi */




















typedef enum 
{
  CBLK, /** Elemental MC */
  RBLK, /** Elemental MR */
  CIDS, /** Distributed according to column ids */
  RIDS, /** Distributed according to    row ids */
  USER, /** Distributed acoording to user defined maps */
  STAR, /** Elemental STAR */
  CIRC  /** Elemental CIRC */
} Distribution_t;


#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<class T, class Allocator = hbw::allocator<T> >
#elif  HMLP_USE_CUDA
/** use pinned (page-lock) memory for NVIDIA GPUs */
template<class T, class Allocator = thrust::system::cuda::experimental::pinned_allocator<T> >
#else
/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
#endif
class DistDataBase : public Data<T, Allocator>
{
  public:

    /** Default constructor */
    DistDataBase( size_t m, size_t n, mpi::Comm comm )
    {
      this->global_m = m;
      this->global_n = n;
      this->comm = comm;
      mpi::Comm_size( comm, &comm_size );
      mpi::Comm_rank( comm, &comm_rank );
    };

    /** Constrcut a zero matrix (but mpi::Comm is still required) */
    DistDataBase( mpi::Comm comm ) { DistDataBase( 0, 0, comm ); };

    /** MPI support */
    mpi::Comm GetComm() { return comm; };
    int GetSize() { return comm_size; };
    int GetRank() { return comm_rank; };

    /** Get total row and column numbers across all MPI ranks */
    size_t row() { return global_m; };
    size_t col() { return global_n; };

    /** Get row and column numbers owned by this MPI rank */
    size_t row_owned() { return Data<T>::row(); };
    size_t col_owned() { return Data<T>::col(); };

  private:

    /** Total row and column numbers across all MPI ranks */
    size_t global_m = 0;
    size_t global_n = 0;

    /** MPI support */
    mpi::Comm comm = MPI_COMM_WORLD;
    int comm_size = 0;
    int comm_rank = 0;

}; /** end class DistDataBase */


/** */
#ifdef HMLP_MIC_AVX512
/** use hbw::allocator for Intel Xeon Phi */
template<Distribution_t ROWDIST, Distribution_t COLDIST, class T, class Allocator = hbw::allocator<T> >
#elif  HMLP_USE_CUDA
/** use pinned (page-lock) memory for NVIDIA GPUs */
template<Distribution_t ROWDIST, Distribution_t COLDIST, class T, class Allocator = thrust::system::cuda::experimental::pinned_allocator<T> >
#else
/** use default stl allocator */
template<Distribution_t ROWDIST, Distribution_t COLDIST, class T, class Allocator = std::allocator<T> >
#endif
class DistData : public DistDataBase<T, Allocator>
{
  public:
  private:
};




template<typename T>
class DistData<CIRC, CIRC, T> : public DistDataBase<T>
{
  public:


    #ifdef HMLP_MIC_AVX512
    /** use hbw::allocator for Intel Xeon Phi */
    using ALLOCATOR = hbw::allocator<T>;
    #elif  HMLP_USE_CUDA
    /** use pinned (page-lock) memory for NVIDIA GPUs */
    using ALLOCATOR = thrust::system::cuda::experimental::pinned_allocator<T>;
    #else
    /** use default stl allocator */
    using ALLOCATOR = std::allocator<T>;
    #endif


    DistData( size_t m, size_t n, int owner, mpi::Comm comm ) :
      DistDataBase<T>( m, n, comm )
    {
      this->owner = owner;
      if ( this->GetRank() == owner ) this->resize( m, n );
    };


    /** redistribute from CIDS to CBLK */
    DistData<CIRC, CIRC, T> & operator = ( DistData<STAR, CIDS, T> &A )
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      /** allocate buffer for ids */
      std::vector<std::vector<size_t>> sendids = A.CIRCOwnership( owner );
      std::vector<std::vector<size_t>> recvids( size );

      /** exchange cids */
      mpi::AlltoallVector( sendids, recvids, comm );
      
      /** allocate buffer for data */
      std::vector<std::vector<T, ALLOCATOR>> senddata( size );
      std::vector<std::vector<T, ALLOCATOR>> recvdata( size );

      std::vector<size_t> amap( this->row() );
      for ( size_t i = 0; i < amap.size(); i ++ ) amap[ i ] = i;

      /** extract rows from A<STAR,CIDS> */
      senddata[ owner ] = A( amap, sendids[ owner ] );

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );

      if ( rank == owner )
      {
        for ( size_t p = 0; p < size; p ++ )
        {
          size_t ld = this->row();

          for ( size_t j = 0; j < recvids[ p ].size(); j ++ )
          {
            size_t cid = recvids[ p ][ j ];

            for ( size_t i = 0; i < this->row(); i ++ )
            {
              (*this)( i, cid ) = recvdata[ p ][ j * ld + i ];
            }
          };
        };
      }

      mpi::Barrier( comm );
      return (*this);
    };



  private:

    int owner = 0;

};


template<typename T>
class DistData<STAR, CBLK, T> : public DistDataBase<T>
{
  public:

    #ifdef HMLP_MIC_AVX512
    /** use hbw::allocator for Intel Xeon Phi */
    using ALLOCATOR = hbw::allocator<T>;
    #elif  HMLP_USE_CUDA
    /** use pinned (page-lock) memory for NVIDIA GPUs */
    using ALLOCATOR = thrust::system::cuda::experimental::pinned_allocator<T>;
    #else
    /** use default stl allocator */
    using ALLOCATOR = std::allocator<T>;
    #endif


    DistData( size_t m, size_t n, mpi::Comm comm ) 
      : DistDataBase<T>( m, n, comm ) 
    {
      /** MPI */
      int size = this->GetSize();
      int rank = this->GetRank();

      size_t edge_n = n % size;
      size_t local_n = ( n - edge_n ) / size;
      if ( rank < edge_n ) local_n ++;

      /** resize the local buffer */
      this->resize( m, local_n );
    };


    DistData( size_t m, size_t n, T initT, mpi::Comm comm )
      : DistDataBase<T>( m, n, comm ) 
    {
      /** MPI */
      int size = this->GetSize();
      int rank = this->GetRank();

      size_t edge_n = n % size;
      size_t local_n = ( n - edge_n ) / size;
      if ( rank < edge_n ) local_n ++;

      /** resize the local buffer */
      this->resize( m, local_n, initT );
    };




    /**
     *  constructor that reads a binary file
     */ 
    DistData( size_t m, size_t n, mpi::Comm comm, string &filename ) 
      : DistData<STAR, CBLK, T>( m, n, comm )
    {
      read( m, n, filename );
    };

    /**
     *  read a dense column-major matrix 
     */ 
    void read( size_t m, size_t n, string &filename )
    {
      assert( this->row() == m );
      assert( this->col() == n );

      /** MPI */
      int size = this->GetSize();
      int rank = this->GetRank();

      /** print out filename */
      cout << filename << endl;

      std::ifstream file( filename.data(), 
          std::ios::in|std::ios::binary|std::ios::ate );

      if ( file.is_open() )
      {
        auto size = file.tellg();
        assert( size == m * n * sizeof(T) );

        //for ( size_t j = rank; j < n; j += size )
        //{
        //  size_t byte_offset = j * m * sizeof(T);
        //  file.seekg( byte_offset, std::ios::beg );
        //  file.read( (char*)this->columndata( j / size ), m * sizeof(T) );
        //}


        file.close();
      }

      #pragma omp parallel
      {
        std::ifstream ompfile( filename.data(), 
          std::ios::in|std::ios::binary|std::ios::ate );

        if ( ompfile.is_open() )
        {
          #pragma omp for
          for ( size_t j = rank; j < n; j += size )
          {
            size_t byte_offset = j * m * sizeof(T);
            ompfile.seekg( byte_offset, std::ios::beg );
            ompfile.read( (char*)this->columndata( j / size ), m * sizeof(T) );
          }
          ompfile.close();
        }
      } /** end omp parallel */





    }; /** end void read() */








    /**
     *  Overload operator () to allow accessing data using gids
     */ 
    T & operator () ( size_t i , size_t j )
    {
      /** assert that Kij is stored on this MPI process */
      assert( j % this->GetSize() == this->GetRank() );

      /** return reference of Kij */
      return DistDataBase<T>::operator () ( i, j / this->GetSize() );
    };


    /**
     *  Overload operator () to return a local submatrix using gids
     */ 
    Data<T> operator () ( vector<size_t> &I, vector<size_t> &J )
    {
      for ( auto it = J.begin(); it != J.end(); it ++ )
      {
        /** assert that Kij is stored on this MPI process */
        assert( (*it) % this->GetSize() == this->GetRank() );
        (*it) = (*it) / this->GetSize();
      }
      return DistDataBase<T>::operator () ( I, J );
    };


    /** redistribute from CIRC to CBLK */
    DistData<STAR, CBLK, T> & operator = ( const DistData<CIRC, CIRC, T> &A )
    {
      printf( "not implemented yet\n" );
      exit( 1 );
      return (*this);
    };

    
    /** Redistribute from CIDS to CBLK */
    DistData<STAR, CBLK, T> & operator = ( DistData<STAR, CIDS, T> &A )
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      /** allocate buffer for ids */
      vector<vector<size_t>> sendids = A.CBLKOwnership();
      vector<vector<size_t>> recvids( size );

      /** exchange cids */
      mpi::AlltoallVector( sendids, recvids, comm );
      
      /** allocate buffer for data */
      vector<vector<T, ALLOCATOR>> senddata( size );
      vector<vector<T, ALLOCATOR>> recvdata( size );

      vector<size_t> amap( this->row() );
      for ( size_t i = 0; i < amap.size(); i ++ ) amap[ i ] = i;

      /** extract rows from A<STAR, CIDS> */
      #pragma omp parallel for
      for ( size_t p = 0; p < size; p ++ )
      {
        /** recvids should be gids (not local posiiton) */
        senddata[ p ] = A( amap, sendids[ p ] );
      }

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );

      /** 
       *  It is possible that the received cid has several copies on
       *  different MPI rank.
       */

      #pragma omp parallel for 
      for ( size_t p = 0; p < size; p ++ )
      {
        size_t ld = this->row();

        for ( size_t j = 0; j < recvids[ p ].size(); j ++ )
        {
          size_t cid = recvids[ p ][ j ];

          for ( size_t i = 0; i < this->row(); i ++ )
          {
            (*this)( i, cid ) = recvdata[ p ][ j * ld + i ];
          }
        };
      };

      mpi::Barrier( comm );
      /** free all buffers and return */
      return (*this);
    };







  private:

}; /** end class DistData<STAR, CBLK, T> */



/**
 *  @brief Ecah MPI process own ( n / size ) rows of A
 *         in a cyclic fashion (Round Robin). i.e.
 *         If there are 3 MPI processes, then
 *
 *         rank0 owns A(0,:), A(3,:), A(6,:), ...
 *         rank1 owns A(1,:), A(4,:), A(7,:), ...
 *         rank2 owns A(2,:), A(5,:), A(8,:), ...
 */ 
template<typename T>
class DistData<RBLK, STAR, T> : public DistDataBase<T>
{
  public:

    #ifdef HMLP_MIC_AVX512
    /** use hbw::allocator for Intel Xeon Phi */
    using ALLOCATOR = hbw::allocator<T>;
    #elif  HMLP_USE_CUDA
    /** use pinned (page-lock) memory for NVIDIA GPUs */
    using ALLOCATOR = thrust::system::cuda::experimental::pinned_allocator<T>;
    #else
    /** use default stl allocator */
    using ALLOCATOR = std::allocator<T>;
    #endif



    DistData( size_t m, size_t n, mpi::Comm comm ) : DistDataBase<T>( m, n, comm ) 
    {
      /** MPI */
      int size = this->GetSize();
      int rank = this->GetRank();
      size_t edge_m = m % size;
      size_t local_m = ( m - edge_m ) / size;
      if ( rank < edge_m ) local_m ++;

      /** resize the local buffer */
      this->resize( local_m, n );
    };



    /**
     *  Overload operator () to allow accessing data using gids
     */ 
    T & operator () ( size_t i , size_t j )
    {
      /** assert that Kij is stored on this MPI process */
      if ( i % this->GetSize() != this->GetRank() )
      {
        printf( "ERROR: accessing %lu on rank %d\n", i, this->GetRank() );
        exit( 1 );
      }
      /** return reference of Kij */
      return DistDataBase<T>::operator () ( i / this->GetSize(), j );
    };


    /**
     *  Overload operator () to return a local submatrix using gids
     */ 
    hmlp::Data<T> operator () ( std::vector<size_t> I, std::vector<size_t> J )
    {
      for ( auto it = I.begin(); it != I.end(); it ++ )
      {
        /** assert that Kij is stored on this MPI process */
        assert( (*it) % this->GetSize() == this->GetRank() );
        (*it) = (*it) / this->GetSize();
      }
      return DistDataBase<T>::operator () ( I, J );
    };


    /** redistribute from CIRC to RBLK */
    DistData<RBLK, STAR, T> & operator = ( const DistData<CIRC, CIRC, T> &A )
    {
      printf( "not implemented yet\n" );
      exit( 1 );
      return (*this);
    };



    /** redistribute from RIDS to RBLK */
    DistData<RBLK, STAR, T> & operator = ( DistData<RIDS, STAR, T> &A )
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      /** allocate buffer for ids */
      vector<vector<size_t>> sendids = A.RBLKOwnership();
      vector<vector<size_t>> recvids( size );

      /** Exchange rids. */
      mpi::AlltoallVector( sendids, recvids, comm );
      
      /** Allocate buffer for data. */
      vector<vector<T, ALLOCATOR>> senddata( size );
      vector<vector<T, ALLOCATOR>> recvdata( size );

      vector<size_t> bmap( this->col() );
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

      /** Extract rows from A<RBLK,STAR> */
      #pragma omp parallel for 
      for ( size_t p = 0; p < size; p ++ )
      {
        /** recvids should be gids (not local posiiton) */
        senddata[ p ] = A( sendids[ p ], bmap );
      }

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );

      #pragma omp parallel for 
      for ( size_t p = 0; p < size; p ++ )
      {
        size_t ld = recvdata[ p ].size() / this->col();
        assert( ld == recvids[ p ].size() );
        for ( size_t i = 0; i < recvids[ p ].size(); i ++ )
        {
          size_t rid = recvids[ p ][ i ];
          for ( size_t j = 0; j < this->col(); j ++ )
          {
            (*this)( rid, j ) = recvdata[ p ][ j * ld + i ];
          }
        };
      };

      mpi::Barrier( comm );
      /** free all buffers and return */
      return (*this);
    };

  private:

}; /** end class DistData<RBLK, START, T> */




/**
 *  @brief Ecah MPI process own ( cids.size() ) columns of A,
 *         and cids denote the distribution. i.e.
 *         ranki owns A(:,cids[0]), 
 *                    A(:,cids[1]), 
 *                    A(:,cids[2]), ...
 */ 
template<typename T>
class DistData<STAR, CIDS, T> : public DistDataBase<T>
{
  public:

    #ifdef HMLP_MIC_AVX512
    /** use hbw::allocator for Intel Xeon Phi */
    using ALLOCATOR = hbw::allocator<T>;
    #elif  HMLP_USE_CUDA
    /** use pinned (page-lock) memory for NVIDIA GPUs */
    using ALLOCATOR = thrust::system::cuda::experimental::pinned_allocator<T>;
    #else
    /** use default stl allocator */
    using ALLOCATOR = std::allocator<T>;
    #endif

    /** Default constructor */
    DistData( size_t m, size_t n, vector<size_t> &cids, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      /** now check if (sum cids.size() >= n) */
      size_t bcast_n = cids.size();
      size_t reduc_n = 0;
      mpi::Allreduce( &bcast_n, &reduc_n, 1, MPI_SUM, comm );
      assert( reduc_n == n );
      this->cids = cids;
      this->resize( m, cids.size() );

      for ( size_t j = 0; j < cids.size(); j ++ )
        cid2col[ cids[ j ] ] = j;      
    };

    /** Default constructor */
    DistData( size_t m, size_t n, vector<size_t> &cids, T initT, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      /** now check if (sum cids.size() >= n) */
      size_t bcast_n = cids.size();
      size_t reduc_n = 0;
      mpi::Allreduce( &bcast_n, &reduc_n, 1, MPI_SUM, comm );
      assert( reduc_n == n );
      this->cids = cids;
      this->resize( m, cids.size(), initT );

      for ( size_t j = 0; j < cids.size(); j ++ )
        cid2col[ cids[ j ] ] = j;      
    };

    DistData( size_t m, size_t n, vector<size_t> &cids, Data<T> &A, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      assert( A.row() == m );
      assert( A.col() == cids.size() );

      /** now check if (sum cids.size() >= n) */
      size_t bcast_n = cids.size();
      size_t reduc_n = 0;
      mpi::Allreduce( &bcast_n, &reduc_n, 1, MPI_SUM, comm );
      assert( reduc_n == n );
      this->cids = cids;

      this->insert( this->end(), A.begin(), A.end() );
      this->resize( A.row(), A.col() );

      for ( size_t j = 0; j < cids.size(); j ++ ) cid2col[ cids[ j ] ] = j;      
    };

    /**
     *  Overload operator () to allow accessing data using gids
     */ 
    T & operator () ( size_t i , size_t j )
    {
      /** assert that Kij is stored on this MPI process */
      assert( cid2col.count( j ) == 1 );
      /** return reference of Kij */
      return DistDataBase<T>::operator () ( i, cid2col[ j ] );
    };


    /**
     *  Overload operator () to return a local submatrix using gids
     */ 
    Data<T> operator () ( vector<size_t> I, vector<size_t> J )
    {
      for ( auto it = J.begin(); it != J.end(); it ++ )
      {
        /** assert that Kij is stored on this MPI process */
        assert( cid2col.count(*it) == 1 );
        (*it) = cid2col[ (*it) ];
      }
      return DistDataBase<T>::operator () ( I, J );
    };

    bool HasColumn( size_t cid ) 
    {
      return cid2col.count( cid );
    };

    T *columndata( size_t cid )
    {
      assert( cid2col.count( cid ) == 1 );
      return Data<T>::columndata( cid2col[ cid ] );
    };

    pair<size_t, T*> GetIDAndColumnPointer( size_t j )
    {
      return pair<size_t, T*>( cids[ j ], Data<T>::columndata( j ) );
    };

    /**
     *  @brief Return a vector of vector that indicates the RBLK ownership
     *         of each MPI rank.
     */ 
    vector<vector<size_t>> CBLKOwnership()
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      vector<vector<size_t>> ownership( size );

      for ( auto it = cids.begin(); it != cids.end(); it ++ )
      {
        size_t cid = (*it);
        /** 
         *  While in CBLK distribution, rid is owned by 
         *  rank ( cid % size ) at position ( cid / size ) 
         */
        ownership[ cid % size ].push_back( cid );
      };

      return ownership;

    }; /** end CBLKOwnership() */


    vector<vector<size_t>> CIRCOwnership( int owner )
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      vector<vector<size_t>> ownership( size );
      ownership[ owner ] = cids;
      return ownership;
    }; /** end CIRCOwnership() */



    /** Redistribution from CBLK to CIDS */
    DistData<STAR, CIDS, T> & operator = ( DistData<STAR, CBLK, T> &A )
    {
      /** assertion: must provide rids */
      assert( cids.size() );

      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      /** allocate buffer for ids */
      vector<vector<size_t>> sendids = CBLKOwnership();
      vector<vector<size_t>> recvids( size );

      /** 
       *  exchange cids: 
       *
       *  sendids contain all  required ids from each rank
       *  recvids contain all requested ids from each rank
       *
       */
      mpi::AlltoallVector( sendids, recvids, comm );


      /** allocate buffer for data */
      vector<vector<T, ALLOCATOR>> senddata( size );
      vector<vector<T, ALLOCATOR>> recvdata( size );

      vector<size_t> amap( this->row() );
      for ( size_t i = 0; i < amap.size(); i ++ ) amap[ i ] = i;


      /** extract columns from A<STAR,CBLK> */
      #pragma omp parallel for 
      for ( size_t p = 0; p < size; p ++ )
      {
        /** recvids should be gids (not local posiiton) */
        senddata[ p ] = A( amap, recvids[ p ] );
        assert( senddata[ p ].size() == amap.size() * recvids[ p ].size() );
      }

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );

      #pragma omp parallel for 
      for ( size_t p = 0; p < size; p ++ )
      {
        assert( recvdata[ p ].size() == sendids[ p ].size() * this->row() );

        size_t ld = this->row();

        for ( size_t j = 0; j < sendids[ p ].size(); j ++ )
        {
          size_t cid = sendids[ p ][ j ];
          for ( size_t i = 0; i < this->row(); i ++ )
          {
            (*this)( i, cid ) = recvdata[ p ][ j * ld + i ];
          }
        };
      };

      mpi::Barrier( comm );
      return *this;
    };

  private:

    vector<size_t> cids;

    vector<size_t> cids_from_other_ranks;

    /** Use hash table: cid2col[ cid ] = local column index */
    unordered_map<size_t, size_t> cid2col;

}; /** end class DistData<STAR, CIDS, T> */


template<typename T>
class DistData<STAR, USER, T> : public DistDataBase<T>
{
  public:

    #ifdef HMLP_MIC_AVX512
    /** use hbw::allocator for Intel Xeon Phi */
    using ALLOCATOR = hbw::allocator<T>;
    #elif  HMLP_USE_CUDA
    /** use pinned (page-lock) memory for NVIDIA GPUs */
    using ALLOCATOR = thrust::system::cuda::experimental::pinned_allocator<T>;
    #else
    /** use default stl allocator */
    using ALLOCATOR = std::allocator<T>;
    #endif

    /** Default constructor */
    DistData( size_t m, size_t n, mpi::Comm comm ) 
    : DistDataBase<T>( m, n, comm ), all_rows( m )
    {
      for ( size_t i = 0; i < m; i ++ ) all_rows[ i ] = i;
    };

    /** Constructed from DistDAta<STAR, CBLK> */
    DistData( DistData<STAR, CBLK, T> &A ) 
    : DistData( A.row(), A.col(), A.GetComm() )
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int comm_size; mpi::Comm_size( comm, &comm_size );
      int comm_rank; mpi::Comm_rank( comm, &comm_rank );

      /** Insert column indecies to hash table (cyclic) */
      for ( size_t j  = comm_rank, i = 0; 
                   j  < this->col(); 
                   j += comm_size, i ++ )
      {
        cid2col[ j ] = i;
      }

      /** Check if matches the column number */
      assert( cid2col.size() == A.col_owned() );
      assert( A.size() == cid2col.size() * this->row() );

      /** Use vector<T>::insert */
      this->insert( this->end(), A.begin(), A.end() );

      /** Resize to update owned_row() and owned_col() */
      this->resize( A.row_owned(), A.col_owned() );
    };



    /**
     *  Insert cids and A in to this object.
     *  Note: this is not thread safe. 
     */ 
    void InsertColumns( vector<size_t> &cids, Data<T> &A )
    {
      int comm_rank = this->GetRank();

      /** Check if all dimensions match */
      assert( cids.size() == A.col() );
      assert( this->row() == A.row() );

      /** Remove duplicated columns */
      set<size_t> unique_cids;
      vector<size_t> new_cids, msk_cids;
      new_cids.reserve( cids.size() );
      msk_cids.reserve( cids.size() );

      //printf( "rank %d Here\n", comm_rank ); fflush( stdout );

      /** 
       * Check if cids already exist using HasColumn(). Make sure
       * there is not duplication using unique_cids.
       */
      //#pragma omp critical
      //{
        for ( size_t i = 0; i < cids.size(); i ++ )
        {
          size_t cid = cids[ i ];
          if ( !HasColumn( cid ) && !unique_cids.count( cid ) )
          {
            new_cids.push_back( cid );
            msk_cids.push_back( i );
            unique_cids.insert( cid );
          }
        }

        if ( new_cids.size() )
       // printf( "rank %d cids[ 0 ] %lu cids %lu new_cids %lu\n", 
       //     comm_rank, cids[ 0 ], cids.size(), new_cids.size() ); fflush( stdout );


        /** Insert to hash table one-by-one */
        for ( size_t i = 0; i < new_cids.size(); i ++ )
          cid2col[ new_cids[ i ] ] = i + this->col_owned();

        //printf( "rank %d new_cids %lu cid2col %lu\n", 
        //    comm_rank, new_cids.size(), cid2col.size() ); fflush( stdout );
        /** Use vector<T>::insert */
        Data<T> new_A = A( all_rows, msk_cids );
        //printf( "rank %d A %lu %lu\n", 
        //    comm_rank, A.row(), A.col() ); fflush( stdout );
        this->insert( this->end(), new_A.begin(), new_A.end() );
        //printf( "rank %d this->size %lu row_owned %lu col_owned %lu\n", 
        //    comm_rank, this->size(), this->row_owned(), this->col_owned() ); fflush( stdout );

        /** Check if total size matches */
        assert( this->row_owned() * cid2col.size() == this->size() );

        /** Resize to update owned_row() and owned_col() */
        //this->resize( this->row(), this->size() / this->row() );
        this->resize( this->row(), cid2col.size() );
        //printf( "rank %d this->size %lu row_owned %lu col_owned %lu done\n", 
        //    comm_rank, this->size(), this->row_owned(), this->col_owned() ); fflush( stdout );

      //} /** end omp critical */
    };





    /**
     *  Overload operator () to allow accessing data using gids
     */ 
    T & operator () ( size_t i , size_t j )
    {
      /** assert that Kij is stored on this MPI process */
      assert( cid2col.count( j ) == 1 );
      /** return reference of Kij */
      return DistDataBase<T>::operator () ( i, cid2col[ j ] );
    };

    /**
     *  Overload operator () to return a local submatrix using gids
     */ 
    Data<T> operator () ( vector<size_t> I, vector<size_t> J )
    {
      int comm_rank = this->GetRank();

      for ( auto it = J.begin(); it != J.end(); it ++ )
      {
        /** Assert that Kij is stored on this MPI process */
        if ( cid2col.count( *it ) != 1 )
        {
          printf( "rank %d cid2col.count(%lu) = %lu\n", 
              comm_rank, *it, cid2col.count( *it ) ); fflush( stdout );
          exit( 1 );
        }
        /** Overwrite J with its corresponding offset */
        (*it) = cid2col[ (*it) ];
      }
      return DistDataBase<T>::operator () ( I, J );
    };

    /** Check if cid already exists */
    bool HasColumn( size_t cid ) 
    {
      return cid2col.count( cid );
    };

    /** Return pointer of cid column using Data<T>::columndata */
    T *columndata( size_t cid )
    {
      assert( cid2col.count( cid ) == 1 );
      return Data<T>::columndata( cid2col[ cid ] );
    };


  private:

    /** all_rows = [ 0, 1, ..., m-1 ] */
    vector<size_t> all_rows;


    /** Use hash table: cid2col[ cid ] = local column index */
    unordered_map<size_t, size_t> cid2col;

}; /** end class DistData<STAR, USER, T> */












/**
 *  @brief Ecah MPI process own ( rids.size() ) rows of A,
 *         and rids denote the distribution. i.e.
 *         ranki owns A(rids[0],:), 
 *                    A(rids[1],:), 
 *                    A(rids[2],:), ...
 */ 
template<typename T>
class DistData<RIDS, STAR, T> : public DistDataBase<T>
{
  public:

    #ifdef HMLP_MIC_AVX512
    /** use hbw::allocator for Intel Xeon Phi */
    using ALLOCATOR = hbw::allocator<T>;
    #elif  HMLP_USE_CUDA
    /** use pinned (page-lock) memory for NVIDIA GPUs */
    using ALLOCATOR = thrust::system::cuda::experimental::pinned_allocator<T>;
    #else
    /** use default stl allocator */
    using ALLOCATOR = std::allocator<T>;
    #endif


    /** default constructor */
    DistData( size_t m, size_t n, std::vector<size_t> &rids, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      /** now check if (sum rids.size() == m) */
      size_t bcast_m = rids.size();
      size_t reduc_m = 0;
      mpi::Allreduce( &bcast_m, &reduc_m, 1, MPI_SUM, comm );

      if ( reduc_m != m ) 
        printf( "%lu %lu\n", reduc_m, m ); fflush( stdout );


      assert( reduc_m == m );
      this->rids = rids;
      this->resize( rids.size(), n );

      for ( size_t i = 0; i < rids.size(); i ++ )
        rid2row[ rids[ i ] ] = i;      
    };


    /**
     *  Overload operator () to allow accessing data using gids
     */ 
    T & operator () ( size_t i , size_t j )
    {
      /** assert that Kij is stored on this MPI process */
      assert( rid2row.count( i ) == 1 );
      /** return reference of Kij */
      return DistDataBase<T>::operator () ( rid2row[ i ], j );
    };


    /**
     *  Overload operator () to return a local submatrix using gids
     */ 
    hmlp::Data<T> operator () ( std::vector<size_t> I, std::vector<size_t> J )
    {
      for ( auto it = I.begin(); it != I.end(); it ++ )
      {
        /** assert that Kij is stored on this MPI process */
        assert( rid2row.count(*it) == 1 );
        (*it) = rid2row[ (*it) ];
      }
      return DistDataBase<T>::operator () ( I, J );
    };






    /**
     *  @brief Return a vector of vector that indicates the RBLK ownership
     *         of each MPI rank.
     */ 
    std::vector<std::vector<size_t>> RBLKOwnership()
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      std::vector<std::vector<size_t>> ownership( size );

      for ( auto it = rids.begin(); it != rids.end(); it ++ )
      {
        size_t rid = (*it);
        /** 
         *  While in RBLK distribution, rid is owned by 
         *  rank ( rid % size ) at position ( rid / size ) 
         */
        ownership[ rid % size ].push_back( rid );
      };

      return ownership;

    }; /** end RBLKOwnership() */



    /**
     *  redistribution from RBLK to RIDS (MPI_Alltoallv) 
     */
    DistData<RIDS, STAR, T> & operator = ( DistData<RBLK, STAR, T> &A )
    {
      /** assertion: must provide rids */
      assert( rids.size() );

      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      //printf( "Enter redistrivution rids.size() %lu\n", rids.size() ); fflush( stdout );
      //mpi::Barrier( comm );

      /** allocate buffer for ids */
      std::vector<std::vector<size_t>> sendids = RBLKOwnership();
      std::vector<std::vector<size_t>> recvids( size );

      //for ( size_t i = 0; i < rids.size(); i ++ )
      //{
      //  /** since A has RBLK distribution, rid is stored at rank (rid) % size */
      //  size_t rid = rids[ i ];
      //  /** ( rid / size ) is the local id of A */
      //  sendids[ rid % size ].push_back( rid );
      //}

      //printf( "before All2allvector1\n" ); fflush( stdout );
      //mpi::Barrier( comm );

      /** 
       *  exchange rids: 
       *
       *  sendids contain all  required ids from each rank
       *  recvids contain all requested ids from each rank
       *
       */
      mpi::AlltoallVector( sendids, recvids, comm );

      //printf( "Finish All2allvector1\n" );fflush( stdout );
      //mpi::Barrier( comm );


      std::vector<std::vector<T, ALLOCATOR>> senddata( size );
      std::vector<std::vector<T, ALLOCATOR>> recvdata( size );

      std::vector<size_t> bmap( this->col() );
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

      /** extract rows from A<RBLK,STAR> */
      #pragma omp parallel for 
      for ( size_t p = 0; p < size; p ++ )
      {
        /** recvids should be gids (not local posiiton) */
        senddata[ p ] = A( recvids[ p ], bmap );
        assert( senddata[ p ].size() == recvids[ p ].size() * bmap.size() );
      }

      //printf( "before All2allvector2\n" ); fflush( stdout );
      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );


      #pragma omp parallel for 
      for ( size_t p = 0; p < size; p ++ )
      {
        assert( recvdata[ p ].size() == sendids[ p ].size() * this->col() );

        size_t ld = recvdata[ p ].size() / this->col();

        assert( ld == sendids[ p ].size() );


        for ( size_t i = 0; i < sendids[ p ].size(); i ++ )
        {
          size_t rid = sendids[ p ][ i ];
          for ( size_t j = 0; j < this->col(); j ++ )
          {
            (*this)( rid, j ) = recvdata[ p ][ j * ld + i ];
          }
        };
      };
      mpi::Barrier( comm );

      return (*this);
    };







  private:

    /** owned row gids */
    vector<size_t> rids;

    map<size_t, size_t> rid2row;


}; /** end class DistData<RIDS, STAR, T> */




template<typename T>
class DistData<STAR, STAR, T> : public DistDataBase<T>
{
  public:

  private:

}; /** end class DistData<STAR, STAR, T> */






}; /** end namespace hmlp */


#endif /** define DISTDATA_HPP */

