#ifndef DISTDATA_HPP
#define DISTDATA_HPP

#include <containers/data.hpp>
#include <mpi/hmlp_mpi.hpp>


namespace hmlp
{




namespace mpi
{
/**
 *  This interface supports hmlp::Data. [ m, n ] will be set
 *  after the routine.
 *
 */ 
template<typename T>
int AlltoallData(
    size_t m,
    std::vector<hmlp::Data<T>> &sendvector, 
    std::vector<hmlp::Data<T>> &recvvector, mpi::Comm comm )
{
  int size = 0;
  int rank = 0;
  Comm_size( comm, &size );
  Comm_rank( comm, &rank );

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



  std::vector<T, ALLOCATOR> sendbuf;
  std::vector<T, ALLOCATOR> recvbuf;
  std::vector<int> sendcounts( size, 0 );
  std::vector<int> recvcounts( size, 0 );
  std::vector<int> sdispls( size + 1, 0 );
  std::vector<int> rdispls( size + 1, 0 );

  for ( size_t p = 0; p < size; p ++ )
  {
    sendcounts[ p ] = sendvector[ p ].size();
    sdispls[ p + 1 ] = sdispls[ p ] + sendcounts[ p ];
    sendbuf.insert( sendbuf.end(), 
        sendvector[ p ].begin(), 
        sendvector[ p ].end() );
  }

  /** exchange sendcount */
  Alltoall( sendcounts.data(), 1, recvcounts.data(), 1, comm );

  //printf( "after Alltoall\n" ); fflush( stdout );


  /** compute total receiving count */
  size_t total_recvcount = 0;
  for ( size_t p = 0; p < size; p ++ )
  {
    rdispls[ p + 1 ] = rdispls[ p ] + recvcounts[ p ];
    total_recvcount += recvcounts[ p ];
  }

  /** resize receving buffer */
  recvbuf.resize( total_recvcount );

  Alltoallv(
      sendbuf.data(), sendcounts.data(), sdispls.data(), 
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
  CIDS, /** distributed according to column ids */
  RIDS, /** distributed according to    row ids */
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

    /**
     *  
     */ 
    DistDataBase( size_t m, size_t n, mpi::Comm comm )
    {
      this->global_m = m;
      this->global_n = n;
      this->comm = comm;
      mpi::Comm_size( comm, &comm_size );
      mpi::Comm_rank( comm, &comm_rank );
    };

    DistDataBase( mpi::Comm comm )
    {
      DistDataBase( 0, 0, comm );
    };

    mpi::Comm GetComm() 
    { 
      return comm; 
    };

    int GetSize() 
    { 
      return comm_size; 
    };
    
    int GetRank() 
    { 
      return comm_rank; 
    };

    size_t row() 
    { 
      return global_m; 
    };

    size_t col() 
    { 
      return global_n; 
    };

    size_t row_owned()
    {
      return Data<T>::row();
    };

    size_t col_owned()
    {
      return Data<T>::col();
    };

  private:

    size_t global_m = 0;

    size_t global_n = 0;

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
    DistData( size_t m, size_t n, mpi::Comm comm, std::string &filename ) 
      : DistData<STAR, CBLK, T>( m, n, comm )
    {
      read( m, n, filename );
    };

    /**
     *  read a dense column-major matrix 
     */ 
    void read( size_t m, size_t n, std::string &filename )
    {
      assert( this->row() == m );
      assert( this->col() == n );

      /** MPI */
      int size = this->GetSize();
      int rank = this->GetRank();

      /** print out filename */
      std::cout << filename << std::endl;

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
    hmlp::Data<T> operator () ( std::vector<size_t> I, std::vector<size_t> J )
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

    
    /** redistribute from CIDS to CBLK */
    DistData<STAR, CBLK, T> & operator = ( DistData<STAR, CIDS, T> &A )
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      /** allocate buffer for ids */
      std::vector<std::vector<size_t>> sendids = A.CBLKOwnership();
      std::vector<std::vector<size_t>> recvids( size );

      /** exchange cids */
      mpi::AlltoallVector( sendids, recvids, comm );
      
      /** allocate buffer for data */
      std::vector<std::vector<T, ALLOCATOR>> senddata( size );
      std::vector<std::vector<T, ALLOCATOR>> recvdata( size );

      std::vector<size_t> amap( this->row() );
      for ( size_t i = 0; i < amap.size(); i ++ ) amap[ i ] = i;

      /** extract rows from A<STAR,CIDS> */
      #pragma omp parallel for
      for ( size_t p = 0; p < size; p ++ )
      {
        /** recvids should be gids (not local posiiton) */
        senddata[ p ] = A( amap, sendids[ p ] );
      }

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );

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



    DistData( size_t m, size_t n, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
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
      assert( i % this->GetSize() == this->GetRank() );
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
      std::vector<std::vector<size_t>> sendids = A.RBLKOwnership();
      std::vector<std::vector<size_t>> recvids( size );

      /** exchange rids */
      mpi::AlltoallVector( sendids, recvids, comm );
      
      /** allocate buffer for data */
      std::vector<std::vector<T, ALLOCATOR>> senddata( size );
      std::vector<std::vector<T, ALLOCATOR>> recvdata( size );

      std::vector<size_t> bmap( this->col() );
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

      /** extract rows from A<RBLK,STAR> */
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

    /** default constructor */
    DistData( size_t m, size_t n, std::vector<size_t> &cids, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      /** now check if (sum cids.size() == n) */
      size_t bcast_n = cids.size();
      size_t reduc_n = 0;
      mpi::Allreduce( &bcast_n, &reduc_n, 1, MPI_SUM, comm );
      assert( reduc_n == n );
      this->cids = cids;
      this->resize( m, cids.size() );

      for ( size_t j = 0; j < cids.size(); j ++ )
        cid2col[ cids[ j ] ] = j;      
    };

    /** default constructor */
    DistData( size_t m, size_t n, std::vector<size_t> &cids, T initT, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      /** now check if (sum cids.size() == n) */
      size_t bcast_n = cids.size();
      size_t reduc_n = 0;
      mpi::Allreduce( &bcast_n, &reduc_n, 1, MPI_SUM, comm );
      assert( reduc_n == n );
      this->cids = cids;
      this->resize( m, cids.size(), initT );

      for ( size_t j = 0; j < cids.size(); j ++ )
        cid2col[ cids[ j ] ] = j;      
    };

    DistData( size_t m, size_t n, std::vector<size_t> &cids, Data<T> &A, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      assert( A.row() == m );
      assert( A.col() == cids.size() );

      /** now check if (sum cids.size() == n) */
      size_t bcast_n = cids.size();
      size_t reduc_n = 0;
      mpi::Allreduce( &bcast_n, &reduc_n, 1, MPI_SUM, comm );
      assert( reduc_n == n );
      this->cids = cids;

      this->insert( this->end(), A.begin(), A.end() );
      this->resize( A.row(), A.col() );

      for ( size_t j = 0; j < cids.size(); j ++ ) cid2col[ cids[ j ] ] = j;      
    };



    /** */
    //void Set( size_t m, size_t n )



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
    hmlp::Data<T> operator () ( std::vector<size_t> I, std::vector<size_t> J )
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

    std::pair<size_t, T*> GetIDAndColumnPointer( size_t j )
    {
      return std::pair<size_t, T*>( cids[ j ], Data<T>::columndata( j ) );
    };

    /**
     *  @brief Return a vector of vector that indicates the RBLK ownership
     *         of each MPI rank.
     */ 
    std::vector<std::vector<size_t>> CBLKOwnership()
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      std::vector<std::vector<size_t>> ownership( size );

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


    std::vector<std::vector<size_t>> CIRCOwnership( int owner )
    {
      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      std::vector<std::vector<size_t>> ownership( size );
      ownership[ owner ] = cids;
      return ownership;
    }; /** end CIRCOwnership() */



    /** redistribution from CBLK to CIDS */
    DistData<STAR, CIDS, T> & operator = ( DistData<STAR, CBLK, T> &A )
    {
      /** assertion: must provide rids */
      assert( cids.size() );

      /** MPI */
      mpi::Comm comm = this->GetComm();
      int size = this->GetSize();
      int rank = this->GetRank();

      /** allocate buffer for ids */
      std::vector<std::vector<size_t>> sendids = CBLKOwnership();
      std::vector<std::vector<size_t>> recvids( size );

      /** 
       *  exchange cids: 
       *
       *  sendids contain all  required ids from each rank
       *  recvids contain all requested ids from each rank
       *
       */
      mpi::AlltoallVector( sendids, recvids, comm );


      /** allocate buffer for data */
      std::vector<std::vector<T, ALLOCATOR>> senddata( size );
      std::vector<std::vector<T, ALLOCATOR>> recvdata( size );

      std::vector<size_t> amap( this->row() );
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

      return *this;
    };

  private:

    std::vector<size_t> cids;

    std::map<size_t, size_t> cid2col;

}; /** end class DistData<STAR, CIDS, T> */




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

      return (*this);
    };







  private:

    /** owned row gids */
    std::vector<size_t> rids;

    std::map<size_t, size_t> rid2row;


}; /** end class DistData<RIDS, STAR, T> */




template<typename T>
class DistData<STAR, STAR, T> : public DistDataBase<T>
{
  public:

  private:

}; /** end class DistData<STAR, STAR, T> */






}; /** end namespace hmlp */


#endif /** define DISTDATA_HPP */

