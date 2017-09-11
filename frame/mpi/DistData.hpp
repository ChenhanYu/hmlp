#ifndef DISTDATA_HPP
#define DISTDATA_HPP

#include <containers/data.hpp>
#include <mpi/hmlp_mpi.hpp>

namespace hmlp
{


typedef enum 
{
  CBLK, /** Elemental MC */
  RBLK, /** Elemental MR */
  CIDS, /** distributed according to column ids */
  RIDS, /** distributed according to    row ids */
  STAR, /** Elemental STAR */
  CIRC  /** Elemental CIRC */
} Distribution_t;


template<typename T>
class DistDataBase : public Data<T>
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

    mpi::Comm GetComm() { return comm; };
    int GetSize() { return comm_size; };
    int GetRank() { return comm_rank; };

    size_t row() { return global_m; };
    size_t col() { return global_n; };

  private:

    size_t global_m = 0;

    size_t global_n = 0;

    mpi::Comm comm = MPI_COMM_WORLD;

    int comm_size = 0;

    int comm_rank = 0;

}; /** end class DistDataBase */


/** */
template<Distribution_t ROWDIST, Distribution_t COLDIST, typename T>
class DistData : public DistDataBase<T>
{
  public:
  private:
};


template<typename T>
class DistData<CIRC, CIRC, T> : public DistDataBase<T>
{
  public:

    DistData( size_t m, size_t n, int owner, mpi::Comm comm ) :
      DistDataBase<T>( m, n, comm )
    {
      this->owner = owner;
    };

  private:

    int owner = 0;

};


template<typename T>
class DistData<STAR, CBLK, T> : public DistDataBase<T>
{
  public:

  private:

};


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


    //DistData( mpi::Comm comm ) : DistDataBase<T>( comm ) {};


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
      std::vector<std::vector<T>> senddata( size );
      std::vector<std::vector<T>> recvdata( size );

      std::vector<size_t> bmap( this->col() );
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

      /** extract rows from A<RBLK,STAR> */
      for ( size_t p = 0; p < size; p ++ )
      {
        /** recvids should be gids (not local posiiton) */
        senddata[ p ] = A( sendids[ p ], bmap );
      }

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );


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


template<typename T>
class DistData<STAR, CIDS, T> : public DistDataBase<T>
{
  public:

    /** redistribution from CBLK to CIDS */
    DistData<STAR, CIDS, T> & operator = ( const DistData<STAR, CBLK, T> &A )
    {
      return *this;
    };

  private:

    std::vector<size_t> cids;
};




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

    /** default constructor */
    DistData( size_t m, size_t n, std::vector<size_t> &rids, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      /** now check if (sum rids.size() == m) */
      size_t bcast_m = rids.size();
      size_t reduc_m = 0;
      mpi::Allreduce( &bcast_m, &reduc_m, 1, MPI_SUM, comm );
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


      std::vector<std::vector<T>> senddata( size );
      std::vector<std::vector<T>> recvdata( size );

      std::vector<size_t> bmap( this->col() );
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

      /** extract rows from A<RBLK,STAR> */
      for ( size_t p = 0; p < size; p ++ )
      {
        /** recvids should be gids (not local posiiton) */
        senddata[ p ] = A( recvids[ p ], bmap );
        assert( senddata[ p ].size() == recvids[ p ].size() * bmap.size() );
      }

      //printf( "before All2allvector2\n" ); fflush( stdout );
      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );


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

};






}; /** end namespace hmlp */


#endif /** define DISTDATA_HPP */

