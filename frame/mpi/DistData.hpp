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
      mpi::Comm_size( comm, &size );
      mpi::Comm_rank( comm, &rank );
    };

    DistDataBase( mpi::Comm comm )
    {
      DistDataBase( 0, 0, comm );
    };


    mpi::Comm GetComm() { return comm; };
    int GetSize() { return size; };
    int GetRank() { return rank; };

    size_t row() { return global_m; };
    size_t col() { return global_n; };

  private:

    size_t global_m = 0;

    size_t global_n = 0;

    mpi::Comm comm = MPI_COMM_WORLD;

    int size = 0;

    int rank = 0;

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


template<typename T>
class DistData<RBLK, STAR, T> : public DistDataBase<T>
{
  public:

    DistData( size_t m, size_t n, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      int size = this->GetSize();
      int rank = this->GetRank();
      size_t edge_m = m % size;
      size_t local_m = ( m - edge_m ) / size;
      if ( rank < edge_m ) local_m ++;

      /** resize the local buffer */
      this->resize( local_m, n );
    };

    DistData( mpi::Comm comm ) : DistDataBase<T>( comm ) {};


    /** redistribution from CIRC to RBLK */
    DistData<RBLK, STAR, T> & operator = ( const DistData<CIRC, CIRC, T> &A )
    {

      return *this;
    };

  private:

};


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

template<typename T>
class DistData<RIDS, STAR, T> : public DistDataBase<T>
{
  public:

    DistData( size_t m, size_t n, std::vector<size_t> &rids, mpi::Comm comm ) : 
      DistDataBase<T>( m, n, comm ) 
    {
      this->rids = rids;
      this->resize( rids.size(), n );
    };

    /** redistribution from RBLK to RIDS (MPI_Alltoallv) */
    DistData<RIDS, STAR, T> & operator = ( const DistData<RBLK, STAR, T> &A )
    {
      /** MPI */
      int size = this->GetSize();
      int rank = this->GetRank();

      std::vector<std::vector<size_t>> sendids( size );
      std::vector<std::vector<size_t>> recvids( size );

      for ( size_t i = 0; i < rids.size(); i ++ )
      {
        /** since A has RBLK distribution, rid is stored at rank (rid) % size */
        size_t rid = rids[ i ];
        sendvdis[ rid % size ].push_back( rid / rank );
      }

      /** exchange rids */
      mpi::AlltoallVector( sendids, recvids, comm );

      std::vector<std::vector<T>> senddata( size );
      std::vector<std::vector<T>> recvdata( size );
      std::vector<size_t> bmap( this->col() );
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

      for ( size_t j = 0; j < size; j ++ )
      {
        senddata[ j ] = (*this)( recvids[ j ], bmap );
      }

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );

      return *this;
    };

  private:

    std::vector<size_t> rids;
};

template<typename T>
class DistData<STAR, STAR, T> : public DistDataBase<T>
{
  public:

  private:

};






}; /** end namespace hmlp */


#endif /** define DISTDATA_HPP */

