/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  


#ifndef GOFMM_MPI_HPP
#define GOFMM_MPI_HPP

#include <mpi/tree_mpi.hpp>
#include <mpi/DistData.hpp>
#include <gofmm/gofmm.hpp>
#include <primitives/combinatorics.hpp>
#include <primitives/gemm.hpp>


using namespace hmlp;

using namespace hmlp::gofmm;

namespace hmlp
{
namespace mpigofmm
{


/**
 *  @biref This class does not have to inherit DistData, but it have to 
 *         inherit DistVirtualMatrix<T>
 *
 */ 
template<typename T>
class DistSPDMatrix : public DistData<STAR, CBLK, T>
{
  public:

    DistSPDMatrix( size_t m, size_t n, mpi::Comm comm ) : 
      DistData<STAR, CBLK, T>( m, n, comm )
    {
    };


    /** ESSENTIAL: this is an abstract function  */
    virtual T operator()( size_t i, size_t j, hmlp::mpi::Comm comm )
    {
      T Kij = 0;

      /** MPI */
      int size, rank;
      hmlp::mpi::Comm_size( comm, &size );
      hmlp::mpi::Comm_rank( comm, &rank );

      std::vector<std::vector<size_t>> sendrids( size );
      std::vector<std::vector<size_t>> recvrids( size );
      std::vector<std::vector<size_t>> sendcids( size );
      std::vector<std::vector<size_t>> recvcids( size );

      /** request Kij from rank ( j % size ) */
      sendrids[ i % size ].push_back( i );    
      sendcids[ j % size ].push_back( j );    

      /** exchange ids */
      mpi::AlltoallVector( sendrids, recvrids, comm );
      mpi::AlltoallVector( sendcids, recvcids, comm );

      /** allocate buffer for data */
      std::vector<std::vector<T>> senddata( size );
      std::vector<std::vector<T>> recvdata( size );

      /** fetch subrows */
      for ( size_t p = 0; p < size; p ++ )
      {
        assert( recvrids[ p ].size() == recvcids[ p ].size() );
        for ( size_t j = 0; j < recvcids[ p ].size(); j ++ )
        {
          size_t rid = recvrids[ p ][ j ];
          size_t cid = recvcids[ p ][ j ];
          senddata[ p ].push_back( (*this)( rid, cid ) );
        }
      }

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );

      for ( size_t p = 0; p < size; p ++ )
      {
        assert( recvdata[ p ].size() <= 1 );
        if ( recvdata[ p ] ) Kij = recvdata[ p ][ 0 ];
      }

      return Kij;
    };


    /** ESSENTIAL: return a submatrix */
    virtual hmlp::Data<T> operator()
		( std::vector<size_t> &imap, std::vector<size_t> &jmap, hmlp::mpi::Comm comm )
    {
      hmlp::Data<T> KIJ( imap.size(), jmap.size() );

      /** MPI */
      int size, rank;
      hmlp::mpi::Comm_size( comm, &size );
      hmlp::mpi::Comm_rank( comm, &rank );



      std::vector<std::vector<size_t>> jmapcids( size );

      std::vector<std::vector<size_t>> sendrids( size );
      std::vector<std::vector<size_t>> recvrids( size );
      std::vector<std::vector<size_t>> sendcids( size );
      std::vector<std::vector<size_t>> recvcids( size );

      /** request KIJ from rank ( j % size ) */
      for ( size_t j = 0; j < jmap.size(); j ++ )
      {
        size_t cid = jmap[ j ];
        sendcids[ cid % size ].push_back( cid );
        jmapcids[ cid % size ].push_back(   j );
      }

      for ( size_t p = 0; p < size; p ++ )
      {
        if ( sendcids[ p ].size() ) sendrids[ p ] = imap;
      }

      /** exchange ids */
      mpi::AlltoallVector( sendrids, recvrids, comm );
      mpi::AlltoallVector( sendcids, recvcids, comm );

      /** allocate buffer for data */
      std::vector<hmlp::Data<T>> senddata( size );
      std::vector<hmlp::Data<T>> recvdata( size );

      /** fetch submatrix */
      for ( size_t p = 0; p < size; p ++ )
      {
        if ( recvcids[ p ].size() && recvrids[ p ].size() )
        {
          senddata[ p ] = (*this)( recvrids[ p ], recvcids[ p ] );
        }
      }

      /** exchange data */
      mpi::AlltoallVector( senddata, recvdata, comm );

      /** merging data */
      for ( size_t p = 0; j < size; p ++ )
      {
        assert( recvdata[ p ].size() == imap.size() * recvcids[ p ].size() );
        recvdata[ p ].resize( imap.size(), recvcids[ p ].size() );
        for ( size_t j = 0; j < recvcids[ p ]; i ++ )
        {
          for ( size_t i = 0; i < imap.size(); i ++ )
          {
            KIJ( i, jmapcids[ p ][ j ] ) = recvdata[ p ]( i, j );
          }
        }
      };

      return KIJ;
    };





    virtual hmlp::Data<T> operator()
		( std::vector<int> &imap, std::vector<int> &jmap, hmlp::mpi::Comm comm )
    {
      printf( "operator() not implemented yet\n" );
      exit( 1 );
    };



    /** overload operator */


  private:

}; /** end class DistSPDMatrix */




/**
 *  @brief These are data that shared by the whole local tree.
 *         Distributed setup inherits mpitree::Setup.
 */ 
template<typename SPDMATRIX, typename SPLITTER, typename T>
class Setup : public hmlp::mpitree::Setup<SPLITTER, T>
{
  public:

		void BackGroundProcess( bool *do_terminate )
		{
			K->BackGroundProcess( do_terminate );
		};

    /** humber of neighbors */
    size_t k = 32;

    /** maximum rank */
    size_t s = 64;

    /** relative error for rank-revealed QR */
    T stol = 1E-3;

		/** (default) distance type */
		DistanceMetric metric = ANGLE_DISTANCE;

    /** the SPDMATRIX (accessed with gids: dense, CSC or OOC) */
    SPDMATRIX *K = NULL;

    /** rhs-by-n all weights */
    hmlp::Data<T> *w = NULL;

    /** n-by-nrhs all potentials */
    hmlp::Data<T> *u = NULL;

    /** buffer space, either dimension needs to be n  */
    hmlp::Data<T> *input = NULL;
    hmlp::Data<T> *output = NULL;

    /** regularization */
    T lambda = 0.0;

    /** whether the matrix is symmetric */
    bool issymmetric = true;

    /** use ULV or Sherman-Morrison-Woodbury */
    bool do_ulv_factorization = true;

}; /** end class Setup */




template<typename T>
class NodeData : public hmlp::hfamily::Factor<T>
{
  public:

    NodeData() : kij_skel( 0.0, 0 ), kij_s2s( 0.0, 0 ), kij_s2n( 0.0, 0 ) {};

    /** the omp (or pthread) lock */
    Lock lock;

    /** whether the node can be compressed */
    bool isskel = false;

    /** whether the coefficient mathx has been computed */
    bool hasproj = false;

    /** my skeletons */
    std::vector<size_t> skels;

    /** (buffer) nsamples row gids */
    std::vector<size_t> candidate_rows;

    /** (buffer) sl+sr column gids of children */
    std::vector<size_t> candidate_cols;

    /** (buffer) nsamples-by-(sl+sr) submatrix of K */
    hmlp::Data<T> KIJ; 

    /** 2s, pivoting order of GEQP3 */
    std::vector<int> jpvt;

    /** s-by-2s */
    hmlp::Data<T> proj;

    /** sampling neighbors ids */
    std::map<std::size_t, T> snids; 

    /* pruning neighbors ids */
    std::unordered_set<std::size_t> pnids; 

    /** skeleton weights and potentials */
    hmlp::Data<T> w_skel;
    hmlp::Data<T> u_skel;

    /** permuted weights and potentials (buffer) */
    hmlp::Data<T> w_leaf;
    hmlp::Data<T> u_leaf[ 20 ];

    /** hierarchical tree view of w<RIDS> and u<RIDS> */
    View<T> w_view;
    View<T> u_view;

    /** cached Kab */
    hmlp::Data<std::size_t> Nearbmap;
    hmlp::Data<T> NearKab;
    hmlp::Data<T> FarKab;







    /** Kij evaluation counter counters */
    std::pair<double, std::size_t> kij_skel;
    std::pair<double, std::size_t> kij_s2s;
    std::pair<double, std::size_t> kij_s2n;

    /** many timers */
    double merge_neighbors_time = 0.0;
    double id_time = 0.0;

    /** recorded events (for HMLP Runtime) */
    hmlp::Event skeletonize;
    hmlp::Event updateweight;
    hmlp::Event skeltoskel;
    hmlp::Event skeltonode;

    hmlp::Event s2s;
    hmlp::Event s2n;

    /** knn accuracy */
    double knn_acc = 0.0;
    size_t num_acc = 0;

}; /** end class NodeData */
























/**
 *  @brief This the main splitter used to build the Spd-Askit tree.
 *         First compute the approximate center using subsamples.
 *         Then find the two most far away points to do the 
 *         projection.
 *
 *  @TODO  This splitter often fails to produce an even split when
 *         the matrix is sparse.
 *
 */ 
template<typename SPDMATRIX, int N_SPLIT, typename T> 
struct centersplit : public hmlp::gofmm::centersplit<SPDMATRIX, N_SPLIT, T>
{

  /** shared-memory operator */
  inline std::vector<std::vector<std::size_t> > operator()
  ( 
    std::vector<std::size_t>& gids,
    std::vector<std::size_t>& lids
  ) const 
  {
    /** MPI */
    int size, rank, global_rank;
    hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

    //printf( "rank %d enter shared centersplit n = %lu\n", 
    //    global_rank, gids.size() ); fflush( stdout );

    return hmlp::gofmm::centersplit<SPDMATRIX, N_SPLIT, T>::operator()
      ( gids, lids );
  };


  /** distributed operator */
  inline std::vector<std::vector<size_t> > operator()
  ( 
    std::vector<size_t>& gids,
    hmlp::mpi::Comm comm
  ) const 
  {
    /** all assertions */
    assert( N_SPLIT == 2 );
    assert( this->Kptr );

    /** declaration */
    int size, rank, global_rank;
    hmlp::mpi::Comm_size( comm, &size );
    hmlp::mpi::Comm_rank( comm, &rank );
    hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
    SPDMATRIX &K = *(this->Kptr);
    std::vector<std::vector<size_t>> split( N_SPLIT );




    /** reduce to get the total size of gids */
    int n = 0;
    int num_points_owned = gids.size();
    std::vector<T> temp( gids.size(), 0.0 );

    //printf( "rank %d before Allreduce\n", global_rank ); fflush( stdout );


    /** n = sum( num_points_owned ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &num_points_owned, &n, 1, 
        MPI_INT, MPI_SUM, comm );


    //printf( "rank %d enter distributed centersplit n = %d\n", global_rank, n ); fflush( stdout );



    /** early return */
    if ( n == 0 ) return split;

    /** diagonal entries */
    hmlp::Data<T> DII( n, (size_t)1 ), DCC( this->n_centroid_samples, (size_t)1 );

    /** collecting DII */
    //for ( size_t i = 0; i < gids.size(); i ++ )
    //{
    //  DII[ i ] = K( gids[ i ], gids[ i ] );
    //}
    DII = K.Diagonal( gids );

    /** collecting column samples of K and DCC */
    std::vector<size_t> column_samples( this->n_centroid_samples );
    for ( size_t j = 0; j < this->n_centroid_samples; j ++ )
    {
      /** just use the first few gids */
      column_samples[ j ] = gids[ j ];
      //DCC[ j ] = K( column_samples[ j ], column_samples[ j ] );
    }
    DCC = K.Diagonal( column_samples );



    //printf( "rank %d enter DCC n = %d\n", rank, n ); fflush( stdout );


    /** collecting KIC */
    auto KIC = K( gids, column_samples );


    //printf( "rank %d after KIC n = %d\n", rank, n ); fflush( stdout );

    /** compute d2c (distance to the approx centroid) for each owned point */
    temp = hmlp::gofmm::AllToCentroid( this->metric, DII, KIC, DCC );

    /** find the f2c (far most to center) from points owned */
    auto itf2c = std::max_element( temp.begin(), temp.end() );
    auto idf2c = std::distance( temp.begin(), itf2c );

    /** create a pair for MPI Allreduce */
    hmlp::mpi::NumberIntPair<T> local_max_pair, max_pair; 
    local_max_pair.val = temp[ idf2c ];
    local_max_pair.key = rank;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** boardcast gidf2c from the MPI process which has the max_pair */
    int gidf2c = gids[ idf2c ];
    hmlp::mpi::Bcast( &gidf2c, 1, MPI_INT, max_pair.key, comm );

    //printf( "rank %d val %E key %d; global val %E key %d\n", 
    //    rank, local_max_pair.val, local_max_pair.key,
    //    max_pair.val, max_pair.key ); fflush( stdout );
    //printf( "rank %d gidf2c %d\n", rank, gidf2c  ); fflush( stdout );

    /** collecting KIP */
    std::vector<size_t> P( 1, gidf2c );
    auto KIP = K( gids, P );


    //printf( "rank %d after KIP n = %d\n", rank, n ); fflush( stdout );



    /** get diagonal entry kpp */
    T kpp = K( gidf2c, gidf2c );

    /** compute the all2f (distance to farthest point) */
    temp = hmlp::gofmm::AllToFarthest( this->metric, DII, KIP, kpp );


    /** find f2f (far most to far most) from owned points */
    auto itf2f = std::max_element( temp.begin(), temp.end() );
    auto idf2f = std::distance( temp.begin(), itf2f );

    /** create a pair for MPI Allreduce */
    local_max_pair.val = temp[ idf2f ];
    local_max_pair.key = rank;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** boardcast gidf2f from the MPI process which has the max_pair */
    int gidf2f = gids[ idf2f ];
    hmlp::mpi::Bcast( &gidf2f, 1, MPI_INT, max_pair.key, comm );

    //printf( "rank %d val %E key %d; global val %E key %d\n", 
    //    rank, local_max_pair.val, local_max_pair.key,
    //    max_pair.val, max_pair.key ); fflush( stdout );
    //printf( "rank %d gidf2f %d\n", rank, gidf2f  ); fflush( stdout );

    /** collecting KIQ */
    std::vector<size_t> Q( 1, gidf2f );
    auto KIQ = K( gids, Q );


    //printf( "rank %d after KIQ n = %d\n", rank, n ); fflush( stdout );


    /** get diagonal entry kpp */
    T kqq = K( gidf2f, gidf2f );

    /** compute all2leftright (projection i.e. dip - diq) */
    temp = hmlp::gofmm::AllToLeftRight( this->metric, DII, KIP, KIQ, kpp, kqq );

    /** parallel median select */
    T  median = hmlp::combinatorics::Select( n / 2, temp, comm );

    //printf( "rank %d median %E\n", rank, median ); fflush( stdout );

    std::vector<size_t> middle;
    middle.reserve( gids.size() );
    split[ 0 ].reserve( gids.size() );
    split[ 1 ].reserve( gids.size() );

    /** TODO: Can be parallelized */
    for ( size_t i = 0; i < gids.size(); i ++ )
    {
      auto val = temp[ i ];

      if ( std::fabs( val - median ) < 1E-6 && !std::isinf( val ) && !std::isnan( val) )
      {
        middle.push_back( i );
      }
      else if ( val < median ) 
      {
        split[ 0 ].push_back( i );
      }
      else
      {
        split[ 1 ].push_back( i );
      }
    }

    int nmid = 0;
    int nlhs = 0;
    int nrhs = 0;
    int num_mid_owned = middle.size();
    int num_lhs_owned = split[ 0 ].size();
    int num_rhs_owned = split[ 1 ].size();

    /** nmid = sum( num_mid_owned ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &num_mid_owned, &nmid, 1, MPI_SUM, comm );
    hmlp::mpi::Allreduce( &num_lhs_owned, &nlhs, 1, MPI_SUM, comm );
    hmlp::mpi::Allreduce( &num_rhs_owned, &nrhs, 1, MPI_SUM, comm );

    //printf( "rank %d [ %d %d %d ] global [ %d %d %d ]\n",
    //    global_rank, num_lhs_owned, num_mid_owned, num_rhs_owned,
    //    nlhs, nmid, nrhs ); fflush( stdout );

    /** assign points in the middle to left or right */
    if ( nmid )
    {
      int nlhs_required, nrhs_required;

			if ( nlhs > nrhs )
			{
        nlhs_required = ( n - 1 ) / 2 + 1 - nlhs;
        nrhs_required = nmid - nlhs_required;
			}
			else
			{
        nrhs_required = ( n - 1 ) / 2 + 1 - nrhs;
        nlhs_required = nmid - nrhs_required;
			}

      assert( nlhs_required >= 0 );
      assert( nrhs_required >= 0 );

      /** now decide the portion */
			double lhs_ratio = ( (double)nlhs_required ) / nmid;
      int nlhs_required_owned = num_mid_owned * lhs_ratio;
      int nrhs_required_owned = num_mid_owned - nlhs_required_owned;


      //printf( "rank %d [ %d %d ] [ %d %d ]\n",
      //  global_rank, 
      //	nlhs_required_owned, nlhs_required,
      //	nrhs_required_owned, nrhs_required ); fflush( stdout );


      assert( nlhs_required_owned >= 0 );
      assert( nrhs_required_owned >= 0 );

      for ( size_t i = 0; i < middle.size(); i ++ )
      {
        if ( i < nlhs_required_owned ) split[ 0 ].push_back( middle[ i ] );
        else                           split[ 1 ].push_back( middle[ i ] );
      }
    };

    return split;
  };


}; /** end struct centersplit */





template<typename SPDMATRIX, int N_SPLIT, typename T> 
struct randomsplit : public gofmm::randomsplit<SPDMATRIX, N_SPLIT, T>
{

  /** shared-memory operator */
  inline std::vector<std::vector<std::size_t> > operator()
  ( 
    std::vector<size_t>& gids,
    std::vector<size_t>& lids
  ) const 
  {
    return gofmm::randomsplit<SPDMATRIX, N_SPLIT, T>::operator()
      ( gids, lids );
  };

  /** distributed operator */
  inline std::vector<std::vector<size_t> > operator()
  (
    std::vector<size_t>& gids,
    mpi::Comm comm
  ) const 
  {
    /** all assertions */
    assert( N_SPLIT == 2 );
    assert( this->Kptr );

    /** declaration */
    int size, rank, global_rank;
    mpi::Comm_size( comm, &size );
    mpi::Comm_rank( comm, &rank );
    mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
    SPDMATRIX &K = *(this->Kptr);
    std::vector<std::vector<size_t>> split( N_SPLIT );


    /** reduce to get the total size of gids */
    int n = 0;
    int num_points_owned = gids.size();
    std::vector<T> temp( gids.size(), 0.0 );

    //printf( "rank %d before Allreduce\n", global_rank ); fflush( stdout );

    /** n = sum( num_points_owned ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &num_points_owned, &n, 1, 
        MPI_INT, MPI_SUM, comm );

    //printf( "rank %d enter distributed randomsplit n = %d\n", global_rank, n ); fflush( stdout );

    /** early return */
    if ( n == 0 ) return split;


    /** randomly select two points p and q */
    size_t idf2c = std::rand() % gids.size();
    size_t idf2f = std::rand() % gids.size();
    while ( idf2c == idf2f ) idf2f = std::rand() % gids.size();

    /** now get their gids */
    auto gidf2c = gids[ idf2c ];
    auto gidf2f = gids[ idf2f ];

    /** Bcast gidf2c and gidf2f from rank-0 to all MPI processes */
    mpi::Bcast( &gidf2c, 1, 0, comm );
    mpi::Bcast( &gidf2f, 1, 0, comm );


    /** collecting DII */
    Data<T> DII = K.Diagonal( gids );

    /** collecting KIP and KIQ */
    std::vector<size_t> P( 1, gidf2c );
    std::vector<size_t> Q( 1, gidf2f );
    auto KIP = K( gids, P );
    auto KIQ = K( gids, Q );

    /** get diagonal entry kpp and kqq */
    auto kpp = K.Diagonal( P );
    auto kqq = K.Diagonal( Q );

    /** compute all2leftright (projection i.e. dip - diq) */
    temp = gofmm::AllToLeftRight( this->metric, DII, KIP, KIQ, kpp[ 0 ], kqq[ 0 ] );













    /** parallel median select */
    T  median = hmlp::combinatorics::Select( n / 2, temp, comm );

    //printf( "rank %d median %E\n", rank, median ); fflush( stdout );

    std::vector<size_t> middle;
    middle.reserve( gids.size() );
    split[ 0 ].reserve( gids.size() );
    split[ 1 ].reserve( gids.size() );

    /** TODO: Can be parallelized */
    for ( size_t i = 0; i < gids.size(); i ++ )
    {
      auto val = temp[ i ];

      if ( std::fabs( val - median ) < 1E-6 && !std::isinf( val ) && !std::isnan( val) )
      {
        middle.push_back( i );
      }
      else if ( val < median ) 
      {
        split[ 0 ].push_back( i );
      }
      else
      {
        split[ 1 ].push_back( i );
      }
    }

    int nmid = 0;
    int nlhs = 0;
    int nrhs = 0;
    int num_mid_owned = middle.size();
    int num_lhs_owned = split[ 0 ].size();
    int num_rhs_owned = split[ 1 ].size();

    /** nmid = sum( num_mid_owned ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &num_mid_owned, &nmid, 1, MPI_SUM, comm );
    hmlp::mpi::Allreduce( &num_lhs_owned, &nlhs, 1, MPI_SUM, comm );
    hmlp::mpi::Allreduce( &num_rhs_owned, &nrhs, 1, MPI_SUM, comm );

    //printf( "rank %d [ %d %d %d ] global [ %d %d %d ]\n",
    //    global_rank, num_lhs_owned, num_mid_owned, num_rhs_owned,
    //    nlhs, nmid, nrhs ); fflush( stdout );

    /** assign points in the middle to left or right */
    if ( nmid )
    {
      int nlhs_required, nrhs_required;

			if ( nlhs > nrhs )
			{
        nlhs_required = ( n - 1 ) / 2 + 1 - nlhs;
        nrhs_required = nmid - nlhs_required;
			}
			else
			{
        nrhs_required = ( n - 1 ) / 2 + 1 - nrhs;
        nlhs_required = nmid - nrhs_required;
			}

      assert( nlhs_required >= 0 );
      assert( nrhs_required >= 0 );

      /** now decide the portion */
			double lhs_ratio = ( (double)nlhs_required ) / nmid;
      int nlhs_required_owned = num_mid_owned * lhs_ratio;
      int nrhs_required_owned = num_mid_owned - nlhs_required_owned;


      //printf( "rank %d [ %d %d ] [ %d %d ]\n",
      //  global_rank, 
			//	nlhs_required_owned, nlhs_required,
			//	nrhs_required_owned, nrhs_required ); fflush( stdout );


      assert( nlhs_required_owned >= 0 );
      assert( nrhs_required_owned >= 0 );

      for ( size_t i = 0; i < middle.size(); i ++ )
      {
        if ( i < nlhs_required_owned ) split[ 0 ].push_back( middle[ i ] );
        else                           split[ 1 ].push_back( middle[ i ] );
      }
    };

    return split;
  };


}; /** end struct randomsplit */












/**
 *  @brief TODO: (Severin)
 *
 */ 
template<typename NODE, typename T>
void FindNeighbors( NODE *node, DistanceMetric metric )
{
  /** in distributed environment, it has type DistData<STAR, CIDS, T> */
  auto &NN   = *(node->setup->NN);
  auto &gids = node->gids;

  /** KII = K( gids, gids ) */
  Data<T> KII;

  /** rho+k nearest neighbors */
  switch ( metric )
  {
    case GEOMETRY_DISTANCE:
    {
      auto &X = *(node->setup->X);
      printf( "GEMETRY DISTNACE not implemented yet\n" );
      break;
    }
    case KERNEL_DISTANCE:
    {
      auto &K = *(node->setup->K);
      KII = K( gids, gids );
      
      /** get digaonal entries */
      Data<T> DII( gids.size(), (size_t)1 );
      for ( size_t i = 0; i < gids.size(); i ++ ) DII[ i ] = KII( i, i );

      for ( size_t j = 0; j < KII.col(); j ++ )
        for ( size_t i = 0; i < KII.row(); i ++ )
          KII( i, j ) = DII[ i ] + DII[ j ] - 2.0 * KII( i, j );


      break;
    }
    case ANGLE_DISTANCE:
    {
      auto &K = *(node->setup->K);
      KII = K( gids, gids );

      /** get digaonal entries */
      Data<T> DII( gids.size(), (size_t)1 );
      for ( size_t i = 0; i < gids.size(); i ++ ) DII[ i ] = KII( i, i );

      for ( size_t j = 0; j < KII.col(); j ++ )
        for ( size_t i = 0; i < KII.row(); i ++ )
          KII( i, j ) = 1.0 - ( KII( i, j ) * KII( i, j ) ) / ( DII[ i ] * DII[ j ] );

      break;
    }
    default:
    {
      exit( 1 );
    }
  }


  for ( size_t j = 0; j < KII.col(); j ++ )
  { 
    /** create a query list for column KII( :, j ) */
    std::vector<std::pair<T, size_t>> query( KII.row() );
    for ( size_t i = 0; i < KII.row(); i ++ ) 
    {
      query[ i ].first  = KII( i, j );
      query[ i ].second = gids[ i ];
    }

    /** sort the query according to distances */
    sort( query.begin(), query.end() );

    /** fill-in the neighbor list */
    auto *NNj = NN.columndata( gids[ j ] );
    for ( size_t i = 0; i < NN.row(); i ++ ) 
    {
      NNj[ i ] = query[ i ];
    }
  }

}; /** end FindNeighbors() */














template<class NODE, typename T>
class NeighborsTask : public hmlp::Task
{
  public:

    NODE *arg;
   
	  /** (default) using angle distance from the Gram vector space */
	  DistanceMetric metric = ANGLE_DISTANCE;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "Neighbors" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();
      // TODO: Need an accurate cost model.
      cost = 1.0;

      /** use the same distance as the tree */
      metric = arg->setup->metric;


      //--------------------------------------
      double flops, mops;
      auto &gids = arg->gids;
      auto &NN = *arg->setup->NN;
      flops = gids.size();
      flops *= 4.0 * gids.size();
      // Heap select worst case
      mops = (size_t)std::log( NN.row() ) * gids.size();
      mops *= gids.size();
      // Access K
      mops += flops;
      event.Set( name + label, flops, mops );
      //--------------------------------------
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      this->TryEnqueue();
    }

    void Execute( Worker* user_worker )
    {
      FindNeighbors<NODE, T>( arg, metric );
    };

}; /** end class NeighborsTask */


















template<typename NODE>
void MergeNeighbors( NODE *node )
{
  /** merge with the existing neighbor list and remove duplication */
};











/**
 *  @brief Compute skeleton weights.
 */
template<typename MPINODE, typename T>
void DistUpdateWeights( MPINODE *node )
{
  /** MPI */
  hmlp::mpi::Status status;
  int global_rank = 0; 
  hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
  hmlp::mpi::Comm comm = node->GetComm();
  int size = node->GetCommSize();
  int rank = node->GetCommRank();

  if ( size < 2 )
  {
    /** this is the root of the local tree */
    gofmm::UpdateWeights<MPINODE, T>( node );
  }
  else
  {
    /** get the child node  */
    auto *child = node->child;
    auto *childFarNodes = &(child->FarNodes);
    auto *child_sibling = (*childFarNodes->begin());

    /** gather shared data and create reference */
    auto &w = *node->setup->w;
    size_t nrhs = w.col();

    /** gather per node data and create reference */
    auto &data = node->data;

    /** this is the corresponding MPI rank */
    if ( rank == 0 )
    {
      auto &proj = data.proj;
      auto &skels = data.skels;
      auto &w_skel = data.w_skel;
      auto &w_lskel = child->data.w_skel;
      auto &w_rskel = child_sibling->data.w_skel;
     
      /** recv child->w_skel from rank / 2 */
      hmlp::mpi::ExchangeVector(
            w_lskel, size / 2, 0, 
            w_rskel, size / 2, 0, comm, &status );

      size_t sl = w_lskel.size() / nrhs;
      size_t sr = w_rskel.size() / nrhs;

      /** resize w_rskel to properly set m and n */
      w_rskel.resize( sr, nrhs );

      //printf( "global rank %d child_sibling morton %lu w_rskel.size() %lu, m %lu n %lu\n", 
      //    global_rank,
      //    child_sibling->morton,
      //    w_rskel.size(), w_rskel.row(), w_rskel.col() );

      /** early return */
      if ( !node->parent || !node->data.isskel ) return;
     

      /** w_skel is s-by-nrhs, initial values are not important */
      w_skel.resize( skels.size(), nrhs );

      if ( 0 )
      {
        /** w_skel = proj( l ) * w_lskel */
        xgemm
        (
          "No-transpose", "No-transpose",
          w_skel.row(), nrhs, sl,
          1.0,    proj.data(),    proj.row(),
               w_lskel.data(),            sl,
          0.0,  w_skel.data(),  w_skel.row()
        );
        /** w_skel += proj( r ) * w_rskel */
        xgemm
        (
          "No-transpose", "No-transpose",
          w_skel.row(), nrhs, sr,
          1.0,    proj.data() + proj.row() * sl, proj.row(),
               w_rskel.data(), sr,
          1.0,  w_skel.data(), w_skel.row()
        );
      }
      else
      {
        /** create a transpose view proj_v */
        View<T> P( false,   proj ), PL,
                                    PR;
        View<T> W( false, w_skel ), WL( false, w_lskel ),
                                    WR( false, w_rskel );
        /** P = [ PL, PR ] */
        P.Partition1x2( PL, PR, sl, LEFT );
        /** W  = PL * WL */
        gemm::xgemm( (T)1.0, PL, WL, (T)0.0, W );
        /** W += PR * WR */
        gemm::xgemm( (T)1.0, PR, WR, (T)1.0, W );
      }
    }

    /** the rank that holds the skeleton weight of the right child */
    if ( rank == size / 2 )
    {
      auto &w_lskel = child_sibling->data.w_skel;
      auto &w_rskel = child->data.w_skel;
      /** send child->w_skel to 0 */
      //hmlp::mpi::SendVector( w_rskel, 0, 0, comm );    
      hmlp::mpi::ExchangeVector(
            w_rskel, 0, 0, 
            w_lskel, 0, 0, comm, &status );
      /** resize l_rskel to properly set m and n */
      w_lskel.resize( w_lskel.size() / nrhs, nrhs );
      //printf( "global rank %d child_sibling morton %lu w_lskel.size() %lu, m %lu n %lu\n", 
      //    global_rank,
      //    child_sibling->morton,
      //    w_lskel.size(), w_lskel.row(), w_lskel.col() );
    }
  }

}; /** end DistUpdateWeights() */




/**
 *  @brief Notice that NODE here is MPITree::Node.
 */ 
template<typename NODE, typename T>
class DistUpdateWeightsTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = std::string( "DistN2S" );
      {
        //label = std::to_string( arg->treelist_id );
        std::ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      /** compute flops and mops */
      double flops = 0.0;
      double mops = 0.0;

      /** setup the event */
      event.Set( label + name, flops, mops );

      /** assume computation bound */
      cost = flops / 1E+9;

      /** "HIGH" priority */
      priority = true;
    };



    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() < 2 )
        {
          arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
          arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        }
        else
        {
          arg->child->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        }
      }

      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };


    void Execute( Worker* user_worker )
    {
      //int global_rank = 0;
      //hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

      //printf( "rank %d level %lu DistUpdateWeights\n", 
      //    global_rank, arg->l ); fflush( stdout );

      mpigofmm::DistUpdateWeights<NODE, T>( arg );

      //printf( "rank %d level %lu DistUpdateWeights end\n", 
      //    global_rank, arg->l ); fflush( stdout );
    };


}; /** end class DistUpdateWeightsTask */




/**
 *
 */ 
template<bool NNPRUNE, typename NODE>
class DistSkeletonsToSkeletonsTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = std::string( "DistS2S" );
      {
        //label = std::to_string( arg->treelist_id );
        std::ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      /** compute flops and mops */
      double flops = 0.0, mops = 0.0;
      auto &w = *arg->setup->w;
      size_t m = arg->data.skels.size();
      size_t n = w.col();

      auto *FarNodes = &arg->FarNodes;
      if ( NNPRUNE ) FarNodes = &arg->NNFarNodes;

      for ( auto it = FarNodes->begin(); it != FarNodes->end(); it ++ )
      {
        size_t k = (*it)->data.skels.size();
        flops += 2.0 * m * n * k;
        mops  += m * k; // cost of Kab
        mops  += 2.0 * ( m * n + n * k + k * n );
      }

      /** setup the event */
      event.Set( label + name, flops, mops );

      /** assume computation bound */
      cost = flops / 1E+9;

      /** "LOW" priority */
      priority = false;
    };



    void DependencyAnalysis()
    {
      //auto *FarNodes = &arg->FarNodes;
      //if ( NNPRUNE ) FarNodes = &arg->NNFarNodes;


      ///** only write to this node while there is Far interaction */
      //if ( FarNodes->size() )
      //  arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      ///** depends on all the far interactions, waiting for skeleton weight */
      //for ( auto it = FarNodes->begin(); it != FarNodes->end(); it ++ )
      //{
      //  (*it)->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      //}



      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      this->TryEnqueue();
    };



    /**
     *  @brief Notice that S2S depends on all Far interactions, which
     *         may include local tree nodes or let nodes. 
     *         For HSS case, the only Far interaction is the sibling.
     *         Skeleton weight of the sibling will always be exchanged
     *         by default in N2S. Thus, currently we do not need 
     *         a distributed S2S, because the skeleton weight is already
     *         in place.
     *
     */ 
    void Execute( Worker* user_worker )
    {
      auto *node = arg;
      /** MPI */
      int size = node->GetCommSize();
      int rank = node->GetCommRank();
      hmlp::mpi::Comm comm = node->GetComm();

      if ( size < 2 )
      {
        hmlp::gofmm::SkeletonsToSkeletons<NNPRUNE>( node );
      }
      else
      {
        /** only 0th rank (owner) will execute this task */
        if ( rank == 0 )
        {
          hmlp::gofmm::SkeletonsToSkeletons<NNPRUNE>( node );
        }
      }
    };

}; /** end class DistSkeletonsToSkeletonsTask */



/**
 *  @brief
 */ 
template<bool NNPRUNE, typename NODE, typename T>
void DistSkeletonsToNodes( NODE *node )
{
  /** MPI */
  int size = node->GetCommSize();
  int rank = node->GetCommRank();
  hmlp::mpi::Comm comm = node->GetComm();
	int global_rank;
	mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

  /** gather shared data and create reference */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;

  /** Gather per node data and create reference */
  auto &lids = node->lids;
  auto &data = node->data;
  auto &proj = data.proj;
  auto &skels = data.skels;
  auto &u_skel = data.u_skel;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  size_t nrhs = w.col();

  if ( size < 2 )
  {
    /** call the shared-memory implementation */
    gofmm::SkeletonsToNodes<NNPRUNE, NODE, T>( node );
  }
  else
  {
    /** early return */
    if ( !node->parent || !node->data.isskel ) return;

    /** get the child node  */
    auto *child = node->child;
    auto *childFarNodes = &(child->FarNodes);
    auto *child_sibling = (*childFarNodes->begin());
    hmlp::mpi::Status status;

		
    //printf( "rank (%d,%d) level %lu DistS2N check1\n", 
		//		rank, global_rank, node->l ); fflush( stdout );
    if ( rank == 0 )
    {
      auto    &proj = data.proj;
      auto  &u_skel = data.u_skel;
      auto &u_lskel = child->data.u_skel;
      auto &u_rskel = child_sibling->data.u_skel;
   
      size_t sl = u_lskel.row();
      size_t sr = proj.col() - sl;

      /** resize */
      u_rskel.resize( sr, nrhs, 0.0 );

      if ( 0 )
      {
        xgemm
        (
          "T", "N",
          u_lskel.row(), u_lskel.col(), proj.row(),
          1.0, proj.data(),    proj.row(),
               u_skel.data(),  u_skel.row(),
          1.0, u_lskel.data(), u_lskel.row()
        );

        xgemm
        (
          "T", "N",
          u_rskel.row(), u_rskel.col(), proj.row(),
          1.0, proj.data() + proj.row() * sl, proj.row(),
               u_skel.data(), u_skel.row(),
          1.0, u_rskel.data(), u_rskel.row()
        );
      }
      else
      {
        /** create a transpose view proj_v */
        View<T> P(  true,   proj ), PL,
                                    PR;
        View<T> U( false, u_skel ), UL( false, u_lskel ),
                                    UR( false, u_rskel );
        /** P' = [ PL, PR ]' */
        P.Partition2x1( PL,
                        PR, sl, TOP );
        T alpha = 1.0;
        T beta  = 1.0;
        /** UL += PL' * U */
        gemm::xgemm( alpha, PL, U, beta, UL );
        /** UR += PR' * U */
        gemm::xgemm( alpha, PR, U, beta, UR );
      }

      hmlp::mpi::SendVector( u_rskel, size / 2, 0, comm );
    }
    //printf( "rank (%d,%d) level %lu DistS2N check2\n", 
		//		rank, global_rank, node->l ); fflush( stdout );

    /**  */
    if ( rank == size / 2 )
    {
      hmlp::Data<T> u_rskel;
      hmlp::mpi::RecvVector( u_rskel, 0, 0, comm, &status );
      /** resize to properly set m and n */
      u_rskel.resize( u_rskel.size() / nrhs, nrhs );

      for ( size_t j = 0; j < u_rskel.col(); j ++ )
        for ( size_t i = 0; i < u_rskel.row(); i ++ )
          child->data.u_skel( i, j ) += u_rskel( i, j );
    }

    //printf( "rank (%d,%d) level %lu DistS2N check3\n", 
		//		rank, global_rank, node->l ); fflush( stdout );
  }

}; /** end DistSkeletonsToNodes() */





template<bool NNPRUNE, typename NODE, typename T>
class DistSkeletonsToNodesTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = std::string( "DistS2N" );
      {
        //label = std::to_string( arg->treelist_id );
        std::ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      double flops = 0.0;
      double mops = 0.0;

      /** setup the event */
      event.Set( label + name, flops, mops );

      /** asuume computation bound */
      cost = flops / 1E+9;

      /** "HIGH" priority */
      priority = true;
    };


    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() > 1 )
        {
          arg->child->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        }
        else
        {
          arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
          arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        }
      }
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
	    int global_rank;
	    mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
      //printf( "rank %d level %lu DistS2N begin\n", global_rank, arg->l ); fflush( stdout );
      DistSkeletonsToNodes<NNPRUNE, NODE, T>( arg );
      //printf( "rank %d level %lu DistS2N   end\n", global_rank, arg->l ); fflush( stdout );
    };

}; /** end class DistSkeletonsToNodesTask */





template<bool NNPRUNE, typename NODE, typename T>
void DistLeavesToLeaves( NODE *node )
{
  assert( node->isleaf );

  /** gather shared data and create reference */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;

  auto &gids = node->gids;
  auto &data = node->data;
  auto &amap = node->gids;
  auto &NearKab = data.NearKab;
  size_t nrhs = w.col();

  auto *NearNodes = &node->NearNodes;
  if ( NNPRUNE ) NearNodes = &node->NNNearNodes;

  /** use a temp buffer */
  auto &u_leaf = data.u_leaf[ 1 ];
  u_leaf.resize( 0, 0 );
  u_leaf.resize( gids.size(), nrhs, 0.0 );

  if ( NearKab.size() ) /** Kab is cached */
  {
    size_t offset = 0;
    for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
    {
      auto wb = (*it)->data.w_leaf;
      /** Kab * wb */
      xgemm
      (
        "N", "N",
        u_leaf.row(), u_leaf.col(), wb.row(),
        1.0, NearKab.data() + offset * NearKab.row(), NearKab.row(),
                  wb.data(),                               wb.row(),
        1.0,  u_leaf.data(),                           u_leaf.row()
      );
      offset += (*it)->gids.size();
    }
  }
  else
  {
    printf( "BUG, NearKab must be cached in the distributed case\n" );
    exit( 1 );
  }

}; /** end LeavesToLeaves() */




template<bool NNPRUNE, typename NODE, typename T>
class DistLeavesToLeavesTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = std::string( "DistL2L" );
      {
        //label = std::to_string( arg->treelist_id );
        std::ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      double flops = 0.0;
      double mops = 0.0;

      /** setup the event */
      event.Set( label + name, flops, mops );

      /** asuume computation bound */
      cost = flops / 1E+9;

      /** "LOW" priority */
      priority = false;
    };


    void DependencyAnalysis()
    {
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      if ( arg->isleaf )
        DistLeavesToLeaves<NNPRUNE, NODE, T>( arg );
    };

}; /** end class DistSkeletonsToNodesTask */

















/**
 *  @brief (FMM specific) Compute Near( leaf nodes ). This is just like
 *         the neighbor list but the granularity is in nodes but not points.
 *         The algorithm is to compute the node morton ids of neighbor points.
 *         Get the pointers of these nodes and insert them into a std::set.
 *         std::set will automatic remove duplication. Here the insertion 
 *         will be performed twice each time to get a symmetric one. That is
 *         if alpha has beta in its list, then beta will also have alpha in
 *         its list.
 *
 *         Only leaf nodes will have the list `` NearNodes''.
 *
 *         This list will later be used to get the FarNodes using a recursive
 *         node traversal scheme.
 *  
 */ 
template<bool SYMMETRIC, typename TREE>
void FindNearNodes( TREE &tree, double budget )
{
  auto &setup = tree.setup;
  auto &NN = *setup.NN;
  size_t n_nodes = ( 1 << tree.depth );
  /** 
   *  The type here is tree::Node but not mpitree::Node.
   *  NearNodes and NNNearNodes also take tree::Node.
   *  This is ok, because they will only contain leaf nodes,
   *  which will never be distributed.
   *  However, FarNodes and NNFarNodes may contain distributed
   *  tree nodes. In this case, we have to do type casting.
   */
  auto level_beg = tree.treelist.begin() + n_nodes - 1;


  /** 
   * traverse all leaf nodes. 
   *
   * TODO: the following omp parallel for will result in segfault on KNL.
   *       Not sure why.
   **/
  for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
  {
    auto *node = *(level_beg + node_ind);
    auto &data = node->data;

    /** if no skeletons, then add every leaf nodes */
    /** TODO: this is not affected by the BUDGET */
    if ( !node->data.isskel )
    {
      for ( size_t i = 0; i < n_nodes; i ++ )
      {
        node->NearNodes.insert(   *(level_beg + i) );
        node->NNNearNodes.insert( *(level_beg + i) );
        node->NearNodeMortonIDs.insert(   (*(level_beg + i))->morton );
        node->NNNearNodeMortonIDs.insert( (*(level_beg + i))->morton );
      }
    }

    //printf( "insert NearNodes (self)\n" ); fflush( stdout );

    /** add myself to the list. */
    node->NearNodes.insert( node );
    node->NNNearNodes.insert( node );
    node->NearNodeMortonIDs.insert( node->morton );
    node->NNNearNodeMortonIDs.insert( node->morton );
 
    /** TODO: ballot table */
    std::vector<std::pair<size_t, size_t>> ballot( n_nodes );
    for ( size_t i = 0; i < n_nodes; i ++ )
    {
      ballot[ i ].first  = 0;
      ballot[ i ].second = i;
    }


    /** 
     *  TODO: traverse all points and their neighbors. NN is stored as k-by-N 
     *       
     *  LET required     
     *
     */


    /** sort the ballot list */
    struct 
    {
      bool operator () ( std::pair<size_t, size_t> a, std::pair<size_t, size_t> b )
      {   
        return a.first > b.first;
      }   
    } BallotMore;


  } /** end for each leaf owned leaf node in the local tree */


  /** TODO: symmetrinize Near( node ) */
  if ( SYMMETRIC )
  {
    /** TODO: LET required */
  }

}; /** end FindNearNodes() */



/**
 *  @brief Task wrapper for FindNearNodes()
 */ 
template<bool SYMMETRIC, typename TREE>
class FindNearNodesTask : public hmlp::Task
{
  public:

    TREE *arg;

    double budget = 0.0;

    void Set( TREE *user_arg, double user_budget )
    {
      arg = user_arg;
      budget = user_budget;
      name = std::string( "near" );

      //--------------------------------------
      double flops = 0.0, mops = 0.0;

      /** setup the event */
      event.Set( label + name, flops, mops );

      /** asuume computation bound */
      cost = 1.0;

      /** low priority */
      priority = true;
    }

    /** depends on all leaf nodes in the local tree */
    void DependencyAnalysis()
    {
      TREE &tree = *arg;
      size_t n_nodes = 1 << tree.depth;
      auto level_beg = tree.treelist.begin() + n_nodes - 1;
      for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
      {
        auto *node = *(level_beg + node_ind);
        node->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      }

      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      //printf( "FindNearNode beg\n" ); fflush( stdout );
      hmlp::mpigofmm::FindNearNodes<SYMMETRIC, TREE>( *arg, budget );
      //printf( "FindNearNode end\n" ); fflush( stdout );
    };

}; /** end class FindNearNodesTask */


/**
 *  @brief (FMM specific) find Far( target ) by traversing all treenodes 
 *         top-down. 
 *         If the visiting ``node'' does not contain any near node
 *         of ``target'' (by MORTON helper function ContainAny() ),
 *         then we add ``node'' to Far( target ).
 *
 *         Otherwise, recurse to two children.
 *
 *         Notice that here target always has type tree::Node, but
 *         node may have different types (e.g. mpitree::Node)
 *         depending on which portion of the tree is visited.
 */ 
template<bool SYMMETRIC, typename NODE, size_t LEVELOFFSET = 4>
void FindFarNodes(
  size_t level, size_t max_depth,
  size_t recurs_morton, NODE *target )
{
  size_t shift = ( 1 << LEVELOFFSET ) - level + LEVELOFFSET;
  size_t source_morton = ( recurs_morton << shift ) + level;

  if ( hmlp::tree::ContainAnyMortonID( target->NearNodeMortonIDs, source_morton ) )
  {
    if ( level < max_depth )
    {
      /** recurse to two children */
      FindFarNodes<SYMMETRIC>( 
          level + 1, max_depth,
          ( recurs_morton << 1 ) + 0, target );
      FindFarNodes( 
          level + 1, max_depth,
          ( recurs_morton << 1 ) + 1, target );
    }
  }
  else
  {
    /** create LETNode, insert ``source'' to Far( target ) */
    //auto *letnode = new hmlp::mpitree::LETNode( level, source_morton );
    target->FarNodeMortonIDs.insert( source_morton );
  }


  /**
   *  case: NNPRUNE
   *
   *  Build NNNearNodes for the FMM approximation.
   *  Near( target ) contains other leaf nodes
   *  
   **/
  if ( hmlp::tree::ContainAnyMortonID( target->NNNearNodeMortonIDs, source_morton ) )
  {
    if ( level < max_depth )
    {
      /** recurse to two children */
      FindFarNodes<SYMMETRIC>( 
          level + 1, max_depth,
          ( recurs_morton << 1 ) + 0, target );
      FindFarNodes( 
          level + 1, max_depth,
          ( recurs_morton << 1 ) + 1, target );
    }
  }
  else
  {
    if ( SYMMETRIC && ( source_morton < target->morton ) )
    {
      /** since target->morton is larger than the visiting node,
       * the interaction between the target and this node has
       * been computed. 
       */ 
    }
    else
    {
      /** create LETNode, insert ``source'' to Far( target ) */
      //auto *letnode = new hmlp::mpitree::LETNode( level, source_morton );
      target->NNFarNodeMortonIDs.insert( source_morton );
    }
  }

}; /** end FindFarNodes() */



template<bool SYMMETRIC, typename NODE>
void MergeFarNodes( NODE *node )
{
  /** if I don't have any skeleton, then I'm nobody's far field */
  if ( !node->data.isskel ) return;

  if ( node->isleaf )
  {
    FindFarNodes<SYMMETRIC>( 0, /** max_depth */, 0, node  );
  }
  else
  {
    /** merge Far( lchild ) and Far( rchild ) from children */
    auto *lchild = node->lchild;
    auto *rchild = node->rchild;

    /** case: !NNPRUNE (HSS specific) */ 
    auto &pFarNodes =   node->FarNodeMortonIDs;
    auto &lFarNodes = lchild->FarNodeMortonIDs;
    auto &rFarNodes = rchild->FarNodeMortonIDs;

    /** Far( parent ) = Far( lchild ) intersects Far( rchild ) */
    for ( auto it = lFarNodes.begin(); it != lFarNodes.end(); ++ it )
    {
      if ( rFarNodes.count( *it ) ) pFarNodes.insert( *it );
    }
    /** Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) */
    for ( auto it = pFarNodes.begin(); it != pFarNodes.end(); it ++ )
    {
      lFarNodes.erase( *it ); rFarNodes.erase( *it );
    }

    /** case: NNPRUNE (FMM specific) */ 
    auto &pNNFarNodes =   node->NNFarNodeMortonIDs;
    auto &lNNFarNodes = lchild->NNFarNodeMortonIDs;
    auto &rNNFarNodes = rchild->NNFarNodeMortonIDs;

    /** Far( parent ) = Far( lchild ) intersects Far( rchild ) */
    for ( auto it = lNNFarNodes.begin(); it != lNNFarNodes.end(); ++ it )
    {
      if ( rNNFarNodes.count( *it ) ) pNNFarNodes.insert( *it );
    }
    /** Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) */
    for ( auto it = pNNFarNodes.begin(); it != pNNFarNodes.end(); it ++ )
    {
      lNNFarNodes.erase( *it ); 
      rNNFarNodes.erase( *it );
    }
  }

  /** TODO: deal with symmetricness for NN case*/
  if ( SYMMETRIC )
  {
  }

}; /** end MergeFarNodes() */




template<bool SYMMETRIC, typename MPINODE>
void ParallelMergeFarNodes( MPINODE *node )
{
  /** if I don't have any skeleton, then I'm nobody's far field */
  if ( !node->data.isskel ) return;

  /** distributed treenode */
  if ( node->GetCommSize() < 2 )
  {
    MergeFarNodes( node );
  }
  else
  {
    /** merge Far( lchild ) and Far( rchild ) from children */
    auto *child = node->child;

    /** recv  */

  }


}; /** end ParallelMergeFarNodes() */




/**
 *
 *
 */ 
template<typename LETNODE, typename NODE>
void SimpleFarNodes( NODE *node )
{
  if ( node->isleaf )
	{
    /** add myself to the list. */
    node->NearNodes.insert( node );
    node->NNNearNodes.insert( node );
    node->NearNodeMortonIDs.insert( node->morton );
    node->NNNearNodeMortonIDs.insert( node->morton );
	}
	else
  {
    auto *lchild = node->lchild;
    auto *rchild = node->rchild;

    lchild->FarNodes.insert( rchild );
    lchild->NNFarNodes.insert( rchild );
    rchild->FarNodes.insert( lchild );
    rchild->NNFarNodes.insert( lchild );

    SimpleFarNodes<LETNODE>( lchild );
    SimpleFarNodes<LETNODE>( rchild );
  }

  //printf( "%lu NNFarNodes: ", node->treelist_id ); 
  //for ( auto it  = node->NNFarNodes.begin();
  //           it != node->NNFarNodes.end(); it ++ )
  //{
  //  printf( "%lu ", (*it)->morton ); fflush( stdout );
  //};
  //printf( "%lu FarNodes: ", node->treelist_id ); 
  //for ( auto it  = node->FarNodes.begin();
  //           it != node->FarNodes.end(); it ++ )
  //{
  //  printf( "%lu ", (*it)->morton ); fflush( stdout );
  //};
  //printf( "\n" );
  
}; /** end SimpleFarNodes() */



/**
 *  TODO: lettreelist is not thread safe.
 *
 */ 
template<typename LETNODE, typename MPINODE>
void ParallelSimpleFarNodes( MPINODE *node, 
    std::map<size_t, LETNODE*> &lettreelist )
{
  //printf( "level %lu beg\n", node->l ); fflush( stdout );
  //hmlp::mpi::Barrier( MPI_COMM_WORLD );

  hmlp::mpi::Status status;
  hmlp::mpi::Comm comm = node->GetComm();
  int size = node->GetCommSize();
  int rank = node->GetCommRank();


  //if ( rank == 0 )
  //{
  //  printf( "level %lu here\n", node->l ); fflush( stdout );
  //}
  //hmlp::mpi::Barrier( MPI_COMM_WORLD );


  if ( size < 2 )
  {
    /** do not create any LET node */
    SimpleFarNodes<LETNODE>( node );
  }
  else
  {
    auto *child = node->child;
    size_t send_morton = child->morton;
    size_t recv_morton = 0;
    LETNODE *letnode = NULL;

    if ( rank == 0 )
    {
      mpi::Sendrecv( 
          &send_morton, 1, size / 2, 0, 
          &recv_morton, 1, size / 2, 0, comm, &status );

      /** initialize a LetNode with sibling's MortonID */
      letnode = new LETNODE( node->setup, recv_morton );
      child->FarNodes.insert( letnode );
      child->NNFarNodes.insert( letnode );
      lettreelist[ recv_morton ] = letnode;

      if ( child->isleaf )
      {
        /** exchange gids */
        auto &send_gids = child->gids;
        auto &recv_gids = letnode->gids;
        mpi::ExchangeVector(
            send_gids, size / 2, 0, 
            recv_gids, size / 2, 0, comm, &status );
        //printf( "0 exchange gids\n" ); fflush( stdout );
      }

      /** exchange skels */
      auto &send_skels = child->data.skels;
      auto &recv_skels = letnode->data.skels;
      mpi::ExchangeVector(
          send_skels, size / 2, 0, 
          recv_skels, size / 2, 0, comm, &status );
      /** exchange interpolative coefficients */
      auto &send_proj = child->data.proj;
      auto &recv_proj = letnode->data.proj;
      mpi::ExchangeVector(
          send_proj, size / 2, 0, 
          recv_proj, size / 2, 0, comm, &status );
      //printf( "0 exchange skels and proj\n" ); fflush( stdout );
    }

    if ( rank == size / 2 )
    {
      mpi::Sendrecv( 
          &send_morton, 1, 0, 0, 
          &recv_morton, 1, 0, 0, comm, &status );

      /** initialize a LetNode with sibling's MortonID */
      letnode = new LETNODE( node->setup, recv_morton );
      child->FarNodes.insert( letnode );
      child->NNFarNodes.insert( letnode );
      lettreelist[ recv_morton ] = letnode;

      if ( child->isleaf )
      {
        /** exchange gids */
        auto &send_gids = child->gids;
        auto &recv_gids = letnode->gids;
        mpi::ExchangeVector(
            send_gids, 0, 0, 
            recv_gids, 0, 0, comm, &status );
        //printf( "size / 2 exchange gids\n" ); fflush( stdout );
      }

      /** exchange skels */
      auto &send_skels = child->data.skels;
      auto &recv_skels = letnode->data.skels;
      mpi::ExchangeVector(
          send_skels, 0, 0, 
          recv_skels, 0, 0, comm, &status );
      /** exchange interpolative coefficients */
      auto &send_proj = child->data.proj;
      auto &recv_proj = letnode->data.proj;
      mpi::ExchangeVector(
          send_proj, 0, 0, 
          recv_proj, 0, 0, comm, &status );
      //printf( "size / 2 exchange skels and proj\n" ); fflush( stdout );
    }


    ParallelSimpleFarNodes<LETNODE>( child, lettreelist );
  }

}; /** end ParallelSimpleFarNodes() */





template<typename NODE>
class SimpleNearFarNodesTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      /** this task contains MPI routines */
      //this->has_mpi_routines = true;

      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "cachef" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
			auto *node = arg;
			if ( node->isleaf )
			{
				/** add myself to the list. */
				node->NearNodes.insert( node );
				node->NNNearNodes.insert( node );
				node->NearNodeMortonIDs.insert( node->morton );
				node->NNNearNodeMortonIDs.insert( node->morton );
			}
			else
			{
				auto *lchild = node->lchild;
				auto *rchild = node->rchild;
				lchild->FarNodes.insert( rchild );
				lchild->NNFarNodes.insert( rchild );
				rchild->FarNodes.insert( lchild );
				rchild->NNFarNodes.insert( lchild );
			}
		};

}; /** end class SimpleNearFarNodesTask */



template<typename LETNODE, typename NODE>
class DistSimpleNearFarNodesTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      /** this task contains MPI routines */
      //this->has_mpi_routines = true;

      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "cachef" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    };

		/** read this node and write to children */
    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() > 1 )
        {
          arg->child->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        }
        else
        {
          arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
          arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        }
      }
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
		{
			auto *node = arg;
			mpi::Status status;
			mpi::Comm comm = node->GetComm();
			int size = node->GetCommSize();
			int rank = node->GetCommRank();

			if ( size < 2 )
			{
			  if ( node->isleaf )
			  {
			  	/** add myself to the list. */
			  	node->NearNodes.insert( node );
			  	node->NNNearNodes.insert( node );
			  	node->NearNodeMortonIDs.insert( node->morton );
			  	node->NNNearNodeMortonIDs.insert( node->morton );
			  }
			  else
			  {
				  auto *lchild = node->lchild;
				  auto *rchild = node->rchild;
			  	lchild->FarNodes.insert( rchild );
			  	lchild->NNFarNodes.insert( rchild );
			  	rchild->FarNodes.insert( lchild );
			  	rchild->NNFarNodes.insert( lchild );
			  }
			}
			else
			{
				auto *child = node->child;
				size_t send_morton = child->morton;
				size_t recv_morton = 0;
				LETNODE *letnode = NULL;

				if ( rank == 0 )
				{
					mpi::Sendrecv( 
							&send_morton, 1, size / 2, 0, 
							&recv_morton, 1, size / 2, 0, comm, &status );

					/** initialize a LetNode with sibling's MortonID */
					letnode = new LETNODE( node->setup, recv_morton );
					child->FarNodes.insert( letnode );
					child->NNFarNodes.insert( letnode );

					if ( child->isleaf )
					{
						/** exchange gids */
						auto &send_gids = child->gids;
						auto &recv_gids = letnode->gids;
						mpi::ExchangeVector(
								send_gids, size / 2, 0, 
								recv_gids, size / 2, 0, comm, &status );
						//printf( "0 exchange gids\n" ); fflush( stdout );
					}

					/** exchange skels */
					auto &send_skels = child->data.skels;
					auto &recv_skels = letnode->data.skels;
					mpi::ExchangeVector(
							send_skels, size / 2, 0, 
							recv_skels, size / 2, 0, comm, &status );
					/** exchange interpolative coefficients */
					auto &send_proj = child->data.proj;
					auto &recv_proj = letnode->data.proj;
					mpi::ExchangeVector(
							send_proj, size / 2, 0, 
							recv_proj, size / 2, 0, comm, &status );
				}

				if ( rank == size / 2 )
				{
					mpi::Sendrecv( 
							&send_morton, 1, 0, 0, 
							&recv_morton, 1, 0, 0, comm, &status );

					/** initialize a LetNode with sibling's MortonID */
					letnode = new LETNODE( node->setup, recv_morton );
					child->FarNodes.insert( letnode );
					child->NNFarNodes.insert( letnode );

					if ( child->isleaf )
					{
						/** exchange gids */
						auto &send_gids = child->gids;
						auto &recv_gids = letnode->gids;
						mpi::ExchangeVector(
								send_gids, 0, 0, 
								recv_gids, 0, 0, comm, &status );
					}

					/** exchange skels */
					auto &send_skels = child->data.skels;
					auto &recv_skels = letnode->data.skels;
					mpi::ExchangeVector(
							send_skels, 0, 0, 
							recv_skels, 0, 0, comm, &status );
					/** exchange interpolative coefficients */
					auto &send_proj = child->data.proj;
					auto &recv_proj = letnode->data.proj;
					mpi::ExchangeVector(
							send_proj, 0, 0, 
							recv_proj, 0, 0, comm, &status );
				}
			}
		};

}; /** end class DistSimpleNearFarNodesTask */









/**
 *
 */ 
template<bool NNPRUNE, typename NODE>
class CacheFarNodesTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      /** this task contains MPI routines */
      //this->has_mpi_routines = true;

      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "cachef" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    };

    void GetEventRecord()
    {
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      auto *node = arg;
      auto *FarNodes = &node->FarNodes;
      if ( NNPRUNE ) FarNodes = &node->NNFarNodes;
      auto &K = *node->setup->K;
      auto &data = node->data;
      auto &amap = data.skels;
      std::vector<size_t> bmap;
      for ( auto it = FarNodes->begin(); it != FarNodes->end(); it ++ )
      {
        bmap.insert( bmap.end(), (*it)->data.skels.begin(), 
                                 (*it)->data.skels.end() );
      }
      data.FarKab = K( amap, bmap );
      //printf( "cache farnode %lu-by-%lu\n", amap.size(), bmap.size() ); fflush( stdout );
    };

}; /** end class CacheFarNodesTask */






/**
 *
 */ 
template<bool NNPRUNE, typename NODE>
class CacheNearNodesTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      /** this task contains MPI routines */
      //this->has_mpi_routines = true;

      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "cachen" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    };

    void DependencyAnalysis()
    {
      if ( arg->isleaf )
        arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      if ( arg->isleaf )
      {
        auto *node = arg;
        auto *NearNodes = &node->NearNodes;
        if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
        auto &K = *node->setup->K;
        auto &data = node->data;
        auto &amap = node->gids;
        std::vector<size_t> bmap;
        for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
        {
          bmap.insert( bmap.end(), (*it)->gids.begin(), 
              (*it)->gids.end() );
        }
        data.NearKab = K( amap, bmap );
        //printf( "cache nearnode %lu-by-%lu\n", amap.size(), bmap.size() ); fflush( stdout );
      }
    };

}; /** end class CacheNearNodesTask */











/**
 *  @brief Work only for no NN pruning.
 *
 */
template<typename NODE>
class WaitLETTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "waitlet" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    }

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      if ( arg->child )
      {
        auto *childFarNodes = &arg->child->NNFarNodes;
        for ( auto it  = childFarNodes->begin(); 
                   it != childFarNodes->end(); it ++ )
        {
          (*it)->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        }
      };
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
    };

}; /** end class WaitLETTask */









template<typename NODE, typename T>
void DistRowSamples( NODE *node, size_t nsamples )
{
  /** MPI */
  mpi::Comm comm = node->GetComm();
  int size = node->GetCommSize();
  int rank = node->GetCommRank();

  /** gather shared data and create reference */
  auto &K = *node->setup->K;

  /** amap contains nsamples of row gids of K */
  std::vector<size_t> &I = node->data.candidate_rows;

  /** clean up candidates from previous iteration */
	I.clear();

  /** fill-on snids first */
	if ( rank == 0 ) 
  {
    /** reserve space */
    I.reserve( nsamples );

    auto &snids = node->data.snids;
    std::multimap<T, size_t> ordered_snids = gofmm::flip_map( snids );

    for ( auto it  = ordered_snids.begin(); 
               it != ordered_snids.end(); it++ )
    {
      /** (*it) has type pair<T, size_t> */
      I.push_back( (*it).second );
      if ( I.size() >= nsamples ) break;
    }
  }

	/** buffer space */
	std::vector<size_t> candidates( nsamples );

	size_t n_required = nsamples - I.size();

	/** bcast the termination criteria */
	mpi::Bcast( &n_required, 1, 0, comm );

	while ( n_required )
	{
		if ( rank == 0 )
		{
  	  for ( size_t i = 0; i < nsamples; i ++ ) 
			  candidates[ i ] = rand() % K.col();
		}

		/** bcast candidates */
		mpi::Bcast( candidates.data(), candidates.size(), 0, comm );

		/** validation */
		std::vector<size_t> vconsensus( nsamples, 0 );
	  std::vector<size_t> validation 
			= node->setup->ContainAny( candidates, node->morton );

		/** reduce validation */
		mpi::Reduce( validation.data(), vconsensus.data(), nsamples,
				MPI_SUM, 0, comm );

	  if ( rank == 0 )
		{
  	  for ( size_t i = 0; i < nsamples; i ++ ) 
			{
				/** exit is there is enough samples */
				if ( I.size() >= nsamples )
				{
					I.resize( nsamples );
					break;
				}
				/** push the candidate to I after validation */
				if ( !vconsensus[ i ] )
				{
					if ( std::find( I.begin(), I.end(), candidates[ i ] ) == I.end() )
						I.push_back( candidates[ i ] );
				}
			};

			/** update n_required */
	    n_required = nsamples - I.size();
		}

	  /** bcast the termination criteria */
	  mpi::Bcast( &n_required, 1, 0, comm );
	}

}; /** end DistRowSamples() */







template<typename NODE, typename T>
void RowSamples( NODE *node, size_t nsamples )
{
  /** gather shared data and create reference */
  auto &K = *node->setup->K;

  /** amap contains nsamples of row gids of K */
  std::vector<size_t> &amap = node->data.candidate_rows;

  /** clean up candidates from previous iteration */
  amap.clear();


  /** construct snids from neighbors */
  if ( node->setup->NN )
  {
    //printf( "construct snids NN.row() %lu NN.col() %lu\n", 
    //    node->setup->NN->row(), node->setup->NN->col() ); fflush( stdout );
    auto &NN = *(node->setup->NN);
    auto &gids = node->gids;
    auto &pnids = node->data.pnids;
    auto &snids = node->data.snids;
    size_t kbeg = ( NN.row() + 1 ) / 2;
    size_t kend = NN.row();
    size_t knum = kend - kbeg;

    if ( node->isleaf )
    {
      pnids.clear();
      for ( size_t j = 0; j < gids.size(); j ++ )
        for ( size_t i = 0; i < NN.row() / 2; i ++ )
          pnids.insert( NN( i, gids[ j ] ).second );

      for ( size_t j = 0; j < gids.size(); j ++ )
        pnids.erase( gids[ j ] );

      snids.clear();
      std::vector<std::pair<T, size_t>> tmp( knum * gids.size() );
      for ( size_t j = 0; j < gids.size(); j ++ )
        for ( size_t i = kbeg; i < kend; i ++ )
          tmp[ j * knum + ( i - kbeg ) ] = NN( i, gids[ j ] );

      /** create a sorted list */
      std::sort( tmp.begin(), tmp.end() );
    
      for ( auto it = tmp.begin(); it != tmp.end(); it ++ )
      {
        /** create a single query */
        std::vector<size_t> sample_query( 1, (*it).second );
				std::vector<size_t> validation = 
					node->setup->ContainAny( sample_query, node->morton );

        if ( !pnids.count( (*it).second ) && !validation[ 0 ] )
        {
          /** duplication is handled by std::map */
          auto ret = snids.insert( std::pair<size_t, T>( (*it).second, (*it).first ) );
        }

        /** while we have enough samples, then exit */
        if ( snids.size() >= nsamples ) break;
      }
    }
    else
    {
      auto &lsnids = node->lchild->data.snids;
      auto &rsnids = node->rchild->data.snids;
      auto &lpnids = node->lchild->data.pnids;
      auto &rpnids = node->rchild->data.pnids;

      /** 
       *  merge children's sampling neighbors...    
       *  start with left sampling neighbor list 
       */
      snids = lsnids;

      /**
       *  Add right sampling neighbor list. If duplicate update distace if nec.
       */
      for ( auto it = rsnids.begin(); it != rsnids.end(); it ++ )
      {
        auto ret = snids.insert( *it );
        if ( !ret.second )
        {
          if ( ret.first->second > (*it).first )
            ret.first->second = (*it).first;
        }
      }

      /** remove "own" points */
      for ( size_t i = 0; i < gids.size(); i ++ ) snids.erase( gids[ i ] );

      for ( auto it = lpnids.begin(); it != lpnids.end(); it ++ )
        snids.erase( *it );
      for ( auto it = rpnids.begin(); it != rpnids.end(); it ++ )
        snids.erase( *it );
    }
  }

  /** create an order snids by flipping the std::map */
  std::multimap<T, size_t> ordered_snids = gofmm::flip_map( node->data.snids );
  
  if ( nsamples < K.col() - node->n )
  {
    amap.reserve( nsamples );

    for ( auto it  = ordered_snids.begin(); 
               it != ordered_snids.end(); it++ )
    {
      /** (*it) has type pair<T, size_t> */
      amap.push_back( (*it).second );
      if ( amap.size() >= nsamples ) break;
    }

    /** uniform samples */
    if ( amap.size() < nsamples )
    {
      while ( amap.size() < nsamples )
      {
        size_t sample = rand() % K.col();

        /** create a single query */
        std::vector<size_t> sample_query( 1, sample );

				std::vector<size_t> validation = 
					node->setup->ContainAny( sample_query, node->morton );

        /**
         *  check duplication using std::find, but check whether the sample
         *  belongs to the diagonal block using Morton ID.
         */ 
        if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() &&
             !validation[ 0 ] )
        {
          amap.push_back( sample );
        }
      }
    }
  }
  else /** use all off-diagonal blocks without samples */
  {
    for ( size_t sample = 0; sample < K.col(); sample ++ )
    {
      if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() )
      {
        amap.push_back( sample );
      }
    }
  }

}; /** end RowSamples() */




/**
 *  @brief Involve MPI routines. All MPI process must devote at least
 *         one thread to participate in this function to avoid deadlock.
 */ 
template<typename NODE, typename T>
void GetSkeletonMatrix( NODE *node )
{
  /** gather shared data and create reference */
  auto &K = *(node->setup->K);

  /** gather per node data and create reference */
  auto &data = node->data;
  auto &candidate_rows = data.candidate_rows;
  auto &candidate_cols = data.candidate_cols;
  auto &KIJ = data.KIJ;

  /** this node belongs to the local tree */
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  if ( node->isleaf )
  {
    /** use all columns */
    candidate_cols = node->gids;
  }
  else
  {
    /** concatinate [ lskels, rskels ] */
    auto &lskels = lchild->data.skels;
    auto &rskels = rchild->data.skels;
    candidate_cols = lskels;
    candidate_cols.insert( candidate_cols.end(), 
        rskels.begin(), rskels.end() );
  }

  size_t nsamples = 2 * candidate_cols.size();

  /** make sure we at least m samples */
  if ( nsamples < 2 * node->setup->m ) nsamples = 2 * node->setup->m;

  /** build  neighbor lists (snids and pnids) */
  //gofmm::BuildNeighbors<NODE, T>( node, nsamples );

  /** sample off-diagonal rows */
  RowSamples<NODE, T>( node, nsamples );

  /** 
   *  get KIJ for skeletonization 
   *
   *  notice that operator () may involve MPI collaborative communication.
   *
   */
  KIJ = K( candidate_rows, candidate_cols );

}; /** end GetSkeletonMatrix() */


/**
 *
 */ 
template<typename NODE, typename T>
class GetSkeletonMatrixTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "par-gskm" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    };

    void GetEventRecord()
    {
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      }

      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      //printf( "%lu GetSkeletonMatrix beg\n", arg->treelist_id );
      GetSkeletonMatrix<NODE, T>( arg );
      //printf( "%lu GetSkeletonMatrix end\n", arg->treelist_id );
    };

}; /** end class GetSkeletonMatrixTask */









/**
 *  @brief Involve MPI routins
 */ 
template<typename NODE, typename T>
void ParallelGetSkeletonMatrix( NODE *node )
{
	/** early return */
	if ( !node->parent ) return;

  /** gather shared data and create reference */
  auto &K = *(node->setup->K);

  /** gather per node data and create reference */
  auto &data = node->data;
  auto &candidate_rows = data.candidate_rows;
  auto &candidate_cols = data.candidate_cols;
  auto &KIJ = data.KIJ;

  /** MPI */
  int size = node->GetCommSize();
  int rank = node->GetCommRank();
  hmlp::mpi::Comm comm = node->GetComm();
	int global_rank;
	mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

  if ( size < 2 )
  {
    /** this node is the root of the local tree */
    GetSkeletonMatrix<NODE, T>( node );
  }
  else
  {
    /** 
     *  this node (mpitree::Node) belongs to the distributed tree 
     *  only executed by 0th and size/2 th rank of 
     *  the local communicator. At this moment, children have been
     *  skeletonized. Thus, we should first update (isskel) to 
     *  all MPI processes. Then we gather information for the
     *  skeletonization.
     *  
     */
    NODE *child = node->child;
    hmlp::mpi::Status status;
		size_t nsamples = 0;


    /**
     *  boardcast (isskel) to all MPI processes using children's comm
     */ 
    int child_isskel = child->data.isskel;
    hmlp::mpi::Bcast( &child_isskel, 1, 0, child->GetComm() );
    child->data.isskel = child_isskel;


    /**
     *  rank-0 owns data of this tree node, and it also owns the left child.
     */ 
    if ( rank == 0 )
    {
      /** concatinate [ lskels, rskels ] */
      candidate_cols = child->data.skels;
      std::vector<size_t> rskel;
      hmlp::mpi::RecvVector( rskel, size / 2, 10, comm, &status );
      candidate_cols.insert( candidate_cols.end(), rskel.begin(), rskel.end() );

			/** use two times of skeletons */
      nsamples = 2 * candidate_cols.size();

      /** make sure we at least m samples */
      if ( nsamples < 2 * node->setup->m ) nsamples = 2 * node->setup->m;

      /** gather rpnids and rsnids */
      auto &lsnids = node->child->data.snids;
      auto &lpnids = node->child->data.pnids;
      std::vector<T>      recv_rsdist;
      std::vector<size_t> recv_rsnids;
      std::vector<size_t> recv_rpnids;

      /** recv rsnids from size / 2 */
      hmlp::mpi::RecvVector( recv_rsdist, size / 2, 20, comm, &status );
      hmlp::mpi::RecvVector( recv_rsnids, size / 2, 30, comm, &status );
      hmlp::mpi::RecvVector( recv_rpnids, size / 2, 40, comm, &status );

      /** merge snids and update the smallest distance */
      auto &snids = node->data.snids;
      snids = lsnids;

      for ( size_t i = 0; i < recv_rsdist.size(); i ++ )
      {
        std::pair<size_t, T> query( recv_rsnids[ i ], recv_rsdist[ i ] );
        auto ret = snids.insert( query );
        if ( !ret.second )
        {
          if ( ret.first->second > recv_rsdist[ i ] )
            ret.first->second = recv_rsdist[ i ];
        }
      }

      /** remove gids from snids */
      auto &gids = node->gids;
      for ( size_t i = 0; i < gids.size(); i ++ ) snids.erase( gids[ i ] );

      /** remove lpnids and rpnids from snids  */
      for ( auto it = lpnids.begin(); it != lpnids.end(); it ++ )
        snids.erase( *it );
      for ( size_t i = 0; i < recv_rpnids.size(); i ++ )
        snids.erase( recv_rpnids[ i ] );
    }

    if ( rank == size / 2 )
    {
      /** send rskel to rank-0 */
      mpi::SendVector( child->data.skels, 0, 10, comm );

      /** gather rsnids */
      auto &rsnids = node->child->data.snids;
      auto &rpnids = node->child->data.pnids;
      std::vector<T>      send_rsdist;
      std::vector<size_t> send_rsnids;
      std::vector<size_t> send_rpnids;

      /** reserve space and push in from map */
      send_rsdist.reserve( rsnids.size() );
      send_rsnids.reserve( rsnids.size() );
      send_rpnids.reserve( rpnids.size() );

      for ( auto it = rsnids.begin(); it != rsnids.end(); it ++ )
      {
        /** (*it) has type std::pair<size_t, T>  */
        send_rsnids.push_back( (*it).first  ); 
        send_rsdist.push_back( (*it).second );
      }

      for ( auto it = rpnids.begin(); it != rpnids.end(); it ++ )
      {
        /** (*it) has type std::size_t  */
        send_rpnids.push_back( *it );
      }

      /** send rsnids and rpnids to rank-0 */
      mpi::SendVector( send_rsdist, 0, 20, comm );
      mpi::SendVector( send_rsnids, 0, 30, comm );
      mpi::SendVector( send_rpnids, 0, 40, comm );
    }

		/** Bcast nsamples */
		mpi::Bcast( &nsamples, 1, 0, comm );

		//printf( "rank %d level %lu nsamples %lu\n",
		//		global_rank, node->l, nsamples ); fflush( stdout );


		/** distributed row samples */
		DistRowSamples<NODE, T>( node, nsamples );


		//printf( "rank %d level %lu nsamples %lu after DistRowSample\n",
		//		global_rank, node->l, nsamples ); fflush( stdout );


    /** only rank-0 has non-empty I and J sets */ 
    if ( rank != 0 ) 
    {
      assert( !candidate_rows.size() );
      assert( !candidate_cols.size() );
    }


    /** 
     *  Now rank-0 has the correct ( I, J ). All other ranks in the
     *  communicator must flush their I and J sets before evaluation.
     *  all MPI process must participate in operator () 
     */
    KIJ = K( candidate_rows, candidate_cols );

		//printf( "rank %d level %lu KIJ %lu, %lu\n",
		//		global_rank, node->l, KIJ.row(), KIJ.col() ); fflush( stdout );
  }

}; /** end ParallelGetSkeletonMatrix() */


/**
 *
 */ 
template<typename NODE, typename T>
class ParallelGetSkeletonMatrixTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "par-gskm" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    };

    void GetEventRecord()
    {
    };


    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() < 2 )
        {
          //printf( "shared case\n" ); fflush( stdout );
          arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
          arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        }
        else
        {
          //printf( "distributed case size %d\n", arg->GetCommSize() ); fflush( stdout );
          arg->child->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        }
      }

      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      //printf( "%lu Par-GetSkeletonMatrix beg\n", arg->treelist_id );
      ParallelGetSkeletonMatrix<NODE, T>( arg );
      //printf( "%lu Par-GetSkeletonMatrix end\n", arg->treelist_id );
    };

}; /** end class ParallelGetSkeletonMatrixTask */












/**
 *  @brief Skeletonization with interpolative decomposition.
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
void DistSkeletonize( NODE *node )
{
  /** early return if we do not need to skeletonize */
  if ( !node->parent ) return;

  /** gather shared data and create reference */
  auto &K   = *(node->setup->K);
  auto &NN  = *(node->setup->NN);
  auto maxs = node->setup->s;
  auto stol = node->setup->stol;

  /** gather per node data and create reference */
  auto &data  = node->data;
  auto &skels = data.skels;
  auto &proj  = data.proj;
  auto &jpvt  = data.jpvt;
  auto &KIJ   = data.KIJ;
  auto &candidate_cols = data.candidate_cols;

  /** interpolative decomposition */
  size_t N = K.col();
  size_t m = KIJ.row();
  size_t n = KIJ.col();
  size_t q = node->n;


  if ( LEVELRESTRICTION )
  {
    /** TODO: need to check of both children's isskel to preceed */
  }


  /** Bill's l2 norm scaling factor */
  T scaled_stol = std::sqrt( (T)n / q ) * std::sqrt( (T)m / (N - q) ) * stol;

  /** account for uniform sampling */
  scaled_stol *= std::sqrt( (T)q / N );

  hmlp::lowrank::id<ADAPTIVE, LEVELRESTRICTION>
  ( 
    KIJ.row(), KIJ.col(), maxs, scaled_stol, /** ignore if !ADAPTIVE */
    KIJ, skels, proj, jpvt
  );


  /** depending on the flag, decide isskel or not */
  if ( LEVELRESTRICTION )
  {
    /** TODO: this needs to be bcast to other nodes */
    data.isskel = (skels.size() != 0);
  }
  else
  {
    assert( skels.size() );
    assert( proj.size() );
    assert( jpvt.size() );
    data.isskel = true;
  }
  
  /** relabel skeletions with the real lids */
  for ( size_t i = 0; i < skels.size(); i ++ )
  {
    skels[ i ] = candidate_cols[ skels[ i ] ];
  }


}; /** end DistSkeletonize() */




template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
class SkeletonizeTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "Skel" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;

      auto &K = *arg->setup->K;
      size_t n = arg->data.proj.col();
      size_t m = 2 * n;
      size_t k = arg->data.proj.row();

      /** Kab */
      flops += K.flops( m, n );

      /** GEQP3 */
      flops += ( 2.0 / 3.0 ) * n * n * ( 3 * m - n );
      mops += ( 2.0 / 3.0 ) * n * n * ( 3 * m - n );

      /* TRSM */
      flops += k * ( k - 1 ) * ( n + 1 );
      mops  += 2.0 * ( k * k + k * n );

      event.Set( label + name, flops, mops );
      arg->data.skeletonize = event;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      //printf( "%d Par-Skel beg\n", global_rank );

      DistSkeletonize<ADAPTIVE, LEVELRESTRICTION, NODE, T>( arg );

      if ( arg->setup->NN )
      {
        auto &skels = arg->data.skels;
        auto &pnids = arg->data.pnids;
        auto &NN = *(arg->setup->NN);
        pnids.clear();
        for ( size_t j = 0; j < skels.size(); j ++ )
          for ( size_t i = 0; i < NN.row(); i ++ )
            pnids.insert( NN( i, skels[ j ] ).second );
      }


      //printf( "%d Par-Skel end\n", global_rank );
    };

}; /** end class SkeletonTask */




/**
 *
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
class DistSkeletonizeTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "DistSkel" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();

      /** we don't know the exact cost here */
      cost = 5.0;

      /** high priority */
      priority = true;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;

      auto &K = *arg->setup->K;
      size_t n = arg->data.proj.col();
      size_t m = 2 * n;
      size_t k = arg->data.proj.row();

      /** Kab */
      flops += K.flops( m, n );

      /** GEQP3 */
      flops += ( 2.0 / 3.0 ) * n * n * ( 3 * m - n );
      mops += ( 2.0 / 3.0 ) * n * n * ( 3 * m - n );

      /* TRSM */
      flops += k * ( k - 1 ) * ( n + 1 );
      mops  += 2.0 * ( k * k + k * n );

      event.Set( label + name, flops, mops );
      arg->data.skeletonize = event;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      int global_rank = 0;
      hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

      if ( arg->GetCommRank() == 0 )
      {
        //printf( "%d Par-Skel beg\n", global_rank );
        DistSkeletonize<ADAPTIVE, LEVELRESTRICTION, NODE, T>( arg );
        //printf( "%d Par-Skel end\n", global_rank );
      }

			/** Bcast isskel to every MPI processes in the same comm */
			int isskel = arg->data.isskel;
			mpi::Bcast( &isskel, 1, 0, arg->GetComm() );

			/** update data.isskel */
			arg->data.isskel = isskel;

      /** Bcast skels to every MPI processes in the same comm */
      //mpi::Bcast( arg->data.skels.data(), arg->data.skels.size(), 0, 
      //    arg->GetComm() );


      if ( arg->setup->NN )
      {
        auto &skels = arg->data.skels;
        auto &pnids = arg->data.pnids;
        auto &NN = *(arg->setup->NN);

        size_t nskels = skels.size();

        /** Bcast skels to every MPI processes in the same comm */
        mpi::Bcast( &nskels, 1, 0, arg->GetComm() );
        if ( skels.size() != nskels ) skels.resize( nskels );
        mpi::Bcast( skels.data(), skels.size(), 0, arg->GetComm() );

        /** create the column distribution using cids */
        std::vector<size_t> cids;
        for ( size_t j = 0; j < skels.size(); j ++ )
          if ( NN.HasColumn( skels[ j ] ) ) cids.push_back( j );

        /** create a k-by-nskels distributed matrix */
        DistData<STAR, CIDS, size_t> X_cids( NN.row(), nskels, cids, arg->GetComm() );
        DistData<CIRC, CIRC, size_t> X_circ( NN.row(), nskels,    0, arg->GetComm() );

        /** redistribute from <STAR, CIDS> to <CIRC, CIRC> on rank-0 */
        X_circ = X_cids;

        if ( arg->GetCommRank() == 0 )
        {
          pnids.clear();
          for ( size_t i = 0; i < X_circ.size(); i ++ ) pnids.insert( X_circ[ i ] );
        }
      }




    };

}; /** end class DistSkeletonTask */




/**
 *  @brief
 */ 
template<typename NODE, typename T>
class InterpolateTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "it" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();
      // Need an accurate cost model.
      cost = 1.0;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
    }

    void Execute( Worker* user_worker )
    {
      /** only executed by rank 0 */
      if ( arg->GetCommRank() == 0  ) Interpolate<NODE, T>( arg );
    };

}; /** end class InterpolateTask */






























/**
 *  @brief ComputeAll
 */ 
template<
  bool     USE_RUNTIME = true, 
  bool     USE_OMP_TASK = false, 
  bool     SYMMETRIC_PRUNE = true, 
  bool     NNPRUNE = true, 
  bool     CACHE = true, 
  typename TREE, 
  typename T>
DistData<RIDS, STAR, T> Evaluate
( 
  TREE &tree,
  /** weights need to be in [RIDS,STAR] distribution */
  DistData<RIDS, STAR, T> &weights,
  mpi::Comm comm
)
{
  /** MPI */
  int size, rank;
  mpi::Comm_size( comm, &size );
  mpi::Comm_rank( comm, &rank );


  /** get type NODE = TREE::NODE */
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;
  using LETNODE = typename TREE::LETNODE;

  /** all timers */
  double beg, time_ratio, evaluation_time = 0.0;
  double allocate_time = 0.0, computeall_time;
  double forward_permute_time, backward_permute_time;

  /** clean up all r/w dependencies left on tree nodes */
  tree.DependencyCleanUp();

  /** n-by-nrhs, initialize potentials */
  size_t n    = weights.row();
  size_t nrhs = weights.col();

  /** potentials must be in [RIDS,STAR] distribution */
  auto &gids_owned = tree.treelist[ 0 ]->gids;
  DistData<RIDS, STAR, T> potentials( n, nrhs, gids_owned, comm );
  potentials.setvalue( 0.0 );

  tree.setup.w = &weights;
  tree.setup.u = &potentials;

  /** permute weights into w_leaf */
  if ( REPORT_EVALUATE_STATUS )
  {
    printf( "Forward permute ...\n" ); fflush( stdout );
  }
  beg = omp_get_wtime();
  int n_nodes = ( 1 << tree.depth );
  auto level_beg = tree.treelist.begin() + n_nodes - 1;
  #pragma omp parallel for
  for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
  {
    auto *node = *(level_beg + node_ind);
    auto &gids = node->gids;
    auto &w_leaf = node->data.w_leaf;
    if ( w_leaf.row() != gids.size() || w_leaf.col() != weights.col() )
    {
      w_leaf.resize( gids.size(), weights.col() );
    }
    for ( size_t j = 0; j < w_leaf.col(); j ++ )
    {
      for ( size_t i = 0; i < w_leaf.row(); i ++ )
      {
        w_leaf( i, j ) = weights( gids[ i ], j ); 
      }
    }
  }
  forward_permute_time = omp_get_wtime() - beg;


  /** Compute all N2S, S2S, S2N, L2L */
  if ( REPORT_EVALUATE_STATUS )
  {
    //hmlp::mpi::Barrier( MPI_COMM_WORLD );
    printf( "N2S, S2S, S2N, L2L (HMLP Runtime) ...\n" ); fflush( stdout );
  }
  if ( SYMMETRIC_PRUNE )
  {
    /** global barrier and timer */
    hmlp::mpi::Barrier( MPI_COMM_WORLD );
    beg = omp_get_wtime();

    /** L2L */
    mpigofmm::DistLeavesToLeavesTask<NNPRUNE, NODE, T> seqL2Ltask;
    tree.LocaTraverseLeafs( seqL2Ltask );

    /** N2S (Loca first, Dist follows) */
    gofmm::UpdateWeightsTask<NODE, T> seqN2Stask;
    mpigofmm::DistUpdateWeightsTask<MPINODE, T> mpiN2Stask;
    tree.LocaTraverseUp( seqN2Stask );
    tree.DistTraverseUp( mpiN2Stask );

    /** S2S */
    gofmm::SkeletonsToSkeletonsTask<NNPRUNE, NODE> seqS2Stask;
    mpigofmm::DistSkeletonsToSkeletonsTask<NNPRUNE, MPINODE> mpiS2Stask;
    tree.LocaTraverseUnOrdered( seqS2Stask );
    tree.DistTraverseUnOrdered( mpiS2Stask );

    /** S2N (Dist first, Loca follows) */
    gofmm::SkeletonsToNodesTask<NNPRUNE, NODE, T> seqS2Ntask;
    mpigofmm::DistSkeletonsToNodesTask<NNPRUNE, MPINODE, T> mpiS2Ntask;
    tree.DistTraverseDown( mpiS2Ntask );
    tree.LocaTraverseDown( seqS2Ntask );

    /** epoch */
    hmlp_run();

    /** reduce direct iteractions from 4 copies */
    #pragma omp parallel for
    for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
    {
      auto *node = *(level_beg + node_ind);
      auto &u_leaf = node->data.u_leaf[ 0 ];
      /** reduce all u_leaf[0:4] */
      for ( size_t p = 1; p < 20; p ++ )
      {
        for ( size_t i = 0; i < node->data.u_leaf[ p ].size(); i ++ )
          u_leaf[ i ] += node->data.u_leaf[ p ][ i ];
      }
    }

    /** global barrier and timer */
    hmlp::mpi::Barrier( MPI_COMM_WORLD );
    computeall_time = omp_get_wtime() - beg;
  }
  else 
  {
    /** TODO: implement unsymmetric prunning */
    printf( "Non symmetric Distributed ComputeAll is not yet implemented\n" );
    exit( 1 );
  }

  /** permute back */
  if ( REPORT_EVALUATE_STATUS )
  {
    printf( "Backward permute ...\n" ); fflush( stdout );
  }
  beg = omp_get_wtime();
  #pragma omp parallel for
  for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
  {
    auto *node = *(level_beg + node_ind);
    auto &amap = node->lids;
    auto &u_leaf = node->data.u_leaf[ 0 ];

    for ( size_t j = 0; j < potentials.col(); j ++ )
      for ( size_t i = 0; i < amap.size(); i ++ )
        potentials( amap[ i ], j ) += u_leaf( i, j );

  }
  backward_permute_time = omp_get_wtime() - beg;

  /** compute the breakdown cost */
  evaluation_time += allocate_time;
  evaluation_time += forward_permute_time;
  evaluation_time += computeall_time;
  evaluation_time += backward_permute_time;
  time_ratio = 100 / evaluation_time;

  if ( rank == 0 && REPORT_EVALUATE_STATUS )
  {
    printf( "========================================================\n");
    printf( "GOFMM evaluation phase\n" );
    printf( "========================================================\n");
    printf( "Allocate ------------------------------ %5.2lfs (%5.1lf%%)\n", 
        allocate_time, allocate_time * time_ratio );
    printf( "Forward permute ----------------------- %5.2lfs (%5.1lf%%)\n", 
        forward_permute_time, forward_permute_time * time_ratio );
    printf( "N2S, S2S, S2N, L2L -------------------- %5.2lfs (%5.1lf%%)\n", 
        computeall_time, computeall_time * time_ratio );
    printf( "Backward permute ---------------------- %5.2lfs (%5.1lf%%)\n", 
        backward_permute_time, backward_permute_time * time_ratio );
    printf( "========================================================\n");
    printf( "Evaluate ------------------------------ %5.2lfs (%5.1lf%%)\n", 
        evaluation_time, evaluation_time * time_ratio );
    printf( "========================================================\n\n");
  }








  tree.DependencyCleanUp(); 

  return potentials;

}; /** end Evaluate() */















/**
 *  @brief template of the compress routine
 */ 
template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  typename    SPLITTER, 
  typename    RKDTSPLITTER, 
  typename    T, 
  typename    SPDMATRIX>
mpitree::Tree<
  mpigofmm::Setup<SPDMATRIX, SPLITTER, T>, 
  mpigofmm::NodeData<T>,
  N_CHILDREN,
  T> 
*Compress
( 
  DistData<STAR, CBLK, T> *X_cblk,
  SPDMATRIX &K, 
  DistData<STAR, CBLK, std::pair<T, std::size_t>> &NN_cblk,
  SPLITTER splitter, 
  RKDTSPLITTER rkdtsplitter,
	Configuration<T> &config
)
{
  /** MPI */
  int size, rank;
  mpi::Comm_size( MPI_COMM_WORLD, &size );
  mpi::Comm_rank( MPI_COMM_WORLD, &rank );




  /** get all user-defined parameters */
  DistanceMetric metric = config.MetricType();
  size_t n = config.ProblemSize();
	size_t m = config.LeafNodeSize();
	size_t k = config.NeighborSize(); 
	size_t s = config.MaximumRank(); 
  T stol = config.Tolerance();
	T budget = config.Budget(); 

  /** options */
  const bool SYMMETRIC = true;
  const bool NNPRUNE   = true;
  const bool CACHE     = true;

  /** instantiation for the GOFMM tree */
  using SETUP       = mpigofmm::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA        = mpigofmm::NodeData<T>;
  using NODE        = tree::Node<SETUP, N_CHILDREN, DATA, T>;
  using MPINODE     = mpitree::Node<SETUP, N_CHILDREN, DATA, T>;
  using LETNODE     = mpitree::LetNode<SETUP, N_CHILDREN, DATA, T>;
  using TREE        = mpitree::Tree<SETUP, DATA, N_CHILDREN, T>;

  /** instantiation for the randomisze Spd-Askit tree */
  using RKDTSETUP   = mpigofmm::Setup<SPDMATRIX, RKDTSPLITTER, T>;
  using RKDTNODE    = tree::Node<RKDTSETUP, N_CHILDREN, DATA, T>;
  using MPIRKDTNODE = tree::Node<RKDTSETUP, N_CHILDREN, DATA, T>;

  /** all timers */
  double beg, omptask45_time, omptask_time, ref_time;
  double time_ratio, compress_time = 0.0, other_time = 0.0;
  double ann_time, tree_time, skel_time, mergefarnodes_time, cachefarnodes_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

  /** original order of the matrix */
  beg = omp_get_wtime();
  std::vector<std::size_t> gids( n ), lids( n );
  #pragma omp parallel for
  for ( auto i = 0; i < n; i ++ ) 
  {
    gids[ i ] = i;
    lids[ i ] = i;
  }
  other_time += omp_get_wtime() - beg;


  /** initialize metric ball tree using approximate center split */
  auto *tree_ptr = new mpitree::Tree<SETUP, DATA, N_CHILDREN, T>();
	auto &tree = *tree_ptr;

	/** global configuration for the metric tree */
  tree.setup.X_cblk = X_cblk;
  tree.setup.K = &K;
	tree.setup.metric = metric; 
  tree.setup.splitter = splitter;
  tree.setup.NN_cblk = &NN_cblk;
  tree.setup.m = m;
  tree.setup.k = k;
  tree.setup.s = s;
  tree.setup.stol = stol;

	/** metric ball tree partitioning */
  printf( "TreePartitioning ...\n" ); fflush( stdout );
  mpi::Barrier( MPI_COMM_WORLD );
  beg = omp_get_wtime();
  tree.TreePartition( n );
  mpi::Barrier( MPI_COMM_WORLD );
  tree_time = omp_get_wtime() - beg;
  printf( "end TreePartitioning ...\n" ); fflush( stdout );
  mpi::Barrier( MPI_COMM_WORLD );

  /** now redistribute K */
  K.Redistribute( tree.treelist[ 0 ]->gids );





  /** iterative all nearnest-neighbor (ANN) */
  const size_t n_iter = 10;
  const bool SORTED = false;
  /** do not change anything below this line */
  hmlp::mpitree::Tree<RKDTSETUP, DATA, N_CHILDREN, T> rkdt;
  rkdt.setup.X_cblk = X_cblk;
  rkdt.setup.K = &K;
	rkdt.setup.metric = metric; 
  rkdt.setup.splitter = rkdtsplitter;
  std::pair<T, std::size_t> initNN( std::numeric_limits<T>::max(), n );

  printf( "NeighborSearch ...\n" ); fflush( stdout );
  beg = omp_get_wtime();
  if ( NN_cblk.row() != k )
  {
    //gofmm::KNNTask<3, RKDTNODE, T> knntask;
    NeighborsTask<RKDTNODE, T> knntask;
    NN_cblk = rkdt.AllNearestNeighbor<SORTED>( n_iter, n, k, 15, initNN, knntask );
  }
  else
  {
    printf( "not performed (precomputed or k=0) ...\n" ); fflush( stdout );
  }
  ann_time = omp_get_wtime() - beg;

  /** check illegle values in NN */
  //for ( size_t j = 0; j < NN.col(); j ++ )
  //{
  //  for ( size_t i = 0; i < NN.row(); i ++ )
  //  {
  //    size_t neighbor_gid = NN( i, j ).second;
  //    if ( neighbor_gid < 0 || neighbor_gid >= n )
  //    {
  //      printf( "NN( %lu, %lu ) has illegle values %lu\n", i, j, neighbor_gid );
  //      break;
  //    }
  //  }
  //}





//  /** initialize metric ball tree using approximate center split */
//  auto *tree_ptr = new mpitree::Tree<SETUP, DATA, N_CHILDREN, T>();
//	auto &tree = *tree_ptr;
//
//	/** global configuration for the metric tree */
//  tree.setup.X_cblk = X_cblk;
//  tree.setup.K = &K;
//	tree.setup.metric = metric; 
//  tree.setup.splitter = splitter;
//  tree.setup.NN_cblk = &NN_cblk;
//  tree.setup.m = m;
//  tree.setup.k = k;
//  tree.setup.s = s;
//  tree.setup.stol = stol;
//
//	/** metric ball tree partitioning */
//  printf( "TreePartitioning ...\n" ); fflush( stdout );
//  mpi::Barrier( MPI_COMM_WORLD );
//  beg = omp_get_wtime();
//  tree.TreePartition( n );
//  mpi::Barrier( MPI_COMM_WORLD );
//  tree_time = omp_get_wtime() - beg;
//  printf( "end TreePartitioning ...\n" ); fflush( stdout );
//  mpi::Barrier( MPI_COMM_WORLD );

//  /** now redistribute K */
//  K.Redistribute( tree.treelist[ 0 ]->gids );

  /** now redistribute NN according to gids */
  DistData<STAR, CIDS, std::pair<T, size_t>> NN( k, n, tree.treelist[ 0 ]->gids, MPI_COMM_WORLD );
  NN = NN_cblk;
  tree.setup.NN = &NN;



  /** skeletonization */
  printf( "Skeletonization (HMLP Runtime) ...\n" ); fflush( stdout );
  mpi::Barrier( MPI_COMM_WORLD );
  beg = omp_get_wtime();

  /** clean up all dependencies for skeletonization */
  tree.DependencyCleanUp();

  /** gather sample rows and skeleton columns, then ID */
  mpigofmm::GetSkeletonMatrixTask<NODE, T> seqGETMTXtask;
  mpigofmm::ParallelGetSkeletonMatrixTask<MPINODE, T> mpiGETMTXtask;
  mpigofmm::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T> seqSKELtask;
  mpigofmm::DistSkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, MPINODE, T> mpiSKELtask;
  tree.LocaTraverseUp( seqGETMTXtask, seqSKELtask );
  tree.DistTraverseUp( mpiGETMTXtask, mpiSKELtask );

  /** compute the coefficient matrix of ID */
  gofmm::InterpolateTask<NODE, T> seqPROJtask;
  mpigofmm::InterpolateTask<MPINODE, T> mpiPROJtask;
  tree.LocaTraverseUnOrdered( seqPROJtask );
  tree.DistTraverseUnOrdered( mpiPROJtask );

	/** create interaction lists */
	SimpleNearFarNodesTask<NODE> seqNEARFARtask;
	DistSimpleNearFarNodesTask<LETNODE, MPINODE> mpiNEARFARtask;
  tree.DistTraverseDown( mpiNEARFARtask );
	tree.LocaTraverseDown( seqNEARFARtask );

	/** cache near KIJ interactions */
  mpigofmm::CacheNearNodesTask<NNPRUNE, NODE> seqNEARKIJtask;
	tree.LocaTraverseLeafs( seqNEARKIJtask );

	/** cache  far KIJ interactions */
  mpigofmm::CacheFarNodesTask<NNPRUNE, NODE> seqFARKIJtask;
  mpigofmm::CacheFarNodesTask<NNPRUNE, MPINODE> mpiFARKIJtask;
  tree.LocaTraverseUnOrdered( seqFARKIJtask );
  tree.DistTraverseUnOrdered( mpiFARKIJtask );

  
  other_time += omp_get_wtime() - beg;
  hmlp_run();
  mpi::Barrier( MPI_COMM_WORLD );
  skel_time = omp_get_wtime() - beg;







  double exact_ratio = (double) m / n; 

  if ( rank == 0 && REPORT_COMPRESS_STATUS )
  {
    compress_time += ann_time;
    compress_time += tree_time;
    compress_time += skel_time;
    compress_time += mergefarnodes_time;
    compress_time += cachefarnodes_time;
    time_ratio = 100.0 / compress_time;
    printf( "========================================================\n");
    printf( "GOFMM compression phase\n" );
    printf( "========================================================\n");
    printf( "NeighborSearch ------------------------ %5.2lfs (%5.1lf%%)\n", ann_time, ann_time * time_ratio );
    printf( "TreePartitioning ---------------------- %5.2lfs (%5.1lf%%)\n", tree_time, tree_time * time_ratio );
    printf( "Skeletonization (HMLP Runtime   ) ----- %5.2lfs (%5.1lf%%)\n", skel_time, skel_time * time_ratio );
    printf( "MergeFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", mergefarnodes_time, mergefarnodes_time * time_ratio );
    printf( "CacheFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", cachefarnodes_time, cachefarnodes_time * time_ratio );
    printf( "========================================================\n");
    printf( "Compress (%4.2lf not compressed) -------- %5.2lfs (%5.1lf%%)\n", 
        exact_ratio, compress_time, compress_time * time_ratio );
    printf( "========================================================\n\n");
  }










  /** cleanup all w/r dependencies on tree nodes */
  tree_ptr->DependencyCleanUp();

  /** global barrier to make sure all processes have completed */
  hmlp::mpi::Barrier( MPI_COMM_WORLD );



  return tree_ptr;

}; /** end Compress() */







/**
 *  @brief Here MPI_Allreduce is required.
 */ 
template<typename TREE, typename T>
T ComputeError( TREE &tree, size_t gid, hmlp::Data<T> potentials, mpi::Comm comm )
{
  auto &K = *tree.setup.K;
  auto &w = *tree.setup.w;

  auto  I = std::vector<size_t>( 1, gid );
  auto &J = tree.treelist[ 0 ]->gids;

  //auto Kab = K( I, J );

	hmlp::Data<T> Kab;

	bool do_terminate = false;

	/** use omp sections to create client-server */
  #pragma omp parallel sections
	{
		/** client thread */
    #pragma omp section
		{
			Kab = K( I, J );
			do_terminate = true;
		}
		/** server thread */
    #pragma omp section
		{
			K.BackGroundProcess( &do_terminate );
		}

	}; /** end omp parallel sections */




  auto loc_exact = potentials;
  auto glb_exact = potentials;

  xgemm
  (
    "N", "N",
    Kab.row(), w.col(), w.row(),
    1.0,       Kab.data(),       Kab.row(),
                 w.data(),         w.row(),
    0.0, loc_exact.data(), loc_exact.row()
  );          

  /** Allreduce K( gid, : ) * w( : ) */
  mpi::Allreduce( 
      loc_exact.data(), glb_exact.data(), 
      loc_exact.size(), MPI_SUM, comm );

  auto nrm2 = hmlp_norm( glb_exact.row(),  glb_exact.col(), 
                         glb_exact.data(), glb_exact.row() ); 

  for ( size_t j = 0; j < potentials.col(); j ++ )
  {
    potentials( (size_t)0, j ) -= glb_exact( (size_t)0, j );
  }

  auto err = hmlp_norm(  potentials.row(), potentials.col(), 
                        potentials.data(), potentials.row() ); 

  return err / nrm2;
}; /** end ComputeError() */










}; /** end namespace gofmm */
}; /** end namespace hmlp */

#endif /** define GOFMM_MPI_HPP */
