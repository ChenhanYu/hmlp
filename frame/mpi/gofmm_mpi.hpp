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

#include <hmlp_util.hpp>
#include <mpi/tree_mpi.hpp>
#include <mpi/DistData.hpp>
#include <gofmm/gofmm.hpp>
#include <primitives/combinatorics.hpp>
#include <primitives/gemm.hpp>


using namespace std;
using namespace hmlp;

//using namespace hmlp::gofmm;

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
class Setup : public mpitree::Setup<SPLITTER, T>
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

    /**
     *  User specific budget for the amount of direct evaluation
     */ 
    double budget = 0.0;

		/** (default) distance type */
		DistanceMetric metric = ANGLE_DISTANCE;

    /** the SPDMATRIX (accessed with gids: dense, CSC or OOC) */
    SPDMATRIX *K = NULL;

    /** rhs-by-n all weights */
    Data<T> *w = NULL;

    /** n-by-nrhs all potentials */
    Data<T> *u = NULL;

    /** buffer space, either dimension needs to be n  */
    Data<T> *input = NULL;
    Data<T> *output = NULL;

    /** regularization */
    T lambda = 0.0;

    /** whether the matrix is symmetric */
    bool issymmetric = true;

    /** use ULV or Sherman-Morrison-Woodbury */
    bool do_ulv_factorization = true;

}; /** end class Setup */




template<typename T>
class NodeData : public hfamily::Factor<T>
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
    vector<size_t> skels;

    /** (buffer) nsamples row gids */
    vector<size_t> candidate_rows;

    /** (buffer) sl+sr column gids of children */
    vector<size_t> candidate_cols;

    /** (buffer) nsamples-by-(sl+sr) submatrix of K */
    Data<T> KIJ; 

    /** 2s, pivoting order of GEQP3 */
    vector<int> jpvt;

    /** s-by-2s */
    Data<T> proj;

    /** sampling neighbors ids */
    map<std::size_t, T> snids; 

    /* pruning neighbors ids */
    unordered_set<std::size_t> pnids; 

    /** skeleton weights and potentials */
    Data<T> w_skel;
    Data<T> u_skel;

    /** permuted weights and potentials (buffer) */
    Data<T> w_leaf;
    Data<T> u_leaf[ 20 ];

    /** hierarchical tree view of w<RIDS> and u<RIDS> */
    View<T> w_view;
    View<T> u_view;

    /** cached Kab */
    Data<std::size_t> Nearbmap;
    Data<T> NearKab;
    Data<T> FarKab;

    /**
     *  This pool contains a subset of gids owned by this node.
     *  Multiple threads may try to update this pool during construction.
     *  We use a lock to prevent race condition.
     */ 
    map<size_t, map<size_t, T>> candidates;
    map<size_t, T> pool;
    multimap<T, size_t> ordered_pool;






    /** Kij evaluation counter counters */
    pair<double, size_t> kij_skel;
    pair<double, size_t> kij_s2s;
    pair<double, size_t> kij_s2n;

    /** many timers */
    double merge_neighbors_time = 0.0;
    double id_time = 0.0;

    /** recorded events (for HMLP Runtime) */
    Event skeletonize;
    Event updateweight;
    Event skeltoskel;
    Event skeltonode;

    Event s2s;
    Event s2n;

    /** knn accuracy */
    double knn_acc = 0.0;
    size_t num_acc = 0;

}; /** end class NodeData */



/** 
 *  @brief This task creates an hierarchical tree view for
 *         weights<RIDS> and potentials<RIDS>.
 */
template<typename NODE>
class DistTreeViewTask : public Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      name = std::string( "TreeView" );
      arg = user_arg;
      cost = 1.0;
      ss << arg->treelist_id;
      label = ss.str();
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    /** preorder dependencies (with a single source node) */
    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
      if ( !arg->isleaf && !arg->child )
      {
        arg->lchild->DependencyAnalysis( RW, this );
        arg->rchild->DependencyAnalysis( RW, this );
      }
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      //printf( "TreeView %lu\n", node->treelist_id );
      auto *node   = arg;
      auto &data   = node->data;
      auto *setup  = node->setup;

      /** w and u can be Data<T> or DistData<RIDS,STAR,T> */
      auto &w = *(setup->w);
      auto &u = *(setup->u);

      /** get the matrix view of this tree node */
      auto &U  = data.u_view;
      auto &W  = data.w_view;

      /** both w and u are column-majored, thus nontranspose */
      U.Set( u );
      W.Set( w );


      if ( !node->isleaf && !node->child )
      {
        assert( node->lchild && node->rchild );
        auto &UL = node->lchild->data.u_view;
        auto &UR = node->rchild->data.u_view;
        auto &WL = node->lchild->data.w_view;
        auto &WR = node->rchild->data.w_view;
        /** 
         *  U = [ UL;    W = [ WL;
         *        UR; ]        WR; ] 
         */
        U.Partition2x1( UL, 
                        UR, node->lchild->n, TOP );
        W.Partition2x1( WL, 
                        WR, node->lchild->n, TOP );

        assert( node->lchild->n == node->lchild->gids.size() );
        assert( UL.row() == node->lchild->n );
        assert( UR.row() == node->rchild->n );
        assert( WL.row() == node->lchild->n );
        assert( WR.row() == node->rchild->n );
      }
    };

}; /** end class DistTreeViewTask */















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
struct centersplit : public gofmm::centersplit<SPDMATRIX, N_SPLIT, T>
{

  /** Shared-memory operator */
  inline vector<vector<size_t> > operator()
  ( 
    vector<size_t>& gids, vector<size_t>& lids
  ) const 
  {
    /** MPI */
    int size, rank, global_rank;
    mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

    //printf( "rank %d enter shared centersplit n = %lu\n", 
    //    global_rank, gids.size() ); fflush( stdout );

    return gofmm::centersplit<SPDMATRIX, N_SPLIT, T>::operator()
      ( gids, lids );
  };


  /** distributed operator */
  inline vector<vector<size_t> > operator()
  ( 
    /** owned gids */
    vector<size_t>& gids,
    /** communicator: */
    mpi::Comm comm
  ) const 
  {
    /** All assertions */
    assert( N_SPLIT == 2 );
    assert( this->Kptr );

    /** Declaration */
    int size, rank, global_rank;
    mpi::Comm_size( comm, &size );
    mpi::Comm_rank( comm, &rank );
    mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
    SPDMATRIX &K = *(this->Kptr);
    vector<vector<size_t>> split( N_SPLIT );

    /** Reduce to get the total size of gids */
    int n = 0;
    int num_points_owned = gids.size();
    vector<T> temp( gids.size(), 0.0 );

    //printf( "rank %d before Allreduce\n", global_rank ); fflush( stdout );


    /** n = sum( num_points_owned ) over all MPI processes in comm */
    mpi::Allreduce( &num_points_owned, &n, 1, MPI_SUM, comm );
    /** Early return */
    if ( n == 0 ) return split;


    //printf( "rank %d enter distributed centersplit n = %d\n", global_rank, n ); fflush( stdout );

    /** Collecting column samples of K and DCC */
    vector<size_t> column_samples( this->n_centroid_samples );
    for ( size_t j = 0; j < this->n_centroid_samples; j ++ )
    {
      /** just use the first few gids */
      column_samples[ j ] = gids[ j ];
    }


    /** Collecting diagonal DII, DCC, KIC */
    Data<T> DII = K.Diagonal( gids );
    Data<T> DCC = K.Diagonal( column_samples );
    Data<T> KIC = K( gids, column_samples );

    /** Compute d2c (distance to the approx centroid) for each owned point */
    temp = gofmm::AllToCentroid( this->metric, DII, KIC, DCC );

    /** Find the f2c (far most to center) from points owned */
    auto idf2c = distance( temp.begin(), max_element( temp.begin(), temp.end() ) );

    /** Create a pair for MPI Allreduce */
    mpi::NumberIntPair<T> local_max_pair, max_pair; 
    local_max_pair.val = temp[ idf2c ];
    local_max_pair.key = rank;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** Boardcast gidf2c from the MPI process which has the max_pair */
    int gidf2c = gids[ idf2c ];
    mpi::Bcast( &gidf2c, 1, MPI_INT, max_pair.key, comm );


    //printf( "rank %d val %E key %d; global val %E key %d\n", 
    //    rank, local_max_pair.val, local_max_pair.key,
    //    max_pair.val, max_pair.key ); fflush( stdout );
    //printf( "rank %d gidf2c %d\n", rank, gidf2c  ); fflush( stdout );

    /** Collecting KIP and kpp */
    vector<size_t> P( 1, gidf2c );
    K.BcastColumns( P, max_pair.key, comm );
    Data<T> KIP = K( gids, P );
    Data<T> kpp = K.Diagonal( P );

    /** Compute the all2f (distance to farthest point) */
    temp = gofmm::AllToFarthest( this->metric, DII, KIP, kpp[ 0 ] );


    /** Find f2f (far most to far most) from owned points */
    auto idf2f = distance( temp.begin(), max_element( temp.begin(), temp.end() ) );

    /** Create a pair for MPI Allreduce */
    local_max_pair.val = temp[ idf2f ];
    local_max_pair.key = rank;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** boardcast gidf2f from the MPI process which has the max_pair */
    int gidf2f = gids[ idf2f ];
    mpi::Bcast( &gidf2f, 1, MPI_INT, max_pair.key, comm );

    //printf( "rank %d val %E key %d; global val %E key %d\n", 
    //    rank, local_max_pair.val, local_max_pair.key,
    //    max_pair.val, max_pair.key ); fflush( stdout );
    //printf( "rank %d gidf2f %d\n", rank, gidf2f  ); fflush( stdout );

    /** Collecting KIQ and kqq */
    vector<size_t> Q( 1, gidf2f );
    K.BcastColumns( Q, max_pair.key, comm );
    auto KIQ = K( gids, Q );
    auto kqq = K.Diagonal( Q );

    /** compute all2leftright (projection i.e. dip - diq) */
    temp = gofmm::AllToLeftRight( this->metric, DII, KIP, KIQ, kpp[ 0 ], kqq[ 0 ] );

    /** parallel median select */
    T  median = combinatorics::Select( n / 2, temp, comm );
    //printf( "kic_t %lfs, a2c_t %lfs, kip_t %lfs, select_t %lfs\n", 
		//		kic_t, a2c_t, kip_t, select_t ); fflush( stdout );

    //printf( "rank %d median %E\n", rank, median ); fflush( stdout );

    vector<size_t> middle;
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
    mpi::Allreduce( &num_mid_owned, &nmid, 1, MPI_SUM, comm );
    mpi::Allreduce( &num_lhs_owned, &nlhs, 1, MPI_SUM, comm );
    mpi::Allreduce( &num_rhs_owned, &nrhs, 1, MPI_SUM, comm );

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


    /** perform P2P redistribution */
    K.RedistributeWithPartner( gids, split[ 0 ], split[ 1 ], comm );

    return split;
  };


}; /** end struct centersplit */





template<typename SPDMATRIX, int N_SPLIT, typename T> 
struct randomsplit : public gofmm::randomsplit<SPDMATRIX, N_SPLIT, T>
{

  /** shared-memory operator */
  inline vector<vector<size_t> > operator()
  ( 
    vector<size_t>& gids,
    vector<size_t>& lids
  ) const 
  {
    return gofmm::randomsplit<SPDMATRIX, N_SPLIT, T>::operator() ( gids, lids );
  };

  /** distributed operator */
  inline vector<vector<size_t> > operator()
  (
    vector<size_t>& gids,
    mpi::Comm comm
  ) const 
  {
    /** all assertions */
    assert( N_SPLIT == 2 );
    assert( this->Kptr );

    /** declaration */
    int size, rank, global_rank, global_size;
    mpi::Comm_size( comm, &size );
    mpi::Comm_rank( comm, &rank );
    mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
    mpi::Comm_size( MPI_COMM_WORLD, &global_size );
    SPDMATRIX &K = *(this->Kptr);
    vector<vector<size_t>> split( N_SPLIT );

    if ( size == global_size )
    {
      for ( size_t i = 0; i < gids.size(); i ++ )
        assert( gids[ i ] == i * size + rank );
    }




    /** reduce to get the total size of gids */
    int n = 0;
    int num_points_owned = gids.size();
    vector<T> temp( gids.size(), 0.0 );

    //printf( "rank %d before Allreduce\n", global_rank ); fflush( stdout );

    /** n = sum( num_points_owned ) over all MPI processes in comm */
    mpi::Allreduce( &num_points_owned, &n, 1, MPI_INT, MPI_SUM, comm );

    //printf( "rank %d enter distributed randomsplit n = %d\n", global_rank, n ); fflush( stdout );

    /** early return */
    if ( n == 0 ) return split;


    /** Randomly select two points p and q */
    //size_t idf2c = std::rand() % gids.size();
    //size_t idf2f = std::rand() % gids.size();
    //while ( idf2c == idf2f ) idf2f = std::rand() % gids.size();

    /** Randomly select two points p and q */
    auto gidf2c = gids[ std::rand() % gids.size() ];
    auto gidf2f = gids[ std::rand() % gids.size() ];

    /** Bcast gidf2c and gidf2f from rank-0 to all MPI processes */
    mpi::Bcast( &gidf2c, 1,        0, comm );
    mpi::Bcast( &gidf2f, 1, size / 2, comm );

    vector<size_t> P( 1, gidf2c );
    vector<size_t> Q( 1, gidf2f );
		vector<size_t> PQ( 2 ); 
    PQ[ 0 ] = gidf2c; 
    PQ[ 1 ] = gidf2f;
    /** Distributed bcast on K */
    K.BcastColumns( P,        0, comm );
    K.BcastColumns( Q, size / 2, comm );

    //printf( "BcastColumns %lu %lu\n", gidf2c, gidf2f ); fflush( stdout ); 
    mpi::Barrier( comm );


    /** Collecting DII, KIP, KIQ, kpp and kqq */
    Data<T> DII = K.Diagonal( gids );
    Data<T> kpp = K.Diagonal( P );
    Data<T> kqq = K.Diagonal( Q );
		Data<T> KIPQ = K( gids, PQ );
		Data<T> KIP( gids.size(), (size_t) 1 );
		Data<T> KIQ( gids.size(), (size_t) 1 );

    //printf( "KIQP %lu %lu\n", gidf2c, gidf2f ); fflush( stdout ); 
    mpi::Barrier( comm );

    for ( size_t i = 0; i < gids.size(); i ++ )
		{
			KIP[ i ] = KIPQ( i, (size_t)0 );
			KIQ[ i ] = KIPQ( i, (size_t)1 );
		}

    /** Compute all2leftright (projection i.e. dip - diq) */
    temp = gofmm::AllToLeftRight( this->metric, DII, KIP, KIQ, kpp[ 0 ], kqq[ 0 ] );


    /** parallel median select */
    /** compute all2leftright (projection i.e. dip - diq) */
    T  median = hmlp::combinatorics::Select( n / 2, temp, comm );

    vector<size_t> middle;
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
    mpi::Allreduce( &num_mid_owned, &nmid, 1, MPI_SUM, comm );
    mpi::Allreduce( &num_lhs_owned, &nlhs, 1, MPI_SUM, comm );
    mpi::Allreduce( &num_rhs_owned, &nrhs, 1, MPI_SUM, comm );

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

      /** Now decide the portion */
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
        if ( i < nlhs_required_owned ) 
          split[ 0 ].push_back( middle[ i ] );
        else                           
          split[ 1 ].push_back( middle[ i ] );
      }
    };

    /** Perform P2P redistribution */
    K.RedistributeWithPartner( gids, split[ 0 ], split[ 1 ], comm );

    printf( "Finish RedistributeWithPartner\n" ); fflush( stdout );
    mpi::Barrier( comm );

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

      /** use the same distance as the tree */
      metric = arg->setup->metric;

      //--------------------------------------
      double flops, mops;
      auto &gids = arg->gids;
      auto &NN = *arg->setup->NN;
      flops = gids.size();
      flops *= ( 4.0 * gids.size() );
      // Heap select worst case
      mops = (size_t)std::log( NN.row() ) * gids.size();
      mops *= gids.size();
      // Access K
      mops += flops;
      event.Set( name + label, flops, mops );
      //--------------------------------------
			
      // TODO: Need an accurate cost model.
      cost = mops / 1E+9;

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
 *
 *  
 */
template<typename MPINODE, typename T>
void DistUpdateWeights( MPINODE *node )
{
  /** MPI */
  mpi::Status status;
  int global_rank = 0; 
  mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
  mpi::Comm comm = node->GetComm();
  int size = node->GetCommSize();
  int rank = node->GetCommRank();

  if ( size < 2 )
  {
    /** this is the root of the local tree */
    gofmm::UpdateWeights<MPINODE, T>( node );
  }
  else
  {
    if ( !node->parent || !node->data.isskel ) return;

    /** get the child node  */
    auto *child = node->child;

    /** gather shared data and create reference */
    auto &w = *node->setup->w;
    size_t nrhs = w.col();

    /** gather per node data and create reference */
    auto &data   = node->data;
    auto &proj   = data.proj;
    auto &skels  = data.skels;
    auto &w_skel = data.w_skel;

    /** this is the corresponding MPI rank */
    if ( rank == 0 )
    {
			size_t sl = child->data.skels.size();
			//size_t sr = child->sibling->data.skels.size();
			size_t sr = proj.col() - sl;

      //printf( "Update weight sl %lu sr %lu sl + sr %lu\n", sl, sr, proj.col()  );

      auto &w_lskel = child->data.w_skel;
      auto &w_rskel = child->sibling->data.w_skel;
    
			/** recv child->w_skel from rank / 2 */
			mpi::ExchangeVector(
					w_lskel, size / 2, 0, 
					w_rskel, size / 2, 0, comm, &status );

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

      if ( 1 )
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
        ///** w_skel += proj( r ) * w_rskel */
        //xgemm
        //(
        //  "No-transpose", "No-transpose",
        //  w_skel.row(), nrhs, sr,
        //  1.0,    proj.data() + proj.row() * sl, proj.row(),
        //       w_rskel.data(), sr,
        //  1.0,  w_skel.data(), w_skel.row()
        //);
      }
      else
      {
        /** create a transpose view proj_v */
        View<T> P( false,   proj ), PL,
                                    PR;
        View<T> W( false, w_skel ), WL( false, w_lskel );
                                    //WR( false, w_rskel );
        /** P = [ PL, PR ] */
        P.Partition1x2( PL, PR, sl, LEFT );
        /** W  = PL * WL */
        gemm::xgemm<GEMM_NB>( (T)1.0, PL, WL, (T)0.0, W );
        //W.DependencyCleanUp();
        ///** W += PR * WR */
        //gemm::xgemm<GEMM_NB>( (T)1.0, PR, WR, (T)1.0, W );
        ////W.DependencyCleanUp();
      }


      Data<T> w_skel_sib;
      mpi::ExchangeVector(
            w_skel,     size / 2, 0, 
            w_skel_sib, size / 2, 0, comm, &status );

			/** reduce */
      #pragma omp parallel for
			for ( size_t i = 0; i < w_skel.size(); i ++ )
				w_skel[ i ] += w_skel_sib[ i ];
    }

    /** the rank that holds the skeleton weight of the right child */
    if ( rank == size / 2 )
    {
			size_t sr = child->data.skels.size();
			size_t sl = proj.col() - sr;

      //printf( "Update weight sl %lu sr %lu sl + sr %lu\n", sl, sr, proj.col()  );


      auto &w_lskel = child->sibling->data.w_skel;
      auto &w_rskel = child->data.w_skel;
      /** send child->w_skel to 0 */
      //hmlp::mpi::SendVector( w_rskel, 0, 0, comm );    
      mpi::ExchangeVector(
            w_rskel, 0, 0, 
            w_lskel, 0, 0, comm, &status );

      /** resize l_rskel to properly set m and n */
      w_lskel.resize( sl, nrhs );

      /** resize l_rskel to properly set m and n */
      //w_lskel.resize( w_lskel.size() / nrhs, nrhs );
      //printf( "global rank %d child_sibling morton %lu w_lskel.size() %lu, m %lu n %lu\n", 
      //    global_rank,
      //    child_sibling->morton,
      //    w_lskel.size(), w_lskel.row(), w_lskel.col() );
			
      /** early return */
      if ( !node->parent || !node->data.isskel ) return;

      /** w_skel is s-by-nrhs, initial values are not important */
      w_skel.resize( skels.size(), nrhs );

      if ( 1 )
      {
        /** w_skel += proj( r ) * w_rskel */
        xgemm
        (
          "No-transpose", "No-transpose",
          w_skel.row(), nrhs, sr,
          1.0,    proj.data() + proj.row() * sl, proj.row(),
               w_rskel.data(), sr,
          0.0,  w_skel.data(), w_skel.row()
        );
      }
      else
      {
        /** create a transpose view proj_v */
        View<T> P( false,   proj ), PL,
                                    PR;
        View<T> W( false, w_skel ), //WL( false, w_lskel ),
                                    WR( false, w_rskel );
        /** P = [ PL, PR ] */
        P.Partition1x2( PL, PR, sl, LEFT );
        /** W += PR * WR */
        gemm::xgemm<GEMM_NB>( (T)1.0, PR, WR, (T)0.0, W );
        //W.DependencyCleanUp();
      }

			Data<T> w_skel_sib;
			mpi::ExchangeVector(
					w_skel,     0, 0, 
					w_skel_sib, 0, 0, comm, &status );
			/** clean_up */
			//w_skel.resize( 0, 0 );
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


      auto &gids = arg->gids;
      auto &skels = arg->data.skels;
      auto &w = *arg->setup->w;


			if ( !arg->child )
			{
        if ( arg->isleaf )
        {
          auto m = skels.size();
          auto n = w.col();
          auto k = gids.size();
          flops = 2.0 * m * n * k;
          mops = 2.0 * ( m * n + m * k + k * n );
        }
        else
        {
          auto &lskels = arg->lchild->data.skels;
          auto &rskels = arg->rchild->data.skels;
          auto m = skels.size();
          auto n = w.col();
          auto k = lskels.size() + rskels.size();
          flops = 2.0 * m * n * k;
          mops  = 2.0 * ( m * n + m * k + k * n );
        }
			}
			else
			{
				if ( arg->GetCommRank() == 0 )
				{
          auto &lskels = arg->child->data.skels;
          auto m = skels.size();
          auto n = w.col();
          auto k = lskels.size();
          flops = 2.0 * m * n * k;
          mops = 2.0 * ( m * n + m * k + k * n );
				}
				if ( arg->GetCommRank() == arg->GetCommSize() / 2 )
				{
          auto &rskels = arg->child->data.skels;
          auto m = skels.size();
          auto n = w.col();
          auto k = rskels.size();
          flops = 2.0 * m * n * k;
          mops = 2.0 * ( m * n + m * k + k * n );
				}
			}


      /** setup the event */
      event.Set( label + name, flops, mops );

      /** assume computation bound */
      cost = flops / 1E+9;

      /** "HIGH" priority */
      priority = true;
    };



    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );

      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() < 2 )
        {
          arg->lchild->DependencyAnalysis( RW, this );
          arg->rchild->DependencyAnalysis( RW, this );
        }
        else
        {
          arg->child->DependencyAnalysis( RW, this );
        }
      }

      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };


    void Execute( Worker* user_worker )
    {
      int global_rank = 0;
      mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

      //printf( "rank %d level %lu DistUpdateWeights\n", global_rank, arg->l ); fflush( stdout );

      mpigofmm::DistUpdateWeights<NODE, T>( arg );

      //printf( "rank %d level %lu DistUpdateWeights end\n", global_rank, arg->l ); fflush( stdout );
    };


}; /** end class DistUpdateWeightsTask */




/**
 *
 */ 
template<bool NNPRUNE, typename NODE, typename T>
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



      arg->DependencyAnalysis( RW, this );
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
      mpi::Comm comm = node->GetComm();

      if ( size < 2 )
      {
        gofmm::SkeletonsToSkeletons<NNPRUNE, NODE, T>( node );
      }
      else
      {
        /** only 0th rank (owner) will execute this task */
        if ( rank == 0 )
        {
          gofmm::SkeletonsToSkeletons<NNPRUNE, NODE, T>( node );
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
  mpi::Status status;
  mpi::Comm comm = node->GetComm();
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

		
    //printf( "rank (%d,%d) level %lu DistS2N check1\n", 
		//		rank, global_rank, node->l ); fflush( stdout );
    if ( rank == 0 )
    {
      auto    &proj = data.proj;
      auto  &u_skel = data.u_skel;
      auto &u_lskel = child->data.u_skel;
   
      //size_t sl = u_lskel.row();
      //size_t sr = proj.col() - sl;
			size_t sl = child->data.skels.size();
			size_t sr = proj.col() - sl;

      //printf( "sl %lu sr %lu\n", sl, sr ); fflush( stdout );


      mpi::SendVector( u_skel, size / 2, 0, comm );


      /** resize */
      //u_rskel.resize( sr, nrhs, 0.0 );

      if ( 1 )
      {
        xgemm
        (
          "T", "N",
          u_lskel.row(), u_lskel.col(), proj.row(),
          1.0, proj.data(),    proj.row(),
               u_skel.data(),  u_skel.row(),
          1.0, u_lskel.data(), u_lskel.row()
        );

        //xgemm
        //(
        //  "T", "N",
        //  u_rskel.row(), u_rskel.col(), proj.row(),
        //  1.0, proj.data() + proj.row() * sl, proj.row(),
        //       u_skel.data(), u_skel.row(),
        //  1.0, u_rskel.data(), u_rskel.row()
        //);
      }
      else
      {
        /** create a transpose view proj_v */
        View<T> P(  true,   proj ), PL,
                                    PR;
        View<T> U( false, u_skel ), UL( false, u_lskel );
                                    //UR( false, u_rskel );
        /** P' = [ PL, PR ]' */
        P.Partition2x1( PL,
                        PR, sl, TOP );
        /** UL += PL' * U */
        gemm::xgemm<GEMM_NB>( (T)1.0, PL, U, (T)1.0, UL );
        /** UR += PR' * U */
        //gemm::xgemm<GEMM_NB>( (T)1.0, PR, U, (T)1.0, UR );
      }

      //mpi::SendVector( u_rskel, size / 2, 0, comm );


    }
    //printf( "rank (%d,%d) level %lu DistS2N check2\n", 
		//		rank, global_rank, node->l ); fflush( stdout );

    /**  */
    if ( rank == size / 2 )
    {
      auto    &proj = data.proj;
      auto  &u_skel = data.u_skel;
      auto &u_rskel = child->data.u_skel;

			size_t sr = child->data.skels.size();
      size_t sl = proj.col() - sr;

      //printf( "sl %lu sr %lu\n", sl, sr ); fflush( stdout );

      /** */
			//Data<T> u_skel_sib;
      mpi::RecvVector( u_skel, 0, 0, comm, &status );			
			u_skel.resize( u_skel.size() / nrhs, nrhs );


      // hmlp::Data<T> u_rskel;
      //mpi::RecvVector( u_rskel, 0, 0, comm, &status );
      /** resize to properly set m and n */
      //u_rskel.resize( u_rskel.size() / nrhs, nrhs );

      //auto &u_rskel = child->data.u_skel;


      //for ( size_t j = 0; j < u_rskel.col(); j ++ )
      //  for ( size_t i = 0; i < u_rskel.row(); i ++ )
      //    child->data.u_skel( i, j ) += u_rskel( i, j );


      if ( 1 )
      {
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
        View<T> U( false, u_skel ), //UL( false, u_lskel ),
                                    UR( false, u_rskel );
        /** P' = [ PL, PR ]' */
        P.Partition2x1( PL,
                        PR, sl, TOP );
        /** UR += PR' * U */
        gemm::xgemm<GEMM_NB>( (T)1.0, PR, U, (T)1.0, UR );
      }
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

      auto &gids = arg->gids;
      auto &skels = arg->data.skels;
      auto &w = *arg->setup->w;

			if ( !arg->child )
			{
        if ( arg->isleaf )
        {
          auto m = skels.size();
          auto n = w.col();
          auto k = gids.size();
          flops = 2.0 * m * n * k;
          mops = 2.0 * ( m * n + m * k + k * n );
        }
        else
        {
          auto &lskels = arg->lchild->data.skels;
          auto &rskels = arg->rchild->data.skels;
          auto m = skels.size();
          auto n = w.col();
          auto k = lskels.size() + rskels.size();
          flops = 2.0 * m * n * k;
          mops  = 2.0 * ( m * n + m * k + k * n );
        }
			}
			else
			{
				if ( arg->GetCommRank() == 0 )
				{
          auto &lskels = arg->child->data.skels;
          auto m = skels.size();
          auto n = w.col();
          auto k = lskels.size();
          flops = 2.0 * m * n * k;
          mops = 2.0 * ( m * n + m * k + k * n );
				}
				if ( arg->GetCommRank() == arg->GetCommSize() / 2 )
				{
          auto &rskels = arg->child->data.skels;
          auto m = skels.size();
          auto n = w.col();
          auto k = rskels.size();
          flops = 2.0 * m * n * k;
          mops = 2.0 * ( m * n + m * k + k * n );
				}
			}







      /** setup the event */
      event.Set( label + name, flops, mops );

      /** asuume computation bound */
      cost = flops / 1E+9;

      /** "HIGH" priority */
      priority = true;
    };


    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( R, this );
      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() > 1 )
        {
          arg->child->DependencyAnalysis( RW, this );
        }
        else
        {
          arg->lchild->DependencyAnalysis( RW, this );
          arg->rchild->DependencyAnalysis( RW, this );
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

  //auto &gids = node->gids;
  auto &data = node->data;
  //auto &amap = node->gids;
  auto &KIJ = data.NearKab;
  size_t nrhs = w.col();

  auto *NearNodes = &node->NearNodes;
  if ( NNPRUNE ) NearNodes = &node->NNNearNodes;

  /** Get U view of this treenode. */
  View<T> &U = node->data.u_view;

  if ( KIJ.HasIllegalValue() ) 
  {
    printf( "KIJ has illegal value in DistLeavesToLeaves\n" ); fflush( stdout );
  }


  if ( KIJ.size() ) /** Kab is cached */
  {
    size_t offset = 0;
    for ( auto it  = NearNodes->begin(); 
               it != NearNodes->end(); it ++ )
    {
      /** Get W view of this treenode. (available for non-LET nodes) */
      View<T> &W = (*it)->data.w_view;

      if ( W.col() == nrhs )
      {
        //printf( "%8lu l2l %8lu U[%lu %lu %lu] W[%lu %lu %lu] KIJ[%lu %lu]\n",
        //    node->morton, (*it)->morton,
        //    U.row(), U.col(), U.ld(),
        //    W.row(), W.col(), W.ld(),
        //    KIJ.row(), KIJ.col() );
        xgemm
        (
          "N", "N", U.row(), U.col(), W.row(),
          1.0, KIJ.data() + offset * KIJ.row(), KIJ.row(),
                 W.data(),                        W.ld(),
          1.0,   U.data(),                        U.ld()
        );
      }
      else
      {
        /** LET nodes have w_leaf stored in Data<T>. */
        auto &w_leaf = (*it)->data.w_leaf;
        assert( w_leaf.col() == nrhs );

        for ( int i = 0; i < (*it)->gids.size(); i ++ )
        {
          if ( w_leaf[ i ] != (*it)->gids[ i ] )
          {
            printf( "%8lu l2l %8lu, w_leaf[ %d ] %E != %lu\n",
            node->morton, (*it)->morton, i, 
            w_leaf[ i ], (*it)->gids[ i ] ); fflush( stdout );
            break;
          }
        }

        //printf( "%8lu l2l %8lu U[%lu %lu %lu] w_leaf[%lu %lu] KIJ[%lu %lu]\n",
        //    node->morton, (*it)->morton,
        //    U.row(), U.col(), U.ld(),
        //    w_leaf.row(), w_leaf.col(),
        //    KIJ.row(), KIJ.col() );

        xgemm
        (
          "N", "N", U.row(), U.col(), w_leaf.row(),
          1.0, KIJ.data() + offset * KIJ.row(), KIJ.row(),
               w_leaf.data(),                w_leaf.row(),
          1.0,   U.data(),                        U.ld()
        );
      }

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


      auto &gids = arg->gids;
      auto &skels = arg->data.skels;
      auto &w = *arg->setup->w;

      size_t m = gids.size();
      size_t n = w.col();

      std::set<NODE*> *NearNodes;
      if ( NNPRUNE ) NearNodes = &arg->NNNearNodes;
      else           NearNodes = &arg->NearNodes;

      for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
      {
        size_t k = (*it)->gids.size();
        flops += 2.0 * m * n * k;
        mops += m * k;
        mops += 2.0 * ( m * n + n * k + m * k );
      }


      /** setup the event */
      event.Set( label + name, flops, mops );

      /** asuume computation bound */
      cost = flops / 1E+9;

      /** "LOW" priority */
      priority = false;
    };


    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
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








template<size_t LEVELOFFSET=4, typename NODE>
void FindFarNodes( size_t morton, size_t l, NODE *target )
{
  /**
   *  Return while reaching the leaf level
   */ 
  if ( l > target->l ) return;

  /** compute the correct shift*/
  size_t shift = ( 1 << LEVELOFFSET ) - l + LEVELOFFSET;
  /** set the node Morton ID */
  size_t node_morton = ( morton << shift ) + l;

  bool prunable = true;
  auto & NearMortonIDs = target->NNNearNodeMortonIDs;

  for ( auto it = NearMortonIDs.begin(); it != NearMortonIDs.end(); it ++ )
  {
    if ( tree::IsMyParent( *it, node_morton ) ) prunable = false;
  }

  if ( prunable )
  {
    if ( node_morton < target->morton )
    {
    }
    else
    {
      /**
       *  If node with node_morton doesn't exist, then allocate
       */
      if ( !target->morton2node->count( node_morton ) )
      {
        target->treelock->Acquire();
        {
          if ( !target->morton2node->count( node_morton ) )
          {
            (*target->morton2node)[ node_morton ] = new NODE( node_morton );
            //printf( "%8lu creates far LET %8lu level %lu\n", target->morton, node_morton, l );
          }
        }
        target->treelock->Release();
      }
      target->NNFarNodes.insert( (*target->morton2node)[ node_morton ] );
      target->NNFarNodeMortonIDs.insert( node_morton );
    }
  }
  else
  {
    /**
     *  Recurs if not yet reaching leaf level
     */
    if ( l < target->l )
    {
      FindFarNodes<LEVELOFFSET>( ( morton << 1 ) + 0, l + 1, target );
      FindFarNodes<LEVELOFFSET>( ( morton << 1 ) + 1, l + 1, target );
    }
  }

}; /** end FindFarNodes() */





template<typename NODE, typename TREE>
void SymmetrizeNearInteractions( TREE & tree )
{
  /**
   *  MPI
   */
  int comm_size; mpi::Comm_size( tree.comm, &comm_size );
  int comm_rank; mpi::Comm_rank( tree.comm, &comm_rank );

  vector<vector<pair<size_t, size_t>>> sendlist( comm_size );
  vector<vector<pair<size_t, size_t>>> recvlist( comm_size );


  /**
   *  Traverse local leaf nodes:
   *
   *  Loop over all near node MortonIDs, create
   *
   */ 
  int n_nodes = 1 << tree.depth;
  auto level_beg = tree.treelist.begin() + n_nodes - 1;

  #pragma omp parallel
  {
    /**
     *  Create a per thread list. Merge them into sendlist afterward.
     */ 
    vector<vector<pair<size_t, size_t>>> list( comm_size );

    #pragma omp for
    for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
    {
      auto *node = *(level_beg + node_ind);
      auto & NearMortonIDs = node->NNNearNodeMortonIDs;
      for ( auto it = NearMortonIDs.begin(); it != NearMortonIDs.end(); it ++ )
      {
        int dest = tree.Morton2Rank( *it );
        if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
        list[ tree.Morton2Rank( *it ) ].push_back( make_pair( *it, node->morton ) );
      }
    } /** end pramga omp for */

    #pragma omp critical
    {
      for ( int p = 0; p < comm_size; p ++ )
        sendlist[ p ].insert( sendlist[ p ].end(), list[ p ].begin(), list[ p ].end() );
    } /** end pragma omp critical*/

  }; /** end pargma omp parallel */

  mpi::Barrier( tree.comm );
  //printf( "rank %d finish first part\n", comm_rank ); fflush( stdout );



  /**
   *  Alltoallv
   */
  mpi::AlltoallVector( sendlist, recvlist, tree.comm );


  //printf( "rank %d finish AlltoallVector\n", comm_rank ); fflush( stdout );

  /**
   *  Loop over queries
   */
  for ( int p = 0; p < comm_size; p ++ )
  {
    #pragma omp parallel for
    for ( size_t i = 0; i < recvlist[ p ].size(); i ++ )
    {
      auto& query = recvlist[ p ][ i ];
      auto* node = tree.morton2node[ query.first ];
      
      /**
       *  Check if query node is allocated?
       */ 
      if ( !tree.morton2node.count( query.second ) )
      {
        #pragma omp critical
        {
          if ( !tree.morton2node.count( query.second ) )
          {
            tree.morton2node[ query.second ] = new NODE( query.second );
            //printf( "rank %d, %8lu level %lu creates near LET %8lu (symmetrize)\n", 
            //    comm_rank, node->morton, node->l, query.second );
          }
        }
      }
      
      node->data.lock.Acquire();
      {
        node->NNNearNodes.insert( tree.morton2node[ query.second ] );
        node->NNNearNodeMortonIDs.insert( query.second );
      }
      node->data.lock.Release();

    }; /** end pargma omp parallel for */
  }

  /** 
   *  Build near interaction list for each MPI rank
   */



}; /** end SymmetrizeNearInteractions() */


template<typename NODE, typename TREE>
void SymmetrizeFarInteractions( TREE & tree )
{
  /**
   *  MPI
   */
  int comm_size;
  int comm_rank;
  mpi::Comm_size( tree.comm, &comm_size );
  mpi::Comm_rank( tree.comm, &comm_rank );

  vector<vector<pair<size_t, size_t>>> sendlist( comm_size );
  vector<vector<pair<size_t, size_t>>> recvlist( comm_size );


  /**
   *  Local traversal
   */ 
  #pragma omp parallel 
  {
    /**
     *  Create a per thread list. Merge them into sendlist afterward.
     */ 
    vector<vector<pair<size_t, size_t>>> list( comm_size );

    #pragma omp for
    for ( size_t i = 1; i < tree.treelist.size(); i ++ )
    {
      auto *node = tree.treelist[ i ];
      for ( auto it  = node->NNFarNodeMortonIDs.begin();
                 it != node->NNFarNodeMortonIDs.end(); it ++ )
      {
        int dest = tree.Morton2Rank( *it );
        if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
        list[ dest ].push_back( make_pair( *it, node->morton ) );
      }
    }

    #pragma omp critical
    {
      for ( int p = 0; p < comm_size; p ++ )
        sendlist[ p ].insert( sendlist[ p ].end(), list[ p ].begin(), list[ p ].end() );
    } /** end pragma omp critical*/
  }


  /**
   *  Distributed traversal
   */ 
  #pragma omp parallel 
  {
    /**
     *  Create a per thread list. Merge them into sendlist afterward.
     */ 
    vector<vector<pair<size_t, size_t>>> list( comm_size );

    #pragma omp for
    for ( size_t i = 0; i < tree.mpitreelists.size(); i ++ )
    {
      auto *node = tree.mpitreelists[ i ];
      for ( auto it  = node->NNFarNodeMortonIDs.begin();
                 it != node->NNFarNodeMortonIDs.end(); it ++ )
      {
        int dest = tree.Morton2Rank( *it );
        //if ( dest >= comm_size ) 
          printf( "%8lu dest %d\n", *it, dest ); fflush( stdout );
        list[ dest ].push_back( make_pair( *it, node->morton ) );
      }
    }

    #pragma omp critical
    {
      for ( int p = 0; p < comm_size; p ++ )
        sendlist[ p ].insert( sendlist[ p ].end(), list[ p ].begin(), list[ p ].end() );
    } /** end pragma omp critical*/
  }

  /**
   *  Alltoallv
   */
  mpi::AlltoallVector( sendlist, recvlist, tree.comm );

  /**
   *  Loop over queries
   */
  for ( int p = 0; p < comm_size; p ++ )
  {
    #pragma omp parallel for
    for ( size_t i = 0; i < recvlist[ p ].size(); i ++ )
    {
      auto& query = recvlist[ p ][ i ];
      auto* node = tree.morton2node[ query.first ];
     
      assert( tree.Morton2Rank( node->morton ) == comm_rank );

      /**
       *  Check if query node is allocated?
       */ 
      if ( !tree.morton2node.count( query.second ) )
      {
        #pragma omp critical
        {
          if ( !tree.morton2node.count( query.second ) )
          {
            tree.morton2node[ query.second ] = new NODE( query.second );
            //printf( "rank %d, %8lu level %lu creates far LET %8lu (symmetrize)\n", 
            //    comm_rank, node->morton, node->l, query.second );
          }
        }
      }
      
      node->data.lock.Acquire();
      {
        node->NNFarNodes.insert( tree.morton2node[ query.second ] );
        node->NNFarNodeMortonIDs.insert( query.second );
      }
      node->data.lock.Release();
    }; /** end pargma omp parallel for */
  }

}; /** end SymmetrizeFarInteractions() */



/**
 *  TODO: need send and recv interaction lists for each rank
 *
 *  SendNNNear[ rank ][ local  morton ]
 *  RecvNNNear[ rank ][ remote morton ]
 *
 *  for each leaf alpha and beta in Near(alpha)
 *    SendNNNear[ rank(beta) ] += Morton(alpha)
 *
 *  Alltoallv( SendNNNear, rbuff );
 *
 *  for each rank
 *    RecvNNNear[ rank ][ remote morton ] = offset in rbuff
 *
 */ 
template<typename TREE>
void BuildInteractionListPerRank( TREE &tree, bool is_near )
{
  /** MPI */
  int comm_size; mpi::Comm_size( tree.comm, &comm_size );
  int comm_rank; mpi::Comm_rank( tree.comm, &comm_rank );

  /** Interaction set per rank in MortonID */
  vector<set<size_t>> lists( comm_size );

  if ( is_near )
  {
    /** Traverse leaf nodes (near interation lists) */ 
    int n_nodes = 1 << tree.depth;
    auto level_beg = tree.treelist.begin() + n_nodes - 1;

    #pragma omp parallel
    {
      /** Create a per thread list. Merge them into sendlist afterward. */
      vector<set<size_t>> list( comm_size );

      #pragma omp for
      for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
      {
        auto *node = *(level_beg + node_ind);
        auto & NearMortonIDs = node->NNNearNodeMortonIDs;
        for ( auto it = NearMortonIDs.begin(); it != NearMortonIDs.end(); it ++ )
        {
          int dest = tree.Morton2Rank( *it );
          if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
          if ( dest != comm_rank ) list[ dest ].insert( node->morton );
        }
      } /** end pramga omp for */

      #pragma omp critical
      {
        for ( int p = 0; p < comm_size; p ++ )
          lists[ p ].insert( list[ p ].begin(), list[ p ].end() );
      } /** end pragma omp critical*/

    }; /** end pargma omp parallel */

    /** TODO: can be removed */
    mpi::Barrier( tree.comm );
    printf( "rank %d finish first part\n", comm_rank ); fflush( stdout );

    /** Cast set to vector */
    vector<vector<size_t>> recvlist( comm_size );
    if ( !tree.NearSentToRank.size() ) tree.NearSentToRank.resize( comm_size );
    if ( !tree.NearRecvFromRank.size() ) tree.NearRecvFromRank.resize( comm_size );
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
    {
      tree.NearSentToRank[ p ].insert( tree.NearSentToRank[ p ].end(), 
          lists[ p ].begin(), lists[ p ].end() );
    }

    //for ( int p = 0; p < comm_size; p ++ )
    //{
    //  if ( tree.NearSentToRank[ p ].size() )
    //  {
    //    printf( "Near %d to %d, ", comm_rank, p ); fflush( stdout );
    //    for ( int i = 0; i < tree.NearSentToRank[ p ].size(); i ++ )
    //      printf( "%8lu ", tree.NearSentToRank[ p ][ i ] ); fflush( stdout );
    //    printf( "\n" ); fflush( stdout );
    //  }
    //}

    /** Use buffer recvlist to catch Alltoallv results */
    mpi::AlltoallVector( tree.NearSentToRank, recvlist, tree.comm );

    //for ( int p = 0; p < comm_size; p ++ )
    //{
    //  if ( recvlist[ p ].size() )
    //  {
    //    printf( "Near %d fr %d, ", comm_rank, p ); fflush( stdout );
    //    for ( int i = 0; i < recvlist[ p ].size(); i ++ )
    //      printf( "%8lu ", recvlist[ p ][ i ] ); fflush( stdout );
    //    printf( "\n" ); fflush( stdout );
    //  }
    //}

    /** Cast vector of vectors to vector of maps */
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
      for ( int i = 0; i < recvlist[ p ].size(); i ++ )
        tree.NearRecvFromRank[ p ][ recvlist[ p ][ i ] ] = i;
  }
  else
  {
    #pragma omp parallel 
    {
      /** Create a per thread list. Merge them into sendlist afterward. */
      vector<set<size_t>> list( comm_size );

      /** Local traversal */
      #pragma omp for
      for ( size_t i = 1; i < tree.treelist.size(); i ++ )
      {
        auto *node = tree.treelist[ i ];
        for ( auto it  = node->NNFarNodeMortonIDs.begin();
                   it != node->NNFarNodeMortonIDs.end(); it ++ )
        {
          int dest = tree.Morton2Rank( *it );
          if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
          if ( dest != comm_rank ) list[ dest ].insert( node->morton );
        }
      }

      /** Distributed traversal */
      #pragma omp for
      for ( size_t i = 0; i < tree.mpitreelists.size(); i ++ )
      {
        auto *node = tree.mpitreelists[ i ];
        /** Add to the list iff this MPI rank owns the distributed node */
        if ( tree.Morton2Rank( node->morton ) == comm_rank )
        {
          for ( auto it  = node->NNFarNodeMortonIDs.begin();
                     it != node->NNFarNodeMortonIDs.end(); it ++ )
          {
            int dest = tree.Morton2Rank( *it );
            if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
            if ( dest != comm_rank ) list[ dest ].insert( node->morton );
          }
        }
      }
      /** Merge lists from all threads */
      #pragma omp critical
      {
        for ( int p = 0; p < comm_size; p ++ )
          lists[ p ].insert( list[ p ].begin(), list[ p ].end() );
      } /** end pragma omp critical*/

    }; /** end pargma omp parallel */

    /** Cast set to vector */
    vector<vector<size_t>> recvlist( comm_size );
    if ( !tree.FarSentToRank.size() ) tree.FarSentToRank.resize( comm_size );
    if ( !tree.FarRecvFromRank.size() ) tree.FarRecvFromRank.resize( comm_size );
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
    {
      tree.FarSentToRank[ p ].insert( tree.FarSentToRank[ p ].end(), 
          lists[ p ].begin(), lists[ p ].end() );
    }

    //for ( int p = 0; p < comm_size; p ++ )
    //{
    //  if ( tree.FarSentToRank[ p ].size() )
    //  {
    //    printf( "\nFar  %d to %d, ", comm_rank, p ); fflush( stdout );
    //    for ( int i = 0; i < tree.FarSentToRank[ p ].size(); i ++ )
    //      printf( "%8lu ", tree.FarSentToRank[ p ][ i ] ); fflush( stdout );
    //    printf( "\n" ); fflush( stdout );
    //  }
    //}

    /** Use buffer recvlist to catch Alltoallv results */
    mpi::AlltoallVector( tree.FarSentToRank, recvlist, tree.comm );

    //for ( int p = 0; p < comm_size; p ++ )
    //{
    //  if ( recvlist[ p ].size() )
    //  {
    //    printf( "\nFar  %d fr %d, ", comm_rank, p ); fflush( stdout );
    //    for ( int i = 0; i < recvlist[ p ].size(); i ++ )
    //      printf( "%8lu ", recvlist[ p ][ i ] ); fflush( stdout );
    //    printf( "\n" ); fflush( stdout );
    //  }
    //}

    /** Cast vector of vectors to vector of maps */
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
      for ( int i = 0; i < recvlist[ p ].size(); i ++ )
        tree.FarRecvFromRank[ p ][ recvlist[ p ][ i ] ] = i;
  }

}; /** BuildInteractionListPerRank() */









/**
 *  Send my skeletons (in gids and params) to other ranks
 *  using FarSentToRank[:].
 *
 *  Recv skeletons from other ranks 
 *  using FarRecvFromRank[:].
 */ 
template<typename T, typename TREE>
void ExchangeLET( TREE &tree, string option )
{
  /** MPI */
  int comm_size; mpi::Comm_size( tree.comm, &comm_size );
  int comm_rank; mpi::Comm_rank( tree.comm, &comm_rank );

  /** Buffers for sizes and skeletons */
  vector<vector<size_t>> sendsizes( comm_size );
  vector<vector<size_t>> recvsizes( comm_size );
  vector<vector<size_t>> sendskels( comm_size );
  vector<vector<size_t>> recvskels( comm_size );
  vector<vector<T>>      sendbuffs( comm_size );
  vector<vector<T>>      recvbuffs( comm_size );
   

  if ( !option.compare( 0, 4, "leaf" ) )
  {
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
    {
      for ( auto it  = tree.NearSentToRank[ p ].begin();
                 it != tree.NearSentToRank[ p ].end();
                 it ++ )
      {
        auto *node = tree.morton2node[ *it ];
        auto &gids = node->gids;
        if ( !option.compare( string( "leafgids" ) ) )
        {
          sendsizes[ p ].push_back( gids.size() );
          sendskels[ p ].insert( sendskels[ p ].end(), gids.begin(), gids.end() );
        }
        else
        {
          auto &w_view = node->data.w_view;
          auto  w_leaf = w_view.toData();
          //printf( "%d send w_leaf to %d [%lu %lu]\n", comm_rank, p, w_leaf.row(), w_leaf.col() );
          sendsizes[ p ].push_back( w_leaf.size() );
          sendbuffs[ p ].insert( sendbuffs[ p ].end(), w_leaf.begin(), w_leaf.end() );
          //printf( "Exchange leafweights not yet implemented\n" );
        }
      }
    } /** end pragma omp parallel for each MPI rank */
  }
  else if ( !option.compare( 0, 4, "skel" ) )
  {
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
    {
      for ( auto it  = tree.FarSentToRank[ p ].begin();
                 it != tree.FarSentToRank[ p ].end();
                 it ++ )
      {
        auto *node = tree.morton2node[ *it ];
        auto &skels = node->data.skels;
        if ( !option.compare( string( "skelgids" ) ) )
        {
          sendsizes[ p ].push_back( skels.size() );
          sendskels[ p ].insert( sendskels[ p ].end(), skels.begin(), skels.end() );
        }
        else
        {
          auto &w_skel = node->data.w_skel;
          //printf( "%d send w_skel (%8lu) to %d [%lu %lu] \n", 
          //    comm_rank, *it, p, w_skel.row(), w_skel.col() ); fflush( stdout );
          //for ( int i = 0; i < 5; i ++ )
          //{
          //   printf( "%E ", w_skel[ i ] ); fflush( stdout );
          //}
          //printf( "\n" ); fflush( stdout );
          sendsizes[ p ].push_back( w_skel.size() );
          sendbuffs[ p ].insert( sendbuffs[ p ].end(), w_skel.begin(), w_skel.end() );
        }
      }
    } /** end pragma omp parallel for each MPI rank */
  }
  else
  {
    printf( "ExchangeLET: option <%s> not available.\n", option.data() );
    exit( 1 );
  }

  /** Alltoallv */
  mpi::AlltoallVector( sendsizes, recvsizes, tree.comm );
  if ( !option.compare( string( "skelgids" ) ) ||
       !option.compare( string( "leafgids" ) ) )
  {
    auto &K = *tree.setup.K;
    mpi::AlltoallVector( sendskels, recvskels, tree.comm );
    K.RequestColumns( recvskels );
  }
  else
  {
    mpi::AlltoallVector( sendbuffs, recvbuffs, tree.comm );
  }

  vector<vector<size_t>> offsets( comm_size );

  #pragma omp parallel for
  for ( int p = 0; p < comm_size; p ++ )
  {
    offsets[ p ].push_back( (size_t)0 );
    for ( auto it  = recvsizes[ p ].begin(); 
               it != recvsizes[ p ].end(); it ++ )
    {
      offsets[ p ].push_back( offsets[ p ].back() + *it );
    }
    /** offsets[ p ] has size recvsizes[ p ].size() + 1 */
  } /** end pragma omp parallel for each MPI rank */


  if ( !option.compare( 0, 4, "leaf" ) )
  {
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
    {
      for ( auto it  = tree.NearRecvFromRank[ p ].begin();
                 it != tree.NearRecvFromRank[ p ].end();
                 it ++ )
      {
        /** Get LET node pointer */
        auto *node = tree.morton2node[ (*it).first ];
        if ( !option.compare( string( "leafgids" ) ) )
        {
          auto &gids = node->gids;
          size_t i = (*it).second;
          gids.reserve( recvsizes[ p ][ i ] );
          for ( auto j = offsets[ p ][ i + 0 ]; 
                     j < offsets[ p ][ i + 1 ]; j ++ )
            gids.push_back( recvskels[ p ][ j ] );
        }
        else
        {
          /** Number of right hand sides */
          size_t nrhs = tree.setup.w->col();
          auto &w_leaf = node->data.w_leaf;
          size_t i = (*it).second;
          w_leaf.resize( recvsizes[ p ][ i ] / nrhs, nrhs );
          //printf( "%d recv w_leaf from %d [%lu %lu]\n", 
          //    comm_rank, p, w_leaf.row(), w_leaf.col() ); fflush( stdout );
          for ( uint64_t j = offsets[ p ][ i + 0 ], jj = 0; 
                         j < offsets[ p ][ i + 1 ]; 
                         j ++,                      jj ++ )
          {
            w_leaf[ jj ] = recvbuffs[ p ][ j ];
          }
        }
      }
    }
  }
  else if ( !option.compare( 0, 4, "skel" ) )
  {
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
    {
      for ( auto it  = tree.FarRecvFromRank[ p ].begin();
                 it != tree.FarRecvFromRank[ p ].end();
                 it ++ )
      {
        /** Get LET node pointer */
        auto *node = tree.morton2node[ (*it).first ];
        if ( !option.compare( string( "skelgids" ) ) )
        {
          auto &skels = node->data.skels;
          size_t i = (*it).second;
          skels.clear();
          skels.reserve( recvsizes[ p ][ i ] );
          for ( auto j = offsets[ p ][ i + 0 ]; 
                     j < offsets[ p ][ i + 1 ]; j ++ )
            skels.push_back( recvskels[ p ][ j ] );
        }
        else
        {
          //printf( "Exchange skelweights not yet implemented\n" );
          /** Number of right hand sides */
          size_t nrhs = tree.setup.w->col();
          auto &w_skel = node->data.w_skel;
          size_t i = (*it).second;
          w_skel.resize( recvsizes[ p ][ i ] / nrhs, nrhs );
          //printf( "%d recv w_skel (%8lu) from %d [%lu %lu], i %lu, offset[%lu %lu] \n", 
          //    comm_rank, (*it).first, p, w_skel.row(), w_skel.col(), i,
          //    offsets[ p ][ i + 0 ], offsets[ p ][ i + 1 ] ); fflush( stdout );
          for ( uint64_t j = offsets[ p ][ i + 0 ], jj = 0; 
                         j < offsets[ p ][ i + 1 ]; 
                         j ++,                      jj ++ )
          {
            w_skel[ jj ] = recvbuffs[ p ][ j ];
            //if ( jj < 5 ) printf( "%E ", w_skel[ jj ] ); fflush( stdout );
          }
          //printf( "\n" ); fflush( stdout );
        }
      }
    } /** end pragma omp parallel for each MPI rank */
  }
  else
  {
    printf( "ExchangeLET: option <%s> not available.\n", option.data() );
    exit( 1 );
  }

}; /** ExchangeSkeletons() */


template<typename T, typename TREE>
void ExchangeNeighbors( TREE &tree )
{
  int comm_rank; mpi::Comm_rank( tree.comm, &comm_rank );
  int comm_size; mpi::Comm_size( tree.comm, &comm_size );

  /** Alltoallv buffers */
  vector<vector<size_t>> send_buff( comm_size );
  vector<vector<size_t>> recv_buff( comm_size );

  /** NN<STAR, CIDS, pair<T, size_t>> */
  unordered_set<size_t> requested_gids;
  auto &NN = *tree.setup.NN;

  /** Remove duplication */
  for ( auto it = NN.begin(); it != NN.end(); it ++ )
  {
    requested_gids.insert( (*it).second );
  }

  /** Remove owned gids */
  for ( auto it  = tree.treelist[ 0 ]->gids.begin();
             it != tree.treelist[ 0 ]->gids.end(); it ++ )
  {
    requested_gids.erase( *it );
  }

  /** Assume gid is owned by (gid % size) */
  for ( auto it  = requested_gids.begin();
             it != requested_gids.end(); it ++ )
  {
    int p = *it % comm_size;
    if ( p != comm_rank ) send_buff[ p ].push_back( *it );
  }

  /** Redistribute K */
  auto &K = *tree.setup.K;
  K.RequestColumns( send_buff );

};









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
//template<bool SYMMETRIC, typename NODE, size_t LEVELOFFSET = 4>
//void FindFarNodes(
//  size_t level, size_t max_depth,
//  size_t recurs_morton, NODE *target )
//{
//  size_t shift = ( 1 << LEVELOFFSET ) - level + LEVELOFFSET;
//  size_t source_morton = ( recurs_morton << shift ) + level;
//
//  if ( hmlp::tree::ContainAnyMortonID( target->NearNodeMortonIDs, source_morton ) )
//  {
//    if ( level < max_depth )
//    {
//      /** recurse to two children */
//      FindFarNodes<SYMMETRIC>( 
//          level + 1, max_depth,
//          ( recurs_morton << 1 ) + 0, target );
//      FindFarNodes( 
//          level + 1, max_depth,
//          ( recurs_morton << 1 ) + 1, target );
//    }
//  }
//  else
//  {
//    /** create LETNode, insert ``source'' to Far( target ) */
//    //auto *letnode = new hmlp::mpitree::LETNode( level, source_morton );
//    target->FarNodeMortonIDs.insert( source_morton );
//  }
//
//
//  /**
//   *  case: NNPRUNE
//   *
//   *  Build NNNearNodes for the FMM approximation.
//   *  Near( target ) contains other leaf nodes
//   *  
//   **/
//  if ( hmlp::tree::ContainAnyMortonID( target->NNNearNodeMortonIDs, source_morton ) )
//  {
//    if ( level < max_depth )
//    {
//      /** recurse to two children */
//      FindFarNodes<SYMMETRIC>( 
//          level + 1, max_depth,
//          ( recurs_morton << 1 ) + 0, target );
//      FindFarNodes( 
//          level + 1, max_depth,
//          ( recurs_morton << 1 ) + 1, target );
//    }
//  }
//  else
//  {
//    if ( SYMMETRIC && ( source_morton < target->morton ) )
//    {
//      /** since target->morton is larger than the visiting node,
//       * the interaction between the target and this node has
//       * been computed. 
//       */ 
//    }
//    else
//    {
//      /** create LETNode, insert ``source'' to Far( target ) */
//      //auto *letnode = new hmlp::mpitree::LETNode( level, source_morton );
//      target->NNFarNodeMortonIDs.insert( source_morton );
//    }
//  }
//
//}; /** end FindFarNodes() */



template<bool SYMMETRIC, typename NODE, typename T>
void MergeFarNodes( NODE *node )
{
  /** if I don't have any skeleton, then I'm nobody's far field */
  //if ( !node->data.isskel ) return;

  /**
   *  Examine "Near" interaction list
   */ 
  if ( node->isleaf )
  {
     auto & NearMortonIDs = node->NNNearNodeMortonIDs;
     #pragma omp critical
     {
       int rank;
       mpi::Comm_rank( MPI_COMM_WORLD, &rank );
       string outfile = to_string( rank );
       FILE * pFile = fopen( outfile.data(), "a+" );
       fprintf( pFile, "(%8lu) ", node->morton );
       for ( auto it = NearMortonIDs.begin(); it != NearMortonIDs.end(); it ++ )
         fprintf( pFile, "%8lu, ", (*it) );
       fprintf( pFile, "\n" ); //fflush( stdout );
     }

     //auto & NearNodes = node->NNNearNodes;
     //for ( auto it = NearNodes.begin(); it != NearNodes.end(); it ++ )
     //{
     //  if ( !(*it)->NNNearNodes.count( node ) )
     //  {
     //    printf( "(%8lu) misses %lu\n", (*it)->morton, node->morton ); fflush( stdout );
     //  }
     //}
  };


  /** Add my sibling (in the same level) to far interaction lists */
  assert( !node->FarNodeMortonIDs.size() );
  assert( !node->FarNodes.size() );
  node->FarNodeMortonIDs.insert( node->sibling->morton );
  node->FarNodes.insert( node->sibling );


  /** Construct NN far interaction lists */
  if ( node->isleaf )
  {
    /** Nearest-neighbor pruning */
    FindFarNodes( 0 /** root morton */, 0 /** level 0 */, node );
  }
  else
  {
    /** merge Far( lchild ) and Far( rchild ) from children */
    auto *lchild = node->lchild;
    auto *rchild = node->rchild;

    ///** case: !NNPRUNE (HSS specific) */ 
    //auto &pFarNodes =   node->FarNodeMortonIDs;
    //auto &lFarNodes = lchild->FarNodeMortonIDs;
    //auto &rFarNodes = rchild->FarNodeMortonIDs;

    ///** Far( parent ) = Far( lchild ) intersects Far( rchild ) */
    //for ( auto it = lFarNodes.begin(); it != lFarNodes.end(); ++ it )
    //{
    //  if ( rFarNodes.count( *it ) ) pFarNodes.insert( *it );
    //}
    ///** Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) */
    //for ( auto it = pFarNodes.begin(); it != pFarNodes.end(); it ++ )
    //{
    //  lFarNodes.erase( *it ); rFarNodes.erase( *it );
    //}

    /** case: NNPRUNE (FMM specific) */ 
    auto &pNNFarNodes =   node->NNFarNodeMortonIDs;
    auto &lNNFarNodes = lchild->NNFarNodeMortonIDs;
    auto &rNNFarNodes = rchild->NNFarNodeMortonIDs;

    /** Far( parent ) = Far( lchild ) intersects Far( rchild ) */
    for ( auto it = lNNFarNodes.begin(); it != lNNFarNodes.end(); ++ it )
    {
      if ( rNNFarNodes.count( *it ) ) 
      {
        pNNFarNodes.insert( *it );
        node->NNFarNodes.insert( (*node->morton2node)[ *it ] );
      }
    }
    /** Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) */
    for ( auto it = pNNFarNodes.begin(); it != pNNFarNodes.end(); it ++ )
    {
      lNNFarNodes.erase( *it ); 
      rNNFarNodes.erase( *it );
      lchild->NNFarNodes.erase( (*node->morton2node)[ *it ] );
      rchild->NNFarNodes.erase( (*node->morton2node)[ *it ] );
    }
  }

}; /** end MergeFarNodes() */



template<bool SYMMETRIC, typename NODE, typename T>
class MergeFarNodesTask : public Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      /** this task contains MPI routines */
      //this->has_mpi_routines = true;

      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "merge" );
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
      arg->DependencyAnalysis( R, this );
      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( RW, this );
        arg->rchild->DependencyAnalysis( RW, this );
      }
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      MergeFarNodes<SYMMETRIC, NODE, T>( arg );
    };

}; /** end class MergeFarNodesTask */












template<bool SYMMETRIC, typename NODE, typename T>
void DistMergeFarNodes( NODE *node )
{
  /**
   *  MPI
   */ 
  mpi::Status status;
  mpi::Comm comm = node->GetComm();
  int comm_size = node->GetCommSize();
  int comm_rank = node->GetCommRank();

  /** if I don't have any skeleton, then I'm nobody's far field */
  //if ( !node->data.isskel ) return;


  /**
   *  Early return
   */ 
  if ( !node->parent ) return;

  /** Distributed treenode */
  if ( node->GetCommSize() < 2 )
  {
    MergeFarNodes<SYMMETRIC, NODE, T>( node );
  }
  else
  {
    /** merge Far( lchild ) and Far( rchild ) from children */
    auto *child = node->child;


    if ( comm_rank == 0 )
    {
      auto &pNNFarNodes =  node->NNFarNodeMortonIDs;
      auto &lNNFarNodes = child->NNFarNodeMortonIDs;
      vector<size_t> recvFarNodes;

      /**
       *  Recv rNNFarNodes
       */ 
      mpi::RecvVector( recvFarNodes, comm_size / 2, 0, comm, &status );

      /** Far( parent ) = Far( lchild ) intersects Far( rchild ) */
      for ( auto it = recvFarNodes.begin(); it != recvFarNodes.end(); ++ it )
      {
        if ( lNNFarNodes.count( *it ) ) 
        {
          pNNFarNodes.insert( *it );
          node->NNFarNodes.insert( (*node->morton2node)[ *it ] );
        }
      }

      printf( "%8lu level %lu Far ", node->morton, node->l ); 
      for ( auto it = pNNFarNodes.begin(); it != pNNFarNodes.end(); it ++ )
      {
        printf( "%8lu ", *it ); fflush( stdout );
      }
      printf( "\n" ); fflush( stdout );


      /**
       *  Reuse space to send pNNFarNodes
       */ 
      recvFarNodes.clear();
      recvFarNodes.reserve( pNNFarNodes.size() );

      /** Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) */
      for ( auto it = pNNFarNodes.begin(); it != pNNFarNodes.end(); it ++ )
      {
        lNNFarNodes.erase( *it ); 
        recvFarNodes.push_back( *it );
        child->NNFarNodes.erase( (*node->morton2node)[ *it ] );
      }

      /**
       *  Send pNNFarNodes
       */
      mpi::SendVector( recvFarNodes, comm_size / 2, 0, comm );
    }


    if ( comm_rank == comm_size / 2 )
    {
      auto &rNNFarNodes = child->NNFarNodeMortonIDs;
      vector<size_t> sendFarNodes;

      sendFarNodes.reserve( rNNFarNodes.size() );
      for ( auto it = rNNFarNodes.begin(); it != rNNFarNodes.end(); it ++ )
        sendFarNodes.push_back( *it );

      /**
       *  Send rNNFarNodes
       */ 
      mpi::SendVector( sendFarNodes, 0, 0, comm );

      /**
       *  Recv pNNFarNodes
       */
      mpi::RecvVector( sendFarNodes, 0, 0, comm, &status );

      /** 
       *  Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) 
       */
      for ( auto it = sendFarNodes.begin(); it != sendFarNodes.end(); ++ it )
      {
        rNNFarNodes.erase( *it );
        child->NNFarNodes.erase( (*node->morton2node)[ *it ] );
      }      
    }
  }

}; /** end DistMergeFarNodes() */



template<bool SYMMETRIC, typename NODE, typename T>
class DistMergeFarNodesTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      /** this task contains MPI routines */
      //this->has_mpi_routines = true;

      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "dist-merge" );
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
      arg->DependencyAnalysis( R, this );
      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() > 1 )
        {
          arg->child->DependencyAnalysis( RW, this );
        }
        else
        {
          arg->lchild->DependencyAnalysis( RW, this );
          arg->rchild->DependencyAnalysis( RW, this );
        }
      }
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      DistMergeFarNodes<SYMMETRIC, NODE, T>( arg );
    };

}; /** end class DistMergeFarNodesTask */





///**
// *
// *
// */ 
//template<typename LETNODE, typename NODE>
//void SimpleFarNodes( NODE *node )
//{
//  if ( node->isleaf )
//	{
//    /** add myself to the list. */
//    node->NearNodes.insert( node );
//    node->NNNearNodes.insert( node );
//    node->NearNodeMortonIDs.insert( node->morton );
//    node->NNNearNodeMortonIDs.insert( node->morton );
//	}
//	else
//  {
//    auto *lchild = node->lchild;
//    auto *rchild = node->rchild;
//
//    lchild->FarNodes.insert( rchild );
//    lchild->NNFarNodes.insert( rchild );
//    rchild->FarNodes.insert( lchild );
//    rchild->NNFarNodes.insert( lchild );
//
//    SimpleFarNodes<LETNODE>( lchild );
//    SimpleFarNodes<LETNODE>( rchild );
//  }
//
//  //printf( "%lu NNFarNodes: ", node->treelist_id ); 
//  //for ( auto it  = node->NNFarNodes.begin();
//  //           it != node->NNFarNodes.end(); it ++ )
//  //{
//  //  printf( "%lu ", (*it)->morton ); fflush( stdout );
//  //};
//  //printf( "%lu FarNodes: ", node->treelist_id ); 
//  //for ( auto it  = node->FarNodes.begin();
//  //           it != node->FarNodes.end(); it ++ )
//  //{
//  //  printf( "%lu ", (*it)->morton ); fflush( stdout );
//  //};
//  //printf( "\n" );
//  
//}; /** end SimpleFarNodes() */
//
//
//
///**
// *  TODO: lettreelist is not thread safe.
// *
// */ 
//template<typename LETNODE, typename MPINODE>
//void ParallelSimpleFarNodes( MPINODE *node, 
//    std::map<size_t, LETNODE*> &lettreelist )
//{
//  //printf( "level %lu beg\n", node->l ); fflush( stdout );
//  //hmlp::mpi::Barrier( MPI_COMM_WORLD );
//
//  hmlp::mpi::Status status;
//  hmlp::mpi::Comm comm = node->GetComm();
//  int size = node->GetCommSize();
//  int rank = node->GetCommRank();
//
//
//  //if ( rank == 0 )
//  //{
//  //  printf( "level %lu here\n", node->l ); fflush( stdout );
//  //}
//  //hmlp::mpi::Barrier( MPI_COMM_WORLD );
//
//
//  if ( size < 2 )
//  {
//    /** do not create any LET node */
//    SimpleFarNodes<LETNODE>( node );
//  }
//  else
//  {
//    auto *child = node->child;
//    size_t send_morton = child->morton;
//    size_t recv_morton = 0;
//    LETNODE *letnode = NULL;
//
//    if ( rank == 0 )
//    {
//      mpi::Sendrecv( 
//          &send_morton, 1, size / 2, 0, 
//          &recv_morton, 1, size / 2, 0, comm, &status );
//
//      /** initialize a LetNode with sibling's MortonID */
//      letnode = new LETNODE( node->setup, recv_morton );
//      child->FarNodes.insert( letnode );
//      child->NNFarNodes.insert( letnode );
//      lettreelist[ recv_morton ] = letnode;
//
//      if ( child->isleaf )
//      {
//        /** exchange gids */
//        auto &send_gids = child->gids;
//        auto &recv_gids = letnode->gids;
//        mpi::ExchangeVector(
//            send_gids, size / 2, 0, 
//            recv_gids, size / 2, 0, comm, &status );
//        //printf( "0 exchange gids\n" ); fflush( stdout );
//      }
//
//      /** exchange skels */
//      auto &send_skels = child->data.skels;
//      auto &recv_skels = letnode->data.skels;
//      mpi::ExchangeVector(
//          send_skels, size / 2, 0, 
//          recv_skels, size / 2, 0, comm, &status );
//      /** exchange interpolative coefficients */
//      auto &send_proj = child->data.proj;
//      auto &recv_proj = letnode->data.proj;
//      mpi::ExchangeVector(
//          send_proj, size / 2, 0, 
//          recv_proj, size / 2, 0, comm, &status );
//      //printf( "0 exchange skels and proj\n" ); fflush( stdout );
//    }
//
//    if ( rank == size / 2 )
//    {
//      mpi::Sendrecv( 
//          &send_morton, 1, 0, 0, 
//          &recv_morton, 1, 0, 0, comm, &status );
//
//      /** initialize a LetNode with sibling's MortonID */
//      letnode = new LETNODE( node->setup, recv_morton );
//      child->FarNodes.insert( letnode );
//      child->NNFarNodes.insert( letnode );
//      lettreelist[ recv_morton ] = letnode;
//
//      if ( child->isleaf )
//      {
//        /** exchange gids */
//        auto &send_gids = child->gids;
//        auto &recv_gids = letnode->gids;
//        mpi::ExchangeVector(
//            send_gids, 0, 0, 
//            recv_gids, 0, 0, comm, &status );
//        //printf( "size / 2 exchange gids\n" ); fflush( stdout );
//      }
//
//      /** exchange skels */
//      auto &send_skels = child->data.skels;
//      auto &recv_skels = letnode->data.skels;
//      mpi::ExchangeVector(
//          send_skels, 0, 0, 
//          recv_skels, 0, 0, comm, &status );
//      /** exchange interpolative coefficients */
//      auto &send_proj = child->data.proj;
//      auto &recv_proj = letnode->data.proj;
//      mpi::ExchangeVector(
//          send_proj, 0, 0, 
//          recv_proj, 0, 0, comm, &status );
//      //printf( "size / 2 exchange skels and proj\n" ); fflush( stdout );
//    }
//
//
//    ParallelSimpleFarNodes<LETNODE>( child, lettreelist );
//  }
//
//}; /** end ParallelSimpleFarNodes() */




template<typename LETNODE, typename NODE, typename T>
void DistBuildPool( NODE *node )
{
  /**
   *  MPI support
   */ 
  mpi::Status status;
  mpi::Comm comm = node->GetComm();
  int size = node->GetCommSize();
  int rank = node->GetCommRank();

  auto & setup = *(node->setup);
  auto & candidates = node->data.candidates;


  if ( node->parent )
  {
    /**
     *  We must type cast here
     */ 
    NODE *parent = (NODE*)node->parent;
    mpi::Comm parent_comm = parent->GetComm();
    int parent_size = parent->GetCommSize();
    int parent_rank = parent->GetCommRank();

		size_t send_morton = node->morton;
		size_t recv_morton = 0;

    /**
     *  Exechage MortonIDs using parent's comm
     */ 
    if ( parent_rank == 0 )
    {
      mpi::Sendrecv( 
          &send_morton, 1, parent_size / 2, 0, 
          &recv_morton, 1, parent_size / 2, 0, parent_comm, &status );
    }
    if ( parent_rank == parent_size / 2 )
    {
      mpi::Sendrecv( 
          &send_morton, 1, 0, 0, 
          &recv_morton, 1, 0, 0, parent_comm, &status );
    }

    /**
     *  Bcast received MortonID using my comm
     */
    mpi::Bcast( &recv_morton, 1, 0, comm );
    node->sibling = new LETNODE( node->setup, recv_morton );
    printf( "Create letnode at level %lu\n", node->l ); fflush( stdout );
  }


  /**
   *  This process is a distributed upward traversal. We need
   *  access to the sibling node, which will be created as a
   *  local essential tree node.
   */
  if ( size < 2 )
  {
    /**
     *  If this is the local tree root, then we call the shared memory
     *  BuildPool to merge children's sample pool.
     */ 
    gofmm::BuildPool<NODE, T>( node );
  }
  else
  {
    vector<pair<T,size_t>> merged_candidates;
    auto *child = node->child;
    auto *sibling = node->sibling;

    if ( rank == 0 )
    {
      /** 
       *  Recv NNNearNodeMortonIDs from right child
       */
      vector<size_t> recv_NearMortonIDs;
      mpi::RecvVector( recv_NearMortonIDs, size / 2, 0, comm, &status );

      /**
       *  For non-leaf nodes, we build the near interaction list to filter
       *  the row samples. Rows in the near iteraction list will be sampled.
       */ 
      for ( auto it  = child->NNNearNodeMortonIDs.begin(); 
                 it != child->NNNearNodeMortonIDs.end(); it ++ )
      {
        if ( !tree::IsMyParent( *it, node->morton ) )
        {
          node->NNNearNodeMortonIDs.insert( *it );
        }
      }

      for ( auto it  = recv_NearMortonIDs.begin(); 
                 it != recv_NearMortonIDs.end(); it ++ )
      {
        if ( !tree::IsMyParent( *it, node->morton ) )
        {
          node->NNNearNodeMortonIDs.insert( *it );
        }
      }

      /** 
       *  Recv candidates from right child as an array of pairs.
       */
      vector<pair<size_t, T>> recv_candidates;
      mpi::RecvVector( recv_candidates, size / 2, 0, comm, &status );

      /** 
       *  Merge left and right candidates 
       */
      candidates = child->data.candidates;
      for ( auto it  = recv_candidates.begin(); 
                 it != recv_candidates.end(); it ++ )
      {
        size_t candidate_morton = setup.morton[ (*it).first ];

        if ( candidates.find( candidate_morton ) != candidates.end() )
        {
          /** 
           *  Try to insert. If gid already exist, update the distance.
           */
          auto & existing_candidate = candidates[ candidate_morton ];
          auto ret = existing_candidate.insert( *it );
          if ( ret.second == false )
          {
            if ( ret.first->second > it->second ) 
              ret.first->second = it->second;
          }
        }
        else
        {
          candidates[ candidate_morton ][ (*it).first ] = (*it).second;
        }
      }

      for ( auto it  = node->NNNearNodeMortonIDs.begin(); 
                 it != node->NNNearNodeMortonIDs.end(); it ++ )
      {
        /** 
         *  Remove candidates in the pruning list 
         */
        candidates.erase( *it );
      }

      /**
       *  Building the sample pool
       */ 
      if ( sibling ) 
      {
        for ( auto it = candidates.begin(); it != candidates.end(); it ++ )
        {
          size_t candidate_morton = (*it).first;
          auto & candidate_pairs  = (*it).second;
          if ( tree::IsMyParent( candidate_morton, sibling->morton ) )
          { 
            for ( auto nn_it  = candidate_pairs.begin(); 
                       nn_it != candidate_pairs.end(); nn_it ++ )
            {
              //merged_candidates.push_back( *nn_it );
              merged_candidates.push_back( flip_pair( *nn_it ) );
            }
          }
        }
        sort( merged_candidates.begin(), merged_candidates.end() );
        size_t nsamples = 4 * std::max( setup.s, setup.m );
        if ( merged_candidates.size() > nsamples )
          merged_candidates.resize( nsamples );
      }

    } /** end if ( rank == 0 ) */

    /**
     *  Partner rank that holds the right child
     */ 
    if ( rank == size / 2 )
    {
      /** 
       *  Send NNNearNodeMortonIDs to left child
       */
      vector<size_t> send_NearMortonIDs;

      for ( auto it  = child->NNNearNodeMortonIDs.begin(); 
                 it != child->NNNearNodeMortonIDs.end(); it ++ )
      {
        send_NearMortonIDs.push_back( *it );
      }

      mpi::SendVector( send_NearMortonIDs, 0, 0, comm );

      /** 
       *  Send candidates to left child as an array of pairs.
       */
      vector<pair<size_t, T>> send_candidates;

      for ( auto it  = child->data.candidates.begin();
                 it != child->data.candidates.end(); it ++ )
      {
         auto & existing_candidate = (*it).second;
         for ( auto query_it  = existing_candidate.begin();
                    query_it != existing_candidate.end(); query_it ++ )
         {
           send_candidates.push_back( *query_it );
         }
      }

      mpi::SendVector( send_candidates, 0, 0, comm );

    } /** end if ( rank == size / 2 ) */

    
    /**
     *  Bcast merge_candidates
     */
    printf( "before merged level %lu size %lu\n", node->l, merged_candidates.size() ); fflush( stdout );
    size_t recv_candidate_size = merged_candidates.size();
    mpi::Bcast( &recv_candidate_size, 1, 0, comm );
    merged_candidates.resize( recv_candidate_size );
    mpi::Bcast( merged_candidates.data(), recv_candidate_size, 0, comm );
    printf( "finish merged lebel %lu\n", node->l ); fflush( stdout );

    /**
     *  Bcast merge_candidates and create pool for sibling
     */ 
    if ( sibling ) 
    {
      auto &pool = sibling->data.pool;
      for ( size_t i = 0; i < merged_candidates.size(); i ++ )
      {
        auto new_candidate = flip_pair( merged_candidates[ i ] );
        auto ret = pool.insert( new_candidate );
        if ( ret.second == false )
        {
          if ( ret.first->second > new_candidate.second ) 
            ret.first->second = new_candidate.second;
        }
      }
    }

    /**
     *  We no longer need these candidates in the parent nodes.
     */ 
    child->data.candidates.clear();
  }



}; /** end DistBuildPool() */




template<typename LETNODE, typename NODE, typename T>
class DistBuildPoolTask : public Task
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
      DistBuildPool<LETNODE, NODE, T>( arg );
    };


};




//template<bool SYMMETRIC, int LEVELOFFSET=4, typename NODE>
//void FindFarNodesByMortonIDs( size_t morton, size_t l, NODE *target )
//{
//  /** compute the correct shift*/
//  size_t shift = ( 1 << LEVELOFFSET ) - l + LEVELOFFSET;
//  /** compute node Morton ID */
//  size_t node_morton = ( morton << shift ) + l;
//  /** Check if node_morton is parent of any target->NNNearNodeMortonIDs */
//  auto & NearIDs = target->NNNearNodeMortonIDs;
//
//  bool prunable = true;
//  for ( auto it = NearIDs.begin(); it != NearIDs.end(); it ++ )
//  {
//    if ( IsMyParent( *it, node_morton ) )
//    {
//      prunable = false;
//      break;
//    }
//  }
// 
//  if ( prunable )
//  {
//    if ( SYMMETRIC && ( node_morton < target->morton ) )
//    {
//    }
//    else
//    {
//      target->NNFarNodeMortonIDs.insert( node_morton );
//    }
//  }
//  else
//  {
//    /**
//     *  Recurs to l + 1 levels
//     */ 
//    if ( l <= target->l )
//    {
//      FindFarNodesByMortonIDs( ( morton << 1 ) + 0, l + 1, target );
//      FindFarNodesByMortonIDs( ( morton << 1 ) + 1, l + 1, target );
//    }
//  }
//
//}; /** end FindFarNodesByMortonIDs() */















//template<typename NODE>
//class SimpleNearFarNodesTask : public hmlp::Task
//{
//  public:
//
//    NODE *arg;
//
//    void Set( NODE *user_arg )
//    {
//      /** this task contains MPI routines */
//      //this->has_mpi_routines = true;
//
//      std::ostringstream ss;
//      arg = user_arg;
//      name = std::string( "cachef" );
//      //label = std::to_string( arg->treelist_id );
//      ss << arg->treelist_id;
//      label = ss.str();
//
//      /** we don't know the exact cost here */
//      cost = 5.0;
//
//      /** high priority */
//      priority = true;
//    };
//
//    void DependencyAnalysis()
//    {
//      if ( arg->parent )
//				arg->parent->DependencyAnalysis( hmlp::ReadWriteType::R, this );
//      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
//      this->TryEnqueue();
//    };
//
//    void Execute( Worker* user_worker )
//    {
//			auto *node = arg;
//
//			if ( node->isleaf )
//			{
//				/** add myself to the list. */
//				node->NearNodes.insert( node );
//				node->NNNearNodes.insert( node );
//				node->NearNodeMortonIDs.insert( node->morton );
//				node->NNNearNodeMortonIDs.insert( node->morton );
//			}
//
//      if ( node->parent )
//			{
//        if ( node->parent->lchild == arg ) 
//        {
//          arg->FarNodes.insert( node->parent->rchild );
//          arg->NNFarNodes.insert( node->parent->rchild );
//        }
//        else
//        {
//          assert( node->parent->rchild == arg );
//          arg->FarNodes.insert( node->parent->lchild );
//          arg->NNFarNodes.insert( node->parent->lchild );
//        }
//			}
//		};
//
//}; /** end class SimpleNearFarNodesTask */
//
//
//
//template<typename LETNODE, typename NODE>
//class DistSimpleNearFarNodesTask : public hmlp::Task
//{
//  public:
//
//    NODE *arg;
//
//    void Set( NODE *user_arg )
//    {
//      /** this task contains MPI routines */
//      //this->has_mpi_routines = true;
//
//      std::ostringstream ss;
//      arg = user_arg;
//      name = std::string( "cachef" );
//      //label = std::to_string( arg->treelist_id );
//      ss << arg->treelist_id;
//      label = ss.str();
//
//      /** we don't know the exact cost here */
//      cost = 5.0;
//
//      /** high priority */
//      priority = true;
//    };
//
//		/** read this node and write to children */
//    void DependencyAnalysis()
//    {
//      arg->DependencyAnalysis( hmlp::ReadWriteType::R, this );
//      if ( !arg->isleaf )
//      {
//        if ( arg->GetCommSize() > 1 )
//        {
//          arg->child->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
//        }
//        else
//        {
//          arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
//          arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
//        }
//      }
//      this->TryEnqueue();
//    };
//
//    void Execute( Worker* user_worker )
//		{
//			auto *node = arg;
//			mpi::Status status;
//			mpi::Comm comm = node->GetComm();
//			int size = node->GetCommSize();
//			int rank = node->GetCommRank();
//
//			if ( size < 2 )
//			{
//			  if ( node->isleaf )
//			  {
//			  	/** add myself to the list. */
//			  	node->NearNodes.insert( node );
//			  	node->NNNearNodes.insert( node );
//			  	node->NearNodeMortonIDs.insert( node->morton );
//			  	node->NNNearNodeMortonIDs.insert( node->morton );
//			  }
//			  //else
//			  //{
//				//  auto *lchild = node->lchild;
//				//  auto *rchild = node->rchild;
//			  //	lchild->FarNodes.insert( rchild );
//			  //	lchild->NNFarNodes.insert( rchild );
//			  //	rchild->FarNodes.insert( lchild );
//			  //	rchild->NNFarNodes.insert( lchild );
//			  //}
//			}
//			else
//			{
//				auto *child = node->child;
//				size_t send_morton = child->morton;
//				size_t recv_morton = 0;
//				//LETNODE *letnode = NULL;
//
//				if ( node->parent )
//				{
//					size_t proj_m = node->data.proj.row();
//					size_t proj_n = node->data.proj.col();
//					mpi::Bcast( &proj_m, 1, 0, comm );
//					mpi::Bcast( &proj_n, 1, 0, comm );
//					if ( node->data.proj.size() != proj_m * proj_n )
//					  node->data.proj.resize( proj_m, proj_n );
//					mpi::Bcast( node->data.proj.data(), proj_m * proj_n, 0, comm );
//				}
//        else
//        {
//          /** Do nothing at root */
//          //return;
//        }
//
//				if ( rank == 0 )
//				{
//					mpi::Sendrecv( 
//							&send_morton, 1, size / 2, 0, 
//							&recv_morton, 1, size / 2, 0, comm, &status );
//
//					/** initialize a LetNode with sibling's MortonID */
//					//letnode = new LETNODE( node->setup, recv_morton );
//					auto *letnode = child->sibling;
//          assert( letnode );
//					child->FarNodes.insert( letnode );
//					child->NNFarNodes.insert( letnode );
//
//					if ( child->isleaf )
//					{
//						/** exchange gids */
//						auto &send_gids = child->gids;
//						auto &recv_gids = letnode->gids;
//						mpi::ExchangeVector(
//								send_gids, size / 2, 0, 
//								recv_gids, size / 2, 0, comm, &status );
//						//printf( "0 exchange gids\n" ); fflush( stdout );
//					}
//
//					/** exchange skels */
//					auto &send_skels = child->data.skels;
//					auto &recv_skels = letnode->data.skels;
//					mpi::ExchangeVector(
//							send_skels, size / 2, 0, 
//							recv_skels, size / 2, 0, comm, &status );
//					/** exchange interpolative coefficients */
//					auto &send_proj = child->data.proj;
//					auto &recv_proj = letnode->data.proj;
//					mpi::ExchangeVector(
//							send_proj, size / 2, 0, 
//							recv_proj, size / 2, 0, comm, &status );
//				}
//
//				if ( rank == size / 2 )
//				{
//					mpi::Sendrecv( 
//							&send_morton, 1, 0, 0, 
//							&recv_morton, 1, 0, 0, comm, &status );
//
//					/** initialize a LetNode with sibling's MortonID */
//					//letnode = new LETNODE( node->setup, recv_morton );
//					auto *letnode = child->sibling;
//          assert( letnode );
//					child->FarNodes.insert( letnode );
//					child->NNFarNodes.insert( letnode );
//
//					if ( child->isleaf )
//					{
//						/** exchange gids */
//						auto &send_gids = child->gids;
//						auto &recv_gids = letnode->gids;
//						mpi::ExchangeVector(
//								send_gids, 0, 0, 
//								recv_gids, 0, 0, comm, &status );
//					}
//
//					/** exchange skels */
//					auto &send_skels = child->data.skels;
//					auto &recv_skels = letnode->data.skels;
//					mpi::ExchangeVector(
//							send_skels, 0, 0, 
//							recv_skels, 0, 0, comm, &status );
//					/** exchange interpolative coefficients */
//					auto &send_proj = child->data.proj;
//					auto &recv_proj = letnode->data.proj;
//					mpi::ExchangeVector(
//							send_proj, 0, 0, 
//							recv_proj, 0, 0, comm, &status );
//				}
//			}
//		};
//
//}; /** end class DistSimpleNearFarNodesTask */
//








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
      int comm_size; mpi::Comm_size( MPI_COMM_WORLD, &comm_size );
      int comm_rank; mpi::Comm_rank( MPI_COMM_WORLD, &comm_rank );


      auto *node = arg;
      auto &data = node->data;
      auto &K = *node->setup->K;
      vector<size_t> &I = data.skels, J;

      if ( NNPRUNE )
      {
        /** Examine that NNFarNodes == NNFarNodeMortonIDs */
        assert( node->NNFarNodes.size() == node->NNFarNodeMortonIDs.size() );

        /** Examine that NNFarNodes == NNFarNodeMortonIDs */
        for ( auto it  = node->NNFarNodes.begin(); 
                   it != node->NNFarNodes.end(); it ++ )
        {
          assert( node->NNFarNodeMortonIDs.count( (*it)->morton ) );

          if ( !(*it)->data.skels.size() )
          {
            printf( "rank %d node %8lu l %lu far %8lu has zero skeletons\n",
                comm_rank,
                node->morton, node->l, (*it)->morton ); fflush( stdout );
          }
          J.insert( J.end(), (*it)->data.skels.begin(), (*it)->data.skels.end() );
        }

        //auto &FarNodes = node->NNFarNodeMortonIDs;
        //for ( auto it = FarNodes.begin(); it != FarNodes.end(); it ++ )
        //{
        //  assert( node->NNFarNodes.count( (*node->morton2node)[ *it ] ) );
        //  assert( (*node->morton2node).count( *it ) );
        //  auto *tar = (*node->morton2node)[ *it ];
        //  if ( !tar->data.skels.size() )
        //  {
        //    printf( "rank %d node %8lu l %lu far %8lu has zero skeletons\n",
        //        comm_rank,
        //        node->morton, node->l, tar->morton ); fflush( stdout );
        //  }
        //  J.insert( J.end(), tar->data.skels.begin(), tar->data.skels.end() );
        //}
      }
      else
      {
        auto &FarNodes = node->FarNodes;
        for ( auto it = FarNodes.begin(); it != FarNodes.end(); it ++ )
        {
          J.insert( J.end(), (*it)->data.skels.begin(), (*it)->data.skels.end() );
        }
      }
      data.FarKab = K( I, J );
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
      auto *node = arg;
      auto &data = node->data;
      auto &K = *node->setup->K;
      assert( node->isleaf );

      vector<size_t> &I = node->gids, J;

      if ( NNPRUNE )
      {
        assert( node->NNNearNodes.size() == node->NNNearNodeMortonIDs.size() );


        for ( auto it  = node->NNNearNodes.begin(); 
                   it != node->NNNearNodes.end(); it ++ )
        {
          assert( (*it)->gids.size() );
          J.insert( J.end(), (*it)->gids.begin(), (*it)->gids.end() );
        }


        //auto &NearMorton = node->NNNearNodeMortonIDs;
        //for ( auto it = NearMorton.begin(); it != NearMorton.end(); it ++ )
        //{
        //  assert( (*node->morton2node).count( *it ) );
        //  auto *tar = (*node->morton2node)[ *it ];
        //  assert( tar->gids.size() );
        //  J.insert( J.end(), tar->gids.begin(), tar->gids.end() );
        //}
        data.NearKab = K( I, J );
      }
      else
      {
        data.NearKab = K( I, I );
      }
    };

}; /** end class CacheNearNodesTask */











///**
// *  @brief Work only for no NN pruning.
// *
// */
//template<typename NODE>
//class WaitLETTask : public hmlp::Task
//{
//  public:
//
//    NODE *arg;
//
//    void Set( NODE *user_arg )
//    {
//      std::ostringstream ss;
//      arg = user_arg;
//      name = std::string( "waitlet" );
//      //label = std::to_string( arg->treelist_id );
//      ss << arg->treelist_id;
//      label = ss.str();
//
//      /** we don't know the exact cost here */
//      cost = 5.0;
//
//      /** high priority */
//      priority = true;
//    }
//
//    void DependencyAnalysis()
//    {
//      arg->DependencyAnalysis( hmlp::ReadWriteType::R, this );
//      if ( arg->child )
//      {
//        auto *childFarNodes = &arg->child->NNFarNodes;
//        for ( auto it  = childFarNodes->begin(); 
//                   it != childFarNodes->end(); it ++ )
//        {
//          (*it)->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
//        }
//      };
//      this->TryEnqueue();
//    };
//
//    void Execute( Worker* user_worker )
//    {
//    };
//
//}; /** end class WaitLETTask */
//
//
//
///**
// *  @brief Work only for no NN pruning.
// *
// */
//template<typename NODE>
//class WaitSiblingTask : public hmlp::Task
//{
//  public:
//
//    NODE *arg;
//
//    void Set( NODE *user_arg )
//    {
//      std::ostringstream ss;
//      arg = user_arg;
//      name = std::string( "WaitSibling" );
//      //label = std::to_string( arg->treelist_id );
//      ss << arg->treelist_id;
//      label = ss.str();
//
//      /** we don't know the exact cost here */
//      cost = 5.0;
//
//      /** high priority */
//      priority = true;
//    }
//
//    void DependencyAnalysis()
//    {
//      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
//
//      if ( !arg->isleaf )
//      {
//        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
//        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
//      }
//
//      /** try to enqueue if there is no dependency */
//      this->TryEnqueue();
//    };
//
//    void Execute( Worker* user_worker )
//    {
//    };
//
//}; /** end class WaitSiblingTask */









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

    //map<T, size_t> candidates;
    //NODE *now = node;
    //while ( now )
    //{
    //  auto *sibling = now->sibling;
    //  if ( sibling )
    //  {
    //    for ( auto it  = sibling->data.pool.begin(); 
    //               it != sibling->data.pool.end(); it ++ )
    //    {
    //      candidates.insert( flip_pair( *it ) );
    //    }
    //  }
    //  printf( "Distributed row sample with candidates %lu\n", candidates.size() ); fflush( stdout );
    //  /** 
    //   *  Move toward parent.
    //   *  We need to cast the pointer here becuase the now->parent may be in
    //   *  type tree::Node, but NODE = mpitree::Node.
    //   */
    //  now = (NODE*)(now->parent);
    //}

    //for ( auto it = candidates.begin(); it != candidates.end(); it ++ )
    //{
    //  if ( I.size() < nsamples ) I.push_back( (*it).second );
    //  else break;
    //}
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
      {
			  //candidates[ i ] = rand() % K.col();
        auto important_sample = K.ImportantSample( 0 );
        candidates[ i ] =  important_sample.second;
      }
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







//template<typename NODE, typename T>
//void RowSamples( NODE *node, size_t nsamples )
//{
//  /** gather shared data and create reference */
//  auto &K = *node->setup->K;
//
//  /** amap contains nsamples of row gids of K */
//  std::vector<size_t> &amap = node->data.candidate_rows;
//
//  /** clean up candidates from previous iteration */
//  amap.clear();
//
//
//  /** construct snids from neighbors */
//  if ( node->setup->NN )
//  {
//    //printf( "construct snids NN.row() %lu NN.col() %lu\n", 
//    //    node->setup->NN->row(), node->setup->NN->col() ); fflush( stdout );
//    auto &NN = *(node->setup->NN);
//    auto &gids = node->gids;
//    auto &pnids = node->data.pnids;
//    auto &snids = node->data.snids;
//    size_t kbeg = ( NN.row() + 1 ) / 2;
//    size_t kend = NN.row();
//    size_t knum = kend - kbeg;
//
//    if ( node->isleaf )
//    {
//      pnids.clear();
//      for ( size_t j = 0; j < gids.size(); j ++ )
//        for ( size_t i = 0; i < NN.row() / 2; i ++ )
//          pnids.insert( NN( i, gids[ j ] ).second );
//
//      for ( size_t j = 0; j < gids.size(); j ++ )
//        pnids.erase( gids[ j ] );
//
//      snids.clear();
//      std::vector<std::pair<T, size_t>> tmp( knum * gids.size() );
//      for ( size_t j = 0; j < gids.size(); j ++ )
//        for ( size_t i = kbeg; i < kend; i ++ )
//          tmp[ j * knum + ( i - kbeg ) ] = NN( i, gids[ j ] );
//
//      /** create a sorted list */
//      std::sort( tmp.begin(), tmp.end() );
//    
//      for ( auto it = tmp.begin(); it != tmp.end(); it ++ )
//      {
//        /** create a single query */
//        std::vector<size_t> sample_query( 1, (*it).second );
//				std::vector<size_t> validation = 
//					node->setup->ContainAny( sample_query, node->morton );
//
//        if ( !pnids.count( (*it).second ) && !validation[ 0 ] )
//        {
//          /** duplication is handled by std::map */
//          auto ret = snids.insert( std::pair<size_t, T>( (*it).second, (*it).first ) );
//        }
//
//        /** while we have enough samples, then exit */
//        if ( snids.size() >= nsamples ) break;
//      }
//    }
//    else
//    {
//      auto &lsnids = node->lchild->data.snids;
//      auto &rsnids = node->rchild->data.snids;
//      auto &lpnids = node->lchild->data.pnids;
//      auto &rpnids = node->rchild->data.pnids;
//
//      /** 
//       *  merge children's sampling neighbors...    
//       *  start with left sampling neighbor list 
//       */
//      snids = lsnids;
//
//      /**
//       *  Add right sampling neighbor list. If duplicate update distace if nec.
//       */
//      for ( auto it = rsnids.begin(); it != rsnids.end(); it ++ )
//      {
//        auto ret = snids.insert( *it );
//        if ( !ret.second )
//        {
//          if ( ret.first->second > (*it).first )
//            ret.first->second = (*it).first;
//        }
//      }
//
//      /** remove "own" points */
//      for ( size_t i = 0; i < gids.size(); i ++ ) snids.erase( gids[ i ] );
//
//      for ( auto it = lpnids.begin(); it != lpnids.end(); it ++ )
//        snids.erase( *it );
//      for ( auto it = rpnids.begin(); it != rpnids.end(); it ++ )
//        snids.erase( *it );
//    }
//  }
//
//  /** create an order snids by flipping the std::map */
//  std::multimap<T, size_t> ordered_snids = gofmm::flip_map( node->data.snids );
//  
//  if ( nsamples < K.col() - node->n )
//  {
//    amap.reserve( nsamples );
//
//    for ( auto it  = ordered_snids.begin(); 
//               it != ordered_snids.end(); it++ )
//    {
//      /** (*it) has type pair<T, size_t> */
//      amap.push_back( (*it).second );
//      if ( amap.size() >= nsamples ) break;
//    }
//
//    /** uniform samples */
//    if ( amap.size() < nsamples )
//    {
//      while ( amap.size() < nsamples )
//      {
//        size_t sample = rand() % K.col();
//
//        /** create a single query */
//        std::vector<size_t> sample_query( 1, sample );
//
//				std::vector<size_t> validation = 
//					node->setup->ContainAny( sample_query, node->morton );
//
//        /**
//         *  check duplication using std::find, but check whether the sample
//         *  belongs to the diagonal block using Morton ID.
//         */ 
//        if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() &&
//             !validation[ 0 ] )
//        {
//          amap.push_back( sample );
//        }
//      }
//    }
//  }
//  else /** use all off-diagonal blocks without samples */
//  {
//    for ( size_t sample = 0; sample < K.col(); sample ++ )
//    {
//      if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() )
//      {
//        amap.push_back( sample );
//      }
//    }
//  }
//
//}; /** end RowSamples() */
//



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
  gofmm::RowSamples<NODE, T>( node, nsamples );

  /** 
   *  get KIJ for skeletonization 
   *
   *  notice that operator () may involve MPI collaborative communication.
   *
   */
  //KIJ = K( candidate_rows, candidate_cols );
  size_t over_size_rank = node->setup->s + 20;
  if ( candidate_rows.size() <= over_size_rank )
  {
    KIJ = K( candidate_rows, candidate_cols );
  }
  else
  {
    auto Ksamples = K( candidate_rows, candidate_cols );
    /**
     *  Compute G * KIJ
     */
    KIJ.resize( over_size_rank, candidate_cols.size() );
    Data<T> G( over_size_rank, nsamples ); G.randn( 0, 1 );

    View<T> Ksamples_v( false, Ksamples );
    View<T> KIJ_v( false, KIJ );
    View<T> G_v( false, G );

    /** KIJ = G * Ksamples */
    gemm::xgemm<GEMM_NB>( (T)1.0, G_v, Ksamples_v, (T)0.0, KIJ_v );
  }

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
      double beg = omp_get_wtime();
      gofmm::GetSkeletonMatrix<NODE, T>( arg );
      double getmtx_t = omp_get_wtime() - beg;
      //printf( "%lu GetSkeletonMatrix end %lfs\n", arg->treelist_id, getmtx_t ); fflush( stdout );
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
  mpi::Comm comm = node->GetComm();
  int size = node->GetCommSize();
  int rank = node->GetCommRank();
	int global_rank;
	mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

  if ( size < 2 )
  {
    /** this node is the root of the local tree */
    gofmm::GetSkeletonMatrix<NODE, T>( node );
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
    mpi::Status status;
		size_t nsamples = 0;


    /**
     *  boardcast (isskel) to all MPI processes using children's comm
     */ 
    int child_isskel = child->data.isskel;
    mpi::Bcast( &child_isskel, 1, 0, child->GetComm() );
    child->data.isskel = child_isskel;


    /**
     *  rank-0 owns data of this tree node, and it also owns the left child.
     */ 
    if ( rank == 0 )
    {
      /** concatinate [ lskels, rskels ] */
      candidate_cols = child->data.skels;
      std::vector<size_t> rskel;
      mpi::RecvVector( rskel, size / 2, 10, comm, &status );
      K.RecvColumns( size / 2, comm, &status );
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
      K.SendColumns( child->data.skels, 0, comm );

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
    //KIJ = K( candidate_rows, candidate_cols );
    size_t over_size_rank = node->setup->s + 20;
    if ( candidate_rows.size() <= over_size_rank )
    {
      KIJ = K( candidate_rows, candidate_cols );
    }
    else
    {
      auto Ksamples = K( candidate_rows, candidate_cols );
      /**
       *  Compute G * KIJ
       */
      KIJ.resize( over_size_rank, candidate_cols.size() );
      Data<T> G( over_size_rank, nsamples ); G.randn( 0, 1 );

      View<T> Ksamples_v( false, Ksamples );
      View<T> KIJ_v( false, KIJ );
      View<T> G_v( false, G );

      /** KIJ = G * Ksamples */
      gemm::xgemm<GEMM_NB>( (T)1.0, G_v, Ksamples_v, (T)0.0, KIJ_v );
    }


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
      printf( "%lu Par-GetSkeletonMatrix beg\n", arg->morton );
      ParallelGetSkeletonMatrix<NODE, T>( arg );
      printf( "%lu Par-GetSkeletonMatrix end\n", arg->morton );
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

  lowrank::id<ADAPTIVE, LEVELRESTRICTION>
  ( 
    KIJ.row(), KIJ.col(), maxs, scaled_stol, /** ignore if !ADAPTIVE */
    KIJ, skels, proj, jpvt
  );

  /** free KIJ for spaces */
  KIJ.resize( 0, 0 );

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

      /** GEQP3 */
      flops += ( 4.0 / 3.0 ) * n * n * ( 3 * m - n );
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

			if ( arg->GetCommRank() == 0 )
			{
        /** GEQP3 */
        flops += ( 4.0 / 3.0 ) * n * n * ( 3 * m - n );
        mops += ( 2.0 / 3.0 ) * n * n * ( 3 * m - n );

        /* TRSM */
        flops += k * ( k - 1 ) * ( n + 1 );
        mops  += 2.0 * ( k * k + k * n );
			}

      event.Set( label + name, flops, mops );
      arg->data.skeletonize = event;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      int global_rank = 0;
      mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
      mpi::Comm comm = arg->GetComm();

      double beg = omp_get_wtime();
      if ( arg->GetCommRank() == 0 )
      {
        //printf( "%d Par-Skel beg\n", global_rank );
        DistSkeletonize<ADAPTIVE, LEVELRESTRICTION, NODE, T>( arg );
        //printf( "%d Par-Skel end\n", global_rank );
      }
      double skel_t = omp_get_wtime() - beg;
      //printf( "rank %d, skeletonize (%4lu, %4lu) %lfs\n", 
      //    global_rank, arg->data.KIJ.row(), arg->data.KIJ.col(), skel_t );

			/** Bcast isskel to every MPI processes in the same comm */
			int isskel = arg->data.isskel;
			mpi::Bcast( &isskel, 1, 0, comm );

			/** update data.isskel */
			arg->data.isskel = isskel;

      /** Bcast skels and proj to every MPI processes in the same comm */
      auto &skels = arg->data.skels;
      size_t nskels = skels.size();
      mpi::Bcast( &nskels, 1, 0, comm );
      if ( skels.size() != nskels ) skels.resize( nskels );
      mpi::Bcast( skels.data(), skels.size(), 0, comm );

      //auto &proj  = arg->data.proj;
      //size_t nrow  = proj.row();
      //size_t ncol  = proj.col();
      //mpi::Bcast( &nrow, 1, 0, comm );
      //mpi::Bcast( &ncol, 1, 0, comm );
      //if ( proj.row() != nrow || proj.col() != ncol ) proj.resize( nrow, ncol );
      //mpi::Bcast( proj.data(), proj.size(), 0, comm );

      if ( arg->setup->NN )
      {
        auto &pnids = arg->data.pnids;
        auto &NN = *(arg->setup->NN);

        /** create the column distribution using cids */
        std::vector<size_t> cids;
        for ( size_t j = 0; j < skels.size(); j ++ )
          if ( NN.HasColumn( skels[ j ] ) ) cids.push_back( j );

        /** create a k-by-nskels distributed matrix */
        DistData<STAR, CIDS, size_t> X_cids( NN.row(), nskels, cids, comm );
        DistData<CIRC, CIRC, size_t> X_circ( NN.row(), nskels,    0, comm );

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
class InterpolateTask : public Task
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
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      /** only executed by rank 0 */
      if ( arg->GetCommRank() == 0  ) gofmm::Interpolate<NODE, T>( arg );

      mpi::Comm comm = arg->GetComm();
      auto &proj  = arg->data.proj;
      size_t nrow  = proj.row();
      size_t ncol  = proj.col();
      mpi::Bcast( &nrow, 1, 0, comm );
      mpi::Bcast( &ncol, 1, 0, comm );
      if ( proj.row() != nrow || proj.col() != ncol ) proj.resize( nrow, ncol );
      mpi::Bcast( proj.data(), proj.size(), 0, comm );
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
  hmlp_redistribute_workers( omp_get_max_threads(), 0, 1 ); 
  //hmlp_redistribute_workers( 4, 0, 17 );
  //hmlp_redistribute_workers( 
  //  hmlp_read_nway_from_env( "HMLP_NORMAL_WORKER" ),
  //  hmlp_read_nway_from_env( "HMLP_SERVER_WORKER" ),
  //  hmlp_read_nway_from_env( "HMLP_NESTED_WORKER" ) );

  /** n-by-nrhs, initialize potentials */
  size_t n    = weights.row();
  size_t nrhs = weights.col();

  /** potentials must be in [RIDS,STAR] distribution */
  auto &gids_owned = tree.treelist[ 0 ]->gids;
  DistData<RIDS, STAR, T> potentials( n, nrhs, gids_owned, comm );
  potentials.setvalue( 0.0 );

  /** Setup */
  tree.setup.w = &weights;
  tree.setup.u = &potentials;

  /** Compute all N2S, S2S, S2N, L2L */
  if ( rank == 0 && REPORT_EVALUATE_STATUS )
  {
    //hmlp::mpi::Barrier( MPI_COMM_WORLD );
    printf( "N2S, S2S, S2N, L2L (HMLP Runtime) ...\n" ); fflush( stdout );
  }
  if ( SYMMETRIC_PRUNE )
  {

    /** 
     *  Algorithms:
     * 
     *  1. TreeView
     *  2. Upward N2S, redistribute w for L2L
     *  3. L2L, redistribute \sk{w} for S2S
     *  4. S2S, S2N
     *
     *  or 
     *
     *  1. TreeView
     *  2. N2S, redistribute w for L2L, 
     *     L2L (on owned_near_node), 
     *     S2S (on owned_far_node)
     *  3. L2L (on recv_near_node), redistribute \sk{w} 
     *  4. S2S (on recv_far_node), S2N
     */ 


			/** global barrier and timer */
			mpi::Barrier( tree.comm );
			beg = omp_get_wtime();

			/** TreeView (downward traversal) */
			gofmm::TreeViewTask<NODE> seqVIEWtask;
			mpigofmm::DistTreeViewTask<MPINODE> mpiVIEWtask;
			tree.DistTraverseDown( mpiVIEWtask );
			tree.LocaTraverseDown( seqVIEWtask );

			mpi::Barrier( tree.comm );
      printf( "Finish Creating TreeView Tasks\n" ); fflush( stdout );


			/** N2S (Loca first, Dist follows) */
			gofmm::UpdateWeightsTask<NODE, T> seqN2Stask;
			mpigofmm::DistUpdateWeightsTask<MPINODE, T> mpiN2Stask;
			tree.LocaTraverseUp( seqN2Stask );
			tree.DistTraverseUp( mpiN2Stask );
			mpi::Barrier( tree.comm );
      printf( "Finish Creating Tasks\n" ); fflush( stdout );

			hmlp_run();
			mpi::Barrier( tree.comm );
      printf( "Finish UpdateWeights\n" ); fflush( stdout );
			mpi::Barrier( tree.comm );

      ExchangeLET<T>( tree, string( "skelweights" ) );
      printf( "Finish Exchange skelweights\n" ); fflush( stdout );
		  mpi::Barrier( tree.comm );
      ExchangeLET<T>( tree, string( "leafweights" ) );
      printf( "Finish Exchange leafweights\n" ); fflush( stdout );
		  mpi::Barrier( tree.comm );




      tree.DependencyCleanUp();
			/** L2L */
			mpigofmm::DistLeavesToLeavesTask<NNPRUNE, NODE, T> seqL2Ltask;
			tree.LocaTraverseLeafs( seqL2Ltask );
			/** epoch */
			hmlp_run();
			mpi::Barrier( tree.comm );
      printf( "Finish L2L\n" ); fflush( stdout );

      //if ( potentials.HasIllegalValue() )
      //{
      //  printf( "Illegal value after L2L\n" ); fflush( stdout );
      //}


      tree.DependencyCleanUp();
			/** S2S */
			gofmm::SkeletonsToSkeletonsTask<NNPRUNE, NODE, T> seqS2Stask;
			mpigofmm::DistSkeletonsToSkeletonsTask<NNPRUNE, MPINODE, T> mpiS2Stask;
			tree.LocaTraverseUnOrdered( seqS2Stask );
			tree.DistTraverseUnOrdered( mpiS2Stask );
			/** epoch */
			hmlp_run();
			mpi::Barrier( tree.comm );
      printf( "Finish S2S\n" ); fflush( stdout );




      tree.DependencyCleanUp();
			/** S2N (Dist first, Loca follows) */
			gofmm::SkeletonsToNodesTask<NNPRUNE, NODE, T> seqS2Ntask;
			mpigofmm::DistSkeletonsToNodesTask<NNPRUNE, MPINODE, T> mpiS2Ntask;
			tree.DistTraverseDown( mpiS2Ntask );
			tree.LocaTraverseDown( seqS2Ntask );

			/** epoch */
			hmlp_run();
			mpi::Barrier( tree.comm );
			computeall_time = omp_get_wtime() - beg;


      //if ( potentials.HasIllegalValue() )
      //{
      //  printf( "Illegal value after S2N\n" ); fflush( stdout );
      //}

  }
  else 
  {
    /** TODO: implement unsymmetric prunning */
    printf( "Non symmetric Distributed ComputeAll is not yet implemented\n" );
    exit( 1 );
  }

  /** compute the breakdown cost */
  evaluation_time += allocate_time;
  evaluation_time += computeall_time;
  time_ratio = 100 / evaluation_time;

  if ( rank == 0 && REPORT_EVALUATE_STATUS )
  {
    printf( "========================================================\n");
    printf( "GOFMM evaluation phase\n" );
    printf( "========================================================\n");
    //printf( "Allocate ------------------------------ %5.2lfs (%5.1lf%%)\n", 
    //    allocate_time, allocate_time * time_ratio );
    //printf( "Forward permute ----------------------- %5.2lfs (%5.1lf%%)\n", 
    //    forward_permute_time, forward_permute_time * time_ratio );
    //printf( "N2S, S2S, S2N, L2L -------------------- %5.2lfs (%5.1lf%%)\n", 
    //    computeall_time, computeall_time * time_ratio );
    //printf( "Backward permute ---------------------- %5.2lfs (%5.1lf%%)\n", 
    //    backward_permute_time, backward_permute_time * time_ratio );
    //printf( "========================================================\n");
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
	gofmm::Configuration<T> &config
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


  /** the following tasks require background tasks */
  int num_background_worker = omp_get_max_threads() / 4 + 1;
  if ( omp_get_max_threads() < 2 )
  {
    printf( "(ERROR!) Distributed GOFMM requires at least 'TWO' threads per MPI process\n" );
    exit( 1 );
  }
  hmlp_set_num_background_worker( num_background_worker );

	if ( rank == 0 )
	{
    printf( "Use %d/%d threads as background workers ...\n", 
				num_background_worker, omp_get_max_threads() ); fflush( stdout );
	}





  /** iterative all nearnest-neighbor (ANN) */
  const size_t n_iter = 10;
  const bool SORTED = false;
  /** do not change anything below this line */
  mpitree::Tree<RKDTSETUP, DATA, N_CHILDREN, T> rkdt;
  rkdt.setup.X_cblk = X_cblk;
  rkdt.setup.K = &K;
	rkdt.setup.metric = metric; 
  rkdt.setup.splitter = rkdtsplitter;
  pair<T, std::size_t> initNN( std::numeric_limits<T>::max(), n );

  if ( rank == 0 ) 
	{
		printf( "NeighborSearch ...\n" ); fflush( stdout );
	}
  beg = omp_get_wtime();
  if ( NN_cblk.row() != k )
  {
    NeighborsTask<RKDTNODE, T> knntask;
    NN_cblk = rkdt.AllNearestNeighbor<SORTED>( n_iter, n, k, 15, initNN, knntask );
  }
  else
  {
		if ( rank == 0 )
		{
      printf( "not performed (precomputed or k=0) ...\n" ); fflush( stdout );
		}
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


  //hmlp_redistribute_workers( 4, 0, 17 );
  //hmlp_redistribute_workers( 1, 0, omp_get_max_threads() );
  //hmlp_redistribute_workers( 2, 0, 10 );










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
  tree.setup.budget = budget;
  tree.setup.stol = stol;

	/** metric ball tree partitioning */
	if ( rank == 0  )
	{
    printf( "TreePartitioning ...\n" ); fflush( stdout );
	}
  mpi::Barrier( tree.comm );
  beg = omp_get_wtime();
  tree.TreePartition( n );
  mpi::Barrier( tree.comm );
  tree_time = omp_get_wtime() - beg;
	if ( rank == 0 )
	{
    printf( "end TreePartitioning ...\n" ); fflush( stdout );
	}
  mpi::Barrier( tree.comm );

  /** 
   *  Now redistribute K. 
   *
   */
  K.Redistribute( true, tree.treelist[ 0 ]->gids );



  /** 
   *  Now redistribute NN according to gids i.e.
   *  NN[ *, CIDS ] = NN[ *, CBLK ];
   */
  DistData<STAR, CIDS, pair<T, size_t>> NN( k, n, tree.treelist[ 0 ]->gids, tree.comm );
  NN = NN_cblk;
  tree.setup.NN = &NN;
  ExchangeNeighbors<T>( tree );
  printf( "Finish redistribute NN\n" ); fflush( stdout );


  /**
   *  Now we need to cache submatirces for imporant samples.
   *
   *  Each MPI process first compute 4s candidates for each leaf node by
   *  merging neighbors.
   *  Then 
   *
   *
   */
	tree.DependencyCleanUp();
  gofmm::NearSamplesTask<NODE, T> NEARSAMPLEStask;
  tree.LocaTraverseLeafs( NEARSAMPLEStask );
  printf( "Finish NearSamplesTask LocalTraverseLeafs\n" ); fflush( stdout );
  hmlp_run();
  mpi::Barrier( tree.comm );
  tree.DependencyCleanUp();
  printf( "Finish NearSamplesTask\n" ); fflush( stdout );


  /**
   *  Symmetrize near interaction list:
   *
   *  for each alpha in owned_leaf and beta in Near(alpha)
   *      sbuff[ rank(beta) ] += pair( morton(beta), morton(alpha) );
   *  Alltoallv( sbuff, rbuff );
   *  for each pair( beta, alpha ) in rbuff
   *      near( beta ) += alpha;
   */ 
  SymmetrizeNearInteractions<NODE>( tree );
  mpi::Barrier( tree.comm );
  BuildInteractionListPerRank( tree, true );
  printf( "Finish SymmetrizeNearInteractions()\n" ); fflush( stdout );

  /**
   *  TODO: need send and recv interaction lists for each rank
   *
   *  SendNNNear[ rank ][ local  morton ]
   *  RecvNNNear[ rank ][ remote morton ]
   *
   *  for each leaf alpha and beta in Near(alpha)
   *    SendNNNear[ rank(beta) ] += Morton(alpha)
   *
   *  Alltoallv( SendNNNear, rbuff );
   *
   *  for each rank
   *    RecvNNNear[ rank ][ remote morton ] = offset in rbuff
   *
   */ 


  /**
   *  Create sample pools for each tree node by merging the near interaction
   *  list. 
   */ 
  //gofmm::BuildPoolTask<NODE, T> seqBUILDPOOLtask;
  //DistBuildPoolTask<LETNODE, MPINODE, T> mpiBUILDPOOLtask;
  //tree.LocaTraverseUp( seqBUILDPOOLtask );
  //tree.DistTraverseUp( mpiBUILDPOOLtask );
  //hmlp_run();
  //tree.DependencyCleanUp();
  //printf( "Finish BuildPoolTask\n" ); fflush( stdout );



  /**
   *  Merge FarNodes
   */
  MergeFarNodesTask<true, NODE, T> seqMERGEtask;
  DistMergeFarNodesTask<true, MPINODE, T> mpiMERGEtask;
  tree.LocaTraverseUp( seqMERGEtask );
  tree.DistTraverseUp( mpiMERGEtask );
  hmlp_run();
  mpi::Barrier( tree.comm );
  tree.DependencyCleanUp();
  printf( "Finish MergeFarNodesTask\n" ); fflush( stdout );




  /**
   *  Symmetrize far interaction list:
   *
   *  for each alpha in owned_node and beta in Far(alpha)
   *      sbuff[ rank(beta) ] += pair( morton(beta), morton(alpha) );
   *  Alltoallv( sbuff, rbuff );
   *  for each pair( beta, alpha ) in rbuff
   *      Far( beta ) += alpha;
   *
   *      
   */ 
  SymmetrizeFarInteractions<NODE>( tree );
  mpi::Barrier( tree.comm );
  BuildInteractionListPerRank( tree, false );
  printf( "Finish SymmetrizeFarInteractions()\n" ); fflush( stdout );




  /** skeletonization */
	if ( rank == 0 )
	{
		printf( "Skeletonization (HMLP Runtime) ...\n" ); fflush( stdout );
	}
	mpi::Barrier( tree.comm );
	beg = omp_get_wtime();

	/** clean up all dependencies for skeletonization */
	tree.DependencyCleanUp();
	//hmlp_redistribute_workers( 
	//		omp_get_max_threads(), 
	//		omp_get_max_threads() / 4 + 1, 1 );
	//hmlp_redistribute_workers( 
	//    hmlp_read_nway_from_env( "HMLP_NORMAL_WORKER" ),
	//    hmlp_read_nway_from_env( "HMLP_SERVER_WORKER" ),
	//    hmlp_read_nway_from_env( "HMLP_NESTED_WORKER" ) );

	//hmlp_set_num_workers( 10 );


	if ( 1 )
	{
		/** gather sample rows and skeleton columns, then ID */
		mpigofmm::GetSkeletonMatrixTask<NODE, T> seqGETMTXtask;
		mpigofmm::ParallelGetSkeletonMatrixTask<MPINODE, T> mpiGETMTXtask;
		mpigofmm::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T> seqSKELtask;
		mpigofmm::DistSkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, MPINODE, T> mpiSKELtask;
		tree.LocaTraverseUp( seqGETMTXtask, seqSKELtask );
		hmlp_run();
		mpi::Barrier( tree.comm );
    tree.DependencyCleanUp();

    printf( "Finish Local Skeletonization\n" ); fflush( stdout );


		tree.DistTraverseUp( mpiGETMTXtask, mpiSKELtask );
		hmlp_run();
		mpi::Barrier( tree.comm );
    tree.DependencyCleanUp();

		/** compute the coefficient matrix of ID */
		gofmm::InterpolateTask<NODE, T> seqPROJtask;
		mpigofmm::InterpolateTask<MPINODE, T> mpiPROJtask;
		tree.LocaTraverseUnOrdered( seqPROJtask );
		tree.DistTraverseUnOrdered( mpiPROJtask );

		hmlp_run();
		mpi::Barrier( tree.comm );
    tree.DependencyCleanUp();
    printf( "Finish Skeletonization\n" ); fflush( stdout );

    /** Exchange {skels} and {params(skels)}  */
    ExchangeLET<T>( tree, string( "skelgids" ) );
    printf( "Finish ExchangeSkeletons\n" ); fflush( stdout );
    ExchangeLET<T>( tree, string( "leafgids" ) );
    printf( "Finish ExchangeLeafs\n" ); fflush( stdout );



		/** create interaction lists */
		//SimpleNearFarNodesTask<NODE> seqNEARFARtask;
		//DistSimpleNearFarNodesTask<LETNODE, MPINODE> mpiNEARFARtask;
		//tree.DistTraverseDown( mpiNEARFARtask );
		//tree.LocaTraverseDown( seqNEARFARtask );


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
		mpi::Barrier( tree.comm );
		skel_time = omp_get_wtime() - beg;
	}



//	if ( 0 )
//  {
//		//hmlp_redistribute_workers( omp_get_max_threads(), 9, 1 );
//
//		mpigofmm::GetSkeletonMatrixTask<NODE, T> seqGETMTXtask;
//		mpigofmm::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T> seqSKELtask;
//		tree.LocaTraverseUp( seqGETMTXtask, seqSKELtask );
//
//		auto *waitsibtask = new WaitSiblingTask<NODE>();
//		waitsibtask->Set( tree.treelist[ 0 ] );
//    waitsibtask->DependencyAnalysis();
//		waitsibtask->Submit();
//
//		/** create interaction lists */
//		SimpleNearFarNodesTask<NODE> seqNEARFARtask;
//		tree.LocaTraverseUp( seqNEARFARtask );
//
//
//		gofmm::InterpolateTask<NODE, T> seqPROJtask;
//		tree.LocaTraverseUnOrdered( seqPROJtask );
//
//		/** cache near KIJ interactions */
//		mpigofmm::CacheNearNodesTask<NNPRUNE, NODE> seqNEARKIJtask;
//		tree.LocaTraverseLeafs( seqNEARKIJtask );
//
//		/** cache  far KIJ interactions */
//		mpigofmm::CacheFarNodesTask<NNPRUNE, NODE> seqFARKIJtask;
//		tree.LocaTraverseUnOrdered( seqFARKIJtask );
//
//		/** */
//		hmlp_run();
//		mpi::Barrier( MPI_COMM_WORLD );
//
//		/** clean up dep */
//		tree.DependencyCleanUp();
//		hmlp_redistribute_workers( 
//				hmlp_read_nway_from_env( "HMLP_NORMAL_WORKER" ),
//				hmlp_read_nway_from_env( "HMLP_SERVER_WORKER" ),
//				hmlp_read_nway_from_env( "HMLP_NESTED_WORKER" ) );
//
//		//hmlp_set_num_workers( 2 );
//
//		mpigofmm::ParallelGetSkeletonMatrixTask<MPINODE, T> mpiGETMTXtask;
//		mpigofmm::DistSkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, MPINODE, T> mpiSKELtask;
//		tree.DistTraverseUp( mpiGETMTXtask, mpiSKELtask );
//
//		mpigofmm::InterpolateTask<MPINODE, T> mpiPROJtask;
//		tree.DistTraverseUnOrdered( mpiPROJtask );
//
//		DistSimpleNearFarNodesTask<LETNODE, MPINODE> mpiNEARFARtask;
//		tree.DistTraverseDown( mpiNEARFARtask );
//
//		mpigofmm::CacheFarNodesTask<NNPRUNE, MPINODE> mpiFARKIJtask;
//		tree.DistTraverseUnOrdered( mpiFARKIJtask );
//
//		other_time += omp_get_wtime() - beg;
//		hmlp_run();
//		mpi::Barrier( MPI_COMM_WORLD );
//		skel_time = omp_get_wtime() - beg;
//	}
//
//
//
//
//
//	if ( 0 )
//	{
//		tree.DependencyCleanUp();
//		mpigofmm::GetSkeletonMatrixTask<NODE, T> seqGETMTXtask;
//		mpigofmm::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T> seqSKELtask;
//		gofmm::InterpolateTask<NODE, T> seqPROJtask;
//		tree.LocaTraverseUp( seqGETMTXtask, seqSKELtask, seqPROJtask );
//		hmlp_run();
//		mpi::Barrier( MPI_COMM_WORLD );
//
//		tree.DependencyCleanUp();
//		/** create interaction lists */
//		SimpleNearFarNodesTask<NODE> seqNEARFARtask;
//		tree.LocaTraverseDown( seqNEARFARtask );
//
//		/** cache near KIJ interactions */
//		mpigofmm::CacheNearNodesTask<NNPRUNE, NODE> seqNEARKIJtask;
//		tree.LocaTraverseLeafs( seqNEARKIJtask );
//
//		/** cache  far KIJ interactions */
//		mpigofmm::CacheFarNodesTask<NNPRUNE, NODE> seqFARKIJtask;
//		tree.LocaTraverseUnOrdered( seqFARKIJtask );
//		hmlp_run();
//		mpi::Barrier( MPI_COMM_WORLD );
//
//
//		tree.DependencyCleanUp();
//		hmlp_redistribute_workers(
//				hmlp_read_nway_from_env( "HMLP_NORMAL_WORKER" ),
//				hmlp_read_nway_from_env( "HMLP_SERVER_WORKER" ),
//				hmlp_read_nway_from_env( "HMLP_NESTED_WORKER" ) );
//
//		mpigofmm::ParallelGetSkeletonMatrixTask<MPINODE, T> mpiGETMTXtask;
//		mpigofmm::DistSkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, MPINODE, T> mpiSKELtask;
//		mpigofmm::InterpolateTask<MPINODE, T> mpiPROJtask;
//		tree.DistTraverseUp( mpiGETMTXtask, mpiSKELtask, mpiPROJtask );
//		hmlp_run();
//		mpi::Barrier( MPI_COMM_WORLD );
//
//
//		/** clean up dep */
//		tree.DependencyCleanUp();
//		hmlp_redistribute_workers(
//				omp_get_max_threads(),
//				omp_get_max_threads() / 4 + 1, 1 );
//
//		/** create interaction lists */
//		DistSimpleNearFarNodesTask<LETNODE, MPINODE> mpiNEARFARtask;
//		tree.DistTraverseDown( mpiNEARFARtask );
//
//		/** cache  far KIJ interactions */
//		mpigofmm::CacheFarNodesTask<NNPRUNE, MPINODE> mpiFARKIJtask;
//		tree.DistTraverseUnOrdered( mpiFARKIJtask );
//
//		hmlp_run();
//		mpi::Barrier( MPI_COMM_WORLD );
//		skel_time = omp_get_wtime() - beg;
//	}





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
    //printf( "MergeFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", mergefarnodes_time, mergefarnodes_time * time_ratio );
    //printf( "CacheFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", cachefarnodes_time, cachefarnodes_time * time_ratio );
    printf( "========================================================\n");
    printf( "Compress (%4.2lf not compressed) -------- %5.2lfs (%5.1lf%%)\n", 
        exact_ratio, compress_time, compress_time * time_ratio );
    printf( "========================================================\n\n");
  }










  /** cleanup all w/r dependencies on tree nodes */
  tree_ptr->DependencyCleanUp();

  /** global barrier to make sure all processes have completed */
  mpi::Barrier( tree.comm );



  return tree_ptr;

}; /** end Compress() */







/**
 *  @brief Here MPI_Allreduce is required.
 */ 
template<typename TREE, typename T>
T ComputeError( TREE &tree, size_t gid, Data<T> potentials, mpi::Comm comm )
{
  int comm_rank; mpi::Comm_rank( comm, &comm_rank );
  int comm_size; mpi::Comm_size( comm, &comm_size );


  auto &K = *tree.setup.K;
  auto &w = *tree.setup.w;

  auto  I = vector<size_t>( 1, gid );
  auto &J = tree.treelist[ 0 ]->gids;

  /** Bcast */
  K.BcastColumns( I, gid % comm_size, comm );

	Data<T> Kab;

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

  //printf( "potential %E exact %E\n", potentials[ 0 ], glb_exact[ 0 ] ); fflush( stdout );

  for ( size_t j = 0; j < potentials.col(); j ++ )
  {
    potentials( (size_t)0, j ) -= glb_exact( (size_t)0, j );
  }

  auto err = hmlp_norm(  potentials.row(), potentials.col(), 
                        potentials.data(), potentials.row() ); 

  //printf( "potentials %E err %E nrm2 %E\n", potentials[ 0 ], err, nrm2 ); fflush( stdout );


  return err / nrm2;
}; /** end ComputeError() */










}; /** end namespace gofmm */
}; /** end namespace hmlp */

#endif /** define GOFMM_MPI_HPP */
