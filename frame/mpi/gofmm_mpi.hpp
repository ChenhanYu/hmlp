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

/** Use STL future, thread, chrono */
#include <future>
#include <thread>
#include <chrono>

#include <hmlp_util.hpp>
#include <mpi/tree_mpi.hpp>
#include <mpi/DistData.hpp>
#include <gofmm/gofmm.hpp>
#include <primitives/combinatorics.hpp>
#include <primitives/gemm.hpp>


using namespace std;
using namespace hmlp;


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
    map<size_t, T> snids; 

    /* pruning neighbors ids */
    unordered_set<size_t> pnids; 

    /** skeleton weights and potentials */
    Data<T> w_skel;
    Data<T> u_skel;

    /** permuted weights and potentials (buffer) */
    Data<T> w_leaf;
    Data<T> u_leaf[ 20 ];

    /** hierarchical tree view of w<RIDS> and u<RIDS> */
    View<T> w_view;
    View<T> u_view;

    /** Cached Kab */
    Data<size_t> Nearbmap;
    Data<T> NearKab;
    Data<T> FarKab;

    /** Distributed dependents */
    //set<ReadWrite*> NearDependents;
    //set<ReadWrite*> FarDependents;
    set<int> NearDependents;
    set<int> FarDependents;

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

    /** Bcast from rank 0 */
    mpi::Bcast( column_samples.data(), column_samples.size(), 0, comm );
    K.BcastColumns( column_samples, 0, comm );


    /** Collecting diagonal DII, DCC, KIC */
    Data<T> DII = K.Diagonal( gids );
    Data<T> DCC = K.Diagonal( column_samples );
    Data<T> KIC = K( gids, column_samples );

    if ( this->metric == GEOMETRY_DISTANCE )
    {
      KIC = K.PairwiseDistances( gids, column_samples );
    }

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

    if ( this->metric == GEOMETRY_DISTANCE )
    {
      KIP = K.PairwiseDistances( gids, P );
    }

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

    if ( this->metric == GEOMETRY_DISTANCE )
    {
      KIQ = K.PairwiseDistances( gids, Q );
    }

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
    vector<size_t>& gids, vector<size_t>& lids
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


    //mpi::Barrier( MPI_COMM_WORLD );
    //if ( rank == 0 )
    //{
    //  printf( "rank %2d enter distributed randomsplit n = %d\n", 
    //      global_rank, n ); fflush( stdout );
    //}

    /** Early return */
    if ( n == 0 ) return split;

    /** Randomly select two points p and q */
    size_t gidf2c, gidf2f;
    if ( gids.size() )
    {
      gidf2c = gids[ std::rand() % gids.size() ];
      gidf2f = gids[ std::rand() % gids.size() ];
    }

    /** Create a pair <gids.size(), rank> for MPI Allreduce */
    mpi::NumberIntPair<T> local_max_pair, max_pair; 
    local_max_pair.val = gids.size();
    local_max_pair.key = rank;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** Bcast gidf2c from the rank that has the most gids */
    mpi::Bcast( &gidf2c, 1, max_pair.key, comm );
    vector<size_t> P( 1, gidf2c );
    K.BcastColumns( P, max_pair.key, comm );

    /** Choose the second MPI rank */
    if ( rank == max_pair.key ) local_max_pair.val = 0;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** Bcast gidf2c from the rank that has the most gids */
    mpi::Bcast( &gidf2f, 1, max_pair.key, comm );
    vector<size_t> Q( 1, gidf2f );
    K.BcastColumns( Q, max_pair.key, comm );

		vector<size_t> PQ( 2 ); 
    PQ[ 0 ] = gidf2c; 
    PQ[ 1 ] = gidf2f;

    //mpi::Barrier( MPI_COMM_WORLD );
    //if ( rank == 0 )
    //{
    //  printf( "rank %2d BcastColumns %lu %lu\n", global_rank, gidf2c, gidf2f ); 
    //  fflush( stdout ); 
    //}

    /** Collecting DII, KIP, KIQ, kpp and kqq */
    Data<T> DII = K.Diagonal( gids );
    Data<T> kpp = K.Diagonal( P );
    Data<T> kqq = K.Diagonal( Q );
		Data<T> KIPQ = K( gids, PQ );
		Data<T> KIP( gids.size(), (size_t) 1 );
		Data<T> KIQ( gids.size(), (size_t) 1 );

    if ( this->metric == GEOMETRY_DISTANCE )
    {
      KIPQ = K.PairwiseDistances( gids, PQ );
    }

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

    //mpi::Barrier( comm );
    //if ( rank == 0 )
    //{
    //  printf( "rank %d [ %d %d %d ] global [ %d %d %d ]\n",
    //      global_rank, num_lhs_owned, num_mid_owned, num_rhs_owned,
    //      nlhs, nmid, nrhs ); fflush( stdout );
    //  fflush( stdout );
    //}


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

    //mpi::Barrier( comm );
    //if ( rank == 0 )
    //{
    //  printf( "rank %2d Begin  RedistributeWithPartner lhr %lu rhs %lu\n", 
    //      global_rank, split[ 0 ].size(), split[ 1 ].size() ); 
    //  fflush( stdout );
    //}

    /** Perform P2P redistribution */
    K.RedistributeWithPartner( gids, split[ 0 ], split[ 1 ], comm );

    //mpi::Barrier( comm );
    //if ( rank == 0 )
    //{
    //  printf( "rank %2d Finish RedistributeWithPartner\n", global_rank ); 
    //  fflush( stdout );
    //}

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
  /** NN has type DistData<STAR, CIDS, T>. */
  auto &NN   = *(node->setup->NN);
  auto &gids = node->gids;

  //printf( "k %lu gids.size() %lu\n", NN.row(), gids.size() );


  /** KII = K( gids, gids ) */
  Data<T> KII;

  /** rho+k nearest neighbors */
  switch ( metric )
  {
    case GEOMETRY_DISTANCE:
    {
      auto &K = *(node->setup->K);
      KII = K.PairwiseDistances( gids, gids );
      //printf( "%lu-by-%lu KII( 0, 0 ) %E\n", gids.size(), gids.size(), 
      //    KII( (size_t)0, (size_t)0 ) );
      //for ( size_t i = 0; i < 10; i ++ ) printf( "%lu ", gids[ i ] );
      //printf( "\n" );
      //assert( KII.row() == gids.size() && KII.col() == gids.size() );
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

      /** Get digaonal entries */
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

  //printf( "KII.row() %lu KII.col() %lu, NN.row() %lu\n", 
  //    KII.row(), KII.col(), NN.row() ); fflush( stdout );
  for ( size_t j = 0; j < KII.col(); j ++ )
  { 
    /** Create a query list for column KII( :, j ) */
    vector<pair<T, size_t>> query( KII.row() );
    for ( size_t i = 0; i < KII.row(); i ++ ) 
    {
      query[ i ].first  = KII( i, j );
      query[ i ].second = gids[ i ];
    }

    /** Sort the query according to distances */
    sort( query.begin(), query.end() );

    /** Fill-in the neighbor list */
    auto *NNj = NN.columndata( gids[ j ] );
    for ( size_t i = 0; i < NN.row(); i ++ ) 
    {
      NNj[ i ] = query[ i ];
    }
  }

}; /** end FindNeighbors() */














template<class NODE, typename T>
class NeighborsTask : public Task
{
  public:

    NODE *arg;
   
	  /** (default) using angle distance from the Gram vector space */
	  DistanceMetric metric = ANGLE_DISTANCE;

    void Set( NODE *user_arg )
    {
      ostringstream ss;
      arg = user_arg;
      name = string( "Neighbors" );
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
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      FindNeighbors<NODE, T>( arg, metric );

      ///** Create a promise and get its future */
      //promise<bool> done;
      //auto future = done.get_future();

      //thread t( [&done] ( NODE *arg, DistanceMetric metric ) -> void {
      //    FindNeighbors<NODE, T>( arg, metric ); 
      //    done.set_value( true );
      //}, arg, metric );
      //
      ///** Polling the future status */
      //while ( future.wait_for( chrono::seconds( 0 ) ) != future_status::ready ) 
      //{
      //  if ( !this->ContextSwitchToNextTask( user_worker ) ) break;
      //}
      //
      ///** Make sure the task is completed */
      //t.join();
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
    /** This is the root of the local tree */
    gofmm::UpdateWeights<MPINODE, T>( node );
  }
  else
  {
    if ( !node->parent || !node->data.isskel ) return;

    /** Get the child node  */
    auto *child = node->child;

    /** Gather shared data and create reference */
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
        /** Create a transpose view proj_v */
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

      if ( 0 )
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
        /** Create a transpose view proj_v */
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
class DistUpdateWeightsTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "DistN2S" );
      label = to_string( arg->treelist_id );

      /** Compute FLOPS and MOPS */
      double flops = 0.0, mops = 0.0;
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

      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Assume computation bound */
      cost = flops / 1E+9;
      /** "HIGH" priority (critical path) */
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
class DistSkeletonsToSkeletonsTask : public Task
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
      for ( auto p : arg->data.FarDependents )
        hmlp_msg_dependency_analysis( 306, p, R, this );

      auto *FarNodes = &arg->FarNodes;
      if ( NNPRUNE ) FarNodes = &arg->NNFarNodes;
      for ( auto it : *FarNodes ) it->DependencyAnalysis( R, this );

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


template<typename NODE, typename LETNODE, typename T>
class S2STask2 : public Task
{
  public:

    NODE *arg = NULL;

    vector<LETNODE*> Sources;

    int p = 0;

    Lock *lock = NULL;

    void Set( NODE *user_arg, vector<LETNODE*> user_src, int user_p, Lock *user_lock )
    {
      assert( user_arg && user_lock );
      arg = user_arg;
      Sources = user_src;
      p = user_p;
      lock = user_lock;
      name = string( "S2S" );
      label = to_string( arg->treelist_id );

      /** Compute FLOPS and MOPS */
      double flops = 0.0, mops = 0.0;
      size_t nrhs = arg->setup->w->col();
      size_t m = arg->data.skels.size();
      for ( auto src : Sources )
      {
        size_t k = src->data.skels.size();
        flops += 2 * m * k * nrhs;
        mops  += 2 * ( m * k + ( m + k ) * nrhs );
        flops += 2 * m * nrhs;
        flops += m * k * ( 2 * 18 + 100 );
      }
      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Assume computation bound */
      cost = flops / 1E+9;
      /** Assume computation bound */
      if ( arg->treelist_id == 0 ) priority = true;
    };

    void DependencyAnalysis()
    {
      int rank; mpi::Comm_rank( MPI_COMM_WORLD, &rank );
      if ( p == rank )
      {
        for ( auto src : Sources ) src->DependencyAnalysis( R, this );
      }
      else hmlp_msg_dependency_analysis( 306, p, R, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      auto *node = arg;
      if ( !node->parent || !node->data.isskel ) return;
      size_t nrhs = node->setup->w->col();
      auto &K = *node->setup->K;
      auto &I = node->data.skels;

      /** Temporary buffer */
      Data<T> u( I.size(), nrhs, 0.0 );

      for ( auto src : Sources )
      {
        auto &J = src->data.skels;
        auto &w = src->data.w_skel;
        bool is_cached = true;

        auto &KIJ = node->DistFar[ p ][ src->morton ];
        if ( KIJ.row() != I.size() || KIJ.col() != J.size() ) 
        {
          //printf( "KIJ %lu %lu I %lu J %lu\n", KIJ.row(), KIJ.col(), I.size(), J.size() );
          KIJ = K( I, J );
          is_cached = false;
        }

        assert( w.col() == nrhs );
        assert( w.row() == J.size() );
        xgemm
        (
          "N", "N", u.row(), u.col(), w.row(),
          1.0, KIJ.data(), KIJ.row(),
                 w.data(),   w.row(),
          1.0,   u.data(),   u.row()
        );

        /** Free KIJ, if !is_cached. */
        if ( !is_cached ) 
        {
          KIJ.resize( 0, 0 );
          KIJ.shrink_to_fit();
        }
      }

      lock->Acquire();
      {
        auto &u_skel = node->data.u_skel;
        for ( int i = 0; i < u.size(); i ++ ) 
          u_skel[ i ] += u[ i ];
      }
      lock->Release();
    };
}; 

template<typename NODE, typename LETNODE, typename T>
class S2SReduceTask2 : public Task
{
  public:

    NODE *arg = NULL;

    vector<S2STask2<NODE, LETNODE, T>*> subtasks;

    Lock lock;

    const size_t batch_size = 2;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "S2SR" );
      label = to_string( arg->treelist_id );

      /** Reset u_skel */
      if ( arg ) 
      {
        size_t nrhs = arg->setup->w->col();
        auto &I = arg->data.skels;
        arg->data.u_skel.resize( 0, 0 );
        arg->data.u_skel.resize( I.size(), nrhs, 0 );
      }

      /** Create subtasks */
      int size; mpi::Comm_size( MPI_COMM_WORLD, &size );
      for ( int p = 0; p < size; p ++ )
      {
        vector<LETNODE*> Sources;
        for ( auto &it : arg->DistFar[ p ] )
        {
          Sources.push_back( (*arg->morton2node)[ it.first ] );
          if ( Sources.size() == batch_size )
          {
            subtasks.push_back( new S2STask2<NODE, LETNODE, T>() );
            subtasks.back()->Submit();
            subtasks.back()->Set( user_arg, Sources, p, &lock );
            subtasks.back()->DependencyAnalysis();
            Sources.clear();
          }
        }
        if ( Sources.size() )
        {
          subtasks.push_back( new S2STask2<NODE, LETNODE, T>() );
          subtasks.back()->Submit();
          subtasks.back()->Set( user_arg, Sources, p, &lock );
          subtasks.back()->DependencyAnalysis();
          Sources.clear();
        }
      }
      /** Compute FLOPS and MOPS. */
      double flops = 0, mops = 0;
      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Assume computation bound */
      priority = true;
    };

    void DependencyAnalysis()
    {
      for ( auto task : subtasks ) Scheduler::DependencyAdd( task, this );
      //if ( arg->NNFarNodes.size() ) 
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker ) { /** Place holder */ };
};




















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


      if ( 0 )
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
class DistSkeletonsToNodesTask : public Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "PS2N" );
      label = to_string( arg->l );

      double flops = 0.0, mops = 0.0;
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

      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Asuume computation bound */
      cost = flops / 1E+9;
      /** "HIGH" priority (critical path) */
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



template<typename NODE, typename T>
class L2LTask2 : public Task
{
  public:

    NODE *arg = NULL;

    /** A list of source node pointers */
    vector<NODE*> Sources;



    int p = 0;

    /** Write lock */
    Lock *lock = NULL;

    void Set( NODE *user_arg, vector<NODE*> user_src, int user_p, Lock *user_lock )
    {
      arg = user_arg;
      Sources = user_src;
      p = user_p;
      lock = user_lock;
      name = string( "L2L" );
      label = to_string( arg->treelist_id );

      /** Compute FLOPS and MOPS. */
      double flops = 0.0, mops = 0.0;
      size_t nrhs = arg->setup->w->col();
      size_t m = arg->gids.size();
      for ( auto src : Sources )
      {
        size_t k = src->gids.size();
        flops += 2 * m * k * nrhs;
        mops  += 2 * ( m * k + ( m + k ) * nrhs );
        flops += 2 * m * nrhs;
        flops += m * k * ( 2 * 18 + 100 );
      }
      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Assume computation bound */
      cost = flops / 1E+9;
      /** "LOW" priority */
      priority = false;
    };

    void DependencyAnalysis()
    {
      int rank; mpi::Comm_rank( MPI_COMM_WORLD, &rank );
      if ( p == rank ) 
      {
        for ( auto src : Sources ) src->DependencyAnalysis( R, this );
      }
      else hmlp_msg_dependency_analysis( 300, p, R, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      auto *node = arg;
      size_t nrhs = node->setup->w->col();
      auto &K = *node->setup->K;
      auto &I = node->gids;

      double beg = omp_get_wtime();
      /** Temporary buffer */
      Data<T> u( I.size(), nrhs, 0.0 );
      size_t k;

      for ( auto src : Sources )
      {
        /** Get W view of this treenode. (available for non-LET nodes) */
        View<T> &W = src->data.w_view;
        Data<T> &w = src->data.w_leaf;
        
        bool is_cached = true;
        auto &J = src->gids;
        auto &KIJ = node->DistNear[ p ][ src->morton ];
        if ( KIJ.row() != I.size() || KIJ.col() != J.size() ) 
        {
          KIJ = K( I, J );
          is_cached = false;
        }

        if ( W.col() == nrhs && W.row() == J.size() )
        {
          k += W.row();
          xgemm
          (
            "N", "N", u.row(), u.col(), W.row(),
            1.0, KIJ.data(), KIJ.row(),
                   W.data(),   W.ld(),
            1.0,   u.data(),   u.row()
          );
        }
        else
        {
          k += w.row();
          xgemm
          (
            "N", "N", u.row(), u.col(), w.row(),
            1.0, KIJ.data(), KIJ.row(),
                   w.data(),   w.row(),
            1.0,   u.data(),   u.row()
          );
        }

        /** Free KIJ, if !is_cached. */
        if ( !is_cached ) 
        {
          KIJ.resize( 0, 0 );
          KIJ.shrink_to_fit();
        }
      }

      double lock_beg = omp_get_wtime();
      lock->Acquire();
      {
        /** Get U view of this treenode. */
        View<T> &U = node->data.u_view;
        for ( int j = 0; j < u.col(); j ++ )
          for ( int i = 0; i < u.row(); i ++ )
            U( i, j ) += u( i, j );
      }
      lock->Release();
      double lock_time = omp_get_wtime() - lock_beg;

      double gemm_time = omp_get_wtime() - beg;
      double GFLOPS = 2.0 * u.row() * u.col() * k / ( 1E+9 * gemm_time );
      //printf( "GEMM %4lu %4lu %4lu %lf GFLOPS, lock(%lf/%lf)\n", 
      //    u.row(), u.col(), k, GFLOPS, lock_time, gemm_time ); fflush( stdout );
    };
};




template<typename NODE, typename T>
class L2LReduceTask2 : public Task
{
  public:

    NODE *arg = NULL;

    vector<L2LTask2<NODE, T>*> subtasks;

    Lock lock;

    const size_t batch_size = 2;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "L2LR" );
      label = to_string( arg->treelist_id );
      /** Create subtasks */
      int size; mpi::Comm_size( MPI_COMM_WORLD, &size );
      for ( int p = 0; p < size; p ++ )
      {
        vector<NODE*> Sources;
        for ( auto &it : arg->DistNear[ p ] )
        {
          Sources.push_back( (*arg->morton2node)[ it.first ] );
          if ( Sources.size() == batch_size )
          {
            subtasks.push_back( new L2LTask2<NODE, T>() );
            subtasks.back()->Submit();
            subtasks.back()->Set( user_arg, Sources, p, &lock );
            subtasks.back()->DependencyAnalysis();
            Sources.clear();
          }
        }
        if ( Sources.size() )
        {
          subtasks.push_back( new L2LTask2<NODE, T>() );
          subtasks.back()->Submit();
          subtasks.back()->Set( user_arg, Sources, p, &lock );
          subtasks.back()->DependencyAnalysis();
          Sources.clear();
        }
      }
      /** Compute FLOPS and MOPS */
      double flops = 0, mops = 0;
      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** "LOW" priority (critical path) */
      priority = false;
    };

    void DependencyAnalysis()
    {
      for ( auto task : subtasks ) Scheduler::DependencyAdd( task, this );
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker ) {};
};























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

  //if ( KIJ.HasIllegalValue() ) 
  //{
  //  printf( "KIJ has illegal value in DistLeavesToLeaves\n" ); fflush( stdout );
  //}


  bool iscached = data.NearKab.size();
  size_t offset = 0;

  for ( auto it  = NearNodes->begin(); 
             it != NearNodes->end(); it ++ )
  {
    /** Get W view of this treenode. (available for non-LET nodes) */
    View<T> &W = (*it)->data.w_view;

    /** Evaluate KIJ if not cached */
    if ( !iscached ) KIJ = K( node->gids, (*it)->gids );

    if ( W.col() == nrhs )
    {
      //printf( "%8lu l2l %8lu U[%lu %lu %lu] W[%lu %lu %lu] KIJ[%lu %lu]\n",
      //    node->morton, (*it)->morton,
      //    U.row(), U.col(), U.ld(),
      //    W.row(), W.col(), W.ld(),
      //    KIJ.row(), KIJ.col() );


      assert( W.ld() >= (*it)->gids.size() );

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
      if ( w_leaf.row() != (*it)->gids.size() )
      {
        printf( "w_leaf.row() %lu gids.size() %lu\n", w_leaf.row(), (*it)->gids.size() );
        fflush( stdout );
      }

      //for ( int i = 0; i < (*it)->gids.size(); i ++ )
      //{
      //  if ( w_leaf[ i ] != (*it)->gids[ i ] )
      //  {
      //    printf( "%8lu l2l %8lu, w_leaf[ %d ] %E != %lu\n",
      //    node->morton, (*it)->morton, i, 
      //    w_leaf[ i ], (*it)->gids[ i ] ); fflush( stdout );
      //    break;
      //  }
      //}

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

    /** Move forward if cached */
    if ( iscached ) offset += (*it)->gids.size();
  }

  /** Clean up */
  if ( !iscached ) data.NearKab.resize( 0, 0 );

}; /** end LeavesToLeaves() */




template<bool NNPRUNE, typename NODE, typename T>
class DistLeavesToLeavesTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "DistL2L" );
      {
        //label = std::to_string( arg->treelist_id );
        ostringstream ss;
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

      set<NODE*> *NearNodes;
      if ( NNPRUNE ) NearNodes = &arg->NNNearNodes;
      else           NearNodes = &arg->NearNodes;

      for ( auto it  = NearNodes->begin(); 
                 it != NearNodes->end(); it ++ )
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
      /** Deal with "RAW" dependencies on distributed messages */
      for ( auto p : arg->data.NearDependents )
        hmlp_msg_dependency_analysis( 300, p, R, this );

      set<NODE*> *NearNodes;
      if ( NNPRUNE ) NearNodes = &arg->NNNearNodes;
      else           NearNodes = &arg->NearNodes;
      for ( auto it : *NearNodes )
      {
        it->DependencyAnalysis( R, this );
      }

      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
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
template<typename NODE, typename TREE>
void FindNearNodes( TREE &tree )
{
  auto &setup = tree.setup;
  auto &NN = *setup.NN;
  double budget = setup.budget;
  size_t n_leafs = ( 1 << tree.depth );
  /** 
   *  The type here is tree::Node but not mpitree::Node.
   *  NearNodes and NNNearNodes also take tree::Node.
   *  This is ok, because they will only contain leaf nodes,
   *  which will never be distributed.
   *  However, FarNodes and NNFarNodes may contain distributed
   *  tree nodes. In this case, we have to do type casting.
   */
  auto level_beg = tree.treelist.begin() + n_leafs - 1;

  /** Traverse all leaf nodes. **/
  #pragma omp parallel for
  for ( size_t node_ind = 0; node_ind < n_leafs; node_ind ++ )
  {
    auto *node = *(level_beg + node_ind);
    auto &data = node->data;
    size_t n_nodes = ( 1 << node->l );

    /** Add myself to the near interaction list.  */
    node->NearNodes.insert( node );
    node->NNNearNodes.insert( node );
    node->NNNearNodeMortonIDs.insert( node->morton );

    /** Compute ballots for all near interactions */
    multimap<size_t, size_t> sorted_ballot = gofmm::NearNodeBallots( node );

    /** Insert near node cadidates until reaching the budget limit. */ 
    for ( auto it  = sorted_ballot.rbegin(); 
               it != sorted_ballot.rend(); it ++ ) 
    {
      /** Exit if we have enough. */ 
      if ( node->NNNearNodes.size() >= n_nodes * budget ) break;

      /**
       *  Get the node pointer from MortonID. 
       *
       *  Two situations:
       *  1. the pointer doesn't exist, then creates a lettreenode
       */ 
      #pragma omp critical
      {
        if ( !(*node->morton2node).count( (*it).second ) )
        {
          (*node->morton2node)[ (*it).second ] = new NODE( (*it).second );
        }
        /** Insert */
        auto *target = (*node->morton2node)[ (*it).second ];
        node->NNNearNodeMortonIDs.insert( (*it).second );
        node->NNNearNodes.insert( target );
      }
    }

  } /** end for each leaf owned leaf node in the local tree */

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








template<size_t LEVELOFFSET=5, typename NODE>
void FindFarNodes( size_t morton, size_t l, NODE *target )
{
  /** Return while reaching the leaf level. */ 
  if ( l > target->l ) return;

  /** Compute the correct shift. */
  size_t shift = ( 1 << LEVELOFFSET ) - l + LEVELOFFSET;
  /** Set the node Morton ID */
  size_t node_morton = ( morton << shift ) + l;

  bool prunable = true;
  auto & NearMortonIDs = target->NNNearNodeMortonIDs;

  //for ( auto it  = NearMortonIDs.begin(); 
  //           it != NearMortonIDs.end(); it ++ )

  for ( auto it : NearMortonIDs )
  {
    if ( tree::IsMyParent( it, node_morton ) ) prunable = false;
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
      ////target->treelock->Acquire();
      //#pragma omp critical
      //{
      //  auto &owned_nodes = *target->morton2node;
      //  if ( !owned_nodes.count( node_morton ) )
      //  {
      //    owned_nodes[ node_morton ] = new NODE( node_morton );
      //    //printf( "%8lu creates far LET %8lu level %lu\n", target->morton, node_morton, l );
      //  }
      //}
      ////target->treelock->Release();
      //target->NNFarNodes.insert( (*target->morton2node)[ node_morton ] );
      target->NNFarNodeMortonIDs.insert( node_morton );
    }
  }
  else
  {
    /** Recurs if not yet reaching leaf level */
    FindFarNodes<LEVELOFFSET>( ( morton << 1 ) + 0, l + 1, target );
    FindFarNodes<LEVELOFFSET>( ( morton << 1 ) + 1, l + 1, target );
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
    for ( size_t i = 0; i < recvlist[ p ].size(); i ++ )
    {
      auto& query = recvlist[ p ][ i ];
      
      /** Check if query node is allocated? */ 
      #pragma omp critical
      {
        auto* node = tree.morton2node[ query.first ];
        if ( !tree.morton2node.count( query.second ) )
        {
          tree.morton2node[ query.second ] = new NODE( query.second );
          //printf( "rank %d, %8lu level %lu creates near LET %8lu (symmetrize)\n", 
          //    comm_rank, node->morton, node->l, query.second );
        }
        node->data.lock.Acquire();
        {
          node->NNNearNodes.insert( tree.morton2node[ query.second ] );
          node->NNNearNodeMortonIDs.insert( query.second );
        }
        node->data.lock.Release();
      }
    }; /** end pargma omp parallel for */
  }
}; /** end SymmetrizeNearInteractions() */


template<typename NODE, typename TREE>
void SymmetrizeFarInteractions( TREE & tree )
{
  /** MPI */
  int comm_size; mpi::Comm_size( tree.comm, &comm_size );
  int comm_rank; mpi::Comm_rank( tree.comm, &comm_rank );

  vector<vector<pair<size_t, size_t>>> sendlist( comm_size );
  vector<vector<pair<size_t, size_t>>> recvlist( comm_size );

  /** Local traversal */ 
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
        /** Allocate if not exist */
        #pragma omp critical
        {
          if ( !tree.morton2node.count( *it ) )
          {
            tree.morton2node[ *it ] = new NODE( *it );
          }
          node->NNFarNodes.insert( tree.morton2node[ *it ] );
        }
        int dest = tree.Morton2Rank( *it );
        if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
        list[ dest ].push_back( make_pair( *it, node->morton ) );
      }
    }

    #pragma omp critical
    {
      for ( int p = 0; p < comm_size; p ++ )
      {
        sendlist[ p ].insert( sendlist[ p ].end(), 
            list[ p ].begin(), list[ p ].end() );
      }
    } /** end pragma omp critical*/
  }


  /** Distributed traversal */ 
  #pragma omp parallel 
  {
    /** Create a per thread list. Merge them into sendlist afterward. */ 
    vector<vector<pair<size_t, size_t>>> list( comm_size );

    #pragma omp for
    for ( size_t i = 0; i < tree.mpitreelists.size(); i ++ )
    {
      auto *node = tree.mpitreelists[ i ];
      for ( auto it  = node->NNFarNodeMortonIDs.begin();
                 it != node->NNFarNodeMortonIDs.end(); it ++ )
      {
        /** Allocate if not exist */
        #pragma omp critical
        {
          if ( !tree.morton2node.count( *it ) )
          {
            tree.morton2node[ *it ] = new NODE( *it );
          }
          node->NNFarNodes.insert( tree.morton2node[ *it ] );
        }
        int dest = tree.Morton2Rank( *it );
        if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest ); fflush( stdout );
        list[ dest ].push_back( make_pair( *it, node->morton ) );
      }
    }

    #pragma omp critical
    {
      for ( int p = 0; p < comm_size; p ++ )
      {
        sendlist[ p ].insert( sendlist[ p ].end(), 
            list[ p ].begin(), list[ p ].end() );
      }
    } /** end pragma omp critical*/
  }

  /** Alltoallv */
  mpi::AlltoallVector( sendlist, recvlist, tree.comm );

  /** Loop over queries */
  for ( int p = 0; p < comm_size; p ++ )
  {
    //#pragma omp parallel for
    for ( size_t i = 0; i < recvlist[ p ].size(); i ++ )
    {
      auto &query = recvlist[ p ][ i ];
     
      /** Check if query node is allocated?  */
      #pragma omp critical
      {
        if ( !tree.morton2node.count( query.second ) )
        {
          tree.morton2node[ query.second ] = new NODE( query.second );
          //printf( "rank %d, %8lu level %lu creates far LET %8lu (symmetrize)\n", 
          //    comm_rank, node->morton, node->l, query.second );
        }
        auto* node = tree.morton2node[ query.first ];
        node->data.lock.Acquire();
        {
          node->NNFarNodes.insert( tree.morton2node[ query.second ] );
          node->NNFarNodeMortonIDs.insert( query.second );
        }
        node->data.lock.Release();
        assert( tree.Morton2Rank( node->morton ) == comm_rank );
      }
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
template<typename T, typename TREE>
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
        node->DistNear.resize( comm_size );
        for ( auto it = NearMortonIDs.begin(); it != NearMortonIDs.end(); it ++ )
        {
          int dest = tree.Morton2Rank( *it );
          if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
          if ( dest != comm_rank ) 
          {
            list[ dest ].insert( node->morton );
            node->data.NearDependents.insert( dest );
          }
          node->DistNear[ dest ][ *it ] = Data<T>();
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
    //printf( "rank %d finish first part\n", comm_rank ); fflush( stdout );

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
        node->DistFar.resize( comm_size );
        for ( auto it  = node->NNFarNodeMortonIDs.begin();
                   it != node->NNFarNodeMortonIDs.end(); it ++ )
        {
          int dest = tree.Morton2Rank( *it );
          if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
          if ( dest != comm_rank ) 
          {
            list[ dest ].insert( node->morton );
            //node->data.FarDependents.insert( tree.FarRecvFrom.data() + dest );
            node->data.FarDependents.insert( dest );
          }
          node->DistFar[ dest ][ *it ] = Data<T>();
        }
      }

      /** Distributed traversal */
      #pragma omp for
      for ( size_t i = 0; i < tree.mpitreelists.size(); i ++ )
      {
        auto *node = tree.mpitreelists[ i ];
        node->DistFar.resize( comm_size );
        /** Add to the list iff this MPI rank owns the distributed node */
        if ( tree.Morton2Rank( node->morton ) == comm_rank )
        {
          for ( auto it  = node->NNFarNodeMortonIDs.begin();
                     it != node->NNFarNodeMortonIDs.end(); it ++ )
          {
            int dest = tree.Morton2Rank( *it );
            if ( dest >= comm_size ) printf( "%8lu dest %d\n", *it, dest );
            if ( dest != comm_rank ) 
            {
              list[ dest ].insert( node->morton );
              //node->data.FarDependents.insert( tree.FarRecvFrom.data() + dest );
              node->data.FarDependents.insert( dest );
            }
            node->DistFar[ dest ][ *it ]  = Data<T>();
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


template<typename T, typename TREE>
void PackNear( TREE &tree, string option, int p, 
    vector<size_t> &sendsizes, 
    vector<size_t> &sendskels, 
    vector<T> &sendbuffs )
{
  vector<size_t> offsets( 1, 0 );

  for ( auto it : tree.NearSentToRank[ p ] )
  {
    auto *node = tree.morton2node[ it ];
    auto &gids = node->gids;
    if ( !option.compare( string( "leafgids" ) ) )
    {
      sendsizes.push_back( gids.size() );
      sendskels.insert( sendskels.end(), gids.begin(), gids.end() );
    }
    else
    {
      auto &w_view = node->data.w_view;
      //auto  w_leaf = w_view.toData();
      ////printf( "%d send w_leaf to %d [%lu %lu]\n", comm_rank, p, w_leaf.row(), w_leaf.col() );

      //if ( !w_leaf.size() )
      //{
      //  printf( "send w_leaf to %d [%lu %lu] gids %lu\n", 
      //      p, w_leaf.row(), w_leaf.col(), gids.size() );
      //  fflush( stdout );
      //}

      //sendsizes.push_back( w_leaf.size() );
      //sendbuffs.insert( sendbuffs.end(), w_leaf.begin(), w_leaf.end() );


      sendsizes.push_back( gids.size() * w_view.col() );
      offsets.push_back( sendsizes.back() + offsets.back() );
    }
  }

  if ( offsets.size() ) sendbuffs.resize( offsets.back() );

  if ( !option.compare( string( "leafweights" ) ) )
  {
    #pragma omp parallel for
    for ( size_t i = 0; i < tree.NearSentToRank[ p ].size(); i ++ )
    {
      auto *node = tree.morton2node[ tree.NearSentToRank[ p ][ i ] ];
      auto &gids = node->gids;
      auto &w_view = node->data.w_view;
      auto  w_leaf = w_view.toData();
      size_t offset = offsets[ i ];
      for ( size_t j = 0; j < w_leaf.size(); j ++ ) 
        sendbuffs[ offset + j ] = w_leaf[ j ];
    }
  }

};


template<typename T, typename TREE>
void UnpackLeaf( TREE &tree, string option, int p, 
    const vector<size_t> &recvsizes, 
    const vector<size_t> &recvskels, 
    const vector<T> &recvbuffs )
{
  vector<size_t> offsets( 1, 0 );
  for ( auto it : recvsizes ) offsets.push_back( offsets.back() + it );

  for ( auto it : tree.NearRecvFromRank[ p ] )
  {
    auto *node = tree.morton2node[ it.first ];
    if ( !option.compare( string( "leafgids" ) ) )
    {
      auto &gids = node->gids;
      size_t i = it.second;
      gids.reserve( recvsizes[ i ] );
      for ( uint64_t j  = offsets[ i + 0 ]; 
                     j  < offsets[ i + 1 ]; 
                     j ++ )
      {
        gids.push_back( recvskels[ j ] );
      }
    }
    else
    {
      /** Number of right hand sides */
      size_t nrhs = tree.setup.w->col();
      auto &w_leaf = node->data.w_leaf;
      size_t i = it.second;
      w_leaf.resize( recvsizes[ i ] / nrhs, nrhs );
      //printf( "%d recv w_leaf from %d [%lu %lu]\n", 
      //    comm_rank, p, w_leaf.row(), w_leaf.col() ); fflush( stdout );
      for ( uint64_t j  = offsets[ i + 0 ], jj = 0; 
                     j  < offsets[ i + 1 ]; 
                     j ++,                 jj ++ )
      {
        w_leaf[ jj ] = recvbuffs[ j ];
      }
    }
  }
};


template<typename T, typename TREE>
void PackFar( TREE &tree, string option, int p, 
    vector<size_t> &sendsizes, 
    vector<size_t> &sendskels, 
    vector<T> &sendbuffs )
{
  for ( auto it : tree.FarSentToRank[ p ] )
  {
    auto *node = tree.morton2node[ it ];
    auto &skels = node->data.skels;
    if ( !option.compare( string( "skelgids" ) ) )
    {
      sendsizes.push_back( skels.size() );
      sendskels.insert( sendskels.end(), skels.begin(), skels.end() );
    }
    else
    {
      auto &w_skel = node->data.w_skel;
      sendsizes.push_back( w_skel.size() );
      sendbuffs.insert( sendbuffs.end(), w_skel.begin(), w_skel.end() );
    }
  }
};


template<typename T, typename TREE>
void UnpackFar( TREE &tree, string option, int p, 
    const vector<size_t> &recvsizes, 
    const vector<size_t> &recvskels, 
    const vector<T> &recvbuffs )
{
  vector<size_t> offsets( 1, 0 );
  for ( auto it : recvsizes ) offsets.push_back( offsets.back() + it );

  for ( auto it : tree.FarRecvFromRank[ p ] )
  {
    /** Get LET node pointer */
    auto *node = tree.morton2node[ it.first ];
    if ( !option.compare( string( "skelgids" ) ) )
    {
      auto &skels = node->data.skels;
      size_t i = it.second;
      skels.clear();
      skels.reserve( recvsizes[ i ] );
      for ( uint64_t j  = offsets[ i + 0 ]; 
                     j  < offsets[ i + 1 ]; 
                     j ++ )
      {
        skels.push_back( recvskels[ j ] );
      }
    }
    else
    {
      /** Number of right hand sides */
      size_t nrhs = tree.setup.w->col();
      auto &w_skel = node->data.w_skel;
      size_t i = it.second;
      w_skel.resize( recvsizes[ i ] / nrhs, nrhs );
      //printf( "%d recv w_skel (%8lu) from %d [%lu %lu], i %lu, offset[%lu %lu] \n", 
      //    comm_rank, (*it).first, p, w_skel.row(), w_skel.col(), i,
      //    offsets[ p ][ i + 0 ], offsets[ p ][ i + 1 ] ); fflush( stdout );
      for ( uint64_t j  = offsets[ i + 0 ], jj = 0; 
                     j  < offsets[ i + 1 ]; 
                     j ++,                  jj ++ )
      {
        w_skel[ jj ] = recvbuffs[ j ];
        //if ( jj < 5 ) printf( "%E ", w_skel[ jj ] ); fflush( stdout );
      }
      //printf( "\n" ); fflush( stdout );
    }
  }
};


template<typename T, typename TREE>
class PackNearTask : public SendTask<T, TREE>
{
  public:

    string option;

    void DependencyAnalysis()
    {
      option = "leafweights";
      PackNear( *this->arg, option, this->tar,
          this->send_sizes, this->send_skels, this->send_buffs );

      TREE &tree = *(this->arg);
      for ( auto it : tree.NearSentToRank[ this->tar ] )
      {
        auto *node = tree.morton2node[ it ];
        node->DependencyAnalysis( R, this );
      }
      this->TryEnqueue();
    };

    void Execute( Worker *user_worker ) 
    {
      double beg = omp_get_wtime();

      mpi::Request req1, req2, req3;
      //option = "leafweights";
      //PackNear( *this->arg, option, this->tar,
      //    this->send_sizes, this->send_skels, this->send_buffs );

      //printf( "Beg PackNear src %d, tar %d, key %d send_sizes %lu send_skels %lu send_buff %lu\n", 
      //    this->src, this->tar, this->key, 
      //    this->send_sizes.size(), this->send_skels.size(), this->send_buffs.size() ); 
      //fflush( stdout );


      mpi::Isend( this->send_sizes.data(), this->send_sizes.size(),
          this->tar, this->key + 0, this->comm, &req1 ); 
      mpi::Isend( this->send_skels.data(), this->send_skels.size(),
          this->tar, this->key + 1, this->comm, &req2 ); 
      mpi::Isend( this->send_buffs.data(), this->send_buffs.size(),
          this->tar, this->key + 2, this->comm, &req3 ); 
      //printf( "End PackNear src %d\n", this->src ); fflush( stdout );
      double pack_time = omp_get_wtime() - beg;
      //printf( "PackNear src %d, tar %d, key %d time %lfs\n",  
      //    this->src, this->tar, this->key, pack_time ); fflush( stdout );
     
    };
};




/**
 *  AlltoallvTask is used perform MPI_Alltoallv in asynchronous.
 *  Overall there will be (p - 1) tasks per MPI rank. Each task
 *  performs Isend while the dependencies toward the destination
 *  is fullfilled. 
 *
 *  To receive the results, each MPI rank also actively runs a
 *  ListenerTask. Listener will keep pulling for incioming message
 *  that matches. Once the received results are secured, it will
 *  release dependent tasks.
 */ 
template<typename T, typename TREE>
class UnpackLeafTask : public RecvTask<T, TREE>
{
  public:

    string option;

    void DependencyAnalysis()
    {
      TREE &tree = *(this->arg);
      /** Read and write NearRecvFrom[ this->src ]. */
      hmlp_msg_dependency_analysis( this->key, this->src, RW, this );
      /** All L2L tasks may dependent on this task. */
      //for ( auto it : tree.NearRecvFromRank[ this->src ] )
      //{
      //  auto *node = tree.morton2node[ it.first ];
      //  node->DependencyAnalysis( RW, this );
      //}
    };

    void Execute( Worker *user_worker ) 
    {
      //printf( "Begin UnpackLeaf src %d\n", this->src ); fflush( stdout );
      double beg = omp_get_wtime();
      option = "leafweights";
      UnpackLeaf( *this->arg, option, this->src,
          this->recv_sizes, this->recv_skels, this->recv_buffs );
      double unpack_time = omp_get_wtime() - beg;
      //printf( "UnpackNear src %d tar %d key %d time %lfs\n",
      //    this->src, this->tar, this->key, unpack_time ); fflush( stdout );

      //printf( "Beg UnpackNear src %d, tar %d, key %d recv_sizes %lu recv_skels %lu recv_buff %lu\n", 
      //    this->src, this->tar, this->key, 
      //    this->recv_sizes.size(), this->recv_skels.size(), this->recv_buffs.size() ); 
      //fflush( stdout );
    };
};



template<typename T, typename TREE>
class PackFarTask : public SendTask<T, TREE>
{
  public:

    string option;

    void DependencyAnalysis()
    {
      TREE &tree = *(this->arg);
      for ( auto it : tree.FarSentToRank[ this->tar ] )
      {
        auto *node = tree.morton2node[ it ];
        node->DependencyAnalysis( R, this );
      }
      this->TryEnqueue();
    };

    void Execute( Worker *user_worker ) 
    {
      double beg = omp_get_wtime();

      mpi::Request req1, req2, req3;
      option = "skelweights";
      PackFar( *this->arg, option, this->tar,
          this->send_sizes, this->send_skels, this->send_buffs );

      //printf( "Beg PackFar src %d, tar %d, key %d send_sizes %lu send_skels %lu send_buff %lu\n", 
      //    this->src, this->tar, this->key, 
      //    this->send_sizes.size(), this->send_skels.size(), this->send_buffs.size() ); 
      //fflush( stdout );

      mpi::Isend( this->send_sizes.data(), this->send_sizes.size(),
          this->tar, this->key + 0, this->comm, &req1 ); 
      mpi::Isend( this->send_skels.data(), this->send_skels.size(),
          this->tar, this->key + 1, this->comm, &req2 ); 
      mpi::Isend( this->send_buffs.data(), this->send_buffs.size(),
          this->tar, this->key + 2, this->comm, &req3 ); 

      double pack_time = omp_get_wtime() - beg;
      //printf( "PackFar src %d, tar %d, key %d time %lfs\n",  
      //    this->src, this->tar, this->key, pack_time ); fflush( stdout );
    };
};


template<typename T, typename TREE>
class UnpackFarTask : public RecvTask<T, TREE>
{
  public:

    string option;

    void DependencyAnalysis()
    {
      /** Read and write NearRecvFrom[ this->src ]. */
      hmlp_msg_dependency_analysis( this->key, this->src, RW, this );
      /** All L2L tasks may dependent on this task. */



      //TREE &tree = *(this->arg);
      //for ( auto it : tree.FarRecvFromRank[ this->src ] )
      //{
      //  auto *node = tree.morton2node[ it.first ];
      //  node->DependencyAnalysis( RW, this );
      //}


    };

    void Execute( Worker *user_worker ) 
    {
      //printf( "End UnpackFar src %d, tar %d, key %d\n", 
      //    this->src, this->tar, this->key );
      //fflush( stdout );
      double beg = omp_get_wtime();
      option = "skelweights";
      UnpackFar( *this->arg, option, this->src,
          this->recv_sizes, this->recv_skels, this->recv_buffs );
      double unpack_time = omp_get_wtime() - beg;
      //printf( "UnpackFar src %d tar %d key %d time %lfs\n",
      //    this->src, this->tar, this->key, unpack_time ); fflush( stdout );

      //printf( "End UnpackFar src %d, tar %d, key %d recv_sizes %lu recv_skels %lu recv_buff %lu\n", 
      //    this->src, this->tar, this->key, 
      //    this->recv_sizes.size(), this->recv_skels.size(), this->recv_buffs.size() ); 
      //fflush( stdout );
    };
};











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
  
  /** Pack */
  #pragma omp parallel for
  for ( int p = 0; p < comm_size; p ++ )
  {
    if ( !option.compare( 0, 4, "leaf" ) )
    {
      PackNear( tree, option, p, sendsizes[ p ], sendskels[ p ], sendbuffs[ p ] );
    }
    else if ( !option.compare( 0, 4, "skel" ) )
    {
      PackFar( tree, option, p, sendsizes[ p ], sendskels[ p ], sendbuffs[ p ] );
    }
    else
    {
      printf( "ExchangeLET: option <%s> not available.\n", option.data() );
      exit( 1 );
    }
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
    double beg = omp_get_wtime();
    mpi::AlltoallVector( sendbuffs, recvbuffs, tree.comm );
    double a2av_time = omp_get_wtime() - beg;
    if ( comm_rank == 0 ) printf( "a2av_time %lfs\n", a2av_time );
  }


  /** Uppack */
  #pragma omp parallel for
  for ( int p = 0; p < comm_size; p ++ )
  {
    if ( !option.compare( 0, 4, "leaf" ) )
    {
      UnpackLeaf( tree, option, p, recvsizes[ p ], recvskels[ p ], recvbuffs[ p ] );
    }
    else if ( !option.compare( 0, 4, "skel" ) )
    {
      UnpackFar( tree, option, p, recvsizes[ p ], recvskels[ p ], recvbuffs[ p ] );
    }
    else
    {
      printf( "ExchangeLET: option <%s> not available.\n", option.data() );
      exit( 1 );
    }
  }


}; /** ExchangeLET() */



template<typename T, typename TREE>
void AsyncExchangeLET( TREE &tree, string option )
{
  /** MPI */
  int comm_size; mpi::Comm_size( tree.comm, &comm_size );
  int comm_rank; mpi::Comm_rank( tree.comm, &comm_rank );

  /** Create sending tasks. */
  for ( int p = 0; p < comm_size; p ++ )
  {
    if ( !option.compare( 0, 4, "leaf" ) )
    {
      auto *task = new PackNearTask<T, TREE>();
      /** Set src, tar, and key (tags). */
      task->Set( &tree, comm_rank, p, 300 );
      task->DependencyAnalysis();
      task->Submit();
    }
    else if ( !option.compare( 0, 4, "skel" ) )
    {
      auto *task = new PackFarTask<T, TREE>();
      /** Set src, tar, and key (tags). */
      task->Set( &tree, comm_rank, p, 306 );
      task->DependencyAnalysis();
      task->Submit();
    }
    else
    {
      printf( "AsyncExchangeLET: option <%s> not available.\n", option.data() );
      exit( 1 );
    }
  }

  /** Create receiving tasks */
  for ( int p = 0; p < comm_size; p ++ )
  {
    if ( !option.compare( 0, 4, "leaf" ) )
    {
      auto *task = new UnpackLeafTask<T, TREE>();
      /** Set src, tar, and key (tags). */
      task->Set( &tree, p, comm_rank, 300 );
      task->DependencyAnalysis();
      task->Submit();
    }
    else if ( !option.compare( 0, 4, "skel" ) )
    {
      auto *task = new UnpackFarTask<T, TREE>();
      /** Set src, tar, and key (tags). */
      task->Set( &tree, p, comm_rank, 306 );
      task->DependencyAnalysis();
      task->Submit();
    }
    else
    {
      printf( "AsyncExchangeLET: option <%s> not available.\n", option.data() );
      exit( 1 );
    }
  }

}; /** AsyncExchangeLET() */




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
    if ( (*it).second >= 0 && (*it).second < tree.n )
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
  //if ( node->isleaf )
  //{
  //   auto & NearMortonIDs = node->NNNearNodeMortonIDs;
  //   #pragma omp critical
  //   {
  //     int rank;
  //     mpi::Comm_rank( MPI_COMM_WORLD, &rank );
  //     string outfile = to_string( rank );
  //     FILE * pFile = fopen( outfile.data(), "a+" );
  //     fprintf( pFile, "(%8lu) ", node->morton );
  //     for ( auto it = NearMortonIDs.begin(); it != NearMortonIDs.end(); it ++ )
  //       fprintf( pFile, "%8lu, ", (*it) );
  //     fprintf( pFile, "\n" ); //fflush( stdout );
  //   }

  //   //auto & NearNodes = node->NNNearNodes;
  //   //for ( auto it = NearNodes.begin(); it != NearNodes.end(); it ++ )
  //   //{
  //   //  if ( !(*it)->NNNearNodes.count( node ) )
  //   //  {
  //   //    printf( "(%8lu) misses %lu\n", (*it)->morton, node->morton ); fflush( stdout );
  //   //  }
  //   //}
  //};


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
    /** Merge Far( lchild ) and Far( rchild ) from children */
    auto *lchild = node->lchild;
    auto *rchild = node->rchild;

    /** case: NNPRUNE (FMM specific) */ 
    auto &pNNFarNodes =   node->NNFarNodeMortonIDs;
    auto &lNNFarNodes = lchild->NNFarNodeMortonIDs;
    auto &rNNFarNodes = rchild->NNFarNodeMortonIDs;

    /** Far( parent ) = Far( lchild ) intersects Far( rchild ) */
    for ( auto it  = lNNFarNodes.begin(); 
               it != lNNFarNodes.end(); it ++ )
    {
      if ( rNNFarNodes.count( *it ) ) 
      {
        pNNFarNodes.insert( *it );
      }
    }
    /** Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) */
    for ( auto it  = pNNFarNodes.begin(); 
               it != pNNFarNodes.end(); it ++ )
    {
      lNNFarNodes.erase( *it ); 
      rNNFarNodes.erase( *it );
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
      arg->DependencyAnalysis( RW, this );
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
  /** MPI */ 
  mpi::Status status;
  mpi::Comm comm = node->GetComm();
  int comm_size = node->GetCommSize();
  int comm_rank = node->GetCommRank();
  int glb_rank; mpi::Comm_rank( MPI_COMM_WORLD, &glb_rank );

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

    //printf( "glb_rank %d level %lu\n", glb_rank, node->l ); fflush( stdout );
    //mpi::Barrier( MPI_COMM_WORLD );
    


    if ( comm_rank == 0 )
    {
      //printf( "glb_rank %d level %lu\n", glb_rank, node->l ); fflush( stdout );
      auto &pNNFarNodes =  node->NNFarNodeMortonIDs;
      auto &lNNFarNodes = child->NNFarNodeMortonIDs;
      vector<size_t> recvFarNodes;

      /** Recv rNNFarNodes */ 
      mpi::RecvVector( recvFarNodes, comm_size / 2, 0, comm, &status );

      /** Far( parent ) = Far( lchild ) intersects Far( rchild ) */
      for ( auto it  = recvFarNodes.begin(); 
                 it != recvFarNodes.end(); it ++ )
      {
        if ( lNNFarNodes.count( *it ) ) 
        {
          pNNFarNodes.insert( *it );
          //node->NNFarNodes.insert( (*node->morton2node)[ *it ] );
        }
      }

      //printf( "%8lu level %lu Far ", node->morton, node->l ); 
      //for ( auto it = pNNFarNodes.begin(); it != pNNFarNodes.end(); it ++ )
      //{
      //  printf( "%8lu ", *it ); fflush( stdout );
      //}
      //printf( "\n" ); fflush( stdout );


      /** Reuse space to send pNNFarNodes */ 
      recvFarNodes.clear();
      recvFarNodes.reserve( pNNFarNodes.size() );

      /** Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) */
      for ( auto it = pNNFarNodes.begin(); it != pNNFarNodes.end(); it ++ )
      {
        lNNFarNodes.erase( *it ); 
        recvFarNodes.push_back( *it );
        //child->NNFarNodes.erase( (*node->morton2node)[ *it ] );
      }

      /** Send pNNFarNodes */
      mpi::SendVector( recvFarNodes, comm_size / 2, 0, comm );
      //printf( "glb_rank %d level %lu done\n", glb_rank, node->l ); fflush( stdout );
    }


    if ( comm_rank == comm_size / 2 )
    {
      //printf( "glb_rank %d level %lu\n", glb_rank, node->l ); fflush( stdout );
      auto &rNNFarNodes = child->NNFarNodeMortonIDs;
      vector<size_t> sendFarNodes;

      sendFarNodes.reserve( rNNFarNodes.size() );
      for ( auto it  = rNNFarNodes.begin(); 
                 it != rNNFarNodes.end(); it ++ )
      {
        sendFarNodes.push_back( *it );
      }

      /** Send rNNFarNodes */ 
      mpi::SendVector( sendFarNodes, 0, 0, comm );

      /** Recv pNNFarNodes */
      mpi::RecvVector( sendFarNodes, 0, 0, comm, &status );

      /** Far( lchild ) \= Far( parent ); Far( rchild ) \= Far( parent ) */
      for ( auto it = sendFarNodes.begin(); it != sendFarNodes.end(); it ++ )
      {
        rNNFarNodes.erase( *it );
        //child->NNFarNodes.erase( (*node->morton2node)[ *it ] );
      }      
      //printf( "glb_rank %d level %lu done\n", glb_rank, node->l ); fflush( stdout );
    }

    //mpi::Barrier( MPI_COMM_WORLD );
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
      arg->DependencyAnalysis( RW, this );
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
      int glb_rank; mpi::Comm_rank( MPI_COMM_WORLD, &glb_rank );
      //printf( "rank %d DistMergeFarNodes begin\n", glb_rank ); fflush( stdout );
      mpi::Barrier( arg->GetComm() );
      DistMergeFarNodes<SYMMETRIC, NODE, T>( arg );
      //printf( "rank %d DistMergeFarNodes end\n", glb_rank ); fflush( stdout );
      mpi::Barrier( arg->GetComm() );
    };

}; /** end class DistMergeFarNodesTask */






/**
 *
 */ 
template<bool NNPRUNE, typename NODE>
class CacheFarNodesTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "FKIJ" );
      label = to_string( arg->treelist_id );
      /** Compute FLOPS and MOPS. */
      double flops = 0, mops = 0;
      /** We don't know the exact cost here. */
      cost = 5.0;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      auto *node = arg;
      auto &K = *node->setup->K;

      for ( int p = 0; p < node->DistFar.size(); p ++ )
      {
        for ( auto &it : node->DistFar[ p ] )
        {
          auto *src = (*node->morton2node)[ it.first ];
          auto &I = node->data.skels;
          auto &J = src->data.skels;
          it.second = K( I, J );
          //printf( "Cache I %lu J %lu\n", I.size(), J.size() ); fflush( stdout );
        }
      }
    };

}; /** end class CacheFarNodesTask */






/**
 *
 */ 
template<bool NNPRUNE, typename NODE>
class CacheNearNodesTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "NKIJ" );
      label = to_string( arg->treelist_id );
      /** We don't know the exact cost here */
      cost = 5.0;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      auto *node = arg;
      auto &K = *node->setup->K;

      for ( int p = 0; p < node->DistNear.size(); p ++ )
      {
        for ( auto &it : node->DistNear[ p ] )
        {
          auto *src = (*node->morton2node)[ it.first ];
          auto &I = node->gids;
          auto &J = src->gids;
          it.second = K( I, J );
          //printf( "Cache I %lu J %lu\n", I.size(), J.size() ); fflush( stdout );
        }
      }
    };

}; /** end class CacheNearNodesTask */












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
  vector<size_t> &I = node->data.candidate_rows;

  /** Clean up candidates from previous iteration */
	I.clear();

  /** Fill-on snids first */
	if ( rank == 0 ) 
  {
    /** reserve space */
    I.reserve( nsamples );

    auto &snids = node->data.snids;
    multimap<T, size_t> ordered_snids = gofmm::flip_map( snids );

    for ( auto it  = ordered_snids.begin(); 
               it != ordered_snids.end(); it++ )
    {
      /** (*it) has type pair<T, size_t> */
      I.push_back( (*it).second );
      if ( I.size() >= nsamples ) break;
    }
  }

	/** buffer space */
	vector<size_t> candidates( nsamples );

	size_t n_required = nsamples - I.size();

	/** bcast the termination criteria */
	mpi::Bcast( &n_required, 1, 0, comm );

	while ( n_required )
	{
		if ( rank == 0 )
		{
  	  for ( size_t i = 0; i < nsamples; i ++ )
      {
        auto important_sample = K.ImportantSample( 0 );
        candidates[ i ] =  important_sample.second;
      }
		}

		/** Bcast candidates */
		mpi::Bcast( candidates.data(), candidates.size(), 0, comm );

		/** validation */
		vector<size_t> vconsensus( nsamples, 0 );
	  vector<size_t> validation = node->setup->ContainAny( candidates, node->morton );

		/** reduce validation */
		mpi::Reduce( validation.data(), vconsensus.data(), nsamples, MPI_SUM, 0, comm );

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
				/** Push the candidate to I after validation */
				if ( !vconsensus[ i ] )
				{
					if ( find( I.begin(), I.end(), candidates[ i ] ) == I.end() )
						I.push_back( candidates[ i ] );
				}
			};

			/** Update n_required */
	    n_required = nsamples - I.size();
		}

	  /** Bcast the termination criteria */
	  mpi::Bcast( &n_required, 1, 0, comm );
	}

}; /** end DistRowSamples() */





/**
 *  @brief Involve MPI routines. All MPI process must devote at least
 *         one thread to participate in this function to avoid deadlock.
 */ 
//template<typename NODE, typename T>
//void GetSkeletonMatrix( NODE *node )
//{
//  /** gather shared data and create reference */
//  auto &K = *(node->setup->K);
//
//  /** gather per node data and create reference */
//  auto &data = node->data;
//  auto &candidate_rows = data.candidate_rows;
//  auto &candidate_cols = data.candidate_cols;
//  auto &KIJ = data.KIJ;
//
//  /** this node belongs to the local tree */
//  auto *lchild = node->lchild;
//  auto *rchild = node->rchild;
//
//  if ( node->isleaf )
//  {
//    /** use all columns */
//    candidate_cols = node->gids;
//  }
//  else
//  {
//    /** concatinate [ lskels, rskels ] */
//    auto &lskels = lchild->data.skels;
//    auto &rskels = rchild->data.skels;
//    candidate_cols = lskels;
//    candidate_cols.insert( candidate_cols.end(), 
//        rskels.begin(), rskels.end() );
//  }
//
//  size_t nsamples = 2 * candidate_cols.size();
//
//  /** make sure we at least m samples */
//  if ( nsamples < 2 * node->setup->m ) nsamples = 2 * node->setup->m;
//
//  /** build  neighbor lists (snids and pnids) */
//  //gofmm::BuildNeighbors<NODE, T>( node, nsamples );
//
//  /** sample off-diagonal rows */
//  gofmm::RowSamples<NODE, T>( node, nsamples );
//
//  /** 
//   *  get KIJ for skeletonization 
//   *
//   *  notice that operator () may involve MPI collaborative communication.
//   *
//   */
//  //KIJ = K( candidate_rows, candidate_cols );
//  size_t over_size_rank = node->setup->s + 20;
//  if ( candidate_rows.size() <= over_size_rank )
//  {
//    KIJ = K( candidate_rows, candidate_cols );
//  }
//  else
//  {
//    auto Ksamples = K( candidate_rows, candidate_cols );
//    /**
//     *  Compute G * KIJ
//     */
//    KIJ.resize( over_size_rank, candidate_cols.size() );
//    Data<T> G( over_size_rank, nsamples ); G.randn( 0, 1 );
//
//    View<T> Ksamples_v( false, Ksamples );
//    View<T> KIJ_v( false, KIJ );
//    View<T> G_v( false, G );
//
//    /** KIJ = G * Ksamples */
//    gemm::xgemm<GEMM_NB>( (T)1.0, G_v, Ksamples_v, (T)0.0, KIJ_v );
//  }
//
//}; /** end GetSkeletonMatrix() */


/**
 *
 */ 
//template<typename NODE, typename T>
//class GetSkeletonMatrixTask : public hmlp::Task
//{
//  public:
//
//    NODE *arg;
//
//    void Set( NODE *user_arg )
//    {
//      std::ostringstream ss;
//      arg = user_arg;
//      name = std::string( "par-gskm" );
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
//    void GetEventRecord()
//    {
//    };
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
//      //printf( "%lu GetSkeletonMatrix beg\n", arg->treelist_id );
//      double beg = omp_get_wtime();
//      
//      /** Create a promise and get its future */
//      promise<bool> done;
//      auto future = done.get_future();
//
//      thread t( [&done] ( NODE *arg ) -> void {
//          gofmm::GetSkeletonMatrix<NODE, T>( arg ); 
//          done.set_value( true );
//      }, arg );
//      
//      /** Polling the future status */
//      while ( future.wait_for( chrono::seconds( 0 ) ) != future_status::ready ) 
//      {
//        this->ContextSwitchToNextTask( user_worker );
//      }
//      
//      /** Make sure the task is completed */
//      t.join();
//
//      //gofmm::GetSkeletonMatrix<NODE, T>( arg );
//      double getmtx_t = omp_get_wtime() - beg;
//      //printf( "%lu GetSkeletonMatrix end %lfs\n", arg->treelist_id, getmtx_t ); fflush( stdout );
//    };
//
//}; /** end class GetSkeletonMatrixTask */









/**
 *  @brief Involve MPI routins
 */ 
template<typename NODE, typename T>
void ParallelGetSkeletonMatrix( NODE *node )
{
	/** Early return */
	if ( !node->parent ) return;

  /** Gather shared data and create reference */
  auto &K = *(node->setup->K);

  /** Gather per node data and create reference */
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
      vector<size_t> rskel;
      mpi::RecvVector( rskel, size / 2, 10, comm, &status );

		  //printf( "rank %d level %lu RecvColumns\n", global_rank, node->l ); fflush( stdout );
      K.RecvColumns( size / 2, comm, &status );
		  //printf( "rank %d level %lu RecvColumns done\n", global_rank, node->l ); fflush( stdout );

      candidate_cols.insert( candidate_cols.end(), rskel.begin(), rskel.end() );

			/** use two times of skeletons */
      nsamples = 2 * candidate_cols.size();

      /** make sure we at least m samples */
      if ( nsamples < 2 * node->setup->m ) nsamples = 2 * node->setup->m;

      /** gather rpnids and rsnids */
      auto &lsnids = node->child->data.snids;
      auto &lpnids = node->child->data.pnids;
      vector<T>      recv_rsdist;
      vector<size_t> recv_rsnids;
      vector<size_t> recv_rpnids;

      /** recv rsnids from size / 2 */
      mpi::RecvVector( recv_rsdist, size / 2, 20, comm, &status );
      mpi::RecvVector( recv_rsnids, size / 2, 30, comm, &status );
      mpi::RecvVector( recv_rpnids, size / 2, 40, comm, &status );


      /** Receive rsnids  */
		  //printf( "rank %d level %lu RecvColumns (rsnids)\n", global_rank, node->l ); fflush( stdout );
      K.RecvColumns( size / 2, comm, &status );
		  //printf( "rank %d level %lu RecvColumns (rsnids) done\n", global_rank, node->l ); fflush( stdout );


      /** merge snids and update the smallest distance */
      auto &snids = node->data.snids;
      snids = lsnids;

      for ( size_t i = 0; i < recv_rsdist.size(); i ++ )
      {
        pair<size_t, T> query( recv_rsnids[ i ], recv_rsdist[ i ] );
        auto ret = snids.insert( query );
        if ( !ret.second )
        {
          if ( ret.first->second > recv_rsdist[ i ] )
            ret.first->second = recv_rsdist[ i ];
        }
      }

      /** Remove gids, lpnids and rpnids from snids */
      for ( auto gid : node->gids ) snids.erase( gid );

      //for ( size_t i = 0; i < gids.size(); i ++ ) snids.erase( gids[ i ] );

      /** Remove lpnids and rpnids from snids  */
      for ( auto lpnid : lpnids ) snids.erase( lpnid );
      //for ( auto it = lpnids.begin(); it != lpnids.end(); it ++ ) snids.erase( *it );
      for ( auto rpnid : recv_rpnids ) snids.erase( rpnid );
      //for ( size_t i = 0; i < recv_rpnids.size(); i ++ ) snids.erase( recv_rpnids[ i ] );
    }

    if ( rank == size / 2 )
    {
      /** Send rskel to rank-0 */
      mpi::SendVector( child->data.skels, 0, 10, comm );
		  //printf( "rank %d level %lu SendColumnn\n", global_rank, node->l ); fflush( stdout );
      K.SendColumns( child->data.skels, 0, comm );

      /** gather rsnids */
      auto &rsnids = node->child->data.snids;
      auto &rpnids = node->child->data.pnids;
      vector<T>      send_rsdist;
      vector<size_t> send_rsnids;
      vector<size_t> send_rpnids;

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


		  //printf( "rank %d level %lu SendColumns (rsnids)\n", global_rank, node->l ); fflush( stdout );
      K.SendColumns( send_rsnids, 0, comm );

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
    //if ( candidate_rows.size() <= over_size_rank )
    if ( 1 )
    {
      KIJ = K( candidate_rows, candidate_cols );
    }
    else
    {
      auto Ksamples = K( candidate_rows, candidate_cols );
      /** Compute G * KIJ */
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
class ParallelGetSkeletonMatrixTask : public Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "par-gskm" );
      label = to_string( arg->treelist_id );
      /** We don't know the exact cost here */
      cost = 5.0;
      /** "High" priority */
      priority = true;
    };

    void GetEventRecord() {};


    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );

      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() < 2 )
        {
          //printf( "shared case\n" ); fflush( stdout );
          arg->lchild->DependencyAnalysis( R, this );
          arg->rchild->DependencyAnalysis( R, this );
        }
        else
        {
          //printf( "distributed case size %d\n", arg->GetCommSize() ); fflush( stdout );
          arg->child->DependencyAnalysis( R, this );
        }
      }

      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      //printf( "%lu Par-GetSkeletonMatrix beg\n", arg->morton );
      ParallelGetSkeletonMatrix<NODE, T>( arg );
      //printf( "%lu Par-GetSkeletonMatrix end\n", arg->morton );
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

  /** Free KIJ for spaces */
  KIJ.resize( 0, 0 );
  KIJ.shrink_to_fit();

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
  
  /** Relabel skeletions with the real gids */
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
      arg = user_arg;
      name = string( "SK" );
      label = to_string( arg->treelist_id );
      /** We don't know the exact cost here */
      cost = 5.0;
      /** "High" priority */
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
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      //printf( "%d Par-Skel beg\n", global_rank );

      DistSkeletonize<ADAPTIVE, LEVELRESTRICTION, NODE, T>( arg );

      /** If neighbors were provided, then update the estimated prunning list. */
      if ( arg->setup->NN )
      {
        auto &skels = arg->data.skels;
        auto &pnids = arg->data.pnids;
        auto &NN = *(arg->setup->NN);
        pnids.clear();

        for ( auto skel : skels )
          //for ( size_t i = 0; i < NN.row() / 2; i ++ )
          for ( size_t i = 0; i < NN.row(); i ++ )
            pnids.insert( NN( i, skel ).second );
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
      arg = user_arg;
      name = string( "PSK" );
      label = to_string( arg->treelist_id );

      /** We don't know the exact cost here */
      cost = 5.0;
      /** "High" priority */
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
			arg->data.isskel = isskel;

      /** Bcast skels and proj to every MPI processes in the same comm */
      auto &skels = arg->data.skels;
      size_t nskels = skels.size();
      mpi::Bcast( &nskels, 1, 0, comm );
      if ( skels.size() != nskels ) skels.resize( nskels );
      mpi::Bcast( skels.data(), skels.size(), 0, comm );


      /** If neighbors were provided, then update the estimated prunning list. */
      if ( arg->setup->NN )
      {
        auto &pnids = arg->data.pnids;
        auto &NN = *(arg->setup->NN);

        /** Create the column distribution using cids */
        vector<size_t> cids;
        for ( size_t j = 0; j < skels.size(); j ++ )
        {
          if ( NN.HasColumn( skels[ j ] ) ) 
          {
            cids.push_back( j );
          }
        }

        /** Create a k-by-nskels distributed matrix */
        DistData<STAR, CIDS, size_t> X_cids( NN.row(), nskels, cids, comm );
        DistData<CIRC, CIRC, size_t> X_circ( NN.row(), nskels,    0, comm );

        /** Redistribute from <STAR, CIDS> to <CIRC, CIRC> on rank-0 */
        X_circ = X_cids;

        if ( arg->GetCommRank() == 0 )
        {
          pnids.clear();
          for ( size_t i = 0; i < X_circ.size(); i ++ )
            pnids.insert( X_circ[ i ] );
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
      arg = user_arg;
      name = string( "PROJ" );
      label = to_string( arg->treelist_id );
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


  /** Get type NODE = TREE::NODE */
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;
  using LETNODE = typename TREE::LETNODE;

  /** All timers */
  double beg, time_ratio, evaluation_time = 0.0;
  double direct_evaluation_time = 0.0, computeall_time, telescope_time, let_exchange_time, async_time;
  double overhead_time;
  double forward_permute_time, backward_permute_time;

  /** Clean up all r/w dependencies left on tree nodes */
  tree.DependencyCleanUp();

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
    printf( "N2S, S2S, S2N, L2L (HMLP Runtime) ...\n" ); 
    fflush( stdout );
  }

  if ( SYMMETRIC_PRUNE )
  {
    /** TreeView (downward traversal) */
    gofmm::TreeViewTask<NODE>           seqVIEWtask;
    mpigofmm::DistTreeViewTask<MPINODE> mpiVIEWtask;
    /** Telescope (upward traversal) */
    gofmm::UpdateWeightsTask<NODE, T>           seqN2Stask;
    mpigofmm::DistUpdateWeightsTask<MPINODE, T> mpiN2Stask;
    /** L2L (sum of direct evaluations) */
    mpigofmm::DistLeavesToLeavesTask<NNPRUNE, NODE, T> seqL2Ltask;
    //mpigofmm::L2LReduceTask<NODE, T> seqL2LReducetask;
    mpigofmm::L2LReduceTask2<NODE, T> seqL2LReducetask2;
    /** S2S (sum of low-rank approximation) */
    gofmm::SkeletonsToSkeletonsTask<NNPRUNE, NODE, T>           seqS2Stask;
    mpigofmm::DistSkeletonsToSkeletonsTask<NNPRUNE, MPINODE, T> mpiS2Stask;
    //mpigofmm::S2SReduceTask<NODE, T>    seqS2SReducetask;
    //mpigofmm::S2SReduceTask<MPINODE, T> mpiS2SReducetask;
    mpigofmm::S2SReduceTask2<NODE, NODE, T>    seqS2SReducetask2;
    mpigofmm::S2SReduceTask2<MPINODE, NODE, T> mpiS2SReducetask2;
    /** Telescope (downward traversal) */
    gofmm::SkeletonsToNodesTask<NNPRUNE, NODE, T>           seqS2Ntask;
    mpigofmm::DistSkeletonsToNodesTask<NNPRUNE, MPINODE, T> mpiS2Ntask;

    /** Global barrier and timer */
    mpi::Barrier( tree.comm );

    //{
    //  /** Stage 1: TreeView and upward telescoping */
    //  beg = omp_get_wtime();
    //  tree.DependencyCleanUp();
    //  tree.DistTraverseDown( mpiVIEWtask );
    //  tree.LocaTraverseDown( seqVIEWtask );
    //  tree.LocaTraverseUp( seqN2Stask );
    //  tree.DistTraverseUp( mpiN2Stask );
    //  hmlp_run();
    //  mpi::Barrier( tree.comm );
    //  telescope_time = omp_get_wtime() - beg;

    //  /** Stage 2: LET exchange */
    //  beg = omp_get_wtime();
    //  ExchangeLET<T>( tree, string( "skelweights" ) );
    //  mpi::Barrier( tree.comm );
    //  ExchangeLET<T>( tree, string( "leafweights" ) );
    //  mpi::Barrier( tree.comm );
    //  let_exchange_time = omp_get_wtime() - beg;

    //  /** Stage 3: L2L */
    //  beg = omp_get_wtime();
    //  tree.DependencyCleanUp();
    //  tree.LocaTraverseLeafs( seqL2LReducetask2 );
    //  hmlp_run();
    //  mpi::Barrier( tree.comm );
    //  direct_evaluation_time = omp_get_wtime() - beg;

    //  /** Stage 4: S2S and downward telescoping */
    //  beg = omp_get_wtime();
    //  tree.DependencyCleanUp();
    //  tree.LocaTraverseUnOrdered( seqS2SReducetask2 );
    //  tree.DistTraverseUnOrdered( mpiS2SReducetask2 );
    //  tree.DistTraverseDown( mpiS2Ntask );
    //  tree.LocaTraverseDown( seqS2Ntask );
    //  hmlp_run();
    //  mpi::Barrier( tree.comm );
    //  computeall_time = omp_get_wtime() - beg;
    //}

    if ( rank == 0 && REPORT_EVALUATE_STATUS )
    {
      printf( "Finish synchronous evaluation ...\n" ); 
      fflush( stdout );
    }

    /** Global barrier and timer */
    potentials.setvalue( 0.0 );
    mpi::Barrier( tree.comm );
    {
      /** Stage 1: TreeView and upward telescoping */
      beg = omp_get_wtime();
      tree.DependencyCleanUp();
      tree.DistTraverseDown( mpiVIEWtask );
      tree.LocaTraverseDown( seqVIEWtask );
      hmlp_run();
      mpi::Barrier( tree.comm );




      tree.DependencyCleanUp();
      AsyncExchangeLET<T>( tree, string( "leafweights" ) );
      //hmlp_run();
      //mpi::Barrier( tree.comm );

      //tree.DependencyCleanUp();
      tree.LocaTraverseUp( seqN2Stask );
      tree.DistTraverseUp( mpiN2Stask );

      /** Stage 2: LET exchange */
      //AsyncExchangeLET<T>( tree, string( "leafweights" ) );
      AsyncExchangeLET<T>( tree, string( "skelweights" ) );

      /** Stage 3: L2L */
      tree.LocaTraverseLeafs( seqL2LReducetask2 );

      /** Stage 4: S2S, S2N */
      tree.LocaTraverseUnOrdered( seqS2SReducetask2 );
      tree.DistTraverseUnOrdered( mpiS2SReducetask2 );
      tree.DistTraverseDown( mpiS2Ntask );
      tree.LocaTraverseDown( seqS2Ntask );
      overhead_time = omp_get_wtime() - beg;
      hmlp_run();
      mpi::Barrier( tree.comm );
      async_time = omp_get_wtime() - beg;
    }

  }
  else 
  {
    /** TODO: implement unsymmetric prunning */
    printf( "Non symmetric Distributed ComputeAll is not yet implemented\n" );
    exit( 1 );
  }

  /** Compute the breakdown cost */
  evaluation_time += direct_evaluation_time;
  evaluation_time += telescope_time;
  evaluation_time += let_exchange_time;
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
    printf( "Upward telescope ---------------------- %5.2lfs (%5.1lf%%)\n", 
        telescope_time, telescope_time * time_ratio );
    printf( "LET exchange -------------------------- %5.2lfs (%5.1lf%%)\n", 
        let_exchange_time, let_exchange_time * time_ratio );
    printf( "L2L ----------------------------------- %5.2lfs (%5.1lf%%)\n", 
        direct_evaluation_time, direct_evaluation_time * time_ratio );
    printf( "S2S, S2N ------------------------------ %5.2lfs (%5.1lf%%)\n", 
        computeall_time, computeall_time * time_ratio );
    //printf( "Backward permute ---------------------- %5.2lfs (%5.1lf%%)\n", 
    //    backward_permute_time, backward_permute_time * time_ratio );
    printf( "========================================================\n");
    printf( "Evaluate ------------------------------ %5.2lfs (%5.1lf%%)\n", 
        evaluation_time, evaluation_time * time_ratio );
    printf( "Evaluate (Async) ---------------------- %5.2lfs (%5.2lfs)\n", 
        async_time, overhead_time );
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
  int size; mpi::Comm_size( MPI_COMM_WORLD, &size );
  int rank; mpi::Comm_rank( MPI_COMM_WORLD, &rank );

  /** Get all user-defined parameters */
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
  double ann_time, tree_time, skel_time, mpi_skel_time, mergefarnodes_time, cachefarnodes_time;
  double local_skel_time, dist_skel_time, let_time; 
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;
  double exchange_neighbor_time, symmetrize_time;

  /** Iterative all nearnest-neighbor (ANN) */
  const size_t n_iter = 3;
  const bool SORTED = false;
  /** Do not change anything below this line */
  mpitree::Tree<RKDTSETUP, DATA, N_CHILDREN, T> rkdt;
  rkdt.setup.X_cblk = X_cblk;
  rkdt.setup.K = &K;
	rkdt.setup.metric = metric; 
  rkdt.setup.splitter = rkdtsplitter;
  pair<T, size_t> initNN( numeric_limits<T>::max(), n );

  mpi::PrintProgress( "NeighborSearch ...\n", MPI_COMM_WORLD );
  beg = omp_get_wtime();
  if ( NN_cblk.row() != k )
  {
    NeighborsTask<RKDTNODE, T> knntask;
    //knntask.metric = metric;
    NN_cblk = rkdt.AllNearestNeighbor<SORTED>( n_iter, n, k, 15, initNN, knntask );
  }
  else
  {
    mpi::PrintProgress( "Not performed (precomputed or k=0) ...\n", MPI_COMM_WORLD );
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



  /** Initialize metric ball tree using approximate center split */
  auto *tree_ptr = new mpitree::Tree<SETUP, DATA, N_CHILDREN, T>();
	auto &tree = *tree_ptr;

	/** Global configuration for the metric tree */
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

	/** Metric ball tree partitioning */
  mpi::PrintProgress( "TreePartitioning ...\n", tree.comm ); 
  mpi::Barrier( tree.comm );
  beg = omp_get_wtime();
  tree.TreePartition( n );
  mpi::Barrier( tree.comm );
  tree_time = omp_get_wtime() - beg;
  mpi::PrintProgress( "End TreePartitioning ...\n", tree.comm ); 

  /** Now redistribute K. */
  K.Redistribute( true, tree.treelist[ 0 ]->gids );
  mpi::PrintProgress( "Finish redistribute K\n", tree.comm );

  /** Redistribute neighbors i.e. NN[ *, CIDS ] = NN[ *, CBLK ]; */
  DistData<STAR, CIDS, pair<T, size_t>> NN( k, n, tree.treelist[ 0 ]->gids, tree.comm );
  NN = NN_cblk;
  tree.setup.NN = &NN;
  beg = omp_get_wtime();
  ExchangeNeighbors<T>( tree );
  exchange_neighbor_time = omp_get_wtime() - beg;
  mpi::PrintProgress( "Finish redistribute neighbors ...\n", tree.comm );

  /** Construct near interaction lists */
  /**
   *  Symmetrize near interaction list:
   *
   *  for each alpha in owned_leaf and beta in Near(alpha)
   *      sbuff[ rank(beta) ] += pair( morton(beta), morton(alpha) );
   *  Alltoallv( sbuff, rbuff );
   *  for each pair( beta, alpha ) in rbuff
   *      near( beta ) += alpha;
   */
  beg = omp_get_wtime();
  FindNearNodes<NODE>( tree );
  mpi::PrintProgress( "Finish FindNearNodes ...\n", tree.comm );
  SymmetrizeNearInteractions<NODE>( tree );
  mpi::Barrier( tree.comm );
  BuildInteractionListPerRank<T>( tree, true );
  mpi::Barrier( tree.comm );
  /** Exchange {leafs} and {paramsleafs)}  */
  ExchangeLET<T>( tree, string( "leafgids" ) );
  symmetrize_time - omp_get_wtime() - beg;
  mpi::PrintProgress( "Finish SymmetrizeNearInteractions ...\n", tree.comm ); 


  /** Merge FarNodes */
  beg = omp_get_wtime();
  tree.DependencyCleanUp();
  MergeFarNodesTask<true, NODE, T> seqMERGEtask;
  DistMergeFarNodesTask<true, MPINODE, T> mpiMERGEtask;
  tree.LocaTraverseUp( seqMERGEtask );
  hmlp_run();
  mpi::Barrier( tree.comm );
  tree.DependencyCleanUp();
  tree.DistTraverseUp( mpiMERGEtask );
  hmlp_run();
  mpi::Barrier( tree.comm );
  mergefarnodes_time += omp_get_wtime() - beg;
  mpi::PrintProgress( "Finish MergeFarNodes ...\n", tree.comm ); 

  /**
   *  Symmetrize far interaction list:
   *
   *  for each alpha in owned_node and beta in Far(alpha)
   *      sbuff[ rank(beta) ] += pair( morton(beta), morton(alpha) );
   *  Alltoallv( sbuff, rbuff );
   *  for each pair( beta, alpha ) in rbuff
   *      Far( beta ) += alpha;
   */
  beg = omp_get_wtime();
  SymmetrizeFarInteractions<NODE>( tree );
  mpi::Barrier( tree.comm );
  BuildInteractionListPerRank<T>( tree, false );
  mpi::Barrier( tree.comm );
  symmetrize_time += omp_get_wtime() - beg;
  mpi::PrintProgress( "Finish SymmetrizeFarInteractions ...\n", tree.comm ); 

  /** Skeletonization */
	beg = omp_get_wtime();
  tree.DependencyCleanUp();
  /** Gather sample rows and skeleton columns, then ID */
  gofmm::GetSkeletonMatrixTask<NODE, T> seqGETMTXtask;
  mpigofmm::ParallelGetSkeletonMatrixTask<MPINODE, T> mpiGETMTXtask;
  mpigofmm::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T> seqSKELtask;
  mpigofmm::DistSkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, MPINODE, T> mpiSKELtask;
  tree.LocaTraverseUp( seqGETMTXtask, seqSKELtask );
  //tree.DistTraverseUp( mpiGETMTXtask, mpiSKELtask );
  /** Compute the coefficient matrix of ID */
  gofmm::InterpolateTask<NODE, T> seqPROJtask;
  mpigofmm::InterpolateTask<MPINODE, T> mpiPROJtask;
  tree.LocaTraverseUnOrdered( seqPROJtask );
  //tree.DistTraverseUnOrdered( mpiPROJtask );

  /** Cache near KIJ interactions */
  mpigofmm::CacheNearNodesTask<NNPRUNE, NODE> seqNEARKIJtask;
  //tree.LocaTraverseLeafs( seqNEARKIJtask );

  hmlp_run();
  mpi::Barrier( tree.comm );
  skel_time = omp_get_wtime() - beg;
  mpi::PrintProgress( "Finish skeletonization and interpolation ...\n", tree.comm ); 



	beg = omp_get_wtime();
  tree.DependencyCleanUp();
  tree.DistTraverseUp( mpiGETMTXtask, mpiSKELtask );
  tree.DistTraverseUnOrdered( mpiPROJtask );
  hmlp_run();
  mpi::Barrier( tree.comm );
  mpi_skel_time = omp_get_wtime() - beg;



  /** Exchange {skels} and {params(skels)}  */
  ExchangeLET<T>( tree, string( "skelgids" ) );
  
  beg = omp_get_wtime();
  tree.DependencyCleanUp();
  /** Cache near KIJ interactions */
  //mpigofmm::CacheNearNodesTask<NNPRUNE, NODE> seqNEARKIJtask;
  //tree.LocaTraverseLeafs( seqNEARKIJtask );
  /** Cache far KIJ interactions */
  mpigofmm::CacheFarNodesTask<NNPRUNE,    NODE> seqFARKIJtask;
  mpigofmm::CacheFarNodesTask<NNPRUNE, MPINODE> mpiFARKIJtask;
  //tree.LocaTraverseUnOrdered( seqFARKIJtask );
  //tree.DistTraverseUnOrdered( mpiFARKIJtask );
  cachefarnodes_time = omp_get_wtime() - beg;
  hmlp_run();
  mpi::Barrier( tree.comm );
  cachefarnodes_time = omp_get_wtime() - beg;



  /** Compute the ratio of exact evaluation */
  double exact_ratio = (double) m / n; 

  if ( rank == 0 && REPORT_COMPRESS_STATUS )
  {
    compress_time += ann_time;
    compress_time += tree_time;
    compress_time += exchange_neighbor_time;
    compress_time += symmetrize_time;
    compress_time += skel_time;
    compress_time += mpi_skel_time;
    compress_time += mergefarnodes_time;
    compress_time += cachefarnodes_time;
    time_ratio = 100.0 / compress_time;
    printf( "========================================================\n");
    printf( "GOFMM compression phase\n" );
    printf( "========================================================\n");
    printf( "NeighborSearch ------------------------ %5.2lfs (%5.1lf%%)\n", ann_time, ann_time * time_ratio );
    printf( "TreePartitioning ---------------------- %5.2lfs (%5.1lf%%)\n", tree_time, tree_time * time_ratio );
    printf( "ExchangeNeighbors --------------------- %5.2lfs (%5.1lf%%)\n", exchange_neighbor_time, exchange_neighbor_time * time_ratio );
    printf( "MergeFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", mergefarnodes_time, mergefarnodes_time * time_ratio );
    printf( "Symmetrize ---------------------------- %5.2lfs (%5.1lf%%)\n", symmetrize_time, symmetrize_time * time_ratio );
    printf( "Skeletonization (HMLP Runtime   ) ----- %5.2lfs (%5.1lf%%)\n", skel_time, skel_time * time_ratio );
    printf( "Skeletonization (MPI            ) ----- %5.2lfs (%5.1lf%%)\n", mpi_skel_time, mpi_skel_time * time_ratio );
    printf( "Cache KIJ ----------------------------- %5.2lfs (%5.1lf%%)\n", cachefarnodes_time, cachefarnodes_time * time_ratio );
    printf( "========================================================\n");
    printf( "Compress (%4.2lf not compressed) -------- %5.2lfs (%5.1lf%%)\n", 
        exact_ratio, compress_time, compress_time * time_ratio );
    printf( "========================================================\n\n");
  }










  /** Cleanup all w/r dependencies on tree nodes */
  tree_ptr->DependencyCleanUp();

  /** Global barrier to make sure all processes have completed */
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
