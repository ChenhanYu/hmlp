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

/** Inherit most of the classes from shared-memory GOFMM. */
#include <gofmm.hpp>
/** Use distributed metric trees. */
#include <tree_mpi.hpp>
#include <igofmm_mpi.hpp>
/** Use distributed matrices inspired by the Elemental notation. */
#include <DistData.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;


namespace hmlp
{
namespace mpigofmm
{


///**
// *  @biref This class does not have to inherit DistData, but it have to 
// *         inherit DistVirtualMatrix<T>
// *
// */ 
//template<typename T>
//class DistSPDMatrix : public DistData<STAR, CBLK, T>
//{
//  public:
//
//    DistSPDMatrix( size_t m, size_t n, mpi::Comm comm ) : 
//      DistData<STAR, CBLK, T>( m, n, comm )
//    {
//    };
//
//
//    /** ESSENTIAL: this is an abstract function  */
//    virtual T operator()( size_t i, size_t j, mpi::Comm comm )
//    {
//      T Kij = 0;
//
//      /** MPI */
//      int size, rank;
//      hmlp::mpi::Comm_size( comm, &size );
//      hmlp::mpi::Comm_rank( comm, &rank );
//
//      std::vector<std::vector<size_t>> sendrids( size );
//      std::vector<std::vector<size_t>> recvrids( size );
//      std::vector<std::vector<size_t>> sendcids( size );
//      std::vector<std::vector<size_t>> recvcids( size );
//
//      /** request Kij from rank ( j % size ) */
//      sendrids[ i % size ].push_back( i );    
//      sendcids[ j % size ].push_back( j );    
//
//      /** exchange ids */
//      mpi::AlltoallVector( sendrids, recvrids, comm );
//      mpi::AlltoallVector( sendcids, recvcids, comm );
//
//      /** allocate buffer for data */
//      std::vector<std::vector<T>> senddata( size );
//      std::vector<std::vector<T>> recvdata( size );
//
//      /** fetch subrows */
//      for ( size_t p = 0; p < size; p ++ )
//      {
//        assert( recvrids[ p ].size() == recvcids[ p ].size() );
//        for ( size_t j = 0; j < recvcids[ p ].size(); j ++ )
//        {
//          size_t rid = recvrids[ p ][ j ];
//          size_t cid = recvcids[ p ][ j ];
//          senddata[ p ].push_back( (*this)( rid, cid ) );
//        }
//      }
//
//      /** exchange data */
//      mpi::AlltoallVector( senddata, recvdata, comm );
//
//      for ( size_t p = 0; p < size; p ++ )
//      {
//        assert( recvdata[ p ].size() <= 1 );
//        if ( recvdata[ p ] ) Kij = recvdata[ p ][ 0 ];
//      }
//
//      return Kij;
//    };
//
//
//    /** ESSENTIAL: return a submatrix */
//    virtual hmlp::Data<T> operator()
//		( std::vector<size_t> &imap, std::vector<size_t> &jmap, hmlp::mpi::Comm comm )
//    {
//      hmlp::Data<T> KIJ( imap.size(), jmap.size() );
//
//      /** MPI */
//      int size, rank;
//      hmlp::mpi::Comm_size( comm, &size );
//      hmlp::mpi::Comm_rank( comm, &rank );
//
//
//
//      std::vector<std::vector<size_t>> jmapcids( size );
//
//      std::vector<std::vector<size_t>> sendrids( size );
//      std::vector<std::vector<size_t>> recvrids( size );
//      std::vector<std::vector<size_t>> sendcids( size );
//      std::vector<std::vector<size_t>> recvcids( size );
//
//      /** request KIJ from rank ( j % size ) */
//      for ( size_t j = 0; j < jmap.size(); j ++ )
//      {
//        size_t cid = jmap[ j ];
//        sendcids[ cid % size ].push_back( cid );
//        jmapcids[ cid % size ].push_back(   j );
//      }
//
//      for ( size_t p = 0; p < size; p ++ )
//      {
//        if ( sendcids[ p ].size() ) sendrids[ p ] = imap;
//      }
//
//      /** exchange ids */
//      mpi::AlltoallVector( sendrids, recvrids, comm );
//      mpi::AlltoallVector( sendcids, recvcids, comm );
//
//      /** allocate buffer for data */
//      std::vector<hmlp::Data<T>> senddata( size );
//      std::vector<hmlp::Data<T>> recvdata( size );
//
//      /** fetch submatrix */
//      for ( size_t p = 0; p < size; p ++ )
//      {
//        if ( recvcids[ p ].size() && recvrids[ p ].size() )
//        {
//          senddata[ p ] = (*this)( recvrids[ p ], recvcids[ p ] );
//        }
//      }
//
//      /** exchange data */
//      mpi::AlltoallVector( senddata, recvdata, comm );
//
//      /** merging data */
//      for ( size_t p = 0; j < size; p ++ )
//      {
//        assert( recvdata[ p ].size() == imap.size() * recvcids[ p ].size() );
//        recvdata[ p ].resize( imap.size(), recvcids[ p ].size() );
//        for ( size_t j = 0; j < recvcids[ p ]; i ++ )
//        {
//          for ( size_t i = 0; i < imap.size(); i ++ )
//          {
//            KIJ( i, jmapcids[ p ][ j ] ) = recvdata[ p ]( i, j );
//          }
//        }
//      };
//
//      return KIJ;
//    };
//
//
//
//
//
//    virtual hmlp::Data<T> operator()
//		( std::vector<int> &imap, std::vector<int> &jmap, hmlp::mpi::Comm comm )
//    {
//      printf( "operator() not implemented yet\n" );
//      exit( 1 );
//    };
//
//
//
//    /** overload operator */
//
//
//  private:
//
//}; /** end class DistSPDMatrix */
//
//


/**
 *  @brief These are data that shared by the whole local tree.
 *         Distributed setup inherits mpitree::Setup.
 */ 
template<typename SPDMATRIX, typename SPLITTER, typename T>
class Setup : public mpitree::Setup<SPLITTER, T>,
              public gofmm::Configuration<T>
{
  public:

    /** Shallow copy from the config. */
    void FromConfiguration( gofmm::Configuration<T> &config,
        SPDMATRIX &K, SPLITTER &splitter,
        DistData<STAR, CBLK, pair<T, size_t>> &NN_cblk, 
        DistData<STAR, CBLK, T> *X_cblk )
    { 
      this->CopyFrom( config ); 
      this->K = &K;
      this->splitter = splitter;
      this->NN_cblk = &NN_cblk;
      this->X_cblk = X_cblk;
    };

    /** The SPDMATRIX (accessed with gids: dense, CSC or OOC) */
    SPDMATRIX *K = NULL;

    /** rhs-by-n, all weights and potentials. */
    Data<T> *w = NULL;
    Data<T> *u = NULL;

    /** buffer space, either dimension needs to be n  */
    Data<T> *input = NULL;
    Data<T> *output = NULL;

    /** regularization */
    T lambda = 0.0;

    /** whether the matrix is symmetric */
    //bool issymmetric = true;

    /** use ULV or Sherman-Morrison-Woodbury */
    bool do_ulv_factorization = true;


  private:

}; /** end class Setup */





/** 
 *  @brief This task creates an hierarchical tree view for
 *         weights<RIDS> and potentials<RIDS>.
 */
template<typename NODE>
class DistTreeViewTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "TreeView" );
      label = to_string( arg->treelist_id );
      cost = 1.0;
    };

    /** Preorder dependencies (with a single source node) */
    void DependencyAnalysis() { arg->DependOnParent( this ); };

    void Execute( Worker* user_worker )
    {
      auto *node   = arg;

      /** w and u can be Data<T> or DistData<RIDS,STAR,T> */
      auto &w = *(node->setup->w);
      auto &u = *(node->setup->u);

      /** get the matrix view of this tree node */
      auto &U = node->data.u_view;
      auto &W = node->data.w_view;

      /** Both w and u are column-majored, thus nontranspose. */
      U.Set( u );
      W.Set( w );

      /** Create sub matrix views for local nodes. */
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
      }
    };

}; /** end class DistTreeViewTask */









/** @brief Split values into two halfs accroding to the median. */ 
template<typename T>
vector<vector<size_t>> DistMedianSplit( vector<T> &values, mpi::Comm comm )
{
  int n = 0;
  int num_points_owned = values.size();
  /** n = sum( num_points_owned ) over all MPI processes in comm */
  mpi::Allreduce( &num_points_owned, &n, 1, MPI_SUM, comm );
  T  median = combinatorics::Select( n / 2, values, comm );

  vector<vector<size_t>> split( 2 );
  vector<size_t> middle;

  if ( n == 0 ) return split;

  for ( size_t i = 0; i < values.size(); i ++ )
  {
    auto v = values[ i ];
    if ( std::fabs( v - median ) < 1E-6 ) middle.push_back( i );
    else if ( v < median ) split[ 0 ].push_back( i );
    else split[ 1 ].push_back( i );
  }

  int nmid = 0;
  int nlhs = 0;
  int nrhs = 0;
  int num_mid_owned = middle.size();
  int num_lhs_owned = split[ 0 ].size();
  int num_rhs_owned = split[ 1 ].size();

  /** nmid = sum( num_mid_owned ) over all MPI processes in comm. */
  mpi::Allreduce( &num_mid_owned, &nmid, 1, MPI_SUM, comm );
  mpi::Allreduce( &num_lhs_owned, &nlhs, 1, MPI_SUM, comm );
  mpi::Allreduce( &num_rhs_owned, &nrhs, 1, MPI_SUM, comm );

  /** Assign points in the middle to left or right. */
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

    assert( nlhs_required >= 0 && nrhs_required >= 0 );

    /** Now decide the portion */
    double lhs_ratio = ( (double)nlhs_required ) / nmid;
    int nlhs_required_owned = num_mid_owned * lhs_ratio;
    int nrhs_required_owned = num_mid_owned - nlhs_required_owned;

    //printf( "rank %d [ %d %d ] [ %d %d ]\n",
    //  global_rank, 
    //  nlhs_required_owned, nlhs_required,
    //  nrhs_required_owned, nrhs_required ); fflush( stdout );

    assert( nlhs_required_owned >= 0 && nrhs_required_owned >= 0 );

    for ( size_t i = 0; i < middle.size(); i ++ )
    {
      if ( i < nlhs_required_owned ) 
        split[ 0 ].push_back( middle[ i ] );
      else                           
        split[ 1 ].push_back( middle[ i ] );
    }
  }

  return split;
}; /** end MedianSplit() */




/**
 *  @brief This the main splitter used to build the Spd-Askit tree.
 *         First compute the approximate center using subsamples.
 *         Then find the two most far away points to do the 
 *         projection.
 */ 
template<typename SPDMATRIX, int N_SPLIT, typename T> 
struct centersplit : public gofmm::centersplit<SPDMATRIX, N_SPLIT, T>
{

  /** Shared-memory operator. */
  inline vector<vector<size_t> > operator() ( vector<size_t>& gids ) const 
  {
    return gofmm::centersplit<SPDMATRIX, N_SPLIT, T>::operator() ( gids );
  };

  /** Distributed operator. */
  inline vector<vector<size_t> > operator() ( vector<size_t>& gids, mpi::Comm comm ) const 
  {
    /** All assertions */
    assert( N_SPLIT == 2 );
    assert( this->Kptr );

    /** MPI Support. */
    int size; mpi::Comm_size( comm, &size );
    int rank; mpi::Comm_rank( comm, &rank );
    auto &K = *(this->Kptr);

    /** */
    vector<T> temp( gids.size(), 0.0 );

    /** Collecting column samples of K. */
    auto column_samples = combinatorics::SampleWithoutReplacement( 
        this->n_centroid_samples, gids );

    /** Bcast column_samples from rank 0. */
    mpi::Bcast( column_samples.data(), column_samples.size(), 0, comm );
    K.BcastIndices( column_samples, 0, comm );

    /** Compute all pairwise distances. */
    auto DIC = K.Distances( this->metric, gids, column_samples );

    /** Zero out the temporary buffer. */
    for ( auto & it : temp ) it = 0;

    /** Accumulate distances to the temporary buffer. */
    for ( size_t j = 0; j < DIC.col(); j ++ )
      for ( size_t i = 0; i < DIC.row(); i ++ ) 
        temp[ i ] += DIC( i, j );

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
    K.BcastIndices( P, max_pair.key, comm );

    /** Compute all pairwise distances. */
    auto DIP = K.Distances( this->metric, gids, P );

    /** Find f2f (far most to far most) from owned points */
    auto idf2f = distance( DIP.begin(), max_element( DIP.begin(), DIP.end() ) );

    /** Create a pair for MPI Allreduce */
    local_max_pair.val = DIP[ idf2f ];
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
    K.BcastIndices( Q, max_pair.key, comm );

    /** Compute all pairwise distances. */
    auto DIQ = K.Distances( this->metric, gids, P );

    /** We use relative distances (dip - diq) for clustering. */
    for ( size_t i = 0; i < temp.size(); i ++ )
      temp[ i ] = DIP[ i ] - DIQ[ i ];

    /** Split gids into two clusters using median split. */
    auto split = DistMedianSplit( temp, comm );

    /** Perform P2P redistribution. */
    mpi::Status status;
    vector<size_t> sent_gids;
    int partner = ( rank + size / 2 ) % size;
    if ( rank < size / 2 )
    {
      for ( auto it : split[ 1 ] ) 
        sent_gids.push_back( gids[ it ] );
      K.SendIndices( sent_gids, partner, comm );
      K.RecvIndices( partner, comm, &status );
    }
    else
    {
      for ( auto it : split[ 0 ] ) 
        sent_gids.push_back( gids[ it ] );
      K.RecvIndices( partner, comm, &status );
      K.SendIndices( sent_gids, partner, comm );
    }

    return split;
  };


}; /** end struct centersplit */





template<typename SPDMATRIX, int N_SPLIT, typename T> 
struct randomsplit : public gofmm::randomsplit<SPDMATRIX, N_SPLIT, T>
{

  /** Shared-memory operator. */
  inline vector<vector<size_t> > operator() ( vector<size_t>& gids ) const 
  {
    return gofmm::randomsplit<SPDMATRIX, N_SPLIT, T>::operator() ( gids );
  };

  /** Distributed operator. */
  inline vector<vector<size_t> > operator() ( vector<size_t>& gids, mpi::Comm comm ) const 
  {
    /** All assertions */
    assert( N_SPLIT == 2 );
    assert( this->Kptr );

    /** Declaration */
    int size, rank, global_rank, global_size;
    mpi::Comm_size( comm, &size );
    mpi::Comm_rank( comm, &rank );
    mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
    mpi::Comm_size( MPI_COMM_WORLD, &global_size );
    SPDMATRIX &K = *(this->Kptr);
    //vector<vector<size_t>> split( N_SPLIT );

    if ( size == global_size )
    {
      for ( size_t i = 0; i < gids.size(); i ++ )
        assert( gids[ i ] == i * size + rank );
    }




    /** Reduce to get the total size of gids. */
    int n = 0;
    int num_points_owned = gids.size();
    vector<T> temp( gids.size(), 0.0 );

    /** n = sum( num_points_owned ) over all MPI processes in comm */
    mpi::Allreduce( &num_points_owned, &n, 1, MPI_INT, MPI_SUM, comm );

    /** Early return */
    //if ( n == 0 ) return split;

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
    K.BcastIndices( P, max_pair.key, comm );

    /** Choose the second MPI rank */
    if ( rank == max_pair.key ) local_max_pair.val = 0;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** Bcast gidf2c from the rank that has the most gids */
    mpi::Bcast( &gidf2f, 1, max_pair.key, comm );
    vector<size_t> Q( 1, gidf2f );
    K.BcastIndices( Q, max_pair.key, comm );


    auto DIP = K.Distances( this->metric, gids, P );
    auto DIQ = K.Distances( this->metric, gids, Q );

    /** We use relative distances (dip - diq) for clustering. */
    for ( size_t i = 0; i < temp.size(); i ++ )
      temp[ i ] = DIP[ i ] - DIQ[ i ];

    /** Split gids into two clusters using median split. */
    auto split = DistMedianSplit( temp, comm );

    /** Perform P2P redistribution. */
    mpi::Status status;
    vector<size_t> sent_gids;
    int partner = ( rank + size / 2 ) % size;
    if ( rank < size / 2 )
    {
      for ( auto it : split[ 1 ] ) 
        sent_gids.push_back( gids[ it ] );
      K.SendIndices( sent_gids, partner, comm );
      K.RecvIndices( partner, comm, &status );
    }
    else
    {
      for ( auto it : split[ 0 ] ) 
        sent_gids.push_back( gids[ it ] );
      K.RecvIndices( partner, comm, &status );
      K.SendIndices( sent_gids, partner, comm );
    }

    return split;
  };


}; /** end struct randomsplit */





















/**
 *  @brief Compute skeleton weights.
 *
 *  
 */
template<typename NODE>
void DistUpdateWeights( NODE *node )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
  /** MPI Support. */
  mpi::Status status;
  auto comm = node->GetComm();
  int  size = node->GetCommSize();
  int  rank = node->GetCommRank();

  /** Early return if this is the root or there is no skeleton. */
  if ( !node->parent || !node->data.isskel ) return;

  if ( size < 2 )
  {
    /** This is the root of the local tree. */
    gofmm::UpdateWeights<NODE, T>( node );
  }
  else
  {
    /** Gather shared data and create reference. */
    auto &w = *node->setup->w;
    size_t nrhs = w.col();

    /** gather per node data and create reference */
    auto &data   = node->data;
    auto &proj   = data.proj;
    auto &w_skel = data.w_skel;

    /** This is the corresponding MPI rank. */
    if ( rank == 0 )
    {
      size_t s  = proj.row();
			size_t sl = node->child->data.skels.size();
			size_t sr = proj.col() - sl;
      /** w_skel is s-by-nrhs, initial values are not important. */
      w_skel.resize( s, nrhs );
      /** Create matrix views. */
      View<T> P( false,   proj ), PL, PR;
      View<T> W( false, w_skel ), WL( false, node->child->data.w_skel );
      /** P = [ PL, PR ] */
      P.Partition1x2( PL, PR, sl, LEFT );
      /** W  = PL * WL */
      gemm::xgemm<GEMM_NB>( (T)1.0, PL, WL, (T)0.0, W );

      Data<T> w_skel_sib;
      mpi::ExchangeVector( w_skel, size / 2, 0, w_skel_sib, size / 2, 0, comm, &status );
			/** Reduce from my sibling. */
      #pragma omp parallel for
			for ( size_t i = 0; i < w_skel.size(); i ++ )
				w_skel[ i ] += w_skel_sib[ i ];
    }

    /** The rank that holds the skeleton weight of the right child. */
    if ( rank == size / 2 )
    {
      size_t s  = proj.row();
			size_t sr = node->child->data.skels.size();
			size_t sl = proj.col() - sr;
      /** w_skel is s-by-nrhs, initial values are not important. */
      w_skel.resize( s, nrhs );
      /** Create a transpose view proj_v */
      View<T> P( false,   proj ), PL, PR;
      View<T> W( false, w_skel ), WR( false, node->child->data.w_skel );
      /** P = [ PL, PR ] */
      P.Partition1x2( PL, PR, sl, LEFT );
      /** W += PR * WR */
      gemm::xgemm<GEMM_NB>( (T)1.0, PR, WR, (T)0.0, W );
      

			Data<T> w_skel_sib;
			mpi::ExchangeVector( w_skel, 0, 0, w_skel_sib, 0, 0, comm, &status );
			w_skel.clear();
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

    void DependencyAnalysis() { arg->DependOnChildren( this ); };

    void Execute( Worker* user_worker ) { DistUpdateWeights( arg ); };

}; /** end class DistUpdateWeightsTask */




/**
 *
 */ 
template<bool NNPRUNE, typename NODE, typename T>
class DistSkeletonsToSkeletonsTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "DistS2S" );
      label = to_string( arg->treelist_id );
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
      /** MPI Support. */
      auto comm = node->GetComm();
      auto size = node->GetCommSize();
      auto rank = node->GetCommRank();

      if ( size < 2 )
      {
        gofmm::SkeletonsToSkeletons<NNPRUNE, NODE, T>( node );
      }
      else
      {
        /** Only 0th rank (owner) will execute this task. */
        if ( rank == 0 ) gofmm::SkeletonsToSkeletons<NNPRUNE, NODE, T>( node );
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




















template<bool NNPRUNE, typename NODE, typename T>
void DistSkeletonsToNodes( NODE *node )
{
  /** MPI Support. */
  auto comm = node->GetComm();
  auto size = node->GetCommSize();
  auto rank = node->GetCommRank();
  mpi::Status status;

  /** gather shared data and create reference */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;


  size_t nrhs = w.col();


  /** Early return if this is the root or has no skeleton. */
  if ( !node->parent || !node->data.isskel ) return;

  if ( size < 2 )
  {
    /** Call the shared-memory implementation. */
    gofmm::SkeletonsToNodes<NNPRUNE, NODE, T>( node );
  }
  else
  {
    auto &data = node->data;
    auto &proj = data.proj;
    auto &u_skel = data.u_skel;

    if ( rank == 0 )
    {
			size_t sl = node->child->data.skels.size();
			size_t sr = proj.col() - sl;
      /** Send u_skel to my sibling. */
      mpi::SendVector( u_skel, size / 2, 0, comm );
      /** Create a transpose matrix view for proj. */
      View<T> P(  true,   proj ), PL, PR;
      View<T> U( false, u_skel ), UL( false, node->child->data.u_skel );
      /** P' = [ PL, PR ]' */
      P.Partition2x1( PL,
                      PR, sl, TOP );
      /** UL += PL' * U */
      gemm::xgemm<GEMM_NB>( (T)1.0, PL, U, (T)1.0, UL );
    }

    /**  */
    if ( rank == size / 2 )
    {
      size_t s  = proj.row();
			size_t sr = node->child->data.skels.size();
      size_t sl = proj.col() - sr;
      /** Receive u_skel from my sibling. */
      mpi::RecvVector( u_skel, 0, 0, comm, &status );			
			u_skel.resize( s, nrhs );
      /** create a transpose view proj_v */
      View<T> P(  true,   proj ), PL, PR;
      View<T> U( false, u_skel ), UR( false, node->child->data.u_skel );
      /** P' = [ PL, PR ]' */
      P.Partition2x1( PL,
                      PR, sl, TOP );
      /** UR += PR' * U */
      gemm::xgemm<GEMM_NB>( (T)1.0, PR, U, (T)1.0, UR );
    }
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

    void DependencyAnalysis() { arg->DependOnParent( this ); };

    void Execute( Worker* user_worker ) { DistSkeletonsToNodes<NNPRUNE, NODE, T>( arg ); };

}; /** end class DistSkeletonsToNodesTask */



template<typename NODE, typename T>
class L2LTask2 : public Task
{
  public:

    NODE *arg = NULL;

    /** A list of source node pointers. */
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
      //if ( p == rank ) 
      //{
      //  for ( auto src : Sources ) src->DependencyAnalysis( R, this );
      //}
      //else hmlp_msg_dependency_analysis( 300, p, R, this );



      /** If p is a distributed process, then depends on the message. */
      if ( p != rank ) hmlp_msg_dependency_analysis( 300, p, R, this );
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
template<typename TREE>
void FindNearInteractions( TREE &tree )
{
  /** Derive type NODE from TREE. */
  using NODE = typename TREE::NODE;
  auto &setup = tree.setup;
  auto &NN = *setup.NN;
  double budget = setup.Budget();
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
      /** Exit if we have enough near interactions. */ 
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
      } /** end pragma omp critical */
    }
  } /** end for each leaf owned leaf node in the local tree */
  mpi::PrintProgress( "Finish FindNearInteractions ...\n", tree.GetComm() );
}; /** end FindNearInteractions() */




template<typename NODE>
void FindFarNodes( MortonHelper::Recursor r, NODE *target )
{
  /** Return while reaching the leaf level (recursion base case). */ 
  if ( r.second > target->l ) return;
  /** Compute the MortonID of the visiting node. */
  size_t node_morton = MortonHelper::MortonID( r );

  //bool prunable = true;
  auto & NearMortonIDs = target->NNNearNodeMortonIDs;

  /** Recur to children if the current node contains near interactions. */
  if ( MortonHelper::ContainAny( node_morton, NearMortonIDs ) )
  {
    FindFarNodes( MortonHelper::RecurLeft( r ), target );
    FindFarNodes( MortonHelper::RecurRight( r ), target );
  }
  else
  {
    if ( node_morton >= target->morton )
      target->NNFarNodeMortonIDs.insert( node_morton );
  }
}; /** end FindFarNodes() */






template<typename TREE>
void SymmetrizeNearInteractions( TREE & tree )
{
  /** Derive type NODE from TREE. */
  using NODE = typename TREE::NODE;
  /** MPI Support */
  int comm_size; mpi::Comm_size( tree.GetComm(), &comm_size );
  int comm_rank; mpi::Comm_rank( tree.GetComm(), &comm_rank );

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
    /** Create a per thread list. Merge them into sendlist afterward. */ 
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
      {
        sendlist[ p ].insert( sendlist[ p ].end(), 
            list[ p ].begin(), list[ p ].end() );
      }
    } /** end pragma omp critical*/
  }; /** end pargma omp parallel */

  //mpi::Barrier( tree.GetComm() );
  //printf( "rank %d finish first part\n", comm_rank ); fflush( stdout );



  /** Alltoallv */
  mpi::AlltoallVector( sendlist, recvlist, tree.GetComm() );


  //printf( "rank %d finish AlltoallVector\n", comm_rank ); fflush( stdout );

  /** Loop over queries. */
  for ( int p = 0; p < comm_size; p ++ )
  {
    for ( auto & query : recvlist[ p ]  )
    {
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
  mpi::Barrier( tree.GetComm() );
  mpi::PrintProgress( "Finish SymmetrizeNearInteractions ...\n", tree.GetComm() ); 
}; /** end SymmetrizeNearInteractions() */


template<typename TREE>
void SymmetrizeFarInteractions( TREE & tree )
{
  /** Derive type NODE from TREE. */
  using NODE = typename TREE::NODE;
  /** MPI Support. */
  int comm_size; mpi::Comm_size( tree.GetComm(), &comm_size );
  int comm_rank; mpi::Comm_rank( tree.GetComm(), &comm_rank );

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
  mpi::AlltoallVector( sendlist, recvlist, tree.GetComm() );

  /** Loop over queries */
  for ( int p = 0; p < comm_size; p ++ )
  {
    //#pragma omp parallel for
    for ( auto & query : recvlist[ p ] )
    {
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
      } /** end pragma omp critical */
    } /** end pargma omp parallel for */
  }

  mpi::Barrier( tree.GetComm() );
  mpi::PrintProgress( "Finish SymmetrizeFarInteractions ...\n", tree.GetComm() ); 
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
  /** Derive type T from TREE. */
  using T = typename TREE::T;
  /** MPI Support. */
  int comm_size; mpi::Comm_size( tree.GetComm(), &comm_size );
  int comm_rank; mpi::Comm_rank( tree.GetComm(), &comm_rank );

  /** Interaction set per rank in MortonID. */
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


    /** Cast set to vector. */
    vector<vector<size_t>> recvlist( comm_size );
    if ( !tree.NearSentToRank.size() ) tree.NearSentToRank.resize( comm_size );
    if ( !tree.NearRecvFromRank.size() ) tree.NearRecvFromRank.resize( comm_size );
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
    {
      tree.NearSentToRank[ p ].insert( tree.NearSentToRank[ p ].end(), 
          lists[ p ].begin(), lists[ p ].end() );
    }

    /** Use buffer recvlist to catch Alltoallv results. */
    mpi::AlltoallVector( tree.NearSentToRank, recvlist, tree.GetComm() );

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


    /** Use buffer recvlist to catch Alltoallv results. */
    mpi::AlltoallVector( tree.FarSentToRank, recvlist, tree.GetComm() );

    /** Cast vector of vectors to vector of maps */
    #pragma omp parallel for
    for ( int p = 0; p < comm_size; p ++ )
      for ( int i = 0; i < recvlist[ p ].size(); i ++ )
        tree.FarRecvFromRank[ p ][ recvlist[ p ][ i ] ] = i;
  }

  mpi::Barrier( tree.GetComm() );
}; /** BuildInteractionListPerRank() */


template<typename TREE>
pair<double, double> NonCompressedRatio( TREE &tree )
{
  /** Tree MPI communicator */
  int comm_size; mpi::Comm_size( tree.GetComm(), &comm_size );
  int comm_rank; mpi::Comm_rank( tree.GetComm(), &comm_rank );

  /** Use double for accumulation. */
  double ratio_n = 0.0;
  double ratio_f = 0.0;


  /** Traverse all nodes in the local tree. */
  for ( auto &tar : tree.treelist )
  {
    if ( tar->isleaf )
    {
      for ( auto nearID : tar->NNNearNodeMortonIDs )
      {
        auto *src = tree.morton2node[ nearID ];
        assert( src );
        double m = tar->gids.size(); 
        double n = src->gids.size();
        double N = tree.n;
        ratio_n += ( m / N ) * ( n / N );
      }
    }

    for ( auto farID : tar->NNFarNodeMortonIDs )
    {
      auto *src = tree.morton2node[ farID ];
      assert( src );
      double m = tar->data.skels.size(); 
      double n = src->data.skels.size(); 
      double N = tree.n;
      ratio_f += ( m / N ) * ( n / N );
    }
  }

  /** Traverse all nodes in the distributed tree. */
  for ( auto &tar : tree.mpitreelists )
  {
    if ( !tar->child || tar->GetCommRank() ) continue;
    for ( auto farID : tar->NNFarNodeMortonIDs )
    {
      auto *src = tree.morton2node[ farID ];
      assert( src );
      double m = tar->data.skels.size(); 
      double n = src->data.skels.size(); 
      double N = tree.n;
      ratio_f += ( m / N ) * ( n / N );
    }
  }

  /** Allreduce total evaluations from all MPI processes. */
  pair<double, double> ret( 0, 0 );
  mpi::Allreduce( &ratio_n, &(ret.first),  1, MPI_SUM, tree.GetComm() );
  mpi::Allreduce( &ratio_f, &(ret.second), 1, MPI_SUM, tree.GetComm() );

  return ret;
};



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
}; /** end PackFar() */












/** @brief Pack a list of weights and their sizes to two messages. */
template<typename TREE, typename T>
void PackWeights( TREE &tree, int p, 
    vector<T> &sendbuffs, vector<size_t> &sendsizes )
{
  for ( auto it : tree.NearSentToRank[ p ] )
  {
    auto *node = tree.morton2node[ it ];
    auto w_leaf = node->data.w_view.toData();
    sendbuffs.insert( sendbuffs.end(), w_leaf.begin(), w_leaf.end() );
    sendsizes.push_back( w_leaf.size() );
  }
}; /** end PackWeights() */


/** @brief Unpack a list of weights and their sizes. */
template<typename TREE, typename T>
void UnpackWeights( TREE &tree, int p,
    const vector<T> recvbuffs, const vector<size_t> &recvsizes )
{
  vector<size_t> offsets( 1, 0 );
  for ( auto it : recvsizes ) offsets.push_back( offsets.back() + it );

  for ( auto it : tree.NearRecvFromRank[ p ] )
  {
    /** Get LET node pointer. */
    auto *node = tree.morton2node[ it.first ];
    /** Number of right hand sides */
    size_t nrhs = tree.setup.w->col();
    auto &w_leaf = node->data.w_leaf;
    size_t i = it.second;
    w_leaf.resize( recvsizes[ i ] / nrhs, nrhs );
    for ( uint64_t j  = offsets[ i + 0 ], jj = 0; 
                   j  < offsets[ i + 1 ]; 
                   j ++,                  jj ++ )
    {
      w_leaf[ jj ] = recvbuffs[ j ];
    }
  }
}; /** end UnpackWeights() */



/** @brief Pack a list of skeletons and their sizes to two messages. */
template<typename TREE>
void PackSkeletons( TREE &tree, int p,
    vector<size_t> &sendbuffs, vector<size_t> &sendsizes )
{
  for ( auto it : tree.FarSentToRank[ p ] )
  {
    /** Get LET node pointer. */
    auto *node = tree.morton2node[ it ];
    auto &skels = node->data.skels;
    sendbuffs.insert( sendbuffs.end(), skels.begin(), skels.end() );
    sendsizes.push_back( skels.size() );
  }
}; /** end PackSkeletons() */


/** @brief Unpack a list of skeletons and their sizes. */
template<typename TREE>
void UnpackSkeletons( TREE &tree, int p,
    const vector<size_t> recvbuffs, const vector<size_t> &recvsizes )
{
  vector<size_t> offsets( 1, 0 );
  for ( auto it : recvsizes ) offsets.push_back( offsets.back() + it );

  for ( auto it : tree.FarRecvFromRank[ p ] )
  {
    /** Get LET node pointer. */
    auto *node = tree.morton2node[ it.first ];
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
}; /** end UnpackSkeletons() */



/** @brief Pack a list of skeleton weights and their sizes to two messages. */
template<typename TREE, typename T>
void PackSkeletonWeights( TREE &tree, int p,
    vector<T> &sendbuffs, vector<size_t> &sendsizes )
{
  for ( auto it : tree.FarSentToRank[ p ] )
  {
    auto *node = tree.morton2node[ it ];
    auto &w_skel = node->data.w_skel;
    sendbuffs.insert( sendbuffs.end(), w_skel.begin(), w_skel.end() );
    sendsizes.push_back( w_skel.size() );
  }
}; /** end PackSkeletonWeights() */


/** @brief Unpack a list of skeletons and their sizes. */
template<typename TREE, typename T>
void UnpackSkeletonWeights( TREE &tree, int p,
    const vector<T> recvbuffs, const vector<size_t> &recvsizes )
{
  vector<size_t> offsets( 1, 0 );
  for ( auto it : recvsizes ) offsets.push_back( offsets.back() + it );

  for ( auto it : tree.FarRecvFromRank[ p ] )
  {
    /** Get LET node pointer. */
    auto *node = tree.morton2node[ it.first ];
    /** Number of right hand sides */
    size_t nrhs = tree.setup.w->col();
    auto &w_skel = node->data.w_skel;
    size_t i = it.second;
    w_skel.resize( recvsizes[ i ] / nrhs, nrhs );
    for ( uint64_t j  = offsets[ i + 0 ], jj = 0; 
                   j  < offsets[ i + 1 ]; 
                   j ++,                  jj ++ )
    {
      w_skel[ jj ] = recvbuffs[ j ];
    }
  }
}; /** end UnpackSkeletonWeights() */






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

    void DependencyAnalysis()
    {
      TREE &tree = *(this->arg);
      tree.DependOnNearInteractions( this->tar, this );
    };

    /** Instansiate Pack() for SendTask. */
    void Pack()
    {
      PackWeights( *this->arg, this->tar, 
          this->send_buffs, this->send_sizes );
    };

}; /** end class PackNearTask */




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

    void Unpack()
    {
      UnpackWeights( *this->arg, this->src, 
          this->recv_buffs, this->recv_sizes );
    };

}; /** end class UnpackLeafTask */


/** @brief */
template<typename T, typename TREE>
class PackFarTask : public SendTask<T, TREE>
{
  public:

    void DependencyAnalysis()
    {
      TREE &tree = *(this->arg);
      tree.DependOnFarInteractions( this->tar, this );
    };

    /** Instansiate Pack() for SendTask. */
    void Pack()
    {
      PackSkeletonWeights( *this->arg, this->tar, 
          this->send_buffs, this->send_sizes );
    };

}; /** end class PackFarTask */


/** @brief */
template<typename T, typename TREE>
class UnpackFarTask : public RecvTask<T, TREE>
{
  public:

    void Unpack()
    {
      UnpackSkeletonWeights( *this->arg, this->src, 
          this->recv_buffs, this->recv_sizes );
    };

}; /** end class UnpackFarTask */











/**
 *  Send my skeletons (in gids and params) to other ranks
 *  using FarSentToRank[:].
 *
 *  Recv skeletons from other ranks 
 *  using FarRecvFromRank[:].
 */ 
template<typename TREE>
void ExchangeLET( TREE &tree, string option )
{
  /** Derive type T from TREE. */
  using T = typename TREE::T;
  /** MPI Support. */
  int comm_size; mpi::Comm_size( tree.GetComm(), &comm_size );
  int comm_rank; mpi::Comm_rank( tree.GetComm(), &comm_rank );

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
  mpi::AlltoallVector( sendsizes, recvsizes, tree.GetComm() );
  if ( !option.compare( string( "skelgids" ) ) ||
       !option.compare( string( "leafgids" ) ) )
  {
    auto &K = *tree.setup.K;
    mpi::AlltoallVector( sendskels, recvskels, tree.GetComm() );
    K.RequestIndices( recvskels );
  }
  else
  {
    double beg = omp_get_wtime();
    mpi::AlltoallVector( sendbuffs, recvbuffs, tree.GetComm() );
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
  int comm_size; mpi::Comm_size( tree.GetComm(), &comm_size );
  int comm_rank; mpi::Comm_rank( tree.GetComm(), &comm_rank );

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
  int comm_rank; mpi::Comm_rank( tree.GetComm(), &comm_rank );
  int comm_size; mpi::Comm_size( tree.GetComm(), &comm_size );

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
  K.RequestIndices( send_buff );

  mpi::PrintProgress( "Finish ExchangeNeighbors ...\n", tree.GetComm() );
}; /** end ExchangeNeighbors() */











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
    //FindFarNodes( 0 /** root morton */, 0 /** level 0 */, node );


    FindFarNodes( MortonHelper::Root(), node );

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
      arg = user_arg;
      name = string( "merge" );
      label = to_string( arg->treelist_id );
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
      arg = user_arg;
      name = string( "dist-merge" );
      label = to_string( arg->treelist_id );
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






/** @brief Involve MPI routins */ 
template<typename NODE>
void DistSkeletonKIJ( NODE *node )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
	/** Early return */
	if ( !node->parent ) return;
  /** Gather shared data and create reference. */
  auto &K = *(node->setup->K);
  /** Gather per node data and create reference. */
  auto &data = node->data;
  auto &candidate_rows = data.candidate_rows;
  auto &candidate_cols = data.candidate_cols;
  auto &KIJ = data.KIJ;

  /** MPI Support. */
  auto comm = node->GetComm();
  auto size = node->GetCommSize();
  auto rank = node->GetCommRank();
  mpi::Status status;

  if ( size < 2 )
  {
    /** This node is the root of the local tree. */
    gofmm::SkeletonKIJ( node );
  }
  else
  {
    /** This node (mpitree::Node) belongs to the distributed tree 
     *  only executed by 0th and size/2 th rank of 
     *  the node communicator. At this moment, children have been
     *  skeletonized. Thus, we should first update (isskel) to 
     *  all MPI processes. Then we gather information for the
     *  skeletonization.
     */
    NODE *child = node->child;
		size_t nsamples = 0;

    /** Bcast (isskel) to all MPI processes using children's communicator. */ 
    int child_isskel = child->data.isskel;
    mpi::Bcast( &child_isskel, 1, 0, child->GetComm() );
    child->data.isskel = child_isskel;


    /** rank-0 owns data of this node, and it also owns the left child. */ 
    if ( rank == 0 )
    {
      candidate_cols = child->data.skels;
      vector<size_t> rskel;
      /** Receive rskel from my sibling. */
      mpi::RecvVector( rskel, size / 2, 10, comm, &status );
      /** Correspondingly, we need to redistribute the matrix K. */
      K.RecvIndices( size / 2, comm, &status );
      /** Concatinate [ lskels, rskels ]. */
      candidate_cols.insert( candidate_cols.end(), rskel.begin(), rskel.end() );
			/** Use two times of skeletons */
      nsamples = 2 * candidate_cols.size();
      /** Make sure we at least m samples */
      if ( nsamples < 2 * node->setup->LeafNodeSize() ) 
        nsamples = 2 * node->setup->LeafNodeSize();

      /** Gather rpnids and rsnids. */
      auto &lsnids = node->child->data.snids;
      auto &lpnids = node->child->data.pnids;
      vector<T>      recv_rsdist;
      vector<size_t> recv_rsnids;
      vector<size_t> recv_rpnids;

      /** Receive rsnids from size / 2. */
      mpi::RecvVector( recv_rsdist, size / 2, 20, comm, &status );
      mpi::RecvVector( recv_rsnids, size / 2, 30, comm, &status );
      mpi::RecvVector( recv_rpnids, size / 2, 40, comm, &status );
      /** Correspondingly, we need to redistribute the matrix K. */
      K.RecvIndices( size / 2, comm, &status );


      /** Merge snids and update the smallest distance. */
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
      /** Remove lpnids and rpnids from snids  */
      for ( auto lpnid : lpnids )      snids.erase( lpnid );
      for ( auto rpnid : recv_rpnids ) snids.erase( rpnid );
    }

    if ( rank == size / 2 )
    {
      /** Send rskel to rank 0. */
      mpi::SendVector( child->data.skels, 0, 10, comm );
      /** Correspondingly, we need to redistribute the matrix K. */
      K.SendIndices( child->data.skels, 0, comm );

      /** Gather rsnids */
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


      /** Correspondingly, we need to redistribute the matrix K. */
      K.SendIndices( send_rsnids, 0, comm );
    }

		/** Bcast nsamples. */
		mpi::Bcast( &nsamples, 1, 0, comm );
		/** Distributed row samples. */
		DistRowSamples<NODE, T>( node, nsamples );
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
  }
}; /** end DistSkeletonKIJ() */


/**
 *
 */ 
template<typename NODE, typename T>
class DistSkeletonKIJTask : public Task
{
  public:

    NODE *arg = NULL;

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

    void DependencyAnalysis() { arg->DependOnChildren( this ); };

    void Execute( Worker* user_worker ) { DistSkeletonKIJ( arg ); };

}; /** end class DistSkeletonKIJTask */












/**
 *  @brief Skeletonization with interpolative decomposition.
 */ 
template<typename NODE, typename T>
void DistSkeletonize( NODE *node )
{
  /** early return if we do not need to skeletonize */
  if ( !node->parent ) return;

  /** gather shared data and create reference */
  auto &K   = *(node->setup->K);
  auto &NN  = *(node->setup->NN);
  auto maxs = node->setup->MaximumRank();
  auto stol = node->setup->Tolerance();
  bool secure_accuracy = node->setup->SecureAccuracy();
  bool use_adaptive_ranks = node->setup->UseAdaptiveRanks();

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

  if ( secure_accuracy )
  {
    /** TODO: need to check of both children's isskel to preceed */
  }


  /** Bill's l2 norm scaling factor */
  T scaled_stol = std::sqrt( (T)n / q ) * std::sqrt( (T)m / (N - q) ) * stol;

  /** account for uniform sampling */
  scaled_stol *= std::sqrt( (T)q / N );

  lowrank::id
  (
    use_adaptive_ranks, secure_accuracy,
    KIJ.row(), KIJ.col(), maxs, scaled_stol, 
    KIJ, skels, proj, jpvt
  );

  /** Free KIJ for spaces */
  KIJ.resize( 0, 0 );
  KIJ.shrink_to_fit();

  /** depending on the flag, decide isskel or not */
  if ( secure_accuracy )
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




template<typename NODE, typename T>
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

      DistSkeletonize<NODE, T>( arg );

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
template<typename NODE, typename T>
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
        DistSkeletonize<NODE, T>( arg );
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

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "PROJ" );
      label = to_string( arg->treelist_id );
      // Need an accurate cost model.
      cost = 1.0;
    };

    void DependencyAnalysis() { arg->DependOnNoOne( this ); };

    void Execute( Worker* user_worker )
    {
      /** MPI Support. */
      auto comm = arg->GetComm();
      auto size = arg->GetCommSize();
      auto rank = arg->GetCommRank();
      /** Only executed by rank 0. */
      if ( rank == 0  ) gofmm::Interpolate<NODE, T>( arg );

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
template<bool NNPRUNE = true, typename TREE, typename T>
DistData<RIDS, STAR, T> Evaluate( TREE &tree, DistData<RIDS, STAR, T> &weights )
{
  /** MPI Support. */
  int size; mpi::Comm_size( tree.GetComm(), &size );
  int rank; mpi::Comm_rank( tree.GetComm(), &rank );
  /** Derive type NODE and MPINODE from TREE. */
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;

  /** All timers */
  double beg, time_ratio, evaluation_time = 0.0;
  double direct_evaluation_time = 0.0, computeall_time, telescope_time, let_exchange_time, async_time;
  double overhead_time;
  double forward_permute_time, backward_permute_time;

  /** Clean up all r/w dependencies left on tree nodes. */
  tree.DependencyCleanUp();

  /** n-by-nrhs, initialize potentials. */
  size_t n    = weights.row();
  size_t nrhs = weights.col();

  /** Potentials must be in [RIDS,STAR] distribution */
  auto &gids_owned = tree.treelist[ 0 ]->gids;
  DistData<RIDS, STAR, T> potentials( n, nrhs, gids_owned, tree.GetComm() );
  potentials.setvalue( 0.0 );

  /** Provide pointers. */
  tree.setup.w = &weights;
  tree.setup.u = &potentials;

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
    mpi::Barrier( tree.GetComm() );

    //{
    //  /** Stage 1: TreeView and upward telescoping */
    //  beg = omp_get_wtime();
    //  tree.DependencyCleanUp();
    //  tree.DistTraverseDown( mpiVIEWtask );
    //  tree.LocaTraverseDown( seqVIEWtask );
    //  tree.LocaTraverseUp( seqN2Stask );
    //  tree.DistTraverseUp( mpiN2Stask );
    //  hmlp_run();
    //  mpi::Barrier( tree.GetComm() );
    //  telescope_time = omp_get_wtime() - beg;

    //  /** Stage 2: LET exchange */
    //  beg = omp_get_wtime();
    //  ExchangeLET<T>( tree, string( "skelweights" ) );
    //  mpi::Barrier( tree.GetComm() );
    //  ExchangeLET<T>( tree, string( "leafweights" ) );
    //  mpi::Barrier( tree.GetComm() );
    //  let_exchange_time = omp_get_wtime() - beg;

    //  /** Stage 3: L2L */
    //  beg = omp_get_wtime();
    //  tree.DependencyCleanUp();
    //  tree.LocaTraverseLeafs( seqL2LReducetask2 );
    //  hmlp_run();
    //  mpi::Barrier( tree.GetComm() );
    //  direct_evaluation_time = omp_get_wtime() - beg;

    //  /** Stage 4: S2S and downward telescoping */
    //  beg = omp_get_wtime();
    //  tree.DependencyCleanUp();
    //  tree.LocaTraverseUnOrdered( seqS2SReducetask2 );
    //  tree.DistTraverseUnOrdered( mpiS2SReducetask2 );
    //  tree.DistTraverseDown( mpiS2Ntask );
    //  tree.LocaTraverseDown( seqS2Ntask );
    //  hmlp_run();
    //  mpi::Barrier( tree.GetComm() );
    //  computeall_time = omp_get_wtime() - beg;
    //}


  /** Global barrier and timer */
  potentials.setvalue( 0.0 );
  mpi::Barrier( tree.GetComm() );
  
  /** Stage 1: TreeView and upward telescoping */
  beg = omp_get_wtime();
  tree.DependencyCleanUp();
  tree.DistTraverseDown( mpiVIEWtask );
  tree.LocaTraverseDown( seqVIEWtask );
  tree.ExecuteAllTasks();
  mpi::PrintProgress( "Finish TreeView\n", tree.GetComm() );
  /** Stage 2: redistribute weights from IDS to LET. */
  AsyncExchangeLET<T>( tree, string( "leafweights" ) );
  /** Stage 3: N2S. */
  tree.LocaTraverseUp( seqN2Stask );
  tree.DistTraverseUp( mpiN2Stask );
  /** Stage 4: redistribute skeleton weights from IDS to LET. */
  AsyncExchangeLET<T>( tree, string( "skelweights" ) );
  /** Stage 5: L2L */
  tree.LocaTraverseLeafs( seqL2LReducetask2 );
  /** Stage 6: S2S */
  tree.LocaTraverseUnOrdered( seqS2SReducetask2 );
  tree.DistTraverseUnOrdered( mpiS2SReducetask2 );
  /** Stage 7: S2N */
  tree.DistTraverseDown( mpiS2Ntask );
  tree.LocaTraverseDown( seqS2Ntask );
  overhead_time = omp_get_wtime() - beg;
  tree.ExecuteAllTasks();
  async_time = omp_get_wtime() - beg;
  


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









  return potentials;

}; /** end Evaluate() */




template<bool NNPRUNE = true, typename TREE, typename T>
DistData<RBLK, STAR, T> Evaluate( TREE &tree, DistData<RBLK, STAR, T> &w_rblk )
{
  size_t n    = w_rblk.row();
  size_t nrhs = w_rblk.col();
  /** Redistribute weights from RBLK to RIDS. */
  DistData<RIDS, STAR, T> w_rids( n, nrhs, tree.treelist[ 0 ]->gids, tree.GetComm() );
  w_rids = w_rblk;
  /** Evaluation with RIDS distribution. */
  auto u_rids = Evaluate<NNPRUNE>( tree, w_rids );
  mpi::Barrier( tree.GetComm() );
  /** Redistribute potentials from RIDS to RBLK. */
  DistData<RBLK, STAR, T> u_rblk( n, nrhs, tree.GetComm() );
  u_rblk = u_rids;
  /** Return potentials in RBLK distribution. */
  return u_rblk;
}; /** end Evaluate() */













/**
 *  @brief template of the compress routine
 */ 
template<typename SPLITTER, typename RKDTSPLITTER, typename T, typename SPDMATRIX>
mpitree::Tree<mpigofmm::Setup<SPDMATRIX, SPLITTER, T>, gofmm::NodeData<T>>
*Compress
( 
  DistData<STAR, CBLK, T> *X_cblk,
  SPDMATRIX &K, 
  DistData<STAR, CBLK, pair<T, size_t>> &NN_cblk,
  SPLITTER splitter, 
  RKDTSPLITTER rkdtsplitter,
	gofmm::Configuration<T> &config
)
{
  /** MPI size ane rank. */
  int size; mpi::Comm_size( MPI_COMM_WORLD, &size );
  int rank; mpi::Comm_rank( MPI_COMM_WORLD, &rank );

  /** Get all user-defined parameters */
  DistanceMetric metric = config.MetricType();
  size_t n = config.ProblemSize();
	size_t m = config.LeafNodeSize();
	size_t k = config.NeighborSize(); 
	size_t s = config.MaximumRank(); 

  /** options */
  const bool SYMMETRIC = true;
  const bool NNPRUNE   = true;
  const bool CACHE     = true;

  /** Instantiation for the GOFMM metric tree. */
  using SETUP       = mpigofmm::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA        = gofmm::NodeData<T>;
  using NODE        = tree::Node<SETUP, DATA>;
  using MPINODE     = mpitree::Node<SETUP, DATA>;
  using TREE        = mpitree::Tree<SETUP, DATA>;

  /** Instantiation for the randomized metric tree. */
  using RKDTSETUP   = mpigofmm::Setup<SPDMATRIX, RKDTSPLITTER, T>;
  using RKDTNODE    = tree::Node<RKDTSETUP, DATA>;
  using MPIRKDTNODE = tree::Node<RKDTSETUP, DATA>;

  /** All timers. */
  double beg, omptask45_time, omptask_time, ref_time;
  double time_ratio, compress_time = 0.0, other_time = 0.0;
  double ann_time, tree_time, skel_time, mpi_skel_time, mergefarnodes_time, cachefarnodes_time;
  double local_skel_time, dist_skel_time, let_time; 
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;
  double exchange_neighbor_time, symmetrize_time;

  /** Iterative all nearnest-neighbor (ANN). */
  const size_t n_iter = 20;
  pair<T, size_t> initNN( numeric_limits<T>::max(), n );
  mpitree::Tree<RKDTSETUP, DATA> rkdt( MPI_COMM_WORLD );
  rkdt.setup.FromConfiguration( config, K, rkdtsplitter, NN_cblk, X_cblk );
  beg = omp_get_wtime();
  gofmm::NeighborsTask<RKDTNODE, T> knntask;
  if ( k && NN_cblk.row() * NN_cblk.col() != k * n )
    NN_cblk = rkdt.AllNearestNeighbor( n_iter, n, k, initNN, knntask );
  ann_time = omp_get_wtime() - beg;
  printf( "ann_time %lf\n", ann_time );


  /** Initialize metric ball tree using approximate center split. */
  auto *tree_ptr = new mpitree::Tree<SETUP, DATA>( MPI_COMM_WORLD );
	auto &tree = *tree_ptr;

	/** Global configuration for the metric tree. */
  tree.setup.FromConfiguration( config, K, splitter, NN_cblk, X_cblk );

	/** Metric ball tree partitioning. */
  beg = omp_get_wtime();
  tree.TreePartition();
  tree_time = omp_get_wtime() - beg;
  printf( "tree_time %lf\n", tree_time );

  /** Get tree permutataion. */
  vector<size_t> perm = tree.GetPermutation();
  if ( rank == 0 )
  {
    ofstream perm_file( "perm.txt" );
    for ( auto &id : perm ) perm_file << id << " ";
    perm_file.close();
  }


  /** Redistribute neighbors i.e. NN[ *, CIDS ] = NN[ *, CBLK ]; */
  DistData<STAR, CIDS, pair<T, size_t>> NN( k, n, tree.treelist[ 0 ]->gids, tree.GetComm() );
  NN = NN_cblk;
  tree.setup.NN = &NN;
  beg = omp_get_wtime();
  ExchangeNeighbors<T>( tree );
  exchange_neighbor_time = omp_get_wtime() - beg;


  beg = omp_get_wtime();
  /** Construct near interaction lists. */
  FindNearInteractions( tree );
  /** Symmetrize interaction pairs by Alltoallv. */
  mpigofmm::SymmetrizeNearInteractions( tree );
  /** Split node interaction lists per MPI rank. */
  BuildInteractionListPerRank( tree, true );
  /** Exchange {leafs} and {paramsleafs)}.  */
  ExchangeLET( tree, string( "leafgids" ) );
  symmetrize_time = omp_get_wtime() - beg;


  /** Find and merge far interactions. */
  beg = omp_get_wtime();
  tree.DependencyCleanUp();
  MergeFarNodesTask<true, NODE, T> seqMERGEtask;
  DistMergeFarNodesTask<true, MPINODE, T> mpiMERGEtask;
  tree.LocaTraverseUp( seqMERGEtask );
  hmlp_run();
  mpi::Barrier( tree.GetComm() );
  tree.DependencyCleanUp();
  tree.DistTraverseUp( mpiMERGEtask );
  hmlp_run();
  mpi::Barrier( tree.GetComm() );
  mergefarnodes_time += omp_get_wtime() - beg;
  mpi::PrintProgress( "Finish MergeFarNodes ...\n", tree.GetComm() ); 

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
  /** Symmetrize interaction pairs by Alltoallv. */
  mpigofmm::SymmetrizeFarInteractions( tree );
  /** Split node interaction lists per MPI rank. */
  BuildInteractionListPerRank( tree, false );
  symmetrize_time += omp_get_wtime() - beg;

  /** Skeletonization */
	beg = omp_get_wtime();
  tree.DependencyCleanUp();
  /** Gather sample rows and skeleton columns, then ID */
  gofmm::SkeletonKIJTask<NODE, T> seqGETMTXtask;
  mpigofmm::DistSkeletonKIJTask<MPINODE, T> mpiGETMTXtask;
  mpigofmm::SkeletonizeTask<NODE, T> seqSKELtask;
  mpigofmm::DistSkeletonizeTask<MPINODE, T> mpiSKELtask;
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
  mpi::Barrier( tree.GetComm() );
  skel_time = omp_get_wtime() - beg;
  mpi::PrintProgress( "Finish skeletonization and interpolation ...\n", tree.GetComm() ); 



	beg = omp_get_wtime();
  tree.DependencyCleanUp();
  tree.DistTraverseUp<false>( mpiGETMTXtask, mpiSKELtask );
  tree.DistTraverseUnOrdered( mpiPROJtask );
  hmlp_run();
  mpi::Barrier( tree.GetComm() );
  mpi_skel_time = omp_get_wtime() - beg;



  /** Exchange {skels} and {params(skels)}.  */
  ExchangeLET( tree, string( "skelgids" ) );
  
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
  mpi::Barrier( tree.GetComm() );
  cachefarnodes_time = omp_get_wtime() - beg;



  /** Compute the ratio of exact evaluation. */
  auto ratio = NonCompressedRatio( tree );

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
    printf( "%5.3lf%% and %5.3lf%% uncompressed--------- %5.2lfs (%5.1lf%%)\n", 
        100 * ratio.first, 100 * ratio.second, compress_time, compress_time * time_ratio );
    printf( "========================================================\n\n");
  }










  /** Cleanup all w/r dependencies on tree nodes */
  tree_ptr->DependencyCleanUp();
  /** Global barrier to make sure all processes have completed */
  mpi::Barrier( tree.GetComm() );

  return tree_ptr;

}; /** end Compress() */



template<typename TREE, typename T>
pair<T, T> ComputeError( TREE &tree, size_t gid, Data<T> potentials )
{
  int comm_rank; mpi::Comm_rank( tree.GetComm(), &comm_rank );
  int comm_size; mpi::Comm_size( tree.GetComm(), &comm_size );

  /** ( sum of square errors, square 2-norm of true values ) */
  pair<T, T> ret( 0, 0 );

  auto &K = *tree.setup.K;
  auto &w = *tree.setup.w;

  auto  I = vector<size_t>( 1, gid );
  auto &J = tree.treelist[ 0 ]->gids;

  /** Bcast gid and its parameter to all MPI processes. */
  K.BcastIndices( I, gid % comm_size, tree.GetComm() );

	Data<T> Kab = K( I, J );

  auto loc_exact = potentials;
  auto glb_exact = potentials;

  xgemm( "N", "N", Kab.row(), w.col(), w.row(),
    1.0,       Kab.data(),       Kab.row(),
                 w.data(),         w.row(),
    0.0, loc_exact.data(), loc_exact.row() );          

  /** Allreduce u( gid, : ) = K( gid, CBLK ) * w( RBLK, : ) */
  mpi::Allreduce( loc_exact.data(), glb_exact.data(), 
      loc_exact.size(), MPI_SUM, tree.GetComm() );

  for ( uint64_t j = 0; j < w.col(); j ++ )
  {
    T exac = glb_exact[ j ];
    T pred = potentials[ j ];
    /** Accumulate SSE and sqaure 2-norm. */
    ret.first  += ( pred - exac ) * ( pred - exac );
    ret.second += exac * exac;
  }

  return ret;
}; /** end ComputeError() */










template<typename TREE>
void SelfTesting( TREE &tree, size_t ntest, size_t nrhs )
{
  /** Derive type T from TREE. */
  using T = typename TREE::T;
  /** MPI Support. */
  int rank; mpi::Comm_rank( tree.GetComm(), &rank );
  int size; mpi::Comm_size( tree.GetComm(), &size );
  /** Size of right hand sides. */
  size_t n = tree.n;
  /** Shrink ntest if ntest > n. */
  if ( ntest > n ) ntest = n;
  /** all_rhs = [ 0, 1, ..., nrhs - 1 ]. */
  vector<size_t> all_rhs( nrhs );
  for ( size_t rhs = 0; rhs < nrhs; rhs ++ ) all_rhs[ rhs ] = rhs;
  /** Input and output in RIDS and RBLK. */
  DistData<RIDS, STAR, T> w_rids( n, nrhs, tree.treelist[ 0 ]->gids, tree.GetComm() );
  DistData<RBLK, STAR, T> u_rblk( n, nrhs, tree.GetComm() );
  /** Initialize with random N( 0, 1 ). */
  w_rids.randn();
  /** Evaluate u ~ K * w. */
  auto u_rids = mpigofmm::Evaluate<true>( tree, w_rids );
  /** Sanity check for INF and NAN. */
  assert( !u_rids.HasIllegalValue() );
  /** Redistribute potentials from RIDS to RBLK. */
  u_rblk = u_rids;
  /** Report elementwise and F-norm accuracy. */
  if ( rank == 0 )
  {
    printf( "========================================================\n");
    printf( "Accuracy report\n" );
    printf( "========================================================\n");
  }
  /** All statistics. */
  T nnerr_avg = 0.0, nonnerr_avg = 0.0, fmmerr_avg = 0.0;
  T sse_2norm = 0.0, ssv_2norm = 0.0;
  /** Loop over all testing gids and right hand sides. */
  for ( size_t i = 0; i < ntest; i ++ )
  {
    size_t tar = i * 500;
    Data<T> potentials( (size_t)1, nrhs );
    if ( rank == ( tar % size ) ) potentials = u_rblk( vector<size_t>( 1, tar ), all_rhs );
    /** Bcast potentials to all MPI processes. */
    mpi::Bcast( potentials.data(), nrhs, tar % size, tree.GetComm() );
    /** Compare potentials with exact MATVEC. */
    auto sse_ssv = mpigofmm::ComputeError( tree, tar, potentials );
    /** Compute element-wise 2-norm error. */
    auto fmmerr  = sqrt( sse_ssv.first / sse_ssv.second ); 
    /** Accumulate element-wise 2-norm error. */
    fmmerr_avg += fmmerr;
    /** Accumulate SSE and SSV. */
    sse_2norm += sse_ssv.first;
    ssv_2norm += sse_ssv.second;
    /** Only print 10 values. */
    if ( i < 10 && rank == 0 )
    {
      printf( "gid %6lu, ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
          tar, 0.0, 0.0, fmmerr );
    }
  }
  if ( rank == 0 )
  {
    printf( "========================================================\n");
    printf( "Elementwise ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
        nnerr_avg / ntest , nonnerr_avg / ntest, fmmerr_avg / ntest );
    printf( "F-norm      ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
        0.0, 0.0, sqrt( sse_2norm / ssv_2norm ) );
    printf( "========================================================\n");
  }
}; /** end SelfTesting() */




}; /** end namespace gofmm */
}; /** end namespace hmlp */

#endif /** define GOFMM_MPI_HPP */
