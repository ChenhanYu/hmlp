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


#ifndef GOFMM_HPP
#define GOFMM_HPP


/** Use STL future, thread, chrono */
#include <future>
#include <thread>
#include <chrono>

/** stl and omp */
#include <set>
#include <vector>
#include <map>
#include <unordered_set>
#include <deque>
#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <functional>
#include <array>
#include <random>
#include <numeric>
#include <sstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <omp.h>
#include <time.h>

/** hmlp */
#include <hmlp.h>
#include <hmlp_blas_lapack.h>
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>
#include <primitives/lowrank.hpp>
#include <primitives/combinatorics.hpp>
#include <primitives/gemm.hpp>

#include <containers/data.hpp>
#include <gofmm/tree.hpp>
#include <gofmm/igofmm.hpp>

/** gpu related */
#ifdef HMLP_USE_CUDA
#include <cuda_runtime.h>
#include <gofmm/gofmm_gpu.hpp>
#endif


/** by default, we use binary tree */
#define N_CHILDREN 2

/** this parameter is used to reserve space for std::vector */
#define MAX_NRHS 1024

/** the block size we use for partitioning GEMM tasks */
#define GEMM_NB 256


//#define DEBUG_SPDASKIT 1
#define REPORT_ANN_ACCURACY 1

#define REPORT_COMPRESS_STATUS 1
#define REPORT_EVALUATE_STATUS 1


#define OMPLEVEL 0
#define OMPRECTASK 0
#define OMPDAGTASK 0

using namespace std;
using namespace hmlp;

/**
 *  @breif GOFMM relies on an arbitrary distance metric that
 *         described the order between matrix element Kij.
 *         Such metric can be the 
 *         Euclidian distance i.e. "GEOMETRY_DISTANCE", or
 *         arbitrary distances defined in the Gram vector space
 *         i.e. "KERNEL_DISTANCE" or "ANGLE_DISTANCE"
 */ 
typedef enum { GEOMETRY_DISTANCE, KERNEL_DISTANCE, ANGLE_DISTANCE } DistanceMetric;


namespace hmlp
{
namespace gofmm
{

/**
 *  @brief Configuration contains all user-defined parameters.
 */ 
template<typename T>
class Configuration
{
	public:

		Configuration
	  ( 
		  DistanceMetric metric,
		  size_t n, size_t m, size_t k, size_t s, 
			T stol, T budget 
	  ) 
		{
			this->metric = metric;
			this->n = n;
			this->m = m;
			this->k = k;
			this->s = s;
			this->stol = stol;
			this->budget = budget;
		};

		DistanceMetric MetricType() { return metric; };

		size_t ProblemSize() { return n; };

		size_t LeafNodeSize() { return m; };

		size_t NeighborSize() { return k; };

		size_t MaximumRank() { return s; };

		T Tolerance() { return stol; };

		T Budget() { return budget; };

	private:

		/** (default) metric type */
		DistanceMetric metric = ANGLE_DISTANCE;

		/** (default) problem size */
		size_t n = 0;

		/** (default) maximum leaf node size */
		size_t m = 64;

		/** (default) number of neighbors */
		size_t k = 32;

		/** (default) maximum off-diagonal ranks */
		size_t s = 64;

		/** (default) user tolerance */
		T stol = 1E-3;

		/** (default) user computation budget */
		T budget = 0.03;

}; /** end class Configuration */



/**
 *  @brief These are data that shared by the whole local tree.
 *
 */ 
template<typename SPDMATRIX, typename SPLITTER, typename T>
class Setup : public tree::Setup<SPLITTER, T>
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

    /** 
     *  User specific relative error
     */
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


/**
 *  @brief This class contains all GOFMM related data.
 *         For Inv-GOFMM, all factors are inherit from hfamily::Factor<T>.
 *
 */ 
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

    /** Skeleton gids */
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

    /**
     *  This pool contains a subset of gids owned by this node.
     *  Multiple threads may try to update this pool during construction.
     *  We use a lock to prevent race condition.
     */ 
    map<size_t, map<size_t, T>> candidates;
    map<size_t, T> pool;
    multimap<T, size_t> ordered_pool;

    /** Skeleton weights and potentials */
    Data<T> w_skel;
    Data<T> u_skel;

    /** Permuted weights and potentials (buffer) */
    Data<T> w_leaf;
    Data<T> u_leaf[ 20 ];

    /** Hierarchical tree view of w<RIDS> and u<RIDS> */
    View<T> w_view;
    View<T> u_view;

    /** Cached Kab */
    Data<size_t> Nearbmap;
    Data<T> NearKab;
    Data<T> FarKab;

    /** Interaction list (in morton) per MPI rank */
    set<int> NearDependents;
    set<int> FarDependents;


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

}; /** end class Data */




/** 
 *  @brief This task creates an hierarchical tree view for
 *         weights<RIDS> and potentials<RIDS>.
 */
template<typename NODE>
class TreeViewTask : public Task
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
      //printf( "TreeView %lu\n", node->treelist_id );
      auto *node   = arg;
      auto &data   = node->data;
      auto *setup  = node->setup;

      /** w and u can be Data<T> or DistData<RIDS,STAR,T> */
      auto &w = *(setup->u);
      auto &u = *(setup->w);

      /** get the matrix view of this tree node */
      auto &U  = data.u_view;
      auto &W  = data.w_view;

      /** create contigious view for u and w at the root level */
      if ( !node->parent ) 
      {
        /** both w and u are column-majored, thus nontranspose */
        U.Set( u );
        W.Set( w );
      }

      /** partition u and w using the hierarchical tree view */
      if ( !node->isleaf )
      {
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

        assert( UL.row() == node->lchild->n );
        assert( UR.row() == node->rchild->n );
        assert( WL.row() == node->lchild->n );
        assert( WR.row() == node->rchild->n );
      }

    };

}; /** end class TreeViewTask */


















/**
 *  @brief This class does not need to inherit hmlp::Data<T>, 
 *         but it should
 *         support two interfaces for data fetching.
 */ 
template<typename T>
class SPDMatrix : public Data<T>
{
  public:

    Data<T> Diagonal( std::vector<size_t> &I )
    {
      Data<T> DII( I.size(), 1 );

      for ( size_t i = 0; i < I.size(); i ++ )
        DII[ i ] = (*this)( I[ i ], I[ i ] );

      return DII;
    };

    Data<T> PairwiseDistances( const vector<size_t> &I, const vector<size_t> &J )
    {
      Data<T> KIJ( I.size(), J.size(), 0.0 );
      return KIJ;
    };


    virtual void BackGroundProcess( bool *do_terminate )
		{
		};

    virtual void SendColumns( vector<size_t> cids, int dest, mpi::Comm comm )
    {
    };

    virtual void RecvColumns( int root, mpi::Comm comm, mpi::Status *status )
    {
    };

    virtual void BcastColumns( vector<size_t> cids, int root, mpi::Comm comm )
    {
    };

    virtual void RequestColumns( vector<vector<size_t>> requ_cids )
    {
    };

    virtual void Redistribute( bool enforce_ordered, vector<size_t> &cids )
    {
    };

    virtual void RedistributeWithPartner
    ( 
     vector<size_t> &gids,
     vector<size_t> &lhs, 
     vector<size_t> &rhs, mpi::Comm comm 
    )
    {
    };

  private:

}; /** end class SPDMatrix */


/**
 *  @brief Provide statistics summary for the execution section.
 *
 */ 
template<typename NODE>
class Summary
{

  public:

    Summary() {};

    deque<Statistic> rank;

    deque<Statistic> merge_neighbors_time;

    deque<Statistic> kij_skel;

    deque<Statistic> kij_skel_time;

    deque<Statistic> id_time;

    deque<Statistic> skeletonize;

    /** n2s */
    deque<Statistic> updateweight;

    /** s2s */
    deque<Statistic> s2s_kij_t;
    deque<Statistic> s2s_t;
    deque<Statistic> s2s_gfp;

    /** s2n */
    deque<Statistic> s2n_kij_t;
    deque<Statistic> s2n_t;
    deque<Statistic> s2n_gfp;


    void operator() ( NODE *node )
    {
      if ( rank.size() <= node->l )
      {
        rank.push_back( hmlp::Statistic() );
        merge_neighbors_time.push_back( hmlp::Statistic() );
        kij_skel.push_back( hmlp::Statistic() );
        kij_skel_time.push_back( hmlp::Statistic() );
        id_time.push_back( hmlp::Statistic() );
        skeletonize.push_back( hmlp::Statistic() );
        updateweight.push_back( hmlp::Statistic() );
        /** s2s */
        s2s_kij_t.push_back( hmlp::Statistic() );
        s2s_t.push_back( hmlp::Statistic() );
        s2s_gfp.push_back( hmlp::Statistic() );
        /** s2n */
        s2n_kij_t.push_back( hmlp::Statistic() );
        s2n_t.push_back( hmlp::Statistic() );
        s2n_gfp.push_back( hmlp::Statistic() );
      }

      rank[ node->l ].Update( (double)node->data.skels.size() );
      merge_neighbors_time[ node->l ].Update( node->data.merge_neighbors_time );
      kij_skel[ node->l ].Update( (double)node->data.kij_skel.second );
      kij_skel_time[ node->l ].Update( node->data.kij_skel.first );
      id_time[ node->l ].Update( node->data.id_time );
      skeletonize[ node->l ].Update( node->data.skeletonize.GetDuration() );
      updateweight[ node->l ].Update( node->data.updateweight.GetDuration() );

      s2s_kij_t[ node->l ].Update( node->data.kij_s2s.first         );
      s2s_t    [ node->l ].Update( node->data.s2s.GetDuration()     );
      s2s_gfp  [ node->l ].Update( node->data.s2s.GflopsPerSecond() );

      s2n_kij_t[ node->l ].Update( node->data.kij_s2n.first         );
      s2n_t    [ node->l ].Update( node->data.s2s.GetDuration()     );
      s2n_gfp  [ node->l ].Update( node->data.s2s.GflopsPerSecond() );

#ifdef DUMP_ANALYSIS_DATA
      if ( node->parent )
      {
        auto *parent = node->parent;
        printf( "@TREE\n" );
        printf( "#%lu (s%lu), #%lu (s%lu), %lu, %lu\n", 
            node->treelist_id, node->data.skels.size(), 
            parent->treelist_id, parent->data.skels.size(),
            node->data.skels.size(), node->l );
      }
      else
      {
        printf( "@TREE\n" );
        printf( "#%lu (s%lu), , %lu, %lu\n", 
            node->treelist_id, node->data.skels.size(), 
            node->data.skels.size(), node->l );
      }
#endif
    };

    void Print()
    {
      for ( size_t l = 1; l < rank.size(); l ++ )
      {
        printf( "@SUMMARY\n" );
        printf( "level %2lu, ", l ); rank[ l ].Print();
        //printf( "merge_neig: " ); merge_neighbors_time[ l ].Print();
        //printf( "kij_skel_n: " ); kij_skel[ l ].Print();
        //printf( "kij_skel_t: " ); kij_skel_time[ l ].Print();
        //printf( "id_t:       " ); id_time[ l ].Print();
        //printf( "skel_t:     " ); skeletonize[ l ].Print();
        //printf( "... ... ...\n" );
        //printf( "n2s_t:      " ); updateweight[ l ].Print();
        ////printf( "s2s_kij_t:  " ); s2s_kij_t[ l ].Print();
        //printf( "s2s_t:      " ); s2s_t[ l ].Print();
        //printf( "s2s_gfp:    " ); s2s_gfp[ l ].Print();
        ////printf( "s2n_kij_t:  " ); s2n_kij_t[ l ].Print();
        //printf( "s2n_t:      " ); s2n_t[ l ].Print();
        //printf( "s2n_gfp:    " ); s2n_gfp[ l ].Print();
      }
    };

}; // end class Summary






/** 
 *  @brief compute all2c (distance to the approximate centroid) 
 *
 *         Notice that DII and DCC are essentially vectors, which
 *         are not sensitive of either they are rows or columns.
 *         However, KIC is a mamtrix and its dimensions must match.
 */
template<typename T>
vector<T> AllToCentroid
( 
  DistanceMetric metric,
  Data<T> &DII, /** diagonals of K( I, I ) */
  Data<T> &KIC, /**              K( I, C ) */
  Data<T> &DCC  /** diagonals of K( C, C ) */
)
{
  /** distances from I to C */
  vector<T> I2C( DII.size(), 0.0 );

  switch ( metric )
  {
    case GEOMETRY_DISTANCE:
    {
      if ( KIC.row() == DII.size() && KIC.col() == DCC.size() )
      {
        for ( size_t j = 0; j < DCC.col(); j ++ )
          for ( size_t i = 0; i < DII.row(); i ++ )
            I2C[ i ] += KIC( i, j );
      }
      else
      {
        for ( size_t j = 0; j < DCC.col(); j ++ )
          for ( size_t i = 0; i < DII.row(); i ++ )
            I2C[ i ] += KIC( j, i );
      }
      break;
    }
    case KERNEL_DISTANCE:
    {
      for ( size_t i = 0; i < DII.size(); i ++ )
        I2C[ i ] = DII[ i ];

      if ( KIC.row() == DII.size() && KIC.col() == DCC.size() )
      {
        #pragma omp parallel for
        for ( size_t i = 0; i < DII.size(); i ++ )
          for ( size_t j = 0; j < DCC.size(); j ++ )
            I2C[ i ] -= ( 2.0 / DCC.size() ) * KIC( i, j );
      }
      else
      {
        /** in this case, this is actually KCI */
        auto &KCI = KIC;
        #pragma omp parallel for
        for ( size_t i = 0; i < DII.size(); i ++ )
          for ( size_t j = 0; j < DCC.size(); j ++ )
            I2C[ i ] -= ( 2.0 / DCC.size() ) * KCI( j, i );
      }
      break;
    }
    case ANGLE_DISTANCE:
    {
      if ( KIC.row() == DII.size() && KIC.col() == DCC.size() )
      {
        #pragma omp parallel for
        for ( size_t i = 0; i < DII.size(); i ++ )
          for ( size_t j = 0; j < DCC.size(); j ++ )
            I2C[ i ] += ( 1.0 - ( KIC( i, j ) * KIC( i, j ) ) / 
                ( DII[ i ] * DCC[ j ] ) );
      }
      else
      {
        printf( "bug!!!!!!!!!\n" ); fflush( stdout );
        /** in this case, this is actually KCI */
        auto &KCI = KIC;
        for ( size_t i = 0; i < DII.size(); i ++ )
          for ( size_t j = 0; j < DCC.size(); j ++ )
            I2C[ i ] += ( 1.0 - ( KCI( j, i ) * KCI( j, i ) ) / 
                ( DII[ i ] * DCC[ j ] ) );
      }
      break;
    }
    default:
    {
      printf( "centersplit() invalid scheme\n" ); fflush( stdout );
      exit( 1 );
    }
  } /** end switch ( metric ) */

  return I2C;

}; /** end AllToCentroid() */




/** compute all2f (distance to the farthest point) */
template<typename T>
vector<T> AllToFarthest
(
  DistanceMetric metric,
  Data<T> &DII, /**  */
  Data<T> &KIP, /**  */
  T       &kpp
)
{
  /** distances from I to P */
  std::vector<T> I2P( KIP.row(), 0.0 );

  switch ( metric )
  {
    case GEOMETRY_DISTANCE:
    {
      #pragma omp parallel for
      for ( size_t i = 0; i < KIP.row(); i ++ ) I2P[ i ] = KIP[ i ];
      break;
    }
    case KERNEL_DISTANCE:
    {
      #pragma omp parallel for
      for ( size_t i = 0; i < KIP.row(); i ++ )
          I2P[ i ] = DII[ i ] - 2.0 * KIP[ i ] + kpp;

      break;
    }
    case ANGLE_DISTANCE:
    {
      #pragma omp parallel for
      for ( size_t i = 0; i < KIP.row(); i ++ )
          I2P[ i ] = ( 1.0 - ( KIP[ i ] * KIP[ i ] ) / 
              ( DII[ i ] * kpp ) );

      break;
    }
    default:
    {
      printf( "centersplit() invalid scheme\n" ); fflush( stdout );
      exit( 1 );
    }
  } /** end switch ( metric ) */

  return I2P;

}; /** end AllToFarthest() */





template<typename T>
vector<T> AllToLeftRight
( 
  DistanceMetric metric,
  Data<T> &DII,
  Data<T> &KIP,
  Data<T> &KIQ,
  T       kpp,
  T       kqq
)
{
  /** distance differences between I to P and I to Q */
  vector<T> I2PQ( KIP.row(), 0.0 );

  switch ( metric )
  {
    case GEOMETRY_DISTANCE:
    {
      #pragma omp parallel for
      for ( size_t i = 0; i < KIP.row(); i ++ ) I2PQ[ i ] = KIP[ i ] - KIQ[ i ];
      break;
    }
    case KERNEL_DISTANCE:
    {
      #pragma omp parallel for
      for ( size_t i = 0; i < KIP.row(); i ++ ) I2PQ[ i ] = KIP[ i ] - KIQ[ i ];

      break;
    }
    case ANGLE_DISTANCE:
    {
      #pragma omp parallel for
      for ( size_t i = 0; i < KIP.row(); i ++ )
        I2PQ[ i ] = ( KIP[ i ] * KIP[ i ] ) / ( DII[ i ] * kpp ) - 
                    ( KIQ[ i ] * KIQ[ i ] ) / ( DII[ i ] * kqq );

      break;
    }
    default:
    {
      printf( "centersplit() invalid scheme\n" ); fflush( stdout );
      exit( 1 );
    }
  } /** end switch ( metric ) */

  return I2PQ;

}; /** end AllToLeftRight() */




//template<typename T>
//std::vector<std::pair<T,size_t>> SamplesToAll
//(
//  DistanceMetric metric,
//  hmlp::Data<T> &DSS,
//  hmlp::Data<T> &KSI,
//  hmlp::Data<T> &DII
//)
//{
//  /** average distances from samples S to all points I */
//  std::vector<std::pair<T, size_t>> S2I( DSS.size() ); 
//
//  /** initialization */
//  for ( size_t s = 0; s < S2I.size(); s ++ )
//  {
//    S2I[ s ].first  = 0;
//    S2I[ s ].second = s;
//  }
//
//  switch ( metric )
//  {
//    case KERNEL_DISTANCE:
//    {
//      #pragma omp parallel for
//      for ( size_t s = 0; s < DSS.size(); s ++ )
//        for ( size_t i = 0; i < DII.size(); i ++ )
//          S2I[ s ] += ( DSS[ s ] * DSS[ s ] - 2 * KSI( s, i ) ) / DII.size();
//      break;
//    }
//    case ANGLE_DISTANCE:
//    {
//      #pragma omp parallel for
//      for ( size_t s = 0; s < DSS.size(); s ++ )
//        for ( size_t i = 0; i < DII.size(); i ++ )
//          S2I[ s ] += ( 1.0 - ( K( s, i ) * K( s, i ) ) / 
//              ( DSS[ s ] * DII[ i ] ) );
//      break;
//    }
//    default:
//    {
//      printf( "centersplit() invalid scheme\n" ); fflush( stdout );
//      exit( 1 );
//    }
//  } /** end switch ( metric ) */
//
//  /** pair sorting */
//  struct 
//  {
//    bool operator () ( std::pair<T, size_t> a, std::pair<T, size_t> b )
//    {   
//      return a.first < b.first;
//    }   
//  } DistanceLess;
//
//  
//
//  return S2I;
//}; /** end SamplesToAll() */











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
struct centersplit
{
  /** closure */
  SPDMATRIX *Kptr = NULL;

  /** (default) using angle distance from the Gram vector space */
  DistanceMetric metric = ANGLE_DISTANCE;

  /** number samples to approximate centroid */
  size_t n_centroid_samples = 1;
  

	/** overload the operator */
  vector<vector<size_t> > operator()
  ( 
    vector<size_t>& gids, vector<size_t>& lids
  ) const 
  {
    /** all assertions */
    assert( N_SPLIT == 2 );
    assert( Kptr );

    /** all timers */
    double beg, d2c_time, d2f_time, projection_time, max_time;

    SPDMATRIX &K = *Kptr;
    std::vector<std::vector<std::size_t>> split( N_SPLIT );
    size_t n = gids.size();
    std::vector<T> temp( n, 0.0 );

    beg = omp_get_wtime();

    /** diagonal entries */
    Data<T> DII( n, (size_t)1 ), DCC( n_centroid_samples, (size_t)1 );


    //printf( "shared enter DII\n" ); fflush( stdout );

    /** Collecting DII */
    DII = K.Diagonal( gids );


    //printf( "shared after DII\n" ); fflush( stdout );


    /** collecting column samples of K and DCC */
    std::vector<size_t> column_samples( n_centroid_samples );
    for ( size_t j = 0; j < n_centroid_samples; j ++ )
    {
      /** just use the first few gids */
      column_samples[ j ] = gids[ j ];
      //DCC[ j ] = K( column_samples[ j ], column_samples[ j ] );
    }
    DCC = K.Diagonal( column_samples );

    /** Collecting KIC */
    Data<T> KIC = K( gids, column_samples );

    if ( metric == GEOMETRY_DISTANCE ) 
    {
      KIC = K.PairwiseDistances( gids, column_samples );
    }


    /** compute d2c (distance to the approximate centroid) */
    temp = AllToCentroid( metric, DII, KIC, DCC );

    d2c_time = omp_get_wtime() - beg;

    /** find f2c (farthest from center) */
    auto itf2c = std::max_element( temp.begin(), temp.end() );
    auto idf2c = std::distance( temp.begin(), itf2c );
    auto gidf2c = gids[ idf2c ];

    /** compute the all2f (distance to farthest point) */
    beg = omp_get_wtime();

    /** collecting KIP */
    std::vector<size_t> P( 1, gidf2c );
    auto KIP = K( gids, P );

    if ( metric == GEOMETRY_DISTANCE ) 
    {
      KIP = K.PairwiseDistances( gids, P );
    }


    /** get diagonal entry kpp */
    //T kpp = K( gidf2c, gidf2c );
    auto kpp = K.Diagonal( P );

    /** compute the all2f (distance to farthest point) */
    temp = AllToFarthest( metric, DII, KIP, kpp[ 0 ] );

    //temp = AllToFarthest( gids, gidf2c );
    //
    d2f_time = omp_get_wtime() - beg;

    /** find f2f (far most to far most) */
    beg = omp_get_wtime();
    auto itf2f = std::max_element( temp.begin(), temp.end() );
    auto idf2f = std::distance( temp.begin(), itf2f );
    auto gidf2f = gids[ idf2f ];
    max_time = omp_get_wtime() - beg;


    /** compute all2leftright (projection i.e. dip - diq) */
    beg = omp_get_wtime();

    /** collecting KIQ */
    std::vector<size_t> Q( 1, gidf2f );
    auto KIQ = K( gids, Q );

    if ( metric == GEOMETRY_DISTANCE ) 
    {
      KIQ = K.PairwiseDistances( gids, Q );
    }

    /** get diagonal entry kpp */
    auto kqq = K.Diagonal( Q );

    /** compute all2leftright (projection i.e. dip - diq) */
    temp = AllToLeftRight( metric, DII, KIP, KIQ, kpp[ 0 ], kqq[ 0 ] );

    projection_time = omp_get_wtime() - beg;







    /** parallel median search */
    T median;

    if ( 1 )
    {
      median = hmlp::combinatorics::Select( n, n / 2, temp );
    }
    else
    {
      auto temp_copy = temp;
      std::sort( temp_copy.begin(), temp_copy.end() );
      median = temp_copy[ n / 2 ];
    }

    split[ 0 ].reserve( n / 2 + 1 );
    split[ 1 ].reserve( n / 2 + 1 );

    /** TODO: Can be parallelized */
    std::vector<std::size_t> middle;
    for ( size_t i = 0; i < n; i ++ )
    {
      //if      ( temp[ i ] < median ) split[ 0 ].push_back( i );
      //else if ( temp[ i ] > median ) split[ 1 ].push_back( i );
      //else                               middle.push_back( i );

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

    for ( size_t i = 0; i < middle.size(); i ++ )
    {
      if ( split[ 0 ].size() <= split[ 1 ].size() ) 
        split[ 0 ].push_back( middle[ i ] );
      else                                          
        split[ 1 ].push_back( middle[ i ] );
    }

    return split;
  };
}; /** end struct centersplit */










/**
 *  @brief This the splitter used in the randomized tree.
 *
 *  @TODO  This splitter often fails to produce an even split when
 *         the matrix is sparse.
 *
 */ 
template<typename SPDMATRIX, int N_SPLIT, typename T> 
struct randomsplit
{
  /** closure */
  SPDMATRIX *Kptr = NULL;

	/** (default) using angle distance from the Gram vector space */
  DistanceMetric metric = ANGLE_DISTANCE;

	/** overload with the operator */
  inline std::vector<std::vector<size_t> > operator()
  ( 
    std::vector<size_t>& gids,
    std::vector<size_t>& lids
  ) const 
  {
    assert( Kptr && ( N_SPLIT == 2 ) );

    SPDMATRIX &K = *Kptr;
    size_t n = gids.size();
    std::vector<std::vector<std::size_t> > split( N_SPLIT );
    std::vector<T> temp( n, 0.0 );

    /** randomly select two points p and q */
    size_t idf2c = std::rand() % n;
    size_t idf2f = std::rand() % n;
    while ( idf2c == idf2f ) idf2f = std::rand() % n;

    /** now get their gids */
    auto gidf2c = gids[ idf2c ];
    auto gidf2f = gids[ idf2f ];

    /** diagonal entries */
    hmlp::Data<T> DII( n, (size_t)1 );

    /** collecting DII */
    DII = K.Diagonal( gids );
    //for ( size_t i = 0; i < gids.size(); i ++ )
    //{
    //  DII[ i ] = K( gids[ i ], gids[ i ] );
    //}


    /** collecting KIP and KIQ */
    vector<size_t> P( 1, gidf2c );
    vector<size_t> Q( 1, gidf2f );
		vector<size_t> PQ( 2 ); PQ[ 0 ] = gidf2c; PQ[ 1 ] = gidf2f;
		auto KIPQ = K( gids, PQ );
    //auto KIP = K( gids, P );
    //auto KIQ = K( gids, Q );

		Data<T> KIP( n, (size_t)1 );
		Data<T> KIQ( n, (size_t)1 );

    if ( metric == GEOMETRY_DISTANCE ) 
    {
      KIPQ = K.PairwiseDistances( gids, PQ );
    }


    for ( size_t i = 0; i < gids.size(); i ++ )
		{
			KIP[ i ] = KIPQ( i, (size_t)0 );
			KIQ[ i ] = KIPQ( i, (size_t)1 );
		}

    /** get diagonal entry kpp and kqq */
    auto kpp = K.Diagonal( P );
    auto kqq = K.Diagonal( Q );

    /** compute all2leftright (projection i.e. dip - diq) */
    temp = AllToLeftRight( metric, DII, KIP, KIQ, kpp[ 0 ], kqq[ 0 ] );








//#ifdef DEBUG_SPDASKIT
//    printf( "randomsplit idf2c %lu idf2f %lu\n", idf2c, idf2f );
//#endif
//
//    /** compute projection */
//    #pragma omp parallel for
//    for ( size_t i = 0; i < n; i ++ )
//    {
//      switch ( metric )
//      {
//        case KERNEL_DISTANCE:
//        {
//          temp[ i ] = K( gids[ i ], gids[ idf2f ] ) - K( gids[ i ], gids[ idf2c ] );
//          break;
//        }
//        case ANGLE_DISTANCE:
//        {
//          T kip = K( gids[ i ],     gids[ idf2f ] );
//          T kiq = K( gids[ i ],     gids[ idf2c ] );
//          T kii = K( gids[ i ],     gids[ i ] );
//          T kpp = K( gids[ idf2f ], gids[ idf2f ] );
//          T kqq = K( gids[ idf2c ], gids[ idf2c ] );
//
//          /** ingore 1 from both terms */
//          temp[ i ] = ( kip * kip ) / ( kii * kpp ) - ( kiq * kiq ) / ( kii * kqq );
//          break;
//        }
//        default:
//        {
//          printf( "randomsplit() invalid splitting scheme\n" ); fflush( stdout );
//          exit( 1 );
//        }
//      }
//    }

    /** parallel median search */
    T median;
    if ( 1 )
    {
      median = hmlp::combinatorics::Select( n, n / 2, temp );
    }
    else
    {
      auto temp_copy = temp;
      std::sort( temp_copy.begin(), temp_copy.end() );
      median = temp_copy[ n / 2 ];
    }

    split[ 0 ].reserve( n / 2 + 1 );
    split[ 1 ].reserve( n / 2 + 1 );

    /** TODO: Can be parallelized */
    std::vector<std::size_t> middle;
		middle.reserve( n );
    for ( size_t i = 0; i < n; i ++ )
    {
      auto val = temp[ i ];
      //if      ( temp[ i ] < median ) split[ 0 ].push_back( i );
      //else if ( temp[ i ] > median ) split[ 1 ].push_back( i );
      //else                           middle.push_back( i );

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

    for ( size_t i = 0; i < middle.size(); i ++ )
    {
      if ( split[ 0 ].size() <= split[ 1 ].size() )
      {
        split[ 0 ].push_back( middle[ i ] );
      }
      else
      {
        split[ 1 ].push_back( middle[ i ] );
      }
    }

//    std::vector<size_t> lflag( n, 0 );
//    std::vector<size_t> rflag( n, 0 );
//    std::vector<size_t> pscan( n + 1, 0 );
//
//    #pragma omp parallel for
//    for ( size_t i = 0; i < n; i ++ )
//    {
//      if ( temp[ i ] > median ) rflag[ i ] = 1;
//      else                      lflag[ i ] = 1;
//    }
//
//    hmlp::tree::Scan( lflag, pscan );
//    split[ 0 ].resize( pscan[ n ] );
//    #pragma omp parallel for 
//    for ( size_t i = 0; i < n; i ++ )
//    {
//      if ( lflag[ i ] ) split[ 0 ][ pscan[ i + 1 ] - 1 ] = i;
//    }
//
//    hmlp::tree::Scan( rflag, pscan );
//    split[ 1 ].resize( pscan[ n ] );
//    #pragma omp parallel for 
//    for ( size_t i = 0; i < n; i ++ )
//    {
//      if ( rflag[ i ] ) split[ 1 ][ pscan[ i + 1 ] - 1 ] = i;
//    }


    return split;
  };
}; // end struct randomsplit


/**
 *  @brief This is the task wrapper of the exact KNN search we
 *         perform in the leaf node of the randomized tree.
 *         Currently our heap select cannot deal with duplicate
 *         id; thus, I need to use a std::set to check for the
 *         duplication before inserting the candidate into the
 *         heap.
 *
 *  @TODO  Improve the heap to deal with unique id.
 *
 */ 
template<int NUM_TEST, class NODE, typename T>
class KNNTask : public Task
{
  public:

    NODE *arg;
   
	  /** (default) using angle distance from the Gram vector space */
	  DistanceMetric metric = ANGLE_DISTANCE;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "nn" );
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

    void Execute( Worker* user_worker )
    {
      auto &K = *arg->setup->K;
      auto &X = *arg->setup->X;
      auto &NN = *arg->setup->NN;
      auto &gids = arg->gids;

      #pragma omp parallel for
      for ( size_t j = 0; j < gids.size(); j ++ )
      {
        std::set<size_t> NNset;

        for ( size_t i = 0; i < NN.row(); i ++ )
        {
          size_t jgid = gids[ j ];
          NNset.insert( NN[ jgid * NN.row() + i ].second );
        }

        for ( size_t i = 0; i < gids.size(); i ++ )
        {
          size_t igid = gids[ i ];
          size_t jgid = gids[ j ];

          if ( !NNset.count( igid ) )
          {
            T dist = 0;
            switch ( metric )
            {
              case GEOMETRY_DISTANCE:
              {
                size_t d = X.row();
                for ( size_t p = 0; p < d; p ++ )
								{
									T xip = X[ igid * d + p ];
									T xjp = X[ jgid * d + p ];
									dist += ( xip - xjp ) * ( xip - xjp );
								}
                break;
              }
              case KERNEL_DISTANCE:
              {
                dist = K( igid, igid ) + K( jgid, jgid ) - 2.0 * K( igid, jgid );
                break;
              }
              case ANGLE_DISTANCE:
              {
                T kij = K( igid, jgid );
                T kii = K( igid, igid );
                T kjj = K( jgid, jgid );

                dist = ( 1.0 - ( kij * kij ) / ( kii * kjj ) );
                break;
              }
              default:
              {
                printf( "KNNTask() invalid splitting scheme\n" ); fflush( stdout );
                exit( 1 );
              }
            }
            std::pair<T, size_t> query( dist, igid );
            hmlp::HeapSelect( 1, NN.row(), &query, NN.data() + jgid * NN.row() );
          }
          else
          {
            /** ignore duplication */
          }
        }

      } /** end omp parallel for */

#ifdef REPORT_ANN_ACCURACY
      /** test the accuracy of NN with exhausted search */
      double knn_acc = 0.0;
      size_t num_acc = 0;

      /** loop over all points in this leaf node */
      for ( size_t j = 0; j < gids.size(); j ++ )
      {
        if ( gids[ j ] >= NUM_TEST ) continue;

        std::set<size_t> NNset;
        hmlp::Data<std::pair<T, size_t>> nn_test( NN.row(), 1 );

        /** initialize nn_test to be the same as NN */
        for ( size_t i = 0; i < NN.row(); i ++ )
        {
          nn_test[ i ] = NN( i, gids[ j ] );
          NNset.insert( nn_test[ i ].second );
        }

        /** loop over all references */
        for ( size_t i = 0; i < K.row(); i ++ )
       {
          size_t igid = i;
          size_t jgid = gids[ j ];
          if ( !NNset.count( igid ) )
          {
            T dist = 0;
            switch ( metric )
            {
              case GEOMETRY_DISTANCE:
							{
                size_t d = X.row();
                for ( size_t p = 0; p < d; p ++ )
								{
									T xip = X[ igid * d + p ];
									T xjp = X[ jgid * d + p ];
									dist += ( xip - xjp ) * ( xip - xjp );
								}
                break;
							}
              case KERNEL_DISTANCE:
							{
                dist = K( igid, igid ) + K( jgid, jgid ) - 2.0 * K( igid, jgid );
                break;
							}
              case ANGLE_DISTANCE:
              {
                T kij = K( igid, jgid );
                T kii = K( igid, igid );
                T kjj = K( jgid, jgid );

                dist = ( 1.0 - ( kij * kij ) / ( kii * kjj ) );
                break;
              }
              default:
							{
                exit( 1 );
							}
            }
            std::pair<T, size_t> query( dist, igid );
            hmlp::HeapSelect( 1, NN.row(), &query, nn_test.data() );
            NNset.insert( igid );
          }
        }

        /** compute the acruracy */
        size_t correct = 0;
        NNset.clear();
        for ( size_t i = 0; i < NN.row(); i ++ ) NNset.insert( nn_test[ i ].second );
        for ( size_t i = 0; i < NN.row(); i ++ ) 
        {
          if ( NNset.count( NN( i, gids[ j ] ).second ) ) correct ++;
        }
        knn_acc += (double)correct / NN.row();
        num_acc ++;
      }
      arg->data.knn_acc = knn_acc;
      arg->data.num_acc = num_acc;
#endif

	};

}; /** end class KNNTask */



/**
 *  @brief This is the ANN routine design for CSC matrices.
 */ 
template<bool DOAPPROXIMATE, bool SORTED, typename T, typename CSCMATRIX>
Data<pair<T, size_t>> SparsePattern( size_t n, size_t k, CSCMATRIX &K )
{
  pair<T, size_t> initNN( numeric_limits<T>::max(), n );
  Data<pair<T, size_t>> NN( k, n, initNN );

  printf( "SparsePattern k %lu n %lu, NN.row %lu NN.col %lu ...", 
      k, n, NN.row(), NN.col() ); fflush( stdout );

  #pragma omp parallel for schedule( dynamic )
  for ( size_t j = 0; j < n; j ++ )
  {
    std::set<size_t> NNset;
    size_t nnz = K.ColPtr( j + 1 ) - K.ColPtr( j );
    if ( DOAPPROXIMATE && nnz > 2 * k ) nnz = 2 * k;

    //printf( "j %lu nnz %lu\n", j, nnz );

    for ( size_t i = 0; i < nnz; i ++ )
    {
      // TODO: this is lid. Need to be gid.
      auto row_ind = K.RowInd( K.ColPtr( j ) + i );
      auto val     = K.Value( K.ColPtr( j ) + i );

      if ( val ) val = 1.0 / std::abs( val );
      else       val = std::numeric_limits<T>::max() - 1.0;

      NNset.insert( row_ind );
      std::pair<T, std::size_t> query( val, row_ind );
      if ( nnz < k ) // not enough candidates
      {
        NN[ j * k + i  ] = query;
      }
      else
      {
        hmlp::HeapSelect( 1, NN.row(), &query, NN.data() + j * NN.row() );
      }
    }

    while ( nnz < k )
    {
      std::size_t row_ind = rand() % n;
      if ( !NNset.count( row_ind ) )
      {
        T val = std::numeric_limits<T>::max() - 1.0;
        std::pair<T, std::size_t> query( val, row_ind );
        NNset.insert( row_ind );
        NN[ j * k + nnz ] = query;
        nnz ++;
      }
    }
  }
  printf( "Done.\n" ); fflush( stdout );

  if ( SORTED )
  {
    printf( "Sorting ... " ); fflush( stdout );
    struct 
    {
      bool operator () ( std::pair<T, size_t> a, std::pair<T, size_t> b )
      {   
        return a.first < b.first;
      }   
    } ANNLess;

    //printf( "SparsePattern k %lu n %lu, NN.row %lu NN.col %lu\n", k, n, NN.row(), NN.col() );

    #pragma omp parallel for
    for ( size_t j = 0; j < NN.col(); j ++ )
    {
      std::sort( NN.data() + j * NN.row(), NN.data() + ( j + 1 ) * NN.row(), ANNLess );
    }
    printf( "Done.\n" ); fflush( stdout );
  }
  
//  for ( size_t j = 0; j < NN.col(); j ++ )
//  {
//	for ( size_t i = 0; i < NN.row(); i ++ )
//	{
//	  printf( "%4lu ", NN[ j * k + i ].second );
//	}
//	printf( "\n" );
//  }

  return NN;
}; /** end SparsePattern() */


/*
 * @brief Helper functions for sorting sampling neighbors.
 */ 
template<typename TA, typename TB>
pair<TB, TA> flip_pair( const pair<TA, TB> &p )
{
  return pair<TB, TA>( p.second, p.first );
}; /** end flip_pair() */


template<typename TA, typename TB>
multimap<TB, TA> flip_map( const map<TA, TB> &src )
{
  multimap<TB, TA> dst;
  transform( src.begin(), src.end(), inserter( dst, dst.begin() ), 
                 flip_pair<TA, TB> );
  return dst;
}; /** end flip_map() */


/**
 *  @brief Building neighbors for each tree node.
 */ 
template<typename NODE, typename T>
void BuildNeighbors( NODE *node, size_t nsamples )
{
  /** early return if no neighbors were provided */
  if ( !node->setup->NN ) return;


  auto &NN = *(node->setup->NN);
  std::vector<size_t> &gids = node->gids;
  auto &snids = node->data.snids;
  auto &pnids = node->data.pnids;
  int n = node->n;
  int k = NN.row();
  if ( node->isleaf )
  {
    /** Pruning neighbor lists/sets: */
    pnids = std::unordered_set<size_t>();
    for ( size_t jj = 0; jj < n; jj ++ )
    {
      for ( size_t ii = 0; ii < k / 2; ii ++ )
      {
        pnids.insert( NN( ii,  gids[ jj ] ).second );
      }
    }
    /** remove "own" points */
    for ( int i = 0; i < n; i ++ )
    {
      pnids.erase( gids[ i ] );
    }
    //printf("Size of pruning neighbor set: %lu \n", pnids.size());
    /**
		 *  Sampling neighbors
     *  To think about: Make building sampling neighbor adaptive.  
     *  E.g. request 0-100 closest neighbors, 
     *  if additional 100 neighbors are requested, return sneighbors 100-200 
		 */ 
    snids = std::map<size_t, T>(); 
    vector<pair<T, size_t>> tmp( k / 2 * n ); 
    set<size_t> nodeIdx( gids.begin() , gids.end() );    
    /** Allocate array for sorting */
    for ( size_t ii = ( k + 1 ) / 2; ii < k; ii ++ )
    {
      for ( size_t jj = 0; jj < n; jj ++ )
      {
        tmp[ ( ii - ( k + 1 ) / 2 ) * n + jj ] = NN( ii, gids[ jj ] );
      }
    }
    sort( tmp.begin() , tmp.end() );
    int i = 0;
    while ( snids.size() < nsamples && i <  (k-1) * n / 2 )
    {
      if ( !pnids.count( tmp[i].second ) && !nodeIdx.count( tmp[i].second ) )
      {
        snids.insert( std::pair<size_t,T>( tmp[i].second , tmp[i].first ) );
      }
      i++;
    } 
    //printf("Size of sampling neighbor list: %lu \n", snids.size());
  }
  else
  {
    /** At interior node */
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
    for ( auto cur = rsnids.begin(); cur != rsnids.end(); cur ++ )
    {
      auto ret = snids.insert( *cur );
      if ( ret.second == false )
      {
        // Update distance?
        if ( ret.first->second > (*cur).first)
        {
          ret.first->second = (*cur).first;
        }
      }
    }

    // Remove "own" points
    for (int i = 0; i < n; i ++ )
    {
      snids.erase( gids[ i ] );
    }

    // Remove pruning neighbors from left and right
    for (auto cur = lpnids.begin(); cur != lpnids.end(); cur++ )
    {
      snids.erase( *cur );
    }
    for (auto cur = rpnids.begin(); cur != rpnids.end(); cur++ )
    {
      snids.erase( *cur );
    }

    //printf("Interior sampling neighbor size: %lu\n", snids.size());
  }
}; // end BuildNeighbors()




















/**
 *  @brief Compute the cofficient matrix by R11^{-1} * proj.
 *
 */ 
template<typename NODE, typename T>
void Interpolate( NODE *node )
{
  /** early return */
  if ( !node ) return;

  auto &K = *node->setup->K;
  auto &data = node->data;
  auto &skels = data.skels;
  auto &proj = data.proj;
  auto &jpvt = data.jpvt;
  auto s = proj.row();
  auto n = proj.col();

  /** early return if the node is incompressible or all zeros */
  if ( !data.isskel || proj[ 0 ] == 0 ) return;

  assert( s );
  assert( s <= n );
  assert( jpvt.size() == n );

  /** if is skeletonized, reserve space for w_skel and u_skel */
  if ( data.isskel )
  {
    data.w_skel.reserve( skels.size(), MAX_NRHS );
    data.u_skel.reserve( skels.size(), MAX_NRHS );
  }



  /** early return if ( s == n ) */
  //if ( s == n )
  //{
  //  for ( int i = 0; i < s; i ++ ) skels[ i ] = i;
  //  for ( int j = 0; j < s; j ++ )
  //  {
  //    for ( int i = 0; i < s; i ++ )
  //    {
  //      if ( i == j ) proj[ j * s + i ] = 1.0;
  //  	else          proj[ j * s + i ] = 0.0;
  //    }
  //  }
  //  return;
  //}

  /** fill in R11 */
  hmlp::Data<T> R1( s, s, 0.0 );

  for ( int j = 0; j < s; j ++ )
  {
    for ( int i = 0; i < s; i ++ )
    {
      if ( i <= j ) R1[ j * s + i ] = proj[ j * s + i ];
    }
  }

  /** copy proj to tmp */
  hmlp::Data<T> tmp = proj;

  /** proj = inv( R1 ) * proj */
  hmlp::xtrsm( "L", "U", "N", "N", s, n, 1.0, R1.data(), s, tmp.data(), s );

  /** Fill in proj */
  for ( int j = 0; j < n; j ++ )
  {
    for ( int i = 0; i < s; i ++ )
    {
  	  proj[ jpvt[ j ] * s + i ] = tmp[ j * s + i ];
    }
  }
 

}; /** end Interpolate() */


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
    }

    void Execute( Worker* user_worker )
    {
      //printf( "%lu Interpolate beg\n", arg->treelist_id );
      Interpolate<NODE, T>( arg );
      //printf( "%lu Interpolate end\n", arg->treelist_id );
    };

}; /** end class InterpolateTask */




/**
 *  TODO: I decided not to use the sampling pool
 */ 
template<typename NODE, typename T>
void RowSamples( NODE *node, size_t nsamples )
{
  /** gather shared data and create reference */
  auto &K = *node->setup->K;

  /** amap contains nsamples of row gids of K */
  vector<size_t> &amap = node->data.candidate_rows;

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
      snids.clear();

      for ( auto gid : gids )
        for ( size_t i = 0; i < NN.row() / 2; i ++ )
          pnids.insert( NN( i, gid ).second );



      //for ( size_t j = 0; j < gids.size(); j ++ )
      //  for ( size_t i = 0; i < NN.row() / 2; i ++ )
      //    pnids.insert( NN( i, gids[ j ] ).second );

      /** Remove self on-diagonal indices */
      for ( auto gid : gids ) pnids.erase( gid );
      //for ( size_t j = 0; j < gids.size(); j ++ )
      //  pnids.erase( gids[ j ] );

      vector<pair<T, size_t>> tmp( knum * gids.size() );

      for ( size_t j = 0; j < gids.size(); j ++ )
        for ( size_t i = kbeg; i < kend; i ++ )
          tmp[ j * knum + ( i - kbeg ) ] = NN( i, gids[ j ] );

      /** Create a sorted list */
      sort( tmp.begin(), tmp.end() );
    
      for ( auto it = tmp.begin(); it != tmp.end(); it ++ )
      {
        /** Create a single query */
        vector<size_t> sample_query( 1, (*it).second );
				vector<size_t> validation = 
					node->setup->ContainAny( sample_query, node->morton );

        if ( !pnids.count( (*it).second ) && !validation[ 0 ] )
        {
          /** Duplication is handled by std::map */
          auto ret = snids.insert( pair<size_t, T>( (*it).second, (*it).first ) );
        }

        /** While we have enough samples, then exit */
        if ( snids.size() >= nsamples ) break;
      }
    }
    else
    {
      auto &lsnids = node->lchild->data.snids;
      auto &rsnids = node->rchild->data.snids;
      auto &lpnids = node->lchild->data.pnids;
      auto &rpnids = node->rchild->data.pnids;

      /** Merge left children's sampling neighbors */
      snids = lsnids;

      /** Merge right child's sample neighbors and update duplicate. */
      for ( auto it = rsnids.begin(); it != rsnids.end(); it ++ )
      {
        auto ret = snids.insert( *it );
        if ( !ret.second )
        {
          if ( ret.first->second > (*it).first )
            ret.first->second = (*it).first;
        }
      }

      /** Remove on-diagonal indicies (gids) */
      for ( auto gid : gids ) snids.erase( gid );
      //for ( size_t i = 0; i < gids.size(); i ++ ) snids.erase( gids[ i ] );

      /** Remove direct evaluation indices */
      for ( auto lpnid : lpnids ) snids.erase( lpnid );
      for ( auto rpnid : lpnids ) snids.erase( rpnid );

      //for ( auto it = lpnids.begin(); it != lpnids.end(); it ++ )
      //  snids.erase( *it );
      //for ( auto it = rpnids.begin(); it != rpnids.end(); it ++ )
      //  snids.erase( *it );
    }

    /** create an order snids by flipping the std::map */
    multimap<T, size_t> ordered_snids = flip_map( node->data.snids );

    if ( nsamples < K.col() - node->n )
    {
      amap.reserve( nsamples );

      for ( auto it  = ordered_snids.begin(); it != ordered_snids.end(); it++ )
      {
        /** (*it) has type pair<T, size_t> */
        amap.push_back( (*it).second );
        if ( amap.size() >= nsamples ) break;
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

      //  if ( now->parent ) 
      //  {
      //      //printf( "node %lu now level %lu sibling->data.pool.size() %lu\n", 
      //      //    node->treelist_id, now->l,
      //      //    sibling->data.pool.size() ); fflush( stdout );
      //    if ( !sibling )
      //      printf( "node %lu now level %lu\n", 
      //          node->treelist_id, now->l); fflush( stdout );
      //    //assert( sibling );
      //  }

      //  /** 
      //   *  Move toward parent.
      //   *  We need to cast the pointer here becuase the now->parent may be in
      //   *  type tree::Node, but NODE = mpitree::Node.
      //   */
      //  now = (NODE*)(now->parent);
      //}


      //for ( auto it = candidates.begin(); it != candidates.end(); it ++ )
      //{
      //  if ( node->isleaf ) 
      //  {
      //    auto &pnids = node->data.pnids;
      //    if ( !pnids.count( (*it).second ) )
      //      amap.push_back( (*it).second );
      //  }
      //  else
      //  {
      //    auto &lpnids = node->lchild->data.pnids;
      //    auto &rpnids = node->rchild->data.pnids;
      //    if ( !lpnids.count( (*it).second ) && !rpnids.count( (*it).second ) )
      //      amap.push_back( (*it).second );
      //  }
      //  if ( amap.size() >= nsamples ) break;
      //}






      /** Uniform samples */
      if ( amap.size() < nsamples )
      {
        while ( amap.size() < nsamples )
        {
          //size_t sample = rand() % K.col();
          auto important_sample = K.ImportantSample( 0 );
          size_t sample = important_sample.second;

          /** create a single query */
          vector<size_t> sample_query( 1, sample );

          vector<size_t> validation = 
            node->setup->ContainAny( sample_query, node->morton );

          /**
           *  check duplication using std::find, but check whether the sample
           *  belongs to the diagonal block using Morton ID.
           */ 
          if ( find( amap.begin(), amap.end(), sample ) == amap.end() &&
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
        if ( find( amap.begin(), amap.end(), sample ) == amap.end() )
        {
          amap.push_back( sample );
        }
      }
    }
  } /** end if ( node->setup->NN ) */

}; /** end RowSamples() */








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
    auto &lskels = lchild->data.skels;
    auto &rskels = rchild->data.skels;

    /** if either child is not skeletonized, then return */
    if ( !lskels.size() || !rskels.size() ) return;

    /** concatinate [ lskels, rskels ] */
    candidate_cols = lskels;
    candidate_cols.insert( candidate_cols.end(), 
        rskels.begin(), rskels.end() );
  }

  /** decide number of rows to sample */
  size_t nsamples = 2 * candidate_cols.size();

  /** make sure we at least m samples */
  if ( nsamples < 2 * node->setup->m ) nsamples = 2 * node->setup->m;

  /** sample off-diagonal rows */
  RowSamples<NODE, T>( node, nsamples );

  /** 
   *  get KIJ for skeletonization 
   *
   *  notice that operator () may involve MPI collaborative communication.
   *
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

    //xgemm
    //(
    //  "No transpose", "No transpose",
    //  over_size_rank, candidate_cols.size(), nsamples,
    //  1.0, G.data(), G.row(),
    //  Ksamples.data(), Ksamples.row(),
    //  0.0, KIJ.data(), KIJ.row()
    //);
  }

}; /** end GetSkeletonMatrix() */







/**
 *
 */ 
template<typename NODE, typename T>
class GetSkeletonMatrixTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      ostringstream ss;
      arg = user_arg;
      name = string( "par-gskm" );
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
      arg->DependencyAnalysis( RW, this );

      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( R, this );
        arg->rchild->DependencyAnalysis( R, this );
      }

      /** Try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      GetSkeletonMatrix<NODE, T>( arg );

      ///** Create a promise and get its future */
      //promise<bool> done;
      //auto future = done.get_future();

      //thread t( [&done] ( NODE *arg ) -> void {
      //    GetSkeletonMatrix<NODE, T>( arg ); 
      //    done.set_value( true );
      //}, arg );
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

}; /** end class GetSkeletonMatrixTask */ 




























/**
 *  @brief Skeletonization with interpolative decomposition.
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
void Skeletonize2( NODE *node )
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
    assert( ADAPTIVE );
    if ( !node->isleaf && ( !node->lchild->data.isskel || !node->rchild->data.isskel ) )
    {
      skels.clear();
      proj.resize( 0, 0 );
      data.isskel = false;
      return;
    }
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
  
  /** Relabel skeletions with the real lids */
  for ( size_t i = 0; i < skels.size(); i ++ )
  {
    skels[ i ] = candidate_cols[ skels[ i ] ];
  }

  /** Update pruning neighbor list */
  data.pnids.clear();
  for ( auto skel : skels )
    for ( size_t i = 0; i < NN.row() / 2; i ++ )
      data.pnids.insert( NN( i, skel ).second );

}; /** end Skeletonize2() */




/**
 *
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
class SkeletonizeTask2 : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      ostringstream ss;
      arg = user_arg;
      name = string( "sk" );
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

      //flops += ( 2.0 / 3.0 ) * k * k * ( 3 * m - k );
      //mops += 2.0 * m * k;
      //flops += 2.0 * m * n * k;
      //mops += 2.0 * ( m * k + k * n + m * n );
      //flops += ( 1.0 / 3.0 ) * k * k * n;
      //mops += 2.0 * ( k * k + k * n );

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
      //printf( "%lu Skel beg\n", arg->treelist_id );
      Skeletonize2<ADAPTIVE, LEVELRESTRICTION, NODE, T>( arg );
      //printf( "%lu Skel end\n", arg->treelist_id );
    };

}; /** end class SkeletonizeTask */































///**
// *  @brief Skeletonization with interpolative decomposition.
// */ 
//template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
//void Skeletonize( NODE *node )
//{
//  /** early return if we do not need to skeletonize */
//  if ( !node->parent ) return;
//
//  double beg = 0.0, kij_skel_time = 0.0, merge_neighbors_time = 0.0, id_time = 0.0;
//
//  /** gather shared data and create reference */
//  auto &K = *node->setup->K;
//  auto maxs = node->setup->s;
//  auto stol = node->setup->stol;
//  auto &NN = *node->setup->NN;
//
//  /** gather per node data and create reference */
//  auto &data = node->data;
//  auto &skels = data.skels;
//  auto &proj = data.proj;
//  auto &jpvt = data.jpvt;
//  auto *lchild = node->lchild;
//  auto *rchild = node->rchild;
//
//  /** early return if fail to skeletonize. */
//  if ( LEVELRESTRICTION )
//  {
//    assert( ADAPTIVE );
//    if ( !node->isleaf && ( !lchild->data.isskel || !rchild->data.isskel ) )
//    {
//      skels.clear();
//      proj.resize( 0, 0 );
//      data.isskel = false;
//      return;
//    }
//  }
//  else
//  {
//    //skels.resize( maxs );
//    //proj.resize( )
//  }
//
//  /** random sampling or importance sampling for rows. */
//  std::vector<size_t> amap;
//  std::vector<size_t> bmap;
//  std::vector<size_t> &gids = node->gids;
//
//
//  /** merge children's skeletons */
//  if ( node->isleaf )
//  {
//    bmap = node->gids;
//  }
//  else
//  {
//    auto &lskels = lchild->data.skels;
//    auto &rskels = rchild->data.skels;
//    bmap = lskels;
//    bmap.insert( bmap.end(), rskels.begin(), rskels.end() );
//  }
//
//  /** decide the number of row samples */
//  auto nsamples = 2 * bmap.size();
//
//  /** make sure we at least m samples */
//  if ( nsamples < 2 * node->setup->m ) nsamples = 2 * node->setup->m;
//
//
//  /** Build Node Neighbors from all nearest neighbors */
//  beg = omp_get_wtime();
//  BuildNeighbors<NODE, T>( node, nsamples );
//  merge_neighbors_time = omp_get_wtime() - beg;
//  
//  /** update merge_neighbors timer */
//  data.merge_neighbors_time = merge_neighbors_time;
//
//
//  auto &snids = data.snids;
//  // Order snids by distance
//  std::multimap<T, size_t > ordered_snids = flip_map( snids );
//  if ( nsamples < K.col() - node->n )
//  {
//    amap.reserve( nsamples );
//
//
//    /** sibling's skeletons */
//    //auto *sibling = node->sibling;
//    //if ( sibling )
//    //if ( 0 )
//    //{
//    //  if ( sibling->isleaf )
//    //  {        
//    //    auto &sibling_gids = sibling->gids;
//    //    for ( size_t i = 0; i < sibling_gids.size(); i ++ )
//    //      amap.push_back( sibling_gids[ i ] );
//    //  }
//    //  else
//    //  {
//    //    auto &sibling_lskel = sibling->lchild->data.skels;
//    //    auto &sibling_rskel = sibling->rchild->data.skels;
//    //    for ( size_t i = 0; i < sibling_lskel.size(); i ++ )
//    //      amap.push_back( sibling_lskel[ i ] );
//    //    for ( size_t i = 0; i < sibling_rskel.size(); i ++ )
//    //      amap.push_back( sibling_rskel[ i ] );
//    //  }
//    //}
//    //if ( amap.size() + ordered_snids.size() > nsamples )
//    //{
//    //  //printf( "exceeding current sample size %lu, enlarge to %lu\n",
//    //  //    nsamples, amap.size() + ordered_snids.size() );
//    //  nsamples = amap.size() + ordered_snids.size();
//    //}
//
//
//    for ( auto cur = ordered_snids.begin(); cur != ordered_snids.end(); cur++ )
//    {
//      amap.push_back( cur->second );
//      if ( amap.size() >= nsamples ) break;
//    }
//    //if ( amap.size() > nsamples ) printf( "amap.size() %lu\n", amap.size() );
//
//
//    // Uniform samples.
//    if ( amap.size() < nsamples )
//    {
//      while ( amap.size() < nsamples )
//      {
//        size_t sample;
//
//        if ( rand() % 5 ) // 80% chances to use important sample
//        {
//          auto importantsample = K.ImportantSample( bmap[ rand() % bmap.size() ] );
//          sample = importantsample.second;
//        }
//        else
//        {
//          sample = rand() % K.col();
//        }
//
//        /** create a single query */
//        std::vector<size_t> sample_query( 1, sample );
//
//        /**
//         *  check duplication using std::find, but check whether the sample
//         *  belongs to the diagonal block using Morton ID.
//         */ 
//        if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() &&
//             !node->ContainAny( sample_query ) )
//        {
//          amap.push_back( sample );
//        }
//      }
//    }
//  }
//  else // Use all off-diagonal blocks without samples.
//  {
//    for ( int sample = 0; sample < K.col(); sample ++ )
//    {
//      if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() )
//      {
//        amap.push_back( sample );
//      }
//    }
//  }
//
//  /** get submatrix Kab from K */
//  beg = omp_get_wtime();
//  auto Kab = K( amap, bmap );
//  kij_skel_time = omp_get_wtime() - beg;
//
//
//  /** update kij counter */
//  data.kij_skel.first  = kij_skel_time;
//  data.kij_skel.second = amap.size() * bmap.size();
//
//  /** interpolative decomposition */
//  beg = omp_get_wtime();
//  size_t N = K.col();
//  size_t m = amap.size();
//  size_t n = bmap.size();
//  size_t q = gids.size();
//  /** Bill's l2 norm scaling factor */
//  T scaled_stol = std::sqrt( (T)n / q ) * std::sqrt( (T)m / (N - q) ) * stol;
//  /** account for uniform sampling */
//  scaled_stol *= std::sqrt( (T)q / N );
//  /** We use a tighter Frobenius norm */
//  //scaled_stol /= std::sqrt( q );
//
//  hmlp::lowrank::id<ADAPTIVE, LEVELRESTRICTION>
//  ( 
//    amap.size(), bmap.size(), maxs, scaled_stol, /** ignore if !ADAPTIVE */
//    Kab, skels, proj, jpvt
//  );
//  id_time = omp_get_wtime() - beg;
//
//  /** update id timer */
//  data.id_time = id_time;
//
//  /** depending on the flag, decide isskel or not */
//  if ( LEVELRESTRICTION )
//  {
//    data.isskel = (skels.size() != 0);
//  }
//  else
//  {
//    assert( skels.size() );
//    assert( proj.size() );
//    assert( jpvt.size() );
//    data.isskel = true;
//  }
//  
//  /** relabel skeletions with the real gids */
//  for ( size_t i = 0; i < skels.size(); i ++ )
//  {
//    skels[ i ] = bmap[ skels[ i ] ];
//  }
//
//  /** separate interpolation of proj */
//  //Interpolate<NODE, T>( node );
//
//  /** update pruning neighbor list */
//  data.pnids.clear();
//  for ( size_t j = 0 ; j < skels.size() ; j ++ )
//    for ( size_t i = 0; i < NN.row() / 2; i ++ )
//      data.pnids.insert( NN( i, skels[ j ] ).second );
//
//}; /** end void Skeletonize() */
//
//
//
///**
// *
// */ 
//template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
//class SkeletonizeTask : public Task
//{
//  public:
//
//    NODE *arg = NULL;
//
//    void Set( NODE *user_arg )
//    {
//      ostringstream ss;
//      arg = user_arg;
//      name = string( "sk" );
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
//      double flops = 0.0, mops = 0.0;
//
//      auto &K = *arg->setup->K;
//      size_t n = arg->data.proj.col();
//      size_t m = 2 * n;
//      size_t k = arg->data.proj.row();
//
//      /** Kab */
//      flops += K.flops( m, n );
//
//      /** GEQP3 */
//      flops += ( 2.0 / 3.0 ) * n * n * ( 3 * m - n );
//      mops += ( 2.0 / 3.0 ) * n * n * ( 3 * m - n );
//
//      /* TRSM */
//      flops += k * ( k - 1 ) * ( n + 1 );
//      mops  += 2.0 * ( k * k + k * n );
//
//      //flops += ( 2.0 / 3.0 ) * k * k * ( 3 * m - k );
//      //mops += 2.0 * m * k;
//      //flops += 2.0 * m * n * k;
//      //mops += 2.0 * ( m * k + k * n + m * n );
//      //flops += ( 1.0 / 3.0 ) * k * k * n;
//      //mops += 2.0 * ( k * k + k * n );
//
//      event.Set( label + name, flops, mops );
//      arg->data.skeletonize = event;
//    };
//
//    void DependencyAnalysis()
//    {
//      arg->DependencyAnalysis( RW, this );
//      if ( !arg->isleaf )
//      {
//        arg->lchild->DependencyAnalysis( R, this );
//        arg->rchild->DependencyAnalysis( R, this );
//      }
//      this->TryEnqueue();
//    };
//
//    void Execute( Worker* user_worker )
//    {
//      //printf( "%lu Skel beg\n", arg->treelist_id );
//      Skeletonize<ADAPTIVE, LEVELRESTRICTION, NODE, T>( arg );
//      //printf( "%lu Skel end\n", arg->treelist_id );
//    };
//
//}; /** end class SkeletonizeTask */


































/**
 *  @brief 
 */
template<typename NODE, typename T>
void UpdateWeights( NODE *node )
{
#ifdef DEBUG_SPDASKIT
  printf( "%lu UpdateWeight\n", node->treelist_id ); fflush( stdout );
#endif
  /** early return */
  if ( !node->parent || !node->data.isskel ) return;

  /** eanble nested parallelism */
  //int num_threads = omp_get_num_threads();
  //if ( node->l < 4 ) omp_set_num_threads( 4 );

  /** gather shared data and create reference */
  auto &w = *node->setup->w;

  /** gather per node data and create reference */
  auto &data = node->data;
  auto &proj = data.proj;
  auto &skels = data.skels;
  auto &w_skel = data.w_skel;
  auto &w_leaf = data.w_leaf;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;


  size_t nrhs = w.col();

  /** w_skel is s-by-nrhs, initial values are not important */
  w_skel.resize( skels.size(), nrhs );

  //printf( "%lu UpdateWeight w_skel.num() %lu\n", node->treelist_id, w_skel.num() );

  if ( node->isleaf )
  {
    //double beg = omp_get_wtime();
    //double w_leaf_time = omp_get_wtime() - beg;

#ifdef DEBUG_SPDASKIT
    printf( "m %lu n %lu k %lu\n", 
      w_skel.row(), w_skel.col(), w_leaf.col());
    printf( "proj.row() %lu w_leaf.row() %lu w_skel.row() %lu\n", 
        proj.row(), w_leaf.row(), w_skel.row() );
#endif

    if ( w_leaf.size() )
    {
      //printf( "%8lu w_leaf allocated [%lu %lu]\n", 
      //    node->morton, w_leaf.row(), w_leaf.col() ); fflush( stdout );
      
      /** w_leaf is allocated */
      xgemm
      (
        "N", "N",
        w_skel.row(), w_skel.col(), w_leaf.row(),
        1.0, proj.data(),   proj.row(),
             w_leaf.data(), w_leaf.row(),
        0.0, w_skel.data(), w_skel.row()
      );
    }
    else
    {
      /** w_leaf is not allocated, use w_view instead */
      View<T> W = data.w_view;
      //printf( "%8lu n2s W[%lu %lu ld %lu]\n", 
      //    node->morton, W.row(), W.col(), W.ld() ); fflush( stdout );
      //for ( int i = 0; i < 10; i ++ )
      //  printf( "%lu W.data() + %d = %E\n", node->gids[ i ], i, *(W.data() + i) );
      xgemm
      (
        "N", "N",
        w_skel.row(), w_skel.col(), W.row(),
        1.0, proj.data(),   proj.row(),
                W.data(),       W.ld(),
        0.0, w_skel.data(), w_skel.row()
      );
    }

    //double update_leaf_time = omp_get_wtime() - beg;
    //printf( "%lu, m %lu n %lu k %lu, total %.3E\n", 
    //  node->treelist_id, 
    //  w_skel.row(), w_skel.col(), w_leaf.col(), update_leaf_time );
  }
  else
  {
    //double beg = omp_get_wtime();
    auto &w_lskel = lchild->data.w_skel;
    auto &w_rskel = rchild->data.w_skel;
    auto &lskel = lchild->data.skels;
    auto &rskel = rchild->data.skels;

    //if ( 1 )
    if ( node->treelist_id > 6 )
    {
      //printf( "%8lu n2s\n", node->morton ); fflush( stdout );
      xgemm
      (
        "N", "N",
        w_skel.row(), w_skel.col(), lskel.size(),
        1.0,    proj.data(),    proj.row(),
             w_lskel.data(), w_lskel.row(),
        0.0,  w_skel.data(),  w_skel.row()
      );
      xgemm
      (
        "N", "N",
        w_skel.row(), w_skel.col(), rskel.size(),
        1.0,    proj.data() + proj.row() * lskel.size(), proj.row(),
             w_rskel.data(), w_rskel.row(),
        1.0,  w_skel.data(),  w_skel.row()
      );
    }
    else
    {
      /** create a view proj_v */
      View<T> P( false,   proj ), PL,
                                  PR;
      View<T> W( false, w_skel ), WL( false, w_lskel ),
                                  WR( false, w_rskel );
      /** P = [ PL, PR ] */
      P.Partition1x2( PL, PR, lskel.size(), LEFT );
      /** W  = PL * WL */
      gemm::xgemm<GEMM_NB>( (T)1.0, PL, WL, (T)0.0, W );
      W.DependencyCleanUp();
      /** W += PR * WR */
      gemm::xgemm<GEMM_NB>( (T)1.0, PR, WR, (T)1.0, W );
      //W.DependencyCleanUp();
    }

    //double update_leaf_time = omp_get_wtime() - beg;
    //printf( "%lu, total %.3E\n", 
    //  node->treelist_id, update_leaf_time );
  }

  //if ( w_skel.HasIllegalValue() )
  //{
  //  printf( "Illegal value in w_skel\n" ); fflush( stdout );
  //}



#ifdef DEBUG_SPDASKIT
  printf( "%lu UpdateWeight done\n", node->treelist_id ); fflush( stdout );
#endif

}; /** end UpdateWeights() */


/**
 *
 */ 
template<typename NODE, typename T>
class UpdateWeightsTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "n2s" );
      label = to_string( arg->treelist_id );

      /** Compute flops and mops */
      double flops, mops;
      auto &gids = arg->gids;
      auto &skels = arg->data.skels;
      auto &w = *arg->setup->w;
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

      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Assume computation bound */
      cost = flops / 1E+9;
      /** "HIGH" priority (critical path) */
      priority = true;
    };

    void Prefetch( Worker* user_worker )
    {
      auto &proj = arg->data.proj;
      __builtin_prefetch( proj.data() );
      auto &w_skel = arg->data.w_skel;
      __builtin_prefetch( w_skel.data() );
      if ( arg->isleaf )
      {
        auto &w_leaf = arg->data.w_leaf;
        __builtin_prefetch( w_leaf.data() );
      }
      else
      {
        auto &w_lskel = arg->lchild->data.w_skel;
        __builtin_prefetch( w_lskel.data() );
        auto &w_rskel = arg->rchild->data.w_skel;
        __builtin_prefetch( w_rskel.data() );
      }
#ifdef HMLP_USE_CUDA
      hmlp::Device *device = NULL;
      if ( user_worker ) device = user_worker->GetDevice();
      if ( device ) 
      {
        proj.CacheD( device );
        proj.PrefetchH2D( device, 1 );
        if ( arg->isleaf )
        {
          auto &w_leaf = arg->data.w_leaf;
          w_leaf.CacheD( device );
          w_leaf.PrefetchH2D( device, 1 );
        }
        else
        {
          auto &w_lskel = arg->lchild->data.w_skel;
          w_lskel.CacheD( device );
          w_lskel.PrefetchH2D( device, 1 );
          auto &w_rskel = arg->rchild->data.w_skel;
          w_rskel.CacheD( device );
          w_rskel.PrefetchH2D( device, 1 );
        }
      }
#endif
    };

    void GetEventRecord()
    {
      arg->data.updateweight = event;
    };

    void DependencyAnalysis()
    {
      if ( arg->parent ) 
      {
        arg->DependencyAnalysis( W, this );
      }
      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( R, this );
        arg->rchild->DependencyAnalysis( R, this );
      }
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
#ifdef HMLP_USE_CUDA 
      hmlp::Device *device = NULL;
      if ( user_worker ) device = user_worker->GetDevice();
      if ( device ) gpu::UpdateWeights( device, arg );
      else               UpdateWeights<NODE, T>( arg );
#else
      UpdateWeights<NODE, T>( arg );
#endif
    };

}; /** end class UpdateWeightsTask */



/**
 *  @brief Compute the interation from column skeletons to row
 *         skeletons. Store the results in the node. Later
 *         there is a SkeletonstoAll function to be called.
 *
 */ 
template<bool NNPRUNE, typename NODE, typename T>
void SkeletonsToSkeletons( NODE *node )
{
#ifdef DEBUG_SPDASKIT
  printf( "%lu Skel2Skel \n", node->treelist_id ); fflush( stdout );
#endif

  if ( !node->parent || !node->data.isskel ) return;

  double beg, kij_s2s_time = 0.0, u_skel_time, s2s_time;

  auto *FarNodes = &node->FarNodes;
  if ( NNPRUNE ) FarNodes = &node->NNFarNodes;

  auto &K = *node->setup->K;
  auto &data = node->data;
  auto &amap = node->data.skels;
  auto &u_skel = node->data.u_skel;
  auto &FarKab = node->data.FarKab;

  size_t nrhs = node->setup->w->col();

  /** initilize u_skel to be zeros( s, nrhs ). */
  beg = omp_get_wtime();
  u_skel.resize( 0, 0 );
  u_skel.resize( amap.size(), nrhs, 0.0 );
  u_skel_time = omp_get_wtime() - beg;

  size_t offset = 0;


  /** create a base view for FarKab */
  View<T> FarKab_v( FarKab );

  /** reduce all u_skel */
  for ( auto it = FarNodes->begin(); it != FarNodes->end(); it ++ )
  {
    auto &bmap = (*it)->data.skels;
    auto &w_skel = (*it)->data.w_skel;
    assert( w_skel.col() == nrhs );
    assert( w_skel.row() == bmap.size() );
    assert( w_skel.size() == nrhs * bmap.size() );

    if ( FarKab.size() ) /** Kab is cached */
    {
      //if ( node->treelist_id > 6 )
      if ( 1 )
      {
        assert( FarKab.row() == amap.size() );
        assert( u_skel.row() * offset <= FarKab.size() );

        //printf( "%8lu s2s %8lu w_skel[%lu %lu]\n", 
        //    node->morton, (*it)->morton, w_skel.row(), w_skel.col() );
        //fflush( stdout );
        xgemm
        (
          "N", "N",
          u_skel.row(), u_skel.col(), w_skel.row(),
          1.0, FarKab.data() + u_skel.row() * offset, FarKab.row(),
               w_skel.data(),          w_skel.row(),
          1.0, u_skel.data(),          u_skel.row()
        );

      }
      else
      {
        /** create views */
        View<T> U( false, u_skel );
        View<T> W( false, w_skel );
        View<T> Kab;
        assert( FarKab.col() >= W.row() + offset );
        Kab.Set( FarKab.row(), W.row(), 0, offset, &FarKab_v );
        gemm::xgemm<GEMM_NB>( (T)1.0, Kab, W, (T)1.0, U );
      }

      /** move to the next submatrix Kab */
      offset += w_skel.row();
    }
    else
    {
      printf( "Far Kab not cached treelist_id %lu, l %lu\n\n",
					node->treelist_id, node->l ); fflush( stdout );

      /** get submatrix Kad from K */
      beg = omp_get_wtime();
      auto Kab = K( amap, bmap );
      kij_s2s_time = omp_get_wtime() - beg;

      /** update kij counter */
      data.kij_s2s.first  += kij_s2s_time;
      data.kij_s2s.second += amap.size() * bmap.size();

      //printf( "%lu (%lu, %lu), ", (*it)->treelist_id, w_skel.row(), w_skel.num() );
      //fflush( stdout );
      xgemm
      (
        "N", "N",
        u_skel.row(), u_skel.col(), w_skel.row(),
        1.0, Kab.data(),       Kab.row(),
             w_skel.data(), w_skel.row(),
        1.0, u_skel.data(), u_skel.row()
      );
    }
  }
  s2s_time = omp_get_wtime() - beg;


  //if ( u_skel.HasIllegalValue() )
  //{
  //  printf( "Illegal value in u_skel\n" ); fflush( stdout );
  //}



  //printf( "u_skel %.3E s2s %.3E\n", u_skel_time, s2s_time );

}; // end void SkeletonsToSkeletons()



/**
 *  @brief There is no dependency between each task. However 
 *         there are raw (read after write) dependencies:
 *
 *         NodesToSkeletons (P*w)
 *         SkeletonsToSkeletons ( Sum( Kab * ))
 *
 *  @TODO  The flops and mops of constructing Kab.
 *
 */ 
template<bool NNPRUNE, typename NODE, typename T>
class SkeletonsToSkeletonsTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "s2s" );
      {
        //label = std::to_string( arg->treelist_id );
        ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      /** compute flops and mops */
      double flops = 0.0, mops = 0.0;
      auto &w = *arg->setup->w;
      size_t m = arg->data.skels.size();
      size_t n = w.col();

      std::set<NODE*> *FarNodes;
      if ( NNPRUNE ) FarNodes = &arg->NNFarNodes;
      else           FarNodes = &arg->FarNodes;

      for ( auto it = FarNodes->begin(); it != FarNodes->end(); it ++ )
      {
        size_t k = (*it)->data.skels.size();
        flops += 2.0 * m * n * k;
        mops  += m * k; // cost of Kab
        mops  += 2.0 * ( m * n + n * k + k * n );
      }

      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Assume computation bound */
      cost = flops / 1E+9;
      /** High priority */
      priority = true;
    };

    void GetEventRecord()
    {
      arg->data.s2s = event;
    };

    void DependencyAnalysis()
    {
      for ( auto p : arg->data.FarDependents )
        hmlp_msg_dependency_analysis( 306, p, R, this );

      set<NODE*> *FarNodes;
      FarNodes = &arg->NNFarNodes;
      for ( auto it : *FarNodes )
      {
        it->DependencyAnalysis( R, this );
      }
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      SkeletonsToSkeletons<NNPRUNE, NODE, T>( arg );
    };
}; // end class SkeletonsToSkeletonsTask


/**
 *  @brief This is a task in Downward traversal. There is data
 *         dependency on u_skel.
 *         
 */ 
template<bool NNPRUNE, typename NODE, typename T>
void SkeletonsToNodes( NODE *node )
{
#ifdef DEBUG_SPDASKIT
  printf( "%lu Skel2Node u_skel.row() %lu\n", 
      node->treelist_id, node->data.u_skel.row() ); 
  fflush( stdout );
#endif

  double beg, kij_s2n_time = 0.0, u_leaf_time, before_writeback_time, after_writeback_time;

  /** gather shared data and create reference */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;

  /** Gather per node data and create reference */
  auto &gids = node->gids;
  auto &data = node->data;
  auto &proj = data.proj;
  auto &skels = data.skels;
  auto &u_skel = data.u_skel;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  size_t nrhs = w.col();





  if ( node->isleaf )
  {
    /** Get U view of this node if initialized */
    View<T> U = data.u_view;

    if ( U.col() == nrhs )
    {
      //printf( "%8lu s2n U[%lu %lu %lu]\n", 
      //    node->morton, U.row(), U.col(), U.ld() ); fflush( stdout );
      xgemm
      (
        "Transpose", "Non-transpose",
        U.row(), U.col(), u_skel.row(),
        1.0,   proj.data(),   proj.row(),
             u_skel.data(), u_skel.row(),
        1.0,      U.data(),       U.ld()
      );
    }
    else
    {
      //printf( "%8lu use u_leaf u_view [%lu %lu ld %lu]\n", 
      //    node->morton, U.row(), U.col(), U.ld()  ); fflush( stdout );

      auto &u_leaf = node->data.u_leaf[ 0 ];

      /** zero-out u_leaf */
      u_leaf.resize( 0, 0 );
      u_leaf.resize( gids.size(), nrhs, 0.0 );

      /** accumulate far interactions */
      if ( data.isskel )
      {
        /** u_leaf += P' * u_skel */
        xgemm
        (
          "T", "N",
          u_leaf.row(), u_leaf.col(), u_skel.row(),
          1.0,   proj.data(),   proj.row(),
               u_skel.data(), u_skel.row(),
          1.0, u_leaf.data(), u_leaf.row()
        );
      }
    }
    after_writeback_time = omp_get_wtime() - beg;

    //printf( "u_leaf %.3E before %.3E after %.3E\n",
    //    u_leaf_time, before_writeback_time, after_writeback_time );
  }
  else
  {
    if ( !node->parent || !node->data.isskel ) return;

    auto &u_lskel = lchild->data.u_skel;
    auto &u_rskel = rchild->data.u_skel;
    auto &lskel = lchild->data.skels;
    auto &rskel = rchild->data.skels;

    //if ( 1 )
    if ( node->treelist_id > 6 )
    {
      //printf( "%8lu s2n\n", node->morton ); fflush( stdout );
      xgemm
      (
        "Transpose", "No transpose",
        u_lskel.row(), u_lskel.col(), proj.row(),
        1.0, proj.data(),    proj.row(),
             u_skel.data(),  u_skel.row(),
        1.0, u_lskel.data(), u_lskel.row()
      );
      xgemm
      (
        "Transpose", "No transpose",
        u_rskel.row(), u_rskel.col(), proj.row(),
        1.0, proj.data() + proj.row() * lskel.size(), proj.row(),
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
                      PR, lskel.size(), TOP );
      /** UL += PL' * U */
      gemm::xgemm<GEMM_NB>( (T)1.0, PL, U, (T)1.0, UL );
      /** UR += PR' * U */
      gemm::xgemm<GEMM_NB>( (T)1.0, PR, U, (T)1.0, UR );
    }
  }
  //printf( "\n" );

}; /** end SkeletonsToNodes() */


template<bool NNPRUNE, typename NODE, typename T>
class SkeletonsToNodesTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "s2n" );
      label = to_string( arg->treelist_id );

      //--------------------------------------
      double flops = 0.0, mops = 0.0;
      auto &gids = arg->gids;
      auto &data = arg->data;
      auto &proj = data.proj;
      auto &skels = data.skels;
      auto &w = *arg->setup->w;

      if ( arg->isleaf )
      {
        size_t m = proj.col();
        size_t n = w.col();
        size_t k = proj.row();
        flops += 2.0 * m * n * k;
        mops  += 2.0 * ( m * n + n * k + m * k );
      }
      else
      {
        if ( !arg->parent || !arg->data.isskel )
        {
          // No computation.
        }
        else
        {
          size_t m = proj.col();
          size_t n = w.col();
          size_t k = proj.row();
          flops += 2.0 * m * n * k;
          mops  += 2.0 * ( m * n + n * k + m * k );
        }
      }

      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Asuume computation bound */
      cost = flops / 1E+9;
      /** "HIGH" priority (critical path) */
      priority = true;
    };

    void Prefetch( Worker* user_worker )
    {
      auto &proj = arg->data.proj;
      __builtin_prefetch( proj.data() );
      auto &u_skel = arg->data.u_skel;
      __builtin_prefetch( u_skel.data() );
      if ( arg->isleaf )
      {
        //__builtin_prefetch( arg->data.u_leaf[ 0 ].data() );
        //__builtin_prefetch( arg->data.u_leaf[ 1 ].data() );
        //__builtin_prefetch( arg->data.u_leaf[ 2 ].data() );
        //__builtin_prefetch( arg->data.u_leaf[ 3 ].data() );
      }
      else
      {
        auto &u_lskel = arg->lchild->data.u_skel;
        __builtin_prefetch( u_lskel.data() );
        auto &u_rskel = arg->rchild->data.u_skel;
        __builtin_prefetch( u_rskel.data() );
      }
#ifdef HMLP_USE_CUDA
      hmlp::Device *device = NULL;
      if ( user_worker ) device = user_worker->GetDevice();
      if ( device )
      {
        int stream_id = arg->treelist_id % 8;
        proj.CacheD( device );
        proj.PrefetchH2D( device, stream_id );
        u_skel.CacheD( device );
        u_skel.PrefetchH2D( device, stream_id );
        if ( arg->isleaf )
        {
        }
        else
        {
          auto &u_lskel = arg->lchild->data.u_skel;
          u_lskel.CacheD( device );
          u_lskel.PrefetchH2D( device, stream_id );
          auto &u_rskel = arg->rchild->data.u_skel;
          u_rskel.CacheD( device );
          u_rskel.PrefetchH2D( device, stream_id );
        }
      }
#endif
    };

    void GetEventRecord()
    {
      arg->data.s2n = event;
    };

    void DependencyAnalysis()
    {
      if ( arg->parent )
      {
        arg->DependencyAnalysis( R, this );
      }
      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( RW, this );
        arg->rchild->DependencyAnalysis( RW, this );
      }
      else
      {
        arg->DependencyAnalysis( W, this );
      }
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
#ifdef HMLP_USE_CUDA 
      Device *device = NULL;
      if ( user_worker ) device = user_worker->GetDevice();
      if ( device ) gpu::SkeletonsToNodes<NNPRUNE, NODE, T>( device, arg );
      else               SkeletonsToNodes<NNPRUNE, NODE, T>( arg );
#else
    SkeletonsToNodes<NNPRUNE, NODE, T>( arg );
#endif
    };

}; // end class SkeletonsToNodesTask



template<int SUBTASKID, bool NNPRUNE, typename NODE, typename T>
void LeavesToLeaves( NODE *node, size_t itbeg, size_t itend )
{
  assert( node->isleaf );

  double beg, kij_s2n_time = 0.0, u_leaf_time, before_writeback_time, after_writeback_time;

  /** gather shared data and create reference */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;

  auto &gids = node->gids;
  auto &data = node->data;
  auto &amap = node->gids;
  auto &NearKab = data.NearKab;

  size_t nrhs = w.col();

  set<NODE*> *NearNodes;
  if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
  else           NearNodes = &node->NearNodes;

  /** TODO: I think there may be a performance bug here. 
   *        Overall there will be 4 task
   **/
  auto &u_leaf = data.u_leaf[ SUBTASKID ];
  u_leaf.resize( 0, 0 );

  /** early return if nothing to do */
  if ( itbeg == itend ) 
  {
    return;
  }
  else
  {
    u_leaf.resize( gids.size(), nrhs, 0.0 );
  }

  if ( NearKab.size() ) /** Kab is cached */
  {
    size_t itptr = 0;
    size_t offset = 0;

    for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
    {
      if ( itbeg <= itptr && itptr < itend )
      {
        //auto wb = w( bmap );
        auto wb = (*it)->data.w_leaf;

        if ( wb.size() )
        {
          /** Kab * wb */
          xgemm
          (
            "N", "N",
            u_leaf.row(), u_leaf.col(), wb.row(),
            1.0, NearKab.data() + offset * NearKab.row(), NearKab.row(),
                      wb.data(),                               wb.row(),
            1.0,  u_leaf.data(),                           u_leaf.row()
          );
        }
        else
        {
          View<T> W = (*it)->data.w_view;
          xgemm
          (
            "N", "N",
            u_leaf.row(), u_leaf.col(), W.row(),
            1.0, NearKab.data() + offset * NearKab.row(), NearKab.row(),
                       W.data(),                                 W.ld(),
            1.0,  u_leaf.data(),                           u_leaf.row()
          );
        }
      }
      offset += (*it)->gids.size();
      itptr ++;
    }
  }
  else /** TODO: make xgemm into NN instead of NT. Kab is not cached */
  {
    size_t itptr = 0;
    for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
    {
      if ( itbeg <= itptr && itptr < itend )
      {
        auto &bmap = (*it)->gids;
        auto wb = (*it)->data.w_leaf;

        /** evaluate the submatrix */
        beg = omp_get_wtime();
        auto Kab = K( amap, bmap );
        kij_s2n_time = omp_get_wtime() - beg;

        /** update kij counter */
        data.kij_s2n.first  += kij_s2n_time;
        data.kij_s2n.second += amap.size() * bmap.size();

        if ( wb.size() )
        {
          /** ( Kab * wb )' = wb' * Kab' */
          xgemm
          (
            "N", "N",
            u_leaf.row(), u_leaf.col(), wb.row(),
            1.0,    Kab.data(),    Kab.row(),
                     wb.data(),     wb.row(),
            1.0, u_leaf.data(), u_leaf.row()
          );
        }
        else
        {
          View<T> W = (*it)->data.w_view;
          xgemm
          (
            "N", "N",
            u_leaf.row(), u_leaf.col(), W.row(),
            1.0,    Kab.data(),    Kab.row(),
                      W.data(),       W.ld(),
            1.0, u_leaf.data(), u_leaf.row()
          );
        }
      }
      itptr ++;
    }
  }
  before_writeback_time = omp_get_wtime() - beg;

}; /** end LeavesToLeaves() */


template<int SUBTASKID, bool NNPRUNE, typename NODE, typename T>
class LeavesToLeavesTask : public Task
{
  public:

    NODE *arg = NULL;

    size_t itbeg;

	size_t itend;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = std::string( "l2l" );
      {
        //label = std::to_string( arg->treelist_id );
        std::ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      /** TODO: fill in flops and mops */
      //--------------------------------------
      double flops = 0.0, mops = 0.0;
      auto &gids = arg->gids;
      auto &data = arg->data;
      auto &proj = data.proj;
      auto &skels = data.skels;
      auto &w = *arg->setup->w;
      auto &K = *arg->setup->K;
      auto &NearKab = data.NearKab;

      assert( arg->isleaf );

      size_t m = gids.size();
      size_t n = w.col();

      std::set<NODE*> *NearNodes;
      if ( NNPRUNE ) NearNodes = &arg->NNNearNodes;
      else           NearNodes = &arg->NearNodes;

      /** TODO: need to better decide the range [itbeg itend] */ 
      size_t itptr = 0;
      size_t itrange = ( NearNodes->size() + 3 ) / 4;
      if ( itrange < 1 ) itrange = 1;
      itbeg = ( SUBTASKID - 1 ) * itrange;
      itend = ( SUBTASKID + 0 ) * itrange;
      if ( itbeg > NearNodes->size() ) itbeg = NearNodes->size();
      if ( itend > NearNodes->size() ) itend = NearNodes->size();
      if ( SUBTASKID == 4 ) itend = NearNodes->size();

      for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
      {
        if ( itbeg <= itptr && itptr < itend )
        {
          size_t k = (*it)->gids.size();
          flops += 2.0 * m * n * k;
          mops += m * k;
          mops += 2.0 * ( m * n + n * k + m * k );
          /** the cost of Kab */
          if ( !NearKab.size() ) flops += K.flops( m, k );
        }
        itptr ++;
      }

      /** setup the event */
      event.Set( label + name, flops, mops );

      /** asuume computation bound */
      cost = flops / 1E+9;
    };

    void Prefetch( Worker* user_worker )
    {
      auto &u_leaf = arg->data.u_leaf[ SUBTASKID ];
      __builtin_prefetch( u_leaf.data() );
    };

    void GetEventRecord()
    {
      /** create l2l event */
      //arg->data.s2n = event;
    };

    void DependencyAnalysis()
    {
      assert( arg->isleaf );
      /** depends on nothing */
      this->TryEnqueue();

      /** impose rw dependencies on multiple copies */
      //auto &u_leaf = arg->data.u_leaf[ SUBTASKID ];
      //u_leaf.DependencyAnalysis( hmlp::ReadWriteType::W, this );
    };

    void Execute( Worker* user_worker )
    {
      LeavesToLeaves<SUBTASKID, NNPRUNE, NODE, T>( arg, itbeg, itend );
    };

}; /** end class LeavesToLeaves */




template<typename NODE>
void PrintSet( set<NODE*> &set )
{
  for ( auto it = set.begin(); it != set.end(); it ++ )
  {
    printf( "%lu, ", (*it)->treelist_id );
  }
  printf( "\n" );
}; /** end PrintSet() */


//template<typename NODE>
//void RemoveClique( NODE *node )
//{
//  assert( node->NNNearNodes.count( node ) );
//  node->NNNearNodes.erase( node );
//};
//
//template<typename NODE>
//void RemoveClique( NODE *node1, NODE *node2 )
//{
//  if ( node1->isleaf )
//  {
//    if ( node2->isleaf )
//    {
//      node1->NNNearNodes.erase( node2 );
//    }
//    else
//    {
//      RemoveClique( node1, node2->lchild );
//      RemoveClique( node1, node2->rchild );
//    }
//  }
//  else
//  {
//    RemoveClique( node1->lchild, node2 );
//    RemoveClique( node1->rchild, node2 );
//  }
//};
//
//
//template<typename NODE>
//bool NearNodeClique( NODE *node1, NODE *node2 )
//{
//  bool isclique = false;
//
//  if ( node1->isleaf )
//  {
//    if ( node2->isleaf )
//    {
//      isclique = node1->NNNearNodes.count( node2 );
//    }
//    else
//    {
//      isclique = 
//        NearNodeClique( node1, node2->lchild ) &&
//        NearNodeClique( node1, node2->rchild );
//    }
//  }
//  else
//  {
//    isclique = 
//      NearNodeClique( node1->lchild, node2 ) &&
//      NearNodeClique( node1->rchild, node2 );
//  }
//
//  return isclique;
//};
//
//template<typename NODE>
//bool NearNodeClique( NODE *node )
//{
//  if ( node->isleaf )
//  {
//    return true;
//  }
//  else
//  {
//    bool ll = NearNodeClique( node->lchild );
//    bool rr = NearNodeClique( node->rchild );
//    bool lr = NearNodeClique( node->lchild, node->rchild );
//    bool rl = NearNodeClique( node->rchild, node->lchild );
//
//    if ( ll && rr && lr && rl )
//    {
//      printf( "clique at level %lu\n", node->l );
//    }
//
//    return ( ll && rr && lr && rl );
//  }
//};
//
//template<typename TREE>
//void FindClique( TREE &tree )
//{
//
//};



/**
 *
 */
template<typename NODE>
multimap<size_t, size_t> NearNodeBallots( NODE *node )
{
  /** Must be a leaf node. */
  assert( node->isleaf );

  auto &setup = *(node->setup);
  auto &NN = *(setup.NN);
  auto &gids = node->gids;

  /** Ballot table ( node MortonID, ids ) */
  //map<size_t, map<size_t, T>> &candidates = node->data.candidates;
  auto &candidates = node->data.candidates;
  map<size_t, size_t> ballot;

  size_t HasMissingNeighbors = 0;


  /** Loop over all neighbors and insert them into tables. */ 
  for ( size_t j = 0; j < gids.size(); j ++ )
  {
    for ( size_t i = 0; i < NN.row(); i ++ )
    {
      auto value = NN( i, gids[ j ] ).first;
      size_t neighbor_gid = NN( i, gids[ j ] ).second;
      /** If this gid is valid, then compute its morton */
      if ( neighbor_gid >= 0 && neighbor_gid < NN.col() )
      {
        size_t neighbor_morton = setup.morton[ neighbor_gid ];
        size_t weighted_ballot = 1.0 / ( value + 1E-3 );
        //printf( "gid %lu i %lu neighbor_gid %lu morton %lu\n", gids[ j ], i, 
        //    neighbor_gid, neighbor_morton );

        if (  i < NN.row() / 2 )
        {
          if ( ballot.find( neighbor_morton ) != ballot.end() )
          {
            //ballot[ neighbor_morton ] ++;
            ballot[ neighbor_morton ] += weighted_ballot;
          }
          else
          {
            //ballot[ neighbor_morton ] = 1;
            ballot[ neighbor_morton ] = weighted_ballot;
          }
        }
        else
        {
          if ( candidates.find( neighbor_morton ) != candidates.end() )
          {
            auto &tar = candidates[ neighbor_morton ];
            auto ret = tar.insert( make_pair( neighbor_gid, value ) );

            /** if already existed, then update distance */
            if ( ret.second == false )
            {
              if ( ret.first->second > value ) ret.first->second = value;
            }
          }
          else
          {
            candidates[ neighbor_morton ][ neighbor_gid ] = value;
          }
        }
      }
      else
      {
        HasMissingNeighbors ++;
      }
    }
  }

  if ( HasMissingNeighbors )
  {
    printf( "Missing %lu neighbor pairs\n", HasMissingNeighbors ); 
    fflush( stdout );
  }

  /** Flip ballot to create sorted_ballot. */ 
  return flip_map( ballot );

}; /** end NearNodeBallots() */




template<typename NODE, typename T>
void NearSamples( NODE *node )
{
  auto &setup = *(node->setup);
  auto &NN = *(setup.NN);

  if ( node->isleaf )
  {
    auto &gids = node->gids;
    double budget = setup.budget;
    size_t n_nodes = ( 1 << node->l );

    /** Add myself to the near interaction list.  */
    node->NearNodes.insert( node );
    node->NNNearNodes.insert( node );
    node->NNNearNodeMortonIDs.insert( node->morton );

    /** Compute ballots for all near interactions */
    multimap<size_t, size_t> sorted_ballot = NearNodeBallots( node );

    /** Insert near node cadidates until reaching the budget limit. */ 
    for ( auto it = sorted_ballot.rbegin(); it != sorted_ballot.rend(); it ++ )
    {
      /** Exit if we have enough. */ 
      if ( node->NNNearNodes.size() >= n_nodes * budget ) break;

      /**
       *  Get the node pointer from MortonID. 
       *
       *  Two situations:
       *  1. the pointer doesn't exist, then creates a lettreenode
       */ 
      /** Acquire the global lock to modify morton2node map. */
      //#pragma omp critical
      //{
      //  node->treelock->Acquire();
      //  {
      //    /** 
      //     *  Double check that the let node does not exist.
      //     *  New tree::Node with MortonID (*it).second 
      //     */
      //    if ( !(*node->morton2node).count( (*it).second ) )
      //    {
      //      (*node->morton2node)[ (*it).second ] = new NODE( (*it).second );
      //    }
          /** Insert */
          auto *target = (*node->morton2node)[ (*it).second ];
          node->NNNearNodeMortonIDs.insert( (*it).second );
          node->NNNearNodes.insert( target );
      //  }
      //  node->treelock->Release();
      //}
    }
    //printf( "%3lu, gids %lu candidates %2lu/%2lu, ballot %2lu budget %lf NNNearNodes %lu k %lu\n", 
    //    node->treelist_id, gids.size(), 
    //    candidates.size(), n_nodes, ballot.size(), budget,
    //    node->NNNearNodes.size(), NN.row() ); fflush( stdout );


    /**
     *  Symmetrify 
     */ 
    for ( auto it = node->NNNearNodes.begin(); it != node->NNNearNodes.end(); it ++ )
    {
      (*it)->data.lock.Acquire();
      {
        (*it)->ProposedNNNearNodes.insert( node );
      }
      (*it)->data.lock.Release();
    }

  }
  else
  {
    //printf( "%3lu pool.size() %lu\n", node->treelist_id, node->data.pool.size() );
    /**  
     *  Merge ballot
     */
  }

}; /** void NearSamples() */



template<typename NODE, typename T>
class NearSamplesTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
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

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( R, this );
        arg->rchild->DependencyAnalysis( R, this );
      }
      if ( !arg->isleaf && arg->sibling )
      {
        arg->sibling->lchild->DependencyAnalysis( R, this );
        arg->sibling->rchild->DependencyAnalysis( R, this );
      }
      /** Try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      NearSamples<NODE, T>( arg );
    };

}; /** end class NearSamplesTask */





template<typename NODE, typename T>
void BuildPool( NODE *node )
{
  auto & setup = *(node->setup);
  auto & candidates = node->data.candidates;

  if ( node->isleaf )
  {
    /**
     *  Merge NNNearNodes with ProposedNNNearNodes.
     */
    for ( auto it  = node->ProposedNNNearNodes.begin();
               it != node->ProposedNNNearNodes.end(); it ++ )
    {
      node->NNNearNodes.insert( *it );
      node->NNNearNodeMortonIDs.insert( (*it)->morton );
    }
  }
  else
  {
    auto *lchild = node->lchild;
    auto *rchild = node->rchild;

    /**
     *  For non-leaf nodes, we build the near interaction list to filter
     *  the row samples. Rows in the near iteraction list will be sampled.
     */ 
    for ( auto it  = lchild->NNNearNodeMortonIDs.begin(); 
               it != lchild->NNNearNodeMortonIDs.end(); it ++ )
    {
      if ( !tree::IsMyParent( *it, node->morton ) )
      {
        node->NNNearNodeMortonIDs.insert( *it );
      }
    }

    for ( auto it  = rchild->NNNearNodeMortonIDs.begin(); 
               it != rchild->NNNearNodeMortonIDs.end(); it ++ )
    {
      if ( !tree::IsMyParent( *it, node->morton ) )
      {
        node->NNNearNodeMortonIDs.insert( *it );
      }
    }

    /** 
     *  Merge left and right candidates 
     */
    candidates = lchild->data.candidates;
    for ( auto it  = rchild->data.candidates.begin(); 
               it != rchild->data.candidates.end(); it ++ )
    {
      if ( candidates.find( (*it).first ) != candidates.end() )
      {
        /** Merge two maps */
        auto & existing_candidate = candidates[ (*it).first ];

        /** Loop over queries */
        for ( auto query_it  = (*it).second.begin();
                   query_it != (*it).second.end(); query_it ++ )
        {
          auto ret = existing_candidate.insert( *query_it );
          if ( ret.second == false )
          {
            if ( ret.first->second > query_it->second ) 
              ret.first->second = query_it->second;
          }
        }
      }
      else
      {
        candidates.insert( *it );
      }
    }

    lchild->data.candidates.clear();
    rchild->data.candidates.clear();
  }

  /**
   *  Remove near interactions from the ballot table. The key value of 
   *  candidates is morton.
   */ 
  for ( auto it  = node->NNNearNodeMortonIDs.begin(); 
             it != node->NNNearNodeMortonIDs.end(); it ++ )
  {
    candidates.erase( *it );
  }








  /**
   *  Building the sample pool
   */ 
  auto *sibling = node->sibling;
  if ( sibling ) 
  {
    if ( node->treelist_id == 0 ) 
      printf( "Construct sibling's sample pool for local root\n" ); fflush( stdout );

    //vector<pair<size_t, T>> merged_candidates;
    vector<pair<T,size_t>> merged_candidates;

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
    //if ( merged_candidates.size() > nsamples )
    //  merged_candidates.resize( nsamples );

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

      //if ( pool.size() >= nsamples ) break;
    }
    //printf( "%3lu pool.size() %lu\n", sibling->treelist_id, pool.size() );
  }


}; /** end BuildPool() */



template<typename NODE, typename T>
class BuildPoolTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
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

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( R, this );
        arg->rchild->DependencyAnalysis( R, this );
      }
      /** Try to enqueue if there is no dependency */
      this->TryEnqueue();

    };

    void Execute( Worker* user_worker )
    {
      BuildPool<NODE, T>( arg );
    };

}; /** end class NearSamplesTask */






















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
  auto level_beg = tree.treelist.begin() + n_nodes - 1;

  //printf( "NN( %lu, %lu ) depth %lu n_nodes %lu treelist.size() %lu\n", 
  //    NN.row(), NN.col(),      
  //    tree.depth, n_nodes, tree.treelist.size() );


  /** 
   * traverse all leaf nodes. 
   *
   * TODO: the following omp parallel for will result in segfault on KNL.
   *       Not sure why.
   **/
  //#pragma omp parallel for
  for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
  {
    auto *node = *(level_beg + node_ind);
    auto &data = node->data;
    auto &gids = node->gids;

    /** if no skeletons, then add every leaf nodes */
    /** TODO: this is not affected by the BUDGET */
    if ( !node->data.isskel )
    {
      for ( size_t i = 0; i < n_nodes; i ++ )
      {
        node->NearNodes.insert(   *(level_beg + i) );
        node->NNNearNodes.insert( *(level_beg + i) );
      }
    }
    
    /** add myself to the list. */
    node->NearNodes.insert( node );
    node->NNNearNodes.insert( node );
 
    /** ballot table */
    std::vector<std::pair<size_t, size_t>> ballot( n_nodes );
    for ( size_t i = 0; i < n_nodes; i ++ )
    {
      ballot[ i ].first  = 0;
      ballot[ i ].second = i;
    }

    //printf( "before Neighbor\n" ); fflush( stdout );

    /** traverse all points and their neighbors. NN is stored as k-by-N */
    for ( size_t j = 0; j < gids.size(); j ++ )
    {
      size_t gid = gids[ j ];
      /** use the second half */
      //for ( size_t i = NN.row() / 2; i < NN.row(); i ++ )
      for ( size_t i = 0; i < NN.row() / 2; i ++ )
      {
        size_t neighbor_gid = NN( i, gid ).second;

        /** if this gid is valid, then compute its morton */
        if ( neighbor_gid >= 0 && neighbor_gid < NN.col() )
        {
          //printf( "lid %lu i %lu neighbor_gid %lu\n", lid, i, neighbor_gid );
          size_t neighbor_morton = setup.morton[ neighbor_gid ];
          //printf( "neighborlid %lu morton %lu\n", neighbor_lid, neighbor_morton );
          //auto *target = tree.Morton2Node( neighbor_morton );
          auto *target = (*node->morton2node)[ neighbor_morton ];
          ballot[ target->treelist_id - ( n_nodes - 1 ) ].first += 1;
        }
        else
        {
          printf( "illegal gid in neighbor pairs\n" ); fflush( stdout );
        }
      }
    }

    //printf( "after Neighbor\n" ); fflush( stdout );

    /** sort the ballot list */
    struct 
    {
      bool operator () ( std::pair<size_t, size_t> a, std::pair<size_t, size_t> b )
      {   
        return a.first > b.first;
      }   
    } BallotMore;

    //printf( "before sorting\n" ); fflush( stdout );
    std::sort( ballot.begin(), ballot.end(), BallotMore );
    //printf( "after sorting\n" ); fflush( stdout );

    /** add leaf nodes with the highest votes util reach the budget */
    for ( size_t i = 0; i < n_nodes; i ++ )
    {
      if ( ballot[ i ].first && node->NNNearNodes.size() < n_nodes * budget )
      {
        //printf( "add %lu ", ballot[ i ].second + ( n_nodes - 1 ) ); fflush( stdout );
        node->NNNearNodes.insert( tree.treelist[ ballot[ i ].second + ( n_nodes - 1 ) ] );
      }
      //printf( "\n" );
    }

    //printf( "After ballot\n" ); fflush( stdout );
  }

  /** symmetrinize Near( node ) */
  if ( SYMMETRIC )
  {
    /** make Near( node ) symmetric */
    for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
    {
      auto *node = *(level_beg + node_ind);
      auto &NNNearNodes = node->NNNearNodes;
      for ( auto it = NNNearNodes.begin(); it != NNNearNodes.end(); it ++ )
      {
        (*it)->NNNearNodes.insert( node );
      }
    }
#ifdef DEBUG_SPDASKIT
    for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
    {
      auto *node = *(level_beg + node_ind);
      auto &NNNearNodes = node->NNNearNodes;
      printf( "Node %lu NearNodes ", node->treelist_id );
      for ( auto it = NNNearNodes.begin(); it != NNNearNodes.end(); it ++ )
      {
        printf( "%lu, ", (*it)->treelist_id );
      }
      printf( "\n" );
    }
#endif
  }

}; /** end FindNearNodes() */



/**
 *  @brief Task wrapper for FindNearNodes()
 */ 
template<bool SYMMETRIC, typename TREE>
class NearNodesTask : public Task
{
  public:

    TREE *arg = NULL;

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
    };

    void Execute( Worker* user_worker )
    {
      //printf( "FindNearNode beg\n" ); fflush( stdout );
      FindNearNodes<SYMMETRIC, TREE>( *arg, budget );
      //printf( "FindNearNode end\n" ); fflush( stdout );
    };

}; /** end class NearNodesTask */


/**
 *  @brief Task wrapper for CacheNearNodes().
 */
template<bool NNPRUNE, typename NODE>
class CacheNearNodesTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = std::string( "c-n" );
      {
        //label = std::to_string( arg->treelist_id );
        std::ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      /** asuume computation bound */
      cost = 1.0;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;

      NODE *node = arg;
      auto *NearNodes = &node->NearNodes;
      if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
      auto &K = *node->setup->K;

      size_t m = node->gids.size();
      size_t n = 0;
      for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
      {
        n += (*it)->gids.size();
      }

      /** Kab */
      flops += K.flops( m, n );

      /** setup the event */
      event.Set( label + name, flops, mops );
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
    };

    void Execute( Worker* user_worker )
    {
      //printf( "%lu CacheNearNodes beg\n", arg->treelist_id ); fflush( stdout );

      NODE *node = arg;
      auto *NearNodes = &node->NearNodes;
      if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
      auto &K = *node->setup->K;
      auto &data = node->data;
      auto &amap = node->gids;
      std::vector<size_t> bmap;
      for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
      {
        bmap.insert( bmap.end(), (*it)->gids.begin(), (*it)->gids.end() );
      }
      data.NearKab = K( amap, bmap );

      /** */
      data.Nearbmap.resize( bmap.size(), 1 );
      for ( size_t i = 0; i < bmap.size(); i ++ ) 
        data.Nearbmap[ i ] = bmap[ i ];

#ifdef HMLP_USE_CUDA
      auto *device = hmlp_get_device( 0 );
      /** prefetch Nearbmap to GPU */
      node->data.Nearbmap.PrefetchH2D( device, 8 );

      size_t preserve_size = 3000000000;
      //if ( data.NearKab.col() * MAX_NRHS < 1200000000 &&
      //     data.NearKab.size() * 8 + preserve_size < device->get_memory_left() &&
      //     data.NearKab.size() * 8 > 4096 * 4096 * 8 * 4 )
      if ( data.NearKab.col() * MAX_NRHS < 1200000000 &&
           data.NearKab.size() * 8 + preserve_size < device->get_memory_left() )
      {
        /** prefetch NearKab to GPU */
        data.NearKab.PrefetchH2D( device, 8 );
      }
      else
      {
        printf( "Kab %lu %lu not cache\n", data.NearKab.row(), data.NearKab.col() );
      }
#endif

      //printf( "%lu CacheNearNodesTask end\n", arg->treelist_id ); fflush( stdout );
    };
}; /** end class CacheNearNodesTask */



/**
 *  @brief (FMM specific) find Far( target ) by traversing all treenodes 
 *         top-down. 
 *         If the visiting ``node'' does not contain any near node
 *         of ``target'' (by MORTON helper function ContainAny() ),
 *         then we add ``node'' to Far( target ).
 *
 *         Otherwise, recurse to two children.
 */ 
template<bool SYMMETRIC, typename NODE>
void FindFarNodes( NODE *node, NODE *target )
{
  /** all assertions, ``target'' must be a leaf node */
  assert( target->isleaf );

  /** get a list of near nodes from target */
  std::set<NODE*> *NearNodes;
  auto &data = node->data;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  /**
   *  case: !NNPRUNE
   *
   *  Build NearNodes for pure hierarchical low-rank approximation.
   *  In this case, Near( target ) only contains target itself.
   *
   **/
  NearNodes = &target->NearNodes;

  /** if this node contains any Near( target ) or isn't skeletonized */
  if ( !data.isskel || node->ContainAny( *NearNodes ) )
  {
    if ( !node->isleaf )
    {
      /** recurse to two children */
      FindFarNodes<SYMMETRIC>( lchild, target );
      FindFarNodes<SYMMETRIC>( rchild, target );
    }
  }
  else
  {
    /** insert ``node'' to Far( target ) */
    target->FarNodes.insert( node );
  }

  /**
   *  case: NNPRUNE
   *
   *  Build NNNearNodes for the FMM approximation.
   *  Near( target ) contains other leaf nodes
   *  
   **/
  NearNodes = &target->NNNearNodes;

  /** if this node contains any Near( target ) or isn't skeletonized */
  if ( !data.isskel || node->ContainAny( *NearNodes ) )
  {
    if ( !node->isleaf )
    {
      /** recurse to two children */
      FindFarNodes<SYMMETRIC>( lchild, target );
      FindFarNodes<SYMMETRIC>( rchild, target );
    }
  }
  else
  {
    if ( SYMMETRIC && ( node->morton < target->morton ) )
    {
      /** since target->morton is larger than the visiting node,
       * the interaction between the target and this node has
       * been computed. 
       */ 
    }
    else
    {
      target->NNFarNodes.insert( node );
    }
  }

}; /** end FindFarNodes() */



/**
 *  @brief (FMM specific) perform an bottom-up traversal to build 
 *         Far( node ) for each node. Leaf nodes call
 *         FindFarNodes(), and inner nodes will merge two Far lists
 *         from lchild and rchild.
 *
 *  @TODO  change to task.
 *
 */
template<bool SYMMETRIC, typename TREE>
void MergeFarNodes( TREE &tree )
{
  for ( int l = tree.depth; l >= 0; l -- )
  {
    std::size_t n_nodes = ( 1 << l );
    auto level_beg = tree.treelist.begin() + n_nodes - 1;

    for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
    {
      auto *node = *(level_beg + node_ind);

      /** if I don't have any skeleton, then I'm nobody's far field */
      if ( !node->data.isskel ) continue;

      if ( node->isleaf )
      {
        FindFarNodes<SYMMETRIC>( tree.treelist[ 0 ] /** root */, node );
      }
      else
      {
        /** merge Far( lchild ) and Far( rchild ) from children */
        auto *lchild = node->lchild;
        auto *rchild = node->rchild;

        /**
         *  case: !NNPRUNE (HSS specific)
         */ 
        auto &pFarNodes =   node->FarNodes;
        auto &lFarNodes = lchild->FarNodes;
        auto &rFarNodes = rchild->FarNodes;
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



        /**
         *  case: NNPRUNE (FMM specific)
         */ 
        auto &pNNFarNodes =   node->NNFarNodes;
        auto &lNNFarNodes = lchild->NNFarNodes;
        auto &rNNFarNodes = rchild->NNFarNodes;

        //printf( "node %lu\n", node->treelist_id );
        //PrintSet( pNNFarNodes );
        //PrintSet( lNNFarNodes );
        //PrintSet( rNNFarNodes );


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

        //PrintSet( pNNFarNodes );
        //PrintSet( lNNFarNodes );
        //PrintSet( rNNFarNodes );
      }
    }
  }

  if ( SYMMETRIC )
  {
    /** symmetrinize FarNodes to FarNodes interaction */
    for ( int l = tree.depth; l >= 0; l -- )
    {
      std::size_t n_nodes = 1 << l;
      auto level_beg = tree.treelist.begin() + n_nodes - 1;

      for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
      {
        auto *node = *(level_beg + node_ind);
        auto &pFarNodes = node->NNFarNodes;
        for ( auto it = pFarNodes.begin(); it != pFarNodes.end(); it ++ )
        {
          (*it)->NNFarNodes.insert( node );
        }
      }
    }
  }
  
#ifdef DEBUG_SPDASKIT
  for ( int l = tree.depth; l >= 0; l -- )
  {
    std::size_t n_nodes = 1 << l;
    auto level_beg = tree.treelist.begin() + n_nodes - 1;

    for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
    {
      auto *node = *(level_beg + node_ind);
      auto &pFarNodes =   node->NNFarNodes;
      for ( auto it = pFarNodes.begin(); it != pFarNodes.end(); it ++ )
      {
        if ( !( (*it)->NNFarNodes.count( node ) ) )
        {
          printf( "Unsymmetric FarNodes %lu, %lu\n", node->treelist_id, (*it)->treelist_id );
          printf( "Near\n" );
          PrintSet(  node->NNNearNodes );
          PrintSet( (*it)->NNNearNodes );
          printf( "Far\n" );
          PrintSet(  node->NNFarNodes );
          PrintSet( (*it)->NNFarNodes );
          printf( "======\n" );
          break;
        }
      }
      if ( pFarNodes.size() )
      {
        printf( "l %2lu FarNodes(%lu) ", node->l, node->treelist_id );
        PrintSet( pFarNodes );
      }
    }
  }
#endif
};


/**
 *  @brief Evaluate and store all submatrices Kba used in the Far 
 *         interaction.
 *
 *  @TODO  Take care the HSS case i.e. (!NNPRUNE)
 *        
 */ 
template<bool NNPRUNE, bool CACHE = true, typename TREE>
void CacheFarNodes( TREE &tree )
{
  /** reserve space for w_leaf and u_leaf */
  #pragma omp parallel for schedule( dynamic )
  for ( size_t i = 0; i < tree.treelist.size(); i ++ )
  {
    auto *node = tree.treelist[ i ];
    if ( node->isleaf )
    {
      node->data.w_leaf.reserve( node->gids.size(), MAX_NRHS );
      node->data.u_leaf[ 0 ].reserve( MAX_NRHS, node->gids.size() );
    }
  }

  /** cache Kab by request */
  if ( CACHE )
  {
    /** cache FarKab */
    #pragma omp parallel for schedule( dynamic )
    for ( size_t i = 0; i < tree.treelist.size(); i ++ )
    {
      auto *node = tree.treelist[ i ];
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
    }
  }
}; /** end CacheFarNodes() */


/**
 *  @brief 
 */ 
template<bool NNPRUNE, typename TREE>
double DrawInteraction( TREE &tree )
{
  double exact_ratio = 0.0;
  FILE * pFile;
  //int n;
  char name[ 100 ];

  pFile = fopen ( "interaction.m", "w" );

  fprintf( pFile, "figure('Position',[100,100,800,800]);" );
  fprintf( pFile, "hold on;" );
  fprintf( pFile, "axis square;" );
  fprintf( pFile, "axis ij;" );

  for ( int l = tree.depth; l >= 0; l -- )
  {
    std::size_t n_nodes = 1 << l;
    auto level_beg = tree.treelist.begin() + n_nodes - 1;

    for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
    {
      auto *node = *(level_beg + node_ind);

      if ( NNPRUNE )
      {
        auto &pNearNodes = node->NNNearNodes;
        auto &pFarNodes = node->NNFarNodes;
        for ( auto it = pFarNodes.begin(); it != pFarNodes.end(); it ++ )
        {
          double gb = (double)std::min( node->l, (*it)->l ) / tree.depth;
          //printf( "node->l %lu (*it)->l %lu depth %lu\n", node->l, (*it)->l, tree.depth );
          fprintf( pFile, "rectangle('position',[%lu %lu %lu %lu],'facecolor',[1.0,%lf,%lf]);\n",
              node->offset,      (*it)->offset,
              node->gids.size(), (*it)->gids.size(),
              gb, gb );
        }
        for ( auto it = pNearNodes.begin(); it != pNearNodes.end(); it ++ )
        {
          fprintf( pFile, "rectangle('position',[%lu %lu %lu %lu],'facecolor',[0.2,0.4,1.0]);\n",
              node->offset,      (*it)->offset,
              node->gids.size(), (*it)->gids.size() );

          /** accumulate exact evaluation */
          exact_ratio += node->gids.size() * (*it)->gids.size();
        }  
      }
      else
      {
      }
    }
  }
  fprintf( pFile, "hold off;" );
  fclose( pFile );

  return exact_ratio / ( tree.n * tree.n );
}; /** end DrawInteration() */


/**
 *  @breif This is a fake evaluation setup aimming to figure out
 *         which tree node will prun which points. The result
 *         will be stored in each node as two lists, prune and noprune.
 *
 */ 
template<bool SYMBOLIC, bool NNPRUNE, typename NODE, typename T>
void Evaluate
( 
  NODE *node, 
  size_t gid, 
  std::vector<size_t> &nnandi, // k + 1 non-prunable lists
  hmlp::Data<T> &potentials 
)
{
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;
  auto &gids = node->gids;
  auto &data = node->data;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  size_t nrhs = w.col();

  auto amap = std::vector<size_t>( 1 );
  amap[ 0 ] = gid;

  if ( !SYMBOLIC ) // No potential evaluation.
  {
    assert( potentials.size() == amap.size() * nrhs );
  }

  if ( !data.isskel || node->ContainAny( nnandi ) )
  {
    if ( node->isleaf )
    {
      if ( SYMBOLIC )
      {
        /** add gid to notprune list. We use a lock */
        data.lock.Acquire();
        {
          if ( NNPRUNE ) node->NNNearIDs.insert( gid );
          else           node->NearIDs.insert(   gid );
        }
        data.lock.Release();
      }
      else
      {
#ifdef DEBUG_SPDASKIT
        printf( "level %lu direct evaluation\n", node->l );
#endif
        /** amap.size()-by-gids.size() */
        auto Kab = K( amap, gids ); 

        /** all right hand sides */
        std::vector<size_t> bmap( nrhs );
        for ( size_t j = 0; j < bmap.size(); j ++ )
          bmap[ j ] = j;

        /** gids.size()-by-nrhs */
        auto wb  = w( gids, bmap ); 

        xgemm
        (
          "N", "N",
          Kab.row(), wb.col(), wb.row(),
          1.0, Kab.data(),        Kab.row(),
                wb.data(),         wb.row(),
          1.0, potentials.data(), potentials.row()
        );
      }
    }
    else
    {
      Evaluate<SYMBOLIC, NNPRUNE>( lchild, gid, nnandi, potentials );      
      Evaluate<SYMBOLIC, NNPRUNE>( rchild, gid, nnandi, potentials );
    }
  }
  else // need gid's morton and neighbors' mortons
  {
    //printf( "level %lu is prunable\n", node->l );
    if ( SYMBOLIC )
    {
      data.lock.Acquire();
      {
        // Add gid to prunable list.
        if ( NNPRUNE ) node->FarIDs.insert(   gid );
        else           node->NNFarIDs.insert( gid );
      }
      data.lock.Release();
    }
    else
    {
#ifdef DEBUG_SPDASKIT
      printf( "level %lu is prunable\n", node->l );
#endif
      auto Kab = K( amap, node->data.skels );
      auto &w_skel = node->data.w_skel;
      xgemm
      (
        "N", "N",
        Kab.row(), w_skel.col(), w_skel.row(),
        1.0, Kab.data(),        Kab.row(),
          w_skel.data(),     w_skel.row(),
        1.0, potentials.data(), potentials.row()
      );          
    }
  }



}; /** end Evaluate() */


/** @brief Evaluate potentials( gid ) using treecode.
 *         Notice in this case, the approximation is unsymmetric.
 *
 **/
template<bool SYMBOLIC, bool NNPRUNE, typename TREE, typename T>
void Evaluate
( 
  TREE &tree, 
  size_t gid, 
  hmlp::Data<T> &potentials
)
{
  std::vector<size_t> nnandi;
  auto &w = *tree.setup.w;

  potentials.clear();
  potentials.resize( 1, w.col(), 0.0 );

  if ( NNPRUNE )
  {
    auto &NN = *tree.setup.NN;
    nnandi.reserve( NN.row() + 1 );
    nnandi.push_back( gid );
    for ( size_t i = 0; i < NN.row(); i ++ )
    {
      nnandi.push_back( NN( i, gid ).second );
    }
#ifdef DEBUG_SPDASKIT
    printf( "nnandi.size() %lu\n", nnandi.size() );
#endif
  }
  else
  {
    nnandi.reserve( 1 );
    nnandi.push_back( gid );
  }

  Evaluate<SYMBOLIC, NNPRUNE>( tree.treelist[ 0 ], gid, nnandi, potentials );

}; /** end Evaluate() */


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
hmlp::Data<T> Evaluate
( 
  TREE &tree,
  hmlp::Data<T> &weights
)
{
  const bool AUTO_DEPENDENCY = true;

  /** get type NODE = TREE::NODE */
  using NODE = typename TREE::NODE;

  /** all timers */
  double beg, time_ratio, evaluation_time = 0.0;
  double allocate_time, computeall_time;
  double forward_permute_time, backward_permute_time;

  /** clean up all r/w dependencies left on tree nodes */
  tree.DependencyCleanUp();

  /** n-by-nrhs initialize potentials */
  size_t n    = weights.row();
  size_t nrhs = weights.col();

  beg = omp_get_wtime();
  hmlp::Data<T> potentials( n, nrhs, 0.0 );
  tree.setup.w = &weights;
  tree.setup.u = &potentials;
  allocate_time = omp_get_wtime() - beg;

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
    };
  }
  forward_permute_time = omp_get_wtime() - beg;



  /** Compute all N2S, S2S, S2N, L2L */
  if ( REPORT_EVALUATE_STATUS )
  {
    printf( "N2S, S2S, S2N, L2L (HMLP Runtime) ...\n" ); fflush( stdout );
  }
  if ( SYMMETRIC_PRUNE )
  {
    beg = omp_get_wtime();
#ifdef HMLP_USE_CUDA
    potentials.AllocateD( hmlp_get_device( 0 ) );
    using LEAFTOLEAFVER2TASK = gpu::LeavesToLeavesVer2Task<CACHE, NNPRUNE, NODE, T>;
    LEAFTOLEAFVER2TASK leaftoleafver2task;
#endif
    using LEAFTOLEAFTASK1 = LeavesToLeavesTask<1, NNPRUNE, NODE, T>;
    using LEAFTOLEAFTASK2 = LeavesToLeavesTask<2, NNPRUNE, NODE, T>;
    using LEAFTOLEAFTASK3 = LeavesToLeavesTask<3, NNPRUNE, NODE, T>;
    using LEAFTOLEAFTASK4 = LeavesToLeavesTask<4, NNPRUNE, NODE, T>;

    using NODETOSKELTASK  = UpdateWeightsTask<NODE, T>;
    using SKELTOSKELTASK  = SkeletonsToSkeletonsTask<NNPRUNE, NODE, T>;
    using SKELTONODETASK  = SkeletonsToNodesTask<NNPRUNE, NODE, T>;

    LEAFTOLEAFTASK1 leaftoleaftask1;
    LEAFTOLEAFTASK2 leaftoleaftask2;
    LEAFTOLEAFTASK3 leaftoleaftask3;
    LEAFTOLEAFTASK4 leaftoleaftask4;

    NODETOSKELTASK  nodetoskeltask;
    SKELTOSKELTASK  skeltoskeltask;
    SKELTONODETASK  skeltonodetask;


//    if ( USE_OMP_TASK )
//    {
//      assert( !USE_RUNTIME );
//      tree.template TraverseLeafs<false, false>( leaftoleaftask1 );
//      tree.template TraverseLeafs<false, false>( leaftoleaftask2 );
//      tree.template TraverseLeafs<false, false>( leaftoleaftask3 );
//      tree.template TraverseLeafs<false, false>( leaftoleaftask4 );
//      tree.template UpDown<true, true, true>( nodetoskeltask, skeltoskeltask, skeltonodetask );
//    }
//    else
//    {
//      assert( !USE_OMP_TASK );
//
//#ifdef HMLP_USE_CUDA
//      tree.template TraverseLeafs<AUTO_DEPENDENCY, USE_RUNTIME>( leaftoleafver2task );
//#else
//      tree.template TraverseLeafs<AUTO_DEPENDENCY, USE_RUNTIME>( leaftoleaftask1 );
//      tree.template TraverseLeafs<AUTO_DEPENDENCY, USE_RUNTIME>( leaftoleaftask2 );
//      tree.template TraverseLeafs<AUTO_DEPENDENCY, USE_RUNTIME>( leaftoleaftask3 );
//      tree.template TraverseLeafs<AUTO_DEPENDENCY, USE_RUNTIME>( leaftoleaftask4 );
//#endif
//
//      /** check scheduler */
//      //hmlp_get_runtime_handle()->scheduler->ReportRemainingTime();
//      tree.template TraverseUp       <AUTO_DEPENDENCY, USE_RUNTIME>( nodetoskeltask );
//      tree.template TraverseUnOrdered<AUTO_DEPENDENCY, USE_RUNTIME>( skeltoskeltask );
//      tree.template TraverseDown     <AUTO_DEPENDENCY, USE_RUNTIME>( skeltonodetask );
//      /** check scheduler */
//      //hmlp_get_runtime_handle()->scheduler->ReportRemainingTime();
//
//      if ( USE_RUNTIME ) hmlp_run();
//
//
//
//#ifdef HMLP_USE_CUDA
//      hmlp::Device *device = hmlp_get_device( 0 );
//      for ( int stream_id = 0; stream_id < 10; stream_id ++ )
//        device->wait( stream_id );
//      //potentials.PrefetchD2H( device, 0 );
//      potentials.FetchD2H( device );
//#endif
//    }




    /** CPU-GPU hybrid uses a different kind of L2L task */
#ifdef HMLP_USE_CUDA
    tree.template TraverseLeafs    <USE_RUNTIME>( leaftoleafver2task );
#else
    tree.template TraverseLeafs    <USE_RUNTIME>( leaftoleaftask1 );
    tree.template TraverseLeafs    <USE_RUNTIME>( leaftoleaftask2 );
    tree.template TraverseLeafs    <USE_RUNTIME>( leaftoleaftask3 );
    tree.template TraverseLeafs    <USE_RUNTIME>( leaftoleaftask4 );
#endif
    tree.template TraverseUp       <USE_RUNTIME>( nodetoskeltask );
    tree.template TraverseUnOrdered<USE_RUNTIME>( skeltoskeltask );
    tree.template TraverseDown     <USE_RUNTIME>( skeltonodetask );
    if ( USE_RUNTIME ) hmlp_run();


    double d2h_beg_t = omp_get_wtime();
#ifdef HMLP_USE_CUDA
    hmlp::Device *device = hmlp_get_device( 0 );
    for ( int stream_id = 0; stream_id < 10; stream_id ++ )
      device->wait( stream_id );
    //potentials.PrefetchD2H( device, 0 );
    potentials.FetchD2H( device );
#endif
    double d2h_t = omp_get_wtime() - d2h_beg_t;
    printf( "d2h_t %lfs\n", d2h_t );


    double aggregate_beg_t = omp_get_wtime();
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
    double aggregate_t = omp_get_wtime() - aggregate_beg_t;
    printf( "aggregate_t %lfs\n", d2h_t );
 
#ifdef HMLP_USE_CUDA
    device->wait( 0 );
#endif
    computeall_time = omp_get_wtime() - beg;
  }
  else // TODO: implement unsymmetric prunning
  {
    /** Not yet implemented. */
    printf( "Non symmetric ComputeAll is not yet implemented\n" );
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
    auto &amap = node->gids;
    auto &u_leaf = node->data.u_leaf[ 0 ];



    /** assemble u_leaf back to u */
    //for ( size_t j = 0; j < amap.size(); j ++ )
    //  for ( size_t i = 0; i < potentials.row(); i ++ )
    //    potentials[ amap[ j ] * potentials.row() + i ] += u_leaf( j, i );


    for ( size_t j = 0; j < potentials.col(); j ++ )
      for ( size_t i = 0; i < amap.size(); i ++ )
        potentials( amap[ i ], j ) += u_leaf( i, j );


  }
  backward_permute_time = omp_get_wtime() - beg;

  evaluation_time += allocate_time;
  evaluation_time += forward_permute_time;
  evaluation_time += computeall_time;
  evaluation_time += backward_permute_time;
  time_ratio = 100 / evaluation_time;

  if ( REPORT_EVALUATE_STATUS )
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


  /** clean up all r/w dependencies left on tree nodes */
  tree.DependencyCleanUp();

  /** return nrhs-by-N outputs */
  return potentials;

}; /** end Evaluate() */





/**
 *  @brielf template of the compress routine
 */ 
template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  typename    SPLITTER, 
  typename    RKDTSPLITTER, 
  typename    T, 
  typename    SPDMATRIX>
tree::Tree<
  gofmm::Setup<SPDMATRIX, SPLITTER, T>, 
  gofmm::NodeData<T>,
  N_CHILDREN,
  T> 
*Compress
( 
  Data<T> *X,
  SPDMATRIX &K, 
  Data<pair<T, size_t>> &NN,
  SPLITTER splitter, 
  RKDTSPLITTER rkdtsplitter,
	Configuration<T> &config
)
{
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

  /** instantiation for the Spd-Askit tree */
  using SETUP     = gofmm::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA      = gofmm::NodeData<T>;
  using NODE      = tree::Node<SETUP, N_CHILDREN, DATA, T>;
  using TREE      = tree::Tree<SETUP, DATA, N_CHILDREN, T>;

  /** instantiation for the randomisze Spd-Askit tree */
  using RKDTSETUP = gofmm::Setup<SPDMATRIX, RKDTSPLITTER, T>;
  using RKDTNODE  = tree::Node<RKDTSETUP, N_CHILDREN, DATA, T>;

  /** all timers */
  double beg, omptask45_time, omptask_time, ref_time;
  double time_ratio, compress_time = 0.0, other_time = 0.0;
  double ann_time, tree_time, skel_time, mergefarnodes_time, cachefarnodes_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;


  /** original order of the matrix */
  beg = omp_get_wtime();
  vector<size_t> gids( n ), lids( n );
  #pragma omp parallel for
  for ( auto i = 0; i < n; i ++ ) 
  {
    gids[ i ] = i;
    lids[ i ] = i;
  }
  other_time += omp_get_wtime() - beg;


  /** iterative all nearnest-neighbor (ANN) */
  const size_t n_iter = 10;
  const bool SORTED = false;
  /** do not change anything below this line */
  tree::Tree<RKDTSETUP, DATA, N_CHILDREN, T> rkdt;
  rkdt.setup.X = X;
  rkdt.setup.K = &K;
	rkdt.setup.metric = metric; 
  rkdt.setup.splitter = rkdtsplitter;
  pair<T, size_t> initNN( numeric_limits<T>::max(), n );
  if ( REPORT_COMPRESS_STATUS )
  {
    printf( "NeighborSearch ...\n" ); fflush( stdout );
  }
  beg = omp_get_wtime();
  if ( NN.size() != n * k )
  {
    gofmm::KNNTask<3, RKDTNODE, T> KNNtask;
    NN = rkdt.template AllNearestNeighbor<SORTED>
         ( n_iter, k, 10, gids, lids, initNN, KNNtask );
  }
  else
  {
    if ( REPORT_COMPRESS_STATUS )
    {
      printf( "not performed (precomputed or k=0) ...\n" ); fflush( stdout );
    }
  }
  ann_time = omp_get_wtime() - beg;

  /** check illegle values in NN */
  for ( size_t j = 0; j < NN.col(); j ++ )
  {
    for ( size_t i = 0; i < NN.row(); i ++ )
    {
      size_t neighbor_gid = NN( i, j ).second;
      if ( neighbor_gid < 0 || neighbor_gid >= n )
      {
        printf( "NN( %lu, %lu ) has illegle values %lu\n", i, j, neighbor_gid );
        break;
      }
    }
  }






  /** initialize metric ball tree using approximate center split */
  auto *tree_ptr = new tree::Tree<SETUP, DATA, N_CHILDREN, T>();
	auto &tree = *tree_ptr;
  tree.setup.X = X;
  tree.setup.K = &K;
	tree.setup.metric = metric; 
  tree.setup.splitter = splitter;
  tree.setup.NN = &NN;
  tree.setup.m = m;
  tree.setup.k = k;
  tree.setup.s = s;
  tree.setup.stol = stol;
  tree.setup.budget = budget;
  if ( REPORT_COMPRESS_STATUS )
  {
    printf( "TreePartitioning ...\n" ); fflush( stdout );
  }
  beg = omp_get_wtime();
  tree.TreePartition( gids, lids );
  tree_time = omp_get_wtime() - beg;


#ifdef HMLP_AVX512
  /** if we are using KNL, use nested omp construct */
  assert( omp_get_max_threads() == 68 );
  //mkl_set_dynamic( 0 );
  //mkl_set_num_threads( 4 );
  hmlp_set_num_workers( 17 );
#else
  //if ( omp_get_max_threads() > 8 )
  //{
  //  hmlp_set_num_workers( omp_get_max_threads() / 2 );
  //}
  if ( REPORT_COMPRESS_STATUS )
  {
    printf( "omp_get_max_threads() %d\n", omp_get_max_threads() );
  }
#endif




  /**
   *  Build the near node list and merge neighbors
   */ 
  NearSamplesTask<NODE, T> NEARSAMPLEStask;
  tree.TraverseLeafs<true>( NEARSAMPLEStask );
  hmlp_run();
  tree.DependencyCleanUp();
  printf( "Finish NearSamplesTask\n" ); fflush( stdout );


  /**
   *  Create sample pools for each tree node by merging the near interaction
   *  list. 
   */ 
  BuildPoolTask<NODE, T> BUILDPOOLtask;
  tree.TraverseUp<true>( BUILDPOOLtask );
  hmlp_run();
  tree.DependencyCleanUp();
  printf( "Finish BuildPoolTask\n" ); fflush( stdout );


  /** Skeletonization */
  if ( REPORT_COMPRESS_STATUS )
  {
    printf( "Skeletonization (HMLP Runtime) ...\n" ); fflush( stdout );
  }
  beg = omp_get_wtime();
  //gofmm::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T> SKELtask;
  //tree.template TraverseUp<true>( SKELtask );
  gofmm::GetSkeletonMatrixTask<NODE, T> GETMTXtask;
  gofmm::SkeletonizeTask2<ADAPTIVE, LEVELRESTRICTION, NODE, T> SKELtask;
  tree.TraverseUp( GETMTXtask, SKELtask );
  gofmm::InterpolateTask<NODE, T> PROJtask;
  tree.template TraverseUnOrdered<true>( PROJtask );
  if ( CACHE )
  {
    gofmm::CacheNearNodesTask<NNPRUNE, NODE> KIJtask;
    tree.template TraverseLeafs<true>( KIJtask );
  }
  other_time += omp_get_wtime() - beg;
  hmlp_run();
  skel_time = omp_get_wtime() - beg;


  /** (optional for comparison) parallel level-by-level traversal */
  beg = omp_get_wtime();
  ref_time = omp_get_wtime() - beg;
 
  /** (optional for comparison) sekeletonization with omp task. */
  beg = omp_get_wtime();
  omptask_time = omp_get_wtime() - beg;

  /** (optional for comparison) sekeletonization with omp task. */
  beg = omp_get_wtime();
  omptask45_time = omp_get_wtime() - beg;


  /** MergeFarNodes */
  beg = omp_get_wtime();
  if ( REPORT_COMPRESS_STATUS )
  {
    printf( "MergeFarNodes ...\n" ); fflush( stdout );
  }
  gofmm::MergeFarNodes<SYMMETRIC>( tree );
  mergefarnodes_time = omp_get_wtime() - beg;

  /** CacheFarNodes */
  beg = omp_get_wtime();
  if ( REPORT_COMPRESS_STATUS )
  {
    printf( "CacheFarNodes ...\n" ); fflush( stdout );
  }
  gofmm::CacheFarNodes<NNPRUNE, CACHE>( tree );
  cachefarnodes_time = omp_get_wtime() - beg;

  /** plot iteraction matrix */  
  auto exact_ratio = hmlp::gofmm::DrawInteraction<true>( tree );

#ifdef HMLP_AVX512
  //mkl_set_dynamic( 1 );
  //mkl_set_num_threads( omp_get_max_threads() );
#else
  //mkl_set_dynamic( 1 );
  //mkl_set_num_threads( omp_get_max_threads() );
#endif

  compress_time += ann_time;
  compress_time += tree_time;
  compress_time += skel_time;
  compress_time += mergefarnodes_time;
  compress_time += cachefarnodes_time;
  time_ratio = 100.0 / compress_time;
  if ( REPORT_COMPRESS_STATUS )
  {
    printf( "========================================================\n");
    printf( "GOFMM compression phase\n" );
    printf( "========================================================\n");
    printf( "NeighborSearch ------------------------ %5.2lfs (%5.1lf%%)\n", ann_time, ann_time * time_ratio );
    printf( "TreePartitioning ---------------------- %5.2lfs (%5.1lf%%)\n", tree_time, tree_time * time_ratio );
    printf( "Skeletonization (HMLP Runtime   ) ----- %5.2lfs (%5.1lf%%)\n", skel_time, skel_time * time_ratio );
    printf( "                (Level-by-Level ) ----- %5.2lfs\n", ref_time );
    printf( "                (omp task       ) ----- %5.2lfs\n", omptask_time );
    printf( "                (Omp task depend) ----- %5.2lfs\n", omptask45_time );
    printf( "MergeFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", mergefarnodes_time, mergefarnodes_time * time_ratio );
    printf( "CacheFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", cachefarnodes_time, cachefarnodes_time * time_ratio );
    printf( "========================================================\n");
    printf( "Compress (%4.2lf not compressed) -------- %5.2lfs (%5.1lf%%)\n", 
        exact_ratio, compress_time, compress_time * time_ratio );
    printf( "========================================================\n\n");
  }

  /** clean up all r/w dependencies left on tree nodes */
  tree_ptr->DependencyCleanUp();

  /** return the hierarhical compreesion of K as a binary tree */
  //return tree;
  return tree_ptr;

}; /** end Compress() */









































/**
 *  @brielf A simple template for the compress routine.
 */ 
template<typename T, typename SPDMATRIX>
tree::Tree<
  gofmm::Setup<SPDMATRIX, centersplit<SPDMATRIX, 2, T>, T>, 
  gofmm::NodeData<T>,
  2,
  T> 
*Compress( SPDMATRIX &K, T stol, T budget, size_t m, size_t k, size_t s )
{
  const bool ADAPTIVE = true;
  const bool LEVELRESTRICTION = false;
  using SPLITTER     = centersplit<SPDMATRIX, 2, T>;
  using RKDTSPLITTER = randomsplit<SPDMATRIX, 2, T>;
  hmlp::Data<T> *X = NULL;
  hmlp::Data<std::pair<T, std::size_t>> NN;
	/** GOFMM tree splitter */
  SPLITTER splitter;
  splitter.Kptr = &K;
	splitter.metric = ANGLE_DISTANCE;
	/** randomized tree splitter */
  RKDTSPLITTER rkdtsplitter;
  rkdtsplitter.Kptr = &K;
	rkdtsplitter.metric = ANGLE_DISTANCE;
  size_t n = K.row();

	/** creatgin configuration for all user-define arguments */
	Configuration<T> config( ANGLE_DISTANCE, n, m, k, s, stol, budget );

	/** call the complete interface and return tree_ptr */
  return Compress<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER>
         ( X, K, NN, //ANGLE_DISTANCE, 
					 splitter, rkdtsplitter, //n, m, k, s, stol, budget, 
					 config );
}; /** end Compress */








/**
 *  @brielf A simple template for the compress routine.
 */ 
template<typename T, typename SPDMATRIX>
tree::Tree<
  gofmm::Setup<SPDMATRIX, centersplit<SPDMATRIX, 2, T>, T>, 
  gofmm::NodeData<T>,
  2,
  T> 
*Compress( SPDMATRIX &K, T stol, T budget )
{
  const bool ADAPTIVE = true;
  const bool LEVELRESTRICTION = false;
  using SPLITTER     = centersplit<SPDMATRIX, 2, T>;
  using RKDTSPLITTER = randomsplit<SPDMATRIX, 2, T>;
  Data<T> *X = NULL;
  Data<pair<T, std::size_t>> NN;
	/** GOFMM tree splitter */
  SPLITTER splitter;
  splitter.Kptr = &K;
  splitter.metric = ANGLE_DISTANCE;
	/** randomized tree splitter */
  RKDTSPLITTER rkdtsplitter;
  rkdtsplitter.Kptr = &K;
  rkdtsplitter.metric = ANGLE_DISTANCE;
  size_t n = K.row();
  size_t m = 128;
  size_t k = 16;
  size_t s = m;

  /** */
  if ( n >= 16384 )
  {
    m = 128;
    k = 20;
    s = 256;
  }

  if ( n >= 32768 )
  {
    m = 256;
    k = 24;
    s = 384;
  }

  if ( n >= 65536 )
  {
    m = 512;
    k = 32;
    s = 512;
  }

	/** creatgin configuration for all user-define arguments */
	Configuration<T> config( ANGLE_DISTANCE, n, m, k, s, stol, budget );

	/** call the complete interface and return tree_ptr */
  return Compress<ADAPTIVE, LEVELRESTRICTION, SPLITTER, RKDTSPLITTER>
         ( X, K, NN, //ANGLE_DISTANCE, 
					 splitter, rkdtsplitter, 
					 //n, m, k, s, stol, budget, 
					 config );

}; /** end Compress() */

/**
 *
 */ 
template<typename T>
tree::Tree<
  gofmm::Setup<SPDMatrix<T>, centersplit<SPDMatrix<T>, 2, T>, T>, 
  gofmm::NodeData<T>,
  2,
  T>
*Compress( SPDMatrix<T> &K, T stol, T budget )
{
	return Compress<T, SPDMatrix<T>>( K, stol, budget );
}; /** end Compress() */















template<typename NODE, typename T>
void ComputeError( NODE *node, hmlp::Data<T> potentials )
{
  auto &K = *node->setup->K;
  auto &w = node->setup->w;

  auto &amap = node->gids;
  std::vector<size_t> bmap = std::vector<size_t>( K.col() );

  for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

  auto Kab = K( amap, bmap );

  auto nrm2 = hmlp_norm( potentials.row(), potentials.col(), 
                         potentials.data(), potentials.row() ); 

  xgemm
  (
    "N", "T",
    Kab.row(), w.row(), w.col(),
    -1.0, Kab.data(),        Kab.row(),
          w.data(),          w.row(),
     1.0, potentials.data(), potentials.row()
  );          

  auto err = hmlp_norm( potentials.row(), potentials.col(), 
                        potentials.data(), potentials.row() ); 
  
  printf( "node relative error %E, nrm2 %E\n", err / nrm2, nrm2 );


}; // end void ComputeError()








/**
 *  @brief 
 */ 
template<typename TREE, typename T>
T ComputeError( TREE &tree, size_t gid, hmlp::Data<T> potentials )
{
  auto &K = *tree.setup.K;
  auto &w = *tree.setup.w;

  auto amap = std::vector<size_t>( 1, gid );
  auto bmap = std::vector<size_t>( K.col() );
  for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

  auto Kab = K( amap, bmap );
  auto exact = potentials;

  xgemm
  (
    "N", "N",
    Kab.row(), w.col(), w.row(),
    1.0,   Kab.data(),   Kab.row(),
             w.data(),     w.row(),
    0.0, exact.data(), exact.row()
  );          


  auto nrm2 = hmlp_norm( exact.row(),  exact.col(), 
                         exact.data(), exact.row() ); 

  xgemm
  (
    "N", "N",
    Kab.row(), w.col(), w.row(),
    -1.0, Kab.data(),       Kab.row(),
          w.data(),          w.row(),
     1.0, potentials.data(), potentials.row()
  );          

  auto err = hmlp_norm( potentials.row(), potentials.col(), 
                        potentials.data(), potentials.row() ); 

  return err / nrm2;
}; /** end ComputeError() */




template<typename T, typename SPDMATRIX>
class SimpleGOFMM
{
  public:

    SimpleGOFMM( SPDMATRIX &K, T stol, T budget )
    {
      tree_ptr = Compress( K, stol, budget );
    };

    ~SimpleGOFMM()
    {
      if ( tree_ptr ) delete tree_ptr;
    };

    void Multiply( hmlp::Data<T> &y, hmlp::Data<T> &x )
    {
      //hmlp::Data<T> weights( x.col(), x.row() );

      //for ( size_t j = 0; j < x.col(); j ++ )
      //  for ( size_t i = 0; i < x.row(); i ++ )
      //    weights( j, i ) = x( i, j );


      y = hmlp::gofmm::Evaluate( *tree_ptr, x );
      //auto potentials = hmlp::gofmm::Evaluate( *tree_ptr, weights );

      //for ( size_t j = 0; j < y.col(); j ++ )
      //  for ( size_t i = 0; i < y.row(); i ++ )
      //    y( i, j ) = potentials( j, i );

    };

  private:

    /** GOFMM tree */
    hmlp::tree::Tree<
      hmlp::gofmm::Setup<SPDMATRIX, centersplit<SPDMATRIX, 2, T>, T>, 
      hmlp::gofmm::NodeData<T>, 2, T> *tree_ptr = NULL; 

}; /** end class SimpleGOFMM */














/**
 *  Instantiation types for double and single precision
 */ 
typedef SPDMatrix<double> dSPDMatrix_t;
typedef SPDMatrix<float > sSPDMatrix_t;

typedef hmlp::gofmm::Setup<SPDMatrix<double>, 
    centersplit<SPDMatrix<double>, 2, double>, double> dSetup_t;

typedef hmlp::gofmm::Setup<SPDMatrix<float>, 
    centersplit<SPDMatrix<float >, 2,  float>,  float> sSetup_t;

typedef hmlp::tree::Tree<dSetup_t, hmlp::gofmm::NodeData<double>, 2, double> dTree_t;
typedef hmlp::tree::Tree<sSetup_t, hmlp::gofmm::NodeData<float >, 2, float > sTree_t;





/** 
 *  PyCompress prototype. Notice that all pass-by-reference
 *  arguments are replaced by pass-by-pointer. There implementaion
 *  can be found at hmlp/package/$HMLP_ARCH/gofmm.gpp
 **/
Data<double> Evaluate( dTree_t *tree, hmlp::Data<double> *weights );
Data<float>  Evaluate( dTree_t *tree, hmlp::Data<float > *weights );

dTree_t *Compress( dSPDMatrix_t *K, double stol, double budget );
sTree_t *Compress( sSPDMatrix_t *K,  float stol,  float budget );

dTree_t *Compress( dSPDMatrix_t *K, double stol, double budget, 
		size_t m, size_t k, size_t s );
sTree_t *Compress( sSPDMatrix_t *K,  float stol,  float budget, 
		size_t m, size_t k, size_t s );


double ComputeError( dTree_t *tree, size_t gid, hmlp::Data<double> *potentials );
float  ComputeError( sTree_t *tree, size_t gid, hmlp::Data<float>  *potentials );










}; /** end namespace gofmm */

}; /** end namespace hmlp */

#endif /** define GOFMM_HPP */
