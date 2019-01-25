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

/** Use HMLP related support. */
#include <hmlp.h>
#include <hmlp_base.hpp>
/** Use HMLP primitives. */
#include <primitives/lowrank.hpp>
#include <primitives/combinatorics.hpp>
#include <primitives/gemm.hpp>
/** Use HMLP containers. */
#include <containers/VirtualMatrix.hpp>
#include <containers/SPDMatrix.hpp>
/** GOFMM templates. */
#include <tree.hpp>
#include <igofmm.hpp>
/** gpu related */
#ifdef HMLP_USE_CUDA
#include <cuda_runtime.h>
#include <gofmm_gpu.hpp>
#endif
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;



/** this parameter is used to reserve space for std::vector */
#define MAX_NRHS 1024
/** the block size we use for partitioning GEMM tasks */
#define GEMM_NB 256


//#define DEBUG_SPDASKIT 1
#define REPORT_ANN_ACCURACY 1
#define REPORT_COMPRESS_STATUS 1
#define REPORT_EVALUATE_STATUS 1


namespace hmlp
{
namespace gofmm
{

/** @brief This is a helper class that parses the arguments from command lines. */
class CommandLineHelper
{
  public:

    /** (Default) constructor. */
    CommandLineHelper( int argc, char *argv[] )
    {
      /** Number of columns and rows, i.e. problem size. */
      sscanf( argv[ 1 ], "%lu", &n );
      /** On-diagonal block size, such that the tree has log(n/m) levels. */
      sscanf( argv[ 2 ], "%lu", &m );
      /** Number of neighbors to use. */
      sscanf( argv[ 3 ], "%lu", &k );
      /** Maximum off-diagonal ranks. */
      sscanf( argv[ 4 ], "%lu", &s );
      /** Number of right-hand sides. */
      sscanf( argv[ 5 ], "%lu", &nrhs );
      /** Desired approximation accuracy. */
      sscanf( argv[ 6 ], "%lf", &stol );
      /** The maximum percentage of direct matrix-multiplication. */
      sscanf( argv[ 7 ], "%lf", &budget );
      /** Specify distance type. */
      distance_type = argv[ 8 ];
      if ( !distance_type.compare( "geometry" ) )
      {
        metric = GEOMETRY_DISTANCE;
      }
      else if ( !distance_type.compare( "kernel" ) )
      {
        metric = KERNEL_DISTANCE;
      }
      else if ( !distance_type.compare( "angle" ) )
      {
        metric = ANGLE_DISTANCE;
      }
      else
      {
        printf( "%s is not supported\n", argv[ 8 ] );
        exit( 1 );
      }
      /** Specify what kind of spdmatrix is used. */
      spdmatrix_type = argv[ 9 ];
      if ( !spdmatrix_type.compare( "testsuit" ) )
      {
        /** NOP */
      }
      else if ( !spdmatrix_type.compare( "userdefine" ) )
      {
        /** NOP */
      }
      else if ( !spdmatrix_type.compare( "pvfmm" ) )
      {
        /** NOP */
      }
      else if ( !spdmatrix_type.compare( "dense" ) || !spdmatrix_type.compare( "ooc" ) )
      {
        /** (Optional) provide the path to the matrix file. */
        user_matrix_filename = argv[ 10 ];
        if ( argc > 11 ) 
        {
          /** (Optional) provide the path to the data file. */
          user_points_filename = argv[ 11 ];
          /** Dimension of the data set. */
          sscanf( argv[ 12 ], "%lu", &d );
        }
      }
      else if ( !spdmatrix_type.compare( "mlp" ) )
      {
        hidden_layers = argv[ 10 ];
        user_points_filename = argv[ 11 ];
        /** Number of attributes (dimensions). */
        sscanf( argv[ 12 ], "%lu", &d );
      }
      else if ( !spdmatrix_type.compare( "cov" ) )
      {
        kernelmatrix_type = argv[ 10 ];
        user_points_filename = argv[ 11 ];
        /** Number of attributes (dimensions) */
        sscanf( argv[ 12 ], "%lu", &d );
        /** Block size (in dimensions) per file */
        sscanf( argv[ 13 ], "%lu", &nb );
      }
      else if ( !spdmatrix_type.compare( "kernel" ) )
      {
        kernelmatrix_type = argv[ 10 ];
        user_points_filename = argv[ 11 ];
        /** Number of attributes (dimensions) */
        sscanf( argv[ 12 ], "%lu", &d );
        /** (Optional) provide Gaussian kernel bandwidth */
        if ( argc > 13 ) sscanf( argv[ 13 ], "%lf", &h );
      }
      else
      {
        printf( "%s is not supported\n", argv[ 9 ] );
        exit( 1 );
      }
    }; /** end CommentLineSupport() */

    /** Basic GOFMM parameters. */
    size_t n, m, k, s, nrhs;
    /** (Default) user-defined approximation toleratnce and budget. */
    double stol = 1E-3;
    double budget = 0.0;
    bool secure_accuracy = false;
    /** (Default) geometric-oblivious scheme. */
    DistanceMetric metric = ANGLE_DISTANCE;

    /** (Optional) */
    size_t d, nb; 
    /** (Optional) set the default Gaussian kernel bandwidth. */
    double h = 1.0;
    string distance_type;
    string spdmatrix_type;
    string kernelmatrix_type;
    string hidden_layers;
    string user_matrix_filename;
    string user_points_filename;
}; /** end class CommandLineHelper */




/** @brief Configuration contains all user-defined parameters. */ 
template<typename T>
class Configuration
{
	public:

    typedef size_t sizeType;
    typedef size_t idType;

    Configuration() {};

		Configuration( DistanceMetric metric_type,
		  size_t problem_size, size_t leaf_node_size, 
      size_t neighbor_size, size_t maximum_rank, 
			T tolerance, T budget, bool secure_accuracy = true ) 
		{
		  try
      {
        HANDLE_ERROR( Set( metric_type, problem_size, leaf_node_size, 
          neighbor_size, maximum_rank, tolerance, budget, secure_accuracy ) );
      }
      catch ( const exception & e )
      {
        cout << e.what() << endl;
        exit( -1 );
      }
		};

		hmlpError_t Set( DistanceMetric metric_type,
		  size_t problem_size, size_t leaf_node_size, 
      size_t neighbor_size, size_t maximum_rank, 
			T tolerance, T budget, bool secure_accuracy ) 
		{
			this->metric_type = metric_type;
			this->problem_size = problem_size;
			//this->leaf_node_size_ = leaf_node_size;
			RETURN_IF_ERROR( setLeafNodeSize( leaf_node_size ) );
			this->neighbor_size = neighbor_size;
			this->maximum_rank = maximum_rank;
			this->tolerance = tolerance;
			this->budget = budget;
			this->secure_accuracy = secure_accuracy;
      /* Return with no error. */
			return HMLP_ERROR_SUCCESS;
		};

    void CopyFrom( Configuration<T> &config ) { *this = config; };

		DistanceMetric MetricType() const noexcept { return metric_type; };

		size_t ProblemSize() const noexcept { return problem_size; };

    hmlpError_t setLeafNodeSize( sizeType leaf_node_size ) noexcept 
    {
      /* Check if arguments are valid. */
      if ( leaf_node_size < 1 ) 
      {
        fprintf( stderr, "[ERROR] leaf node size must be > 0\n" );
        return HMLP_ERROR_INVALID_VALUE;
      }
      /* Set the value. */
      leaf_node_size_ = leaf_node_size;
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    };

		size_t getLeafNodeSize() const noexcept 
		{ 
		  return leaf_node_size_; 
		};

		size_t NeighborSize() const noexcept { return neighbor_size; };

		size_t MaximumRank() const noexcept { return maximum_rank; };

		T Tolerance() const noexcept { return tolerance; };

		T Budget() const noexcept { return budget; };

    bool IsSymmetric() const noexcept { return is_symmetric; };

    bool UseAdaptiveRanks() const noexcept { return use_adaptive_ranks; };

    bool SecureAccuracy() const noexcept { return secure_accuracy; };

	private:

		/** (Default) metric type. */
		DistanceMetric metric_type = ANGLE_DISTANCE;

		/** (Default) problem size. */
		size_t problem_size = 0;

		/** (Default) maximum leaf node size. */
		sizeType leaf_node_size_ = 64;

		/** (Default) number of neighbors. */
		size_t neighbor_size = 32;

		/** (Default) maximum off-diagonal ranks. */
		size_t maximum_rank = 64;

		/** (Default) user error tolerance. */
		T tolerance = 1E-3;

		/** (Default) user computation budget. */
		T budget = 0.03;

    /** (Default, Advanced) whether the matrix is symmetric. */
    bool is_symmetric = true;

		/** (Default, Advanced) whether or not using adaptive ranks. */
		bool use_adaptive_ranks = true;

    /** (Default, Advanced) whether or not securing the accuracy. */
    bool secure_accuracy = true;

}; /** end class Configuration */



/** @brief These are data that shared by the whole local tree. */ 
template<typename SPDMATRIX, typename SPLITTER, typename T>
class Setup : public tree::Setup<SPLITTER, T>,
              public Configuration<T>
{
  public:

    Setup() {};

    /** Shallow copy from the config. */
    void FromConfiguration( Configuration<T> &config,
        SPDMATRIX &K, SPLITTER &splitter, Data<pair<T, size_t>> *NN )
    { 
      this->CopyFrom( config ); 
      this->K = &K;
      this->splitter = splitter;
      this->NN = NN;
    };

    /** The SPDMATRIX (accessed with gids: dense, CSC or OOC). */
    SPDMATRIX *K = NULL;

    /** rhs-by-n, weights and potentials. */
    Data<T> *w = NULL;
    Data<T> *u = NULL;

    /** Buffer space, either dimension needs to be n. */
    Data<T> *input = NULL;
    Data<T> *output = NULL;

    /** Regularization for factorization. */
    T lambda = 0.0;

    /** Use ULV or Sherman-Morrison-Woodbury */
    bool do_ulv_factorization = true;

  private:




}; /** end class Setup */


/** @brief This class contains all GOFMM related data carried by a tree node. */ 
template<typename T>
class NodeData : public Factor<T>
{
  public:

    /** (Default) constructor. */
    NodeData() {};

    /** The OpenMP (or pthread) lock that grants exclusive right. */
    Lock lock;

    /** Whether the node can be compressed (with skel and proj). */
    bool is_compressed = false;

    /** Skeleton gids (subset of gids). */
    vector<size_t> skels;

    /** 2s, pivoting order of GEQP3 (or GEQP4). */
    vector<int> jpvt;

    /** s-by-2s, interpolative coefficients. */
    Data<T> proj;

    /** Sampling neighbors gids. */
    map<size_t, T> snids; 

    /** (Buffer) nsamples row gids, and sl + sr skeleton columns of children. */
    vector<size_t> candidate_rows;
    vector<size_t> candidate_cols;

    /** (Buffer) nsamples-by-(sl+sr) submatrix of K. */
    Data<T> KIJ; 

    /** (Buffer) skeleton weights and potentials. */
    Data<T> w_skel;
    Data<T> u_skel;

    /** (Buffer) permuted weights and potentials. */
    Data<T> w_leaf;
    Data<T> u_leaf[ 20 ];

    /** Hierarchical tree view of w<RIDS, STAR> and u<RIDS, STAR>. */
    View<T> w_view;
    View<T> u_view;

    /** Cached Kab */
    Data<size_t> Nearbmap;
    Data<T> NearKab;
    Data<T> FarKab;


    /** recorded events (for HMLP Runtime) */
    Event skeletonize;
    Event updateweight;
    Event skeltoskel;
    Event skeltonode;
    Event s2s;
    Event s2n;

    /** knn accuracy */
    //double knn_acc = 0.0;
    //size_t num_acc = 0;

}; /** end class Data */




/** @brief This task creates an hierarchical tree view for w<RIDS> and u<RIDS>. */
template<typename NODE>
class TreeViewTask : public Task
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

    /** Preorder dependencies (with a single source node). */
    void DependencyAnalysis() { arg->DependOnParent( this ); };

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
      }
    };
}; /** end class TreeViewTask */



















/** @brief Provide statistics summary for the execution section.  */ 
template<typename NODE>
class Summary
{

  public:

    Summary() {};

    deque<Statistic> rank;

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
        skeletonize.push_back( hmlp::Statistic() );
        updateweight.push_back( hmlp::Statistic() );
      }

      rank[ node->l ].Update( (double)node->data.skels.size() );
      skeletonize[ node->l ].Update( node->data.skeletonize.GetDuration() );
      updateweight[ node->l ].Update( node->data.updateweight.GetDuration() );

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

}; /** end class Summary */



/**
 *  @brief This the main splitter used to build the Spd-Askit tree.
 *         First compute the approximate center using subsamples.
 *         Then find the two most far away points to do the 
 *         projection.
 */ 
template<typename SPDMATRIX, int N_SPLIT, typename T> 
struct centersplit
{
  /** Closure */
  SPDMATRIX *Kptr = NULL;
  /** (Default) use angle distance from the Gram vector space. */
  DistanceMetric metric = ANGLE_DISTANCE;
  /** Number samples to approximate centroid. */
  size_t n_centroid_samples = 5;
 
  centersplit() {};

  centersplit( SPDMATRIX& K ) { this->Kptr = &K; };

	/** Overload the operator (). */
  vector<vector<size_t>> operator() ( vector<size_t>& gids ) const 
  {
    /** all assertions */
    assert( N_SPLIT == 2 );
    assert( Kptr );


    SPDMATRIX &K = *Kptr;
    vector<vector<size_t>> split( N_SPLIT );
    size_t n = gids.size();
    vector<T> temp( n, 0.0 );

    /** Collecting column samples of K. */
    auto column_samples = combinatorics::SampleWithoutReplacement( 
        n_centroid_samples, gids );


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

    /** Collecting KIP */
    vector<size_t> P( 1, gids[ idf2c ] );

    /** Compute all pairwise distances. */
    auto DIP = K.Distances( this->metric, gids, P );

    /** Find f2f (far most to far most) from owned points */
    auto idf2f = distance( DIP.begin(), max_element( DIP.begin(), DIP.end() ) );

    /** collecting KIQ */
    vector<size_t> Q( 1, gids[ idf2f ] );

    /** Compute all pairwise distances. */
    auto DIQ = K.Distances( this->metric, gids, P );

    for ( size_t i = 0; i < temp.size(); i ++ )
      temp[ i ] = DIP[ i ] - DIQ[ i ];

    return combinatorics::MedianSplit( temp );
  };
}; /** end struct centersplit */










/** @brief This the splitter used in the randomized tree.  */ 
template<typename SPDMATRIX, int N_SPLIT, typename T> 
struct randomsplit
{
  /** closure */
  SPDMATRIX *Kptr = NULL;

	/** (default) using angle distance from the Gram vector space */
  DistanceMetric metric = ANGLE_DISTANCE;

  randomsplit() {};

  randomsplit( SPDMATRIX& K ) { this->Kptr = &K; };

	/** overload with the operator */
  inline vector<vector<size_t> > operator() ( vector<size_t>& gids ) const 
  {
    assert( Kptr && ( N_SPLIT == 2 ) );

    SPDMATRIX &K = *Kptr;
    size_t n = gids.size();
    vector<vector<size_t> > split( N_SPLIT );
    vector<T> temp( n, 0.0 );

    /** Randomly select two points p and q. */
    size_t idf2c = std::rand() % n;
    size_t idf2f = std::rand() % n;
    while ( idf2c == idf2f ) idf2f = std::rand() % n;


    vector<size_t> P( 1, gids[ idf2c ] );
    vector<size_t> Q( 1, gids[ idf2f ] );


    /** Compute all pairwise distances. */
    auto DIP = K.Distances( this->metric, gids, P );
    auto DIQ = K.Distances( this->metric, gids, Q );

    for ( size_t i = 0; i < temp.size(); i ++ )
      temp[ i ] = DIP[ i ] - DIQ[ i ];

    return combinatorics::MedianSplit( temp );

  };
}; /** end struct randomsplit */




/**
 *  \brief Compute the all kappa-nearest neighbors for indices
 *         in node using the target distance metric.
 *  \return the error code
 */
template<typename NODE>
hmlpError_t FindNeighbors( NODE *node, DistanceMetric metric )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
  auto & setup = *(node->setup);
  auto & K = *(setup.K);
  auto & NN = *(setup.NN);
  auto & I = node->gids;
  /** Number of neighbors to search for. */
  size_t kappa = NN.row();
  /** Initial value for the neighbor select. */
  pair<T, size_t> init( numeric_limits<T>::max(), NN.col() );
  Data<pair<T, size_t>> candidates;
  /** k-nearest neighbor search kernel. */
  RETURN_IF_ERROR( K.NeighborSearch( metric, kappa, I, I, candidates, init ) );
  /** Merge and update neighbors. */
	#pragma omp parallel
  {
    vector<pair<T, size_t> > aux( 2 * kappa );
    #pragma omp for
    for ( size_t j = 0; j < I.size(); j ++ )
    {
      MergeNeighbors( kappa, NN.columndata( I[ j ] ), 
          candidates.columndata( j ), aux );
    }
  }
  return HMLP_ERROR_SUCCESS;
}; /* end FindNeighbors() */





template<class NODE, typename T>
class NeighborsTask : public Task
{
  public:

    NODE *arg = NULL;
   
	  /** (Default) using angle distance from the Gram vector space. */
	  DistanceMetric metric = ANGLE_DISTANCE;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "Neighbors" );
      label = to_string( arg->treelist_id );
      /** Use the same distance as the tree. */
      metric = arg->setup->MetricType();

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

    void DependencyAnalysis() { arg->DependOnNoOne( this ); };

    void Execute( Worker* user_worker ) 
    { 
      HANDLE_ERROR( FindNeighbors( arg, metric ) ); 
    };

}; /** end class NeighborsTask */



/** @brief This is the ANN routine design for CSC matrices. */ 
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

  return NN;
}; /** end SparsePattern() */


/* @brief Helper functions for sorting sampling neighbors. */ 
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




/** @brief Compute the cofficient matrix by R11^{-1} * proj. */ 
template<typename NODE>
void Interpolate( NODE *node )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
  /** Early return if possible. */
  if ( !node ) return;

  auto &K = *node->setup->K;
  auto &data = node->data;
  auto &skels = data.skels;
  auto &proj = data.proj;
  auto &jpvt = data.jpvt;
  auto s = proj.row();
  auto n = proj.col();

  /** Early return if the node is incompressible or all zeros. */
  if ( !data.is_compressed || proj[ 0 ] == 0 ) return;

  assert( s );
  assert( s <= n );
  assert( jpvt.size() == n );

  /** If is skeletonized, reserve space for w_skel and u_skel */
  if ( data.is_compressed )
  {
    data.w_skel.reserve( skels.size(), MAX_NRHS );
    data.u_skel.reserve( skels.size(), MAX_NRHS );
  }


  /** Fill in R11. */
  Data<T> R1( s, s, 0.0 );

  for ( int j = 0; j < s; j ++ )
  {
    for ( int i = 0; i < s; i ++ )
    {
      if ( i <= j ) R1[ j * s + i ] = proj[ j * s + i ];
    }
  }

  /** Copy proj to tmp. */
  Data<T> tmp = proj;

  /** proj = inv( R1 ) * proj */
  xtrsm( "L", "U", "N", "N", s, n, 1.0, R1.data(), s, tmp.data(), s );

  /** Fill in proj. */
  for ( int j = 0; j < n; j ++ )
  {
    for ( int i = 0; i < s; i ++ )
    {
  	  proj[ jpvt[ j ] * s + i ] = tmp[ j * s + i ];
    }
  }
 

}; /** end Interpolate() */


/** @brief The correponding task of Interpolate(). */ 
template<typename NODE>
class InterpolateTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "it" );
      label = to_string( arg->treelist_id );
      // Need an accurate cost model.
      cost = 1.0;
    };

    void DependencyAnalysis() { arg->DependOnNoOne( this ); };

    void Execute( Worker* user_worker ) { Interpolate( arg ); };

}; /** end class InterpolateTask */




/**
 *  TODO: I decided not to use the sampling pool
 */ 
template<bool NNPRUNE, typename NODE>
void RowSamples( NODE *node, size_t nsamples )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
  auto &setup = *(node->setup);
  auto &data = node->data;
  auto &K = *(setup.K);

  /** amap contains nsamples of row gids of K. */
  auto &amap = data.candidate_rows;

  /** Clean up candidates from previous iteration. */
  amap.clear();

  /** Construct snids from neighbors. */
  if ( setup.NN )
  {
    //printf( "construct snids NN.row() %lu NN.col() %lu\n", 
    //    node->setup->NN->row(), node->setup->NN->col() ); fflush( stdout );
    auto &NN = *(setup.NN);
    auto &gids = node->gids;
    auto &snids = data.snids;
    size_t knum = NN.row();

    if ( node->isleaf )
    {
      snids.clear();

      vector<pair<T, size_t>> tmp( knum * gids.size() );
      for ( size_t j = 0; j < gids.size(); j ++ )
        for ( size_t i = 0; i < knum; i ++ )
          tmp[ j * knum + i ] = NN( i, gids[ j ] );

      /** Create a sorted list. */
      sort( tmp.begin(), tmp.end() );
    
      /** Each candidate is a pair of (distance, gid). */
      for ( auto it : tmp )
      {
        size_t it_gid = it.second;
        size_t it_morton = setup.morton[ it_gid ];

        if ( snids.size() >= nsamples ) break;

        /** Accept the sample if it does not belong to any near node */
        bool is_near;
        if ( NNPRUNE ) is_near = node->NNNearNodeMortonIDs.count( it_morton );
        else           is_near = (it_morton == node->morton );

        if ( !is_near )
        {
          /** Duplication is handled by std::map. */
          auto ret = snids.insert( make_pair( it.second, it.first ) );
        }
      }
    }
    else
    {
      auto &lsnids = node->lchild->data.snids;
      auto &rsnids = node->rchild->data.snids;

      /** Merge left children's sampling neighbors */
      snids = lsnids;

      /**
       *  TODO: Exclude lsnids (rsnids) that are near rchild (lchild),
       *  perhaps using a NearNodes list defined for interior nodes.
       **/
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

      /** Remove on-diagonal indices (gids) */
      for ( auto gid : gids ) snids.erase( gid );
    }


    if ( nsamples < K.col() - node->n )
    {
      /** Create an order snids by flipping the std::map */
      multimap<T, size_t> ordered_snids = flip_map( snids );
      /** Reserve space for insertion. */
      amap.reserve( nsamples );

      /** First we use important samples from snids. */
      for ( auto it : ordered_snids )
      {
        if ( amap.size() >= nsamples ) break;
        /** it has type pair<T, size_t> */
        amap.push_back( it.second );
      }

      /** Use uniform samples with replacement if there are not enough samples. */
      while ( amap.size() < nsamples )
      {
        //size_t sample = rand() % K.col();
        auto important_sample = K.ImportantSample( 0 );
        size_t sample_gid = important_sample.second;
        size_t sample_morton = setup.morton[ sample_gid ];

        if ( !MortonHelper::IsMyParent( sample_morton, node->morton ) )
        {
          amap.push_back( sample_gid );
        }
      }
    }
    else /** use all off-diagonal blocks without samples */
    {
      for ( size_t sample = 0; sample < K.col(); sample ++ )
      {
        size_t sample_morton = setup.morton[ sample ];
        if ( !MortonHelper::IsMyParent( sample_morton, node->morton ) )
        {
          amap.push_back( sample );
        }
      }
    }
  } /** end if ( node->setup->NN ) */

}; /** end RowSamples() */








template<bool NNPRUNE, typename NODE>
void SkeletonKIJ( NODE *node )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
  /** Gather shared data and create reference. */
  auto &K = *(node->setup->K);
  /** Gather per node data and create reference. */
  auto &data = node->data;
  auto &candidate_rows = data.candidate_rows;
  auto &candidate_cols = data.candidate_cols;
  auto &KIJ = data.KIJ;
  /** This node belongs to the local tree. */
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  if ( node->isleaf )
  {
    /** Use all columns. */
    candidate_cols = node->gids;
  }
  else
  {
    auto &lskels = lchild->data.skels;
    auto &rskels = rchild->data.skels;
    /** If either child is not skeletonized, then return. */
    if ( !lskels.size() || !rskels.size() ) return;
    /** Concatinate [ lskels, rskels ]. */
    candidate_cols = lskels;
    candidate_cols.insert( candidate_cols.end(), 
        rskels.begin(), rskels.end() );
  }

  /** Decide number of rows to sample. */
  size_t nsamples = 2 * candidate_cols.size();

  /** Make sure we at least m samples. */
  if ( nsamples < 2 * node->setup->getLeafNodeSize() ) 
  {
    nsamples = 2 * node->setup->getLeafNodeSize();
  }

  /** Sample off-diagonal rows. */
  RowSamples<NNPRUNE>( node, nsamples );
  
  /** Compute (or fetch) submatrix KIJ. */
  KIJ = K( candidate_rows, candidate_cols );



}; /** end SkeletonKIJ() */







/**
 *
 */ 
template<bool NNPRUNE, typename NODE, typename T>
class SkeletonKIJTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "par-gskm" );
      label = to_string( arg->treelist_id );
      /** we don't know the exact cost here */
      cost = 5.0;
      /** high priority */
      priority = true;
    };

    void DependencyAnalysis() { arg->DependOnChildren( this ); };

    void Execute( Worker* user_worker ) { SkeletonKIJ<NNPRUNE>( arg ); };

}; /** end class SkeletonKIJTask */ 


/** @brief Compress with interpolative decomposition (ID). */ 
template<typename NODE>
void Skeletonize( NODE *node )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
  /** Gather shared data and create reference. */
  auto &K   = *(node->setup->K);
  auto maxs = node->setup->MaximumRank();
  auto stol = node->setup->Tolerance();
  bool secure_accuracy = node->setup->SecureAccuracy();
  bool use_adaptive_ranks = node->setup->UseAdaptiveRanks();
  /** Gather per node data and create reference. */
  auto &data  = node->data;
  auto &skels = data.skels;
  auto &proj  = data.proj;
  auto &jpvt  = data.jpvt;
  auto &KIJ   = data.KIJ;
  auto &candidate_cols = data.candidate_cols;
  /** Interpolative decomposition (ID). */
  size_t N = K.col();
  size_t m = KIJ.row();
  size_t n = KIJ.col();
  size_t q = node->n;
  /** Bill's l2 norm scaling factor. */
  T scaled_stol = std::sqrt( (T)n / q ) * std::sqrt( (T)m / (N - q) ) * stol;
  /** TODO: check if this is needed? Account for uniform sampling. */
  if ( true ) scaled_stol *= std::sqrt( (T)q / N );
  /** Call adaptive interpolative decomposition primitive. */
  lowrank::id( use_adaptive_ranks, secure_accuracy,
    KIJ.row(), KIJ.col(), maxs, scaled_stol, KIJ, skels, proj, jpvt );
  /** Free KIJ for spaces. */
  KIJ.clear();
  /** Relabel skeletions with the real gids. */
  for ( size_t i = 0; i < skels.size(); i ++ ) 
  {
    skels[ i ] = candidate_cols[ skels[ i ] ];
  }
}; /* end Skeletonize() */


template<typename NODE, typename T>
class SkeletonizeTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "sk" );
      label = to_string( arg->treelist_id );
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

    void DependencyAnalysis() { arg->DependOnNoOne( this ); };

    void Execute( Worker* user_worker ) 
    { 
      /** Early return if we do not need to skeletonize. */
      if ( !arg->parent ) return;
      /* Check if we need to secure the accuracy? */
      bool secure_accuracy = arg->setup->SecureAccuracy();
      /* Gather per node data and create reference. */
      auto &data  = arg->data;
      /* Clear skels and proj. */
      data.skels.clear();
      data.proj.clear();
      /* If one of my children is not compreesed, so am I. */
      if ( secure_accuracy && !arg->isleaf ) 
      {
        if ( !arg->lchild->data.is_compressed || !arg->rchild->data.is_compressed )
        {
          data.is_compressed = false;
          return;
        }
      }
      /* Skeletonization using interpolative decomposition. */
      Skeletonize( arg );
      /* Depending on the flag, decide is_compressed or not. */
      data.is_compressed = ( secure_accuracy ) ? data.skels.size() : true;
      /* Sanity check. */
      if ( data.is_compressed ) assert( data.skels.size() && data.proj.size() && data.jpvt.size() );

    };

}; /** end class SkeletonizeTask */

































/** @brief Compute skeleton weights for each node. */
template<typename NODE>
void UpdateWeights( NODE *node )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
  /** Early return if possible. */
  if ( !node->parent || !node->data.is_compressed ) return;

  /** Gather shared data and create reference */
  auto &w = *node->setup->w;

  /** Gather per node data and create reference */
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
  }
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

    void DependencyAnalysis() { arg->DependOnChildren( this ); };

    void Execute( Worker* user_worker )
    {
#ifdef HMLP_USE_CUDA 
      hmlp::Device *device = NULL;
      if ( user_worker ) device = user_worker->GetDevice();
      if ( device ) gpu::UpdateWeights( device, arg );
      else               UpdateWeights<NODE, T>( arg );
#else
      UpdateWeights( arg );
#endif
    };

}; /** end class UpdateWeightsTask */



/**
 *  @brief Compute the interation from column skeletons to row
 *         skeletons. Store the results in the node. Later
 *         there is a SkeletonstoAll function to be called.
 *
 */ 
template<typename NODE>
void SkeletonsToSkeletons( NODE *node )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;
  /** Early return if possible. */
  if ( !node->parent || !node->data.is_compressed ) return;

  double beg, u_skel_time, s2s_time;

  auto *FarNodes = &node->NNFarNodes;

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
      auto Kab = K( amap, bmap );
      xgemm( "N", "N", u_skel.row(), u_skel.col(), w_skel.row(),
        1.0, Kab.data(),       Kab.row(),
             w_skel.data(), w_skel.row(),
        1.0, u_skel.data(), u_skel.row() );
    }
  }

}; /** end SkeletonsToSkeletons() */



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

    void DependencyAnalysis()
    {
      for ( auto it : arg->NNFarNodes ) it->DependencyAnalysis( R, this );
      arg->DependencyAnalysis( RW, this );
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker ) { SkeletonsToSkeletons( arg ); };
}; /** end class SkeletonsToSkeletonsTask */


/**
 *  @brief This is a task in Downward traversal. There is data
 *         dependency on u_skel.
 *         
 */ 
template<typename NODE>
void SkeletonsToNodes( NODE *node )
{
  /** Derive type T from NODE. */
  using T = typename NODE::T;

  /** Gather shared data and create reference. */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;

  /** Gather per node data and create reference. */
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
      if ( data.is_compressed )
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
  }
  else
  {
    if ( !node->parent || !node->data.is_compressed ) return;

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
        if ( !arg->parent || !arg->data.is_compressed )
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

    void DependencyAnalysis() { arg->DependOnParent( this ); };

    void Execute( Worker* user_worker )
    {
#ifdef HMLP_USE_CUDA 
      Device *device = NULL;
      if ( user_worker ) device = user_worker->GetDevice();
      if ( device ) gpu::SkeletonsToNodes<NNPRUNE, NODE, T>( device, arg );
      else               SkeletonsToNodes<NNPRUNE, NODE, T>( arg );
#else
    SkeletonsToNodes( arg );
#endif
    };

}; /** end class SkeletonsToNodesTask */



template<int SUBTASKID, bool NNPRUNE, typename NODE, typename T>
void LeavesToLeaves( NODE *node, size_t itbeg, size_t itend )
{
  assert( node->isleaf );

  double beg, u_leaf_time, before_writeback_time, after_writeback_time;

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
        auto Kab = K( amap, bmap );

        if ( wb.size() )
        {
          /** ( Kab * wb )' = wb' * Kab' */
          xgemm( "N", "N", u_leaf.row(), u_leaf.col(), wb.row(),
            1.0,    Kab.data(),    Kab.row(),
                     wb.data(),     wb.row(),
            1.0, u_leaf.data(), u_leaf.row());
        }
        else
        {
          View<T> W = (*it)->data.w_view;
          xgemm( "N", "N", u_leaf.row(), u_leaf.col(), W.row(),
            1.0,    Kab.data(),    Kab.row(),
                      W.data(),       W.ld(),
            1.0, u_leaf.data(), u_leaf.row() );
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
      name = string( "l2l" );
      label = to_string( arg->treelist_id );

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

      set<NODE*> *NearNodes;
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
    //double budget = setup.budget;
    double budget = setup.Budget();
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
      /** Insert */
      auto *target = (*node->morton2node)[ (*it).second ];
      node->NNNearNodeMortonIDs.insert( (*it).second );
      node->NNNearNodes.insert( target );
    }
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
      name = string( "near" );

      //--------------------------------------
      double flops = 0.0, mops = 0.0;

      /** setup the event */
      event.Set( label + name, flops, mops );
      /** asuume computation bound */
      cost = 1.0;
      /** low priority */
      priority = true;
    }

    void DependencyAnalysis() { this->TryEnqueue(); };

    void Execute( Worker* user_worker )
    {
      NearSamples<NODE, T>( arg );
    };

}; /** end class NearSamplesTask */


template<typename TREE>
void SymmetrizeNearInteractions( TREE & tree )
{
  int n_nodes = 1 << tree.depth;
  auto level_beg = tree.treelist.begin() + n_nodes - 1;

  for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
  {
    auto *node = *(level_beg + node_ind);
    auto & NearMortonIDs = node->NNNearNodeMortonIDs;
    for ( auto & it : NearMortonIDs )
    {
      auto *target = tree.morton2node[ it ];
      target->NNNearNodes.insert( node );
      target->NNNearNodeMortonIDs.insert( it );
    }
  }
}; /** end SymmetrizeNearInteractions() */


/** @brief Task wrapper for CacheNearNodes(). */
template<bool NNPRUNE, typename NODE>
class CacheNearNodesTask : public Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "c-n" );
      label = to_string( arg->treelist_id );
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
      /** setup the event */
      event.Set( label + name, flops, mops );
    };

    void DependencyAnalysis() { arg->DependOnNoOne( this ); };

    void Execute( Worker* user_worker )
    {
      //printf( "%lu CacheNearNodes beg\n", arg->treelist_id ); fflush( stdout );

      NODE *node = arg;
      auto *NearNodes = &node->NearNodes;
      if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
      auto &K = *node->setup->K;
      auto &data = node->data;
      auto &amap = node->gids;
      vector<size_t> bmap;
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
template<typename NODE>
void FindFarNodes( NODE *node, NODE *target )
{
  /** all assertions, ``target'' must be a leaf node */
  assert( target->isleaf );

  /** get a list of near nodes from target */
  set<NODE*> *NearNodes;
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

  /** If this node contains any Near( target ) or isn't skeletonized */
  if ( !data.is_compressed || node->ContainAny( *NearNodes ) )
  {
    if ( !node->isleaf )
    {
      /** Recurs to two children */
      FindFarNodes( lchild, target );
      FindFarNodes( rchild, target );
    }
  }
  else
  {
    /** Insert ``node'' to Far( target ) */
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

  /** If this node contains any Near( target ) or isn't skeletonized */
  if ( !data.is_compressed || node->ContainAny( *NearNodes ) )
  {
    if ( !node->isleaf )
    {
      /** Recurs to two children */
      FindFarNodes( lchild, target );
      FindFarNodes( rchild, target );
    }
  }
  else
  {
    if ( node->setup->IsSymmetric() && ( node->morton < target->morton ) )
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
template<typename TREE>
void MergeFarNodes( TREE &tree )
{
  for ( int l = tree.depth; l >= 0; l -- )
  {
    size_t n_nodes = ( 1 << l );
    auto level_beg = tree.treelist.begin() + n_nodes - 1;

    for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
    {
      auto *node = *(level_beg + node_ind);

      /** if I don't have any skeleton, then I'm nobody's far field */
      if ( !node->data.is_compressed ) continue;

      if ( node->isleaf )
      {
        FindFarNodes( tree.treelist[ 0 ] /** root */, node );
      }
      else
      {
        /** merge Far( lchild ) and Far( rchild ) from children */
        auto *lchild = node->lchild;
        auto *rchild = node->rchild;

        /** case: !NNPRUNE (HSS specific) */ 
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


        /** case: NNPRUNE (FMM specific) */ 
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

  if ( tree.setup.IsSymmetric() )
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
 *
 */ 
typedef enum
{
  EVALUATE_OPTION_EXACT,
  EVALUATE_OPTION_NEIGHBOR_PRUNING,
  EVALUATE_OPTION_SELF_PRUNING
} evaluateOption_t;

/**
 *  @breif This is a fake evaluation setup aimming to figure out
 *         which tree node will prun which points. The result
 *         will be stored in each node as two lists, prune and noprune.
 *
 */ 
template<typename NODE, typename T>
hmlpError_t Evaluate( NODE *node, const size_t gid, Data<T> & potentials, const vector<size_t> & neighbors )
{
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;
  auto &data = node->data;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  size_t nrhs = w.col();

  assert( potentials.size() == nrhs );

  if ( !data.is_compressed || node->ContainAny( neighbors ) )
  {
    auto   I = vector<size_t>( 1, gid );
    auto & J = node->gids;
    if ( node->isleaf )
    {
      /** I.size()-by-J.size(). */
      auto Kab = K( I, J ); 
      /** All right hand sides [ 0, 1,..., nrhs -1 ]. */
      vector<size_t> bmap( nrhs );
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;
      /** J.size()-by-nrhs */
      auto wb  = w( J, bmap ); 

      xgemm( "No transpose", "No transpose",
        Kab.row(), wb.col(), wb.row(),
        1.0, Kab.data(),        Kab.row(),
              wb.data(),         wb.row(),
        1.0, potentials.data(), potentials.row() );
    }
    else
    {
      RETURN_IF_ERROR( Evaluate( lchild, gid, potentials, neighbors ) );
      RETURN_IF_ERROR( Evaluate( rchild, gid, potentials, neighbors ) );
    }
  }
  else 
  {
    auto   I = vector<size_t>( 1, gid );
    auto & J = node->data.skels;
    auto Kab = K( I, J );
    auto & w_skel = node->data.w_skel;
    xgemm( "No transpose", "No transpose",
      Kab.row(), w_skel.col(), w_skel.row(),
      1.0, Kab.data(),        Kab.row(),
        w_skel.data(),     w_skel.row(),
      1.0, potentials.data(), potentials.row() );          
  }
  return HMLP_ERROR_SUCCESS;
}; /** end Evaluate() */


/** @brief Evaluate potentials( gid ) using treecode.
 *         Notice in this case, the approximation is unsymmetric.
 *
 **/
template<typename TREE, typename T>
hmlpError_t Evaluate( TREE &tree, const size_t gid, Data<T> & potentials, const evaluateOption_t option )
{
  /* Put gid itself into the neighbor list. */
  vector<size_t> neighbors( 1, gid );
  auto &w = *tree.setup.w;
  /* Clean up and properly initialize the output vector. */
  potentials.clear();
  potentials.resize( 1, w.col(), 0.0 );

  switch ( option )
  {
    case EVALUATE_OPTION_NEIGHBOR_PRUNING:
    {
      auto & all_neighbors = *tree.setup.NN;
      /* Get the number of neighbors. */
      size_t kappa = all_neighbors.row();
      neighbors.reserve( kappa + 1 );
      /* Insert all neighbor gids into neighbors. */
      for ( size_t i = 0; i < kappa; i ++ )
      {
        neighbors.push_back( all_neighbors( i, gid ).second );
      }
      return Evaluate( tree.treelist[ 0 ], gid, potentials, neighbors );
    }
    case EVALUATE_OPTION_SELF_PRUNING:
    {
      return Evaluate( tree.treelist[ 0 ], gid, potentials, neighbors );
    }
    case EVALUATE_OPTION_EXACT:
    {
      return HMLP_ERROR_INVALID_VALUE;
    }
    default:
    {
      return HMLP_ERROR_INVALID_VALUE;
    }
  }
  return HMLP_ERROR_INVALID_VALUE;
}; /* end Evaluate() */


/**
 *  @brief ComputeAll
 */ 
template<
  bool     USE_RUNTIME = true, 
  bool     USE_OMP_TASK = false, 
  bool     NNPRUNE = true, 
  bool     CACHE = true, 
  typename TREE, 
  typename T>
Data<T> Evaluate
( 
  TREE &tree,
  Data<T> &weights
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
  if ( tree.setup.IsSymmetric() )
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
    tree.TraverseLeafs( leaftoleafver2task );
#else
    tree.TraverseLeafs( leaftoleaftask1 );
    tree.TraverseLeafs( leaftoleaftask2 );
    tree.TraverseLeafs( leaftoleaftask3 );
    tree.TraverseLeafs( leaftoleaftask4 );
#endif
    tree.TraverseUp( nodetoskeltask );
    tree.TraverseUnOrdered( skeltoskeltask );
    tree.TraverseDown( skeltonodetask );
    tree.ExecuteAllTasks();
    //if ( USE_RUNTIME ) hmlp_run();


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



template<typename SPLITTER, typename T, typename SPDMATRIX>
Data<pair<T, size_t>> FindNeighbors( SPDMATRIX &K, SPLITTER splitter, 
    Configuration<T> &config, size_t n_iter = 20 )
{
  try
  {
    /** Instantiation for the randomisze tree. */
    using DATA  = gofmm::NodeData<T>;
    using SETUP = gofmm::Setup<SPDMATRIX, SPLITTER, T>;
    using TREE  = tree::Tree<SETUP, DATA>;
    /** Derive type NODE from TREE. */
    using NODE  = typename TREE::NODE;
    /** Get all user-defined parameters. */
    DistanceMetric metric = config.MetricType();
    size_t n = config.ProblemSize();
	  size_t k = config.NeighborSize(); 
    /** Iterative all nearnest-neighbor (ANN). */
    pair<T, size_t> init( numeric_limits<T>::max(), n );
    gofmm::NeighborsTask<NODE, T> NEIGHBORStask;
    TREE rkdt;
    rkdt.setup.FromConfiguration( config, K, splitter, NULL );
    Data<pair<T, size_t>> neighbors;
    HANDLE_ERROR( rkdt.AllNearestNeighbor( n_iter, k, n_iter, neighbors, init, NEIGHBORStask ) );
    return neighbors;
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    exit( -1 );
  }
}; /** end FindNeighbors() */


/**
 *  @brielf template of the compress routine
 */ 
template<typename SPLITTER, typename RKDTSPLITTER, typename T, typename SPDMATRIX>
tree::Tree< gofmm::Setup<SPDMATRIX, SPLITTER, T>, gofmm::NodeData<T>>
*Compress( SPDMATRIX &K, Data<pair<T, size_t>> &NN, 
    SPLITTER splitter, RKDTSPLITTER rkdtsplitter, Configuration<T> &config )
{
  try
  {
    /** Get all user-defined parameters. */
    DistanceMetric metric = config.MetricType();
    size_t n = config.ProblemSize();
    size_t m = config.getLeafNodeSize();
    size_t k = config.NeighborSize(); 
    size_t s = config.MaximumRank(); 
    T stol = config.Tolerance();
    T budget = config.Budget(); 

    /** options */
    const bool NNPRUNE   = true;
    const bool CACHE     = true;

    /** instantiation for the Spd-Askit tree */
    using SETUP = gofmm::Setup<SPDMATRIX, SPLITTER, T>;
    using DATA  = gofmm::NodeData<T>;
    using TREE  = tree::Tree<SETUP, DATA>;
    /** Derive type NODE from TREE. */
    using NODE  = typename TREE::NODE;


    /** all timers */
    double beg, omptask45_time, omptask_time, ref_time;
    double time_ratio, compress_time = 0.0, other_time = 0.0;
    double ann_time, tree_time, skel_time, mergefarnodes_time, cachefarnodes_time;
    double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

    /** Iterative all nearnest-neighbor (ANN). */
    beg = omp_get_wtime();
    if ( NN.size() != n * k ) 
    {
      NN = gofmm::FindNeighbors( K, rkdtsplitter, config );
    }
    ann_time = omp_get_wtime() - beg;


    /** Initialize metric ball tree using approximate center split. */
    auto *tree_ptr = new TREE();
    auto &tree = *tree_ptr;
    tree.setup.FromConfiguration( config, K, splitter, &NN );


    if ( REPORT_COMPRESS_STATUS )
    {
      printf( "TreePartitioning ...\n" ); fflush( stdout );
    }
    beg = omp_get_wtime();
    tree.TreePartition();
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




    /** Build near interaction lists. */ 
    NearSamplesTask<NODE, T> NEARSAMPLEStask;
    tree.DependencyCleanUp();
    printf( "Dependency clean up\n" ); fflush( stdout );
    tree.TraverseLeafs( NEARSAMPLEStask );
    tree.ExecuteAllTasks();
    //hmlp_run();
    printf( "Finish NearSamplesTask\n" ); fflush( stdout );
    SymmetrizeNearInteractions( tree );
    printf( "Finish SymmetrizeNearInteractions\n" ); fflush( stdout );



    /** Skeletonization */
    if ( REPORT_COMPRESS_STATUS )
    {
      printf( "Skeletonization (HMLP Runtime) ...\n" ); fflush( stdout );
    }
    beg = omp_get_wtime();
    gofmm::SkeletonKIJTask<NNPRUNE, NODE, T> GETMTXtask;
    gofmm::SkeletonizeTask<NODE, T> SKELtask;
    gofmm::InterpolateTask<NODE> PROJtask;
    tree.DependencyCleanUp();
    tree.TraverseUp( GETMTXtask, SKELtask );
    tree.TraverseUnOrdered( PROJtask );
    if ( CACHE )
    {
      gofmm::CacheNearNodesTask<NNPRUNE, NODE> KIJtask;
      tree.template TraverseLeafs( KIJtask );
    }
    other_time += omp_get_wtime() - beg;
    hmlp_run();
    skel_time = omp_get_wtime() - beg;




    /** MergeFarNodes */
    beg = omp_get_wtime();
    if ( REPORT_COMPRESS_STATUS )
    {
      printf( "MergeFarNodes ...\n" ); fflush( stdout );
    }
    gofmm::MergeFarNodes( tree );
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
      printf( "Skeletonization ----------------------- %5.2lfs (%5.1lf%%)\n", skel_time, skel_time * time_ratio );
      printf( "MergeFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", mergefarnodes_time, mergefarnodes_time * time_ratio );
      printf( "CacheFarNodes ------------------------- %5.2lfs (%5.1lf%%)\n", cachefarnodes_time, cachefarnodes_time * time_ratio );
      printf( "========================================================\n");
      printf( "Compress (%4.2lf not compressed) -------- %5.2lfs (%5.1lf%%)\n", 
          exact_ratio, compress_time, compress_time * time_ratio );
      printf( "========================================================\n\n");
    }

    /** Clean up all r/w dependencies left on tree nodes. */
    tree_ptr->DependencyCleanUp();
    /** Return the hierarhical compreesion of K as a binary tree. */
    return tree_ptr;
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    exit( -1 );
  }
}; /** end Compress() */









































/**
 *  @brielf A simple template for the compress routine.
 */ 
template<typename T, typename SPDMATRIX>
tree::Tree<
  gofmm::Setup<SPDMATRIX, centersplit<SPDMATRIX, 2, T>, T>, 
  gofmm::NodeData<T>>
*Compress( SPDMATRIX &K, T stol, T budget, size_t m, size_t k, size_t s )
{
  using SPLITTER     = centersplit<SPDMATRIX, 2, T>;
  using RKDTSPLITTER = randomsplit<SPDMATRIX, 2, T>;
  Data<pair<T, size_t>> NN;
	/** GOFMM tree splitter */
  SPLITTER splitter( K );
  splitter.Kptr = &K;
	splitter.metric = ANGLE_DISTANCE;
	/** randomized tree splitter */
  RKDTSPLITTER rkdtsplitter( K );
  rkdtsplitter.Kptr = &K;
	rkdtsplitter.metric = ANGLE_DISTANCE;
  size_t n = K.row();

	/** creatgin configuration for all user-define arguments */
	Configuration<T> config( ANGLE_DISTANCE, n, m, k, s, stol, budget );

	/** call the complete interface and return tree_ptr */
  return Compress<SPLITTER, RKDTSPLITTER>
         ( K, NN, //ANGLE_DISTANCE, 
					 splitter, rkdtsplitter, //n, m, k, s, stol, budget, 
					 config );
}; /** end Compress */








/**
 *  @brielf A simple template for the compress routine.
 */ 
template<typename T, typename SPDMATRIX>
tree::Tree<
  gofmm::Setup<SPDMATRIX, centersplit<SPDMATRIX, 2, T>, T>, 
  gofmm::NodeData<T>>
*Compress( SPDMATRIX &K, T stol, T budget )
{
  using SPLITTER     = centersplit<SPDMATRIX, 2, T>;
  using RKDTSPLITTER = randomsplit<SPDMATRIX, 2, T>;
  Data<pair<T, std::size_t>> NN;
	/** GOFMM tree splitter */
  SPLITTER splitter( K );
  splitter.Kptr = &K;
  splitter.metric = ANGLE_DISTANCE;
	/** randomized tree splitter */
  RKDTSPLITTER rkdtsplitter( K );
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
  return Compress<SPLITTER, RKDTSPLITTER>
         ( K, NN, //ANGLE_DISTANCE, 
					 splitter, rkdtsplitter, config );

}; /** end Compress() */

/**
 *
 */ 
template<typename T>
tree::Tree<
  gofmm::Setup<SPDMatrix<T>, centersplit<SPDMatrix<T>, 2, T>, T>, 
  gofmm::NodeData<T>>
*Compress( SPDMatrix<T> &K, T stol, T budget )
{
	return Compress<T, SPDMatrix<T>>( K, stol, budget );
}; /** end Compress() */







template<typename NODE, typename T>
void ComputeError( NODE *node, Data<T> potentials )
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
T ComputeError( TREE &tree, size_t gid, Data<T> potentials )
{
  auto &K = *tree.setup.K;
  auto &w = *tree.setup.w;

  auto amap = std::vector<size_t>( 1, gid );
  auto bmap = std::vector<size_t>( K.col() );
  for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

  auto Kab = K( amap, bmap );
  auto exact = potentials;

  xgemm( "No transpose", "No transpose", 
      Kab.row(), w.col(), w.row(),
    1.0,   Kab.data(),   Kab.row(),
             w.data(),     w.row(),
    0.0, exact.data(), exact.row() );          


  auto nrm2 = hmlp_norm( exact.row(),  exact.col(), exact.data(), exact.row() ); 

  xgemm( "No transpose", "No transpose",
    Kab.row(), w.col(), w.row(),
    -1.0, Kab.data(),       Kab.row(),
          w.data(),          w.row(),
     1.0, potentials.data(), potentials.row() );          

  auto err = hmlp_norm( potentials.row(), potentials.col(), potentials.data(), potentials.row() ); 

  //return err / nrm2;
  return relative_error_from_rse_and_nrm( err, nrm2 );
}; /* end ComputeError() */



template<typename TREE>
hmlpError_t SelfTesting( TREE &tree, size_t ntest, size_t nrhs )
{
  /** Derive type T from TREE. */
  using T = typename TREE::T;
  /** Size of right hand sides. */
  size_t n = tree.n;
  /** Shrink ntest if ntest > n. */
  if ( ntest > n ) ntest = n;
  /** all_rhs = [ 0, 1, ..., nrhs - 1 ]. */
  vector<size_t> all_rhs( nrhs );
  for ( size_t rhs = 0; rhs < nrhs; rhs ++ ) all_rhs[ rhs ] = rhs;

  //auto A = tree.CheckAllInteractions();

  /** Evaluate u ~ K * w. */
  Data<T> w( n, nrhs ); w.rand();
  auto u = Evaluate<true, false, true, true>( tree, w );

  /** Examine accuracy with 3 setups, ASKIT, HODLR, and GOFMM. */
  T nnerr_avg = 0.0;
  T nonnerr_avg = 0.0;
  T fmmerr_avg = 0.0;
  printf( "========================================================\n");
  printf( "Accuracy report\n" );
  printf( "========================================================\n");
  for ( size_t i = 0; i < ntest; i ++ )
  {
    //size_t tar = i * n / ntest;
    size_t tar = i * 1000;
    Data<T> potentials;
    /** ASKIT treecode with NN pruning. */
    Evaluate( tree, tar, potentials, EVALUATE_OPTION_NEIGHBOR_PRUNING );
    auto nnerr = ComputeError( tree, tar, potentials );
    /** ASKIT treecode without NN pruning. */
    Evaluate( tree, tar, potentials, EVALUATE_OPTION_SELF_PRUNING );
    auto nonnerr = ComputeError( tree, tar, potentials );
    /** Get results from GOFMM */
    //potentials = u( vector<size_t>( i ), all_rhs );
    for ( size_t p = 0; p < potentials.col(); p ++ )
    {
      potentials[ p ] = u( tar, p );
    }
    auto fmmerr = ComputeError( tree, tar, potentials );

    /** Only print 10 values. */
    if ( i < 10 )
    {
      printf( "gid %6lu, ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
          tar, nnerr, nonnerr, fmmerr );
    }
    nnerr_avg += nnerr;
    nonnerr_avg += nonnerr;
    fmmerr_avg += fmmerr;
  }
  printf( "========================================================\n");
  printf( "            ASKIT %3.1E, HODLR %3.1E, GOFMM %3.1E\n", 
      nnerr_avg / ntest , nonnerr_avg / ntest, fmmerr_avg / ntest );
  printf( "========================================================\n");

  if ( !tree.setup.SecureAccuracy() )
  {
    /** Factorization */
    T lambda = 5.0;
    RETURN_IF_ERROR( gofmm::Factorize( tree, lambda ) );
    /** Compute error. */
    gofmm::ComputeError( tree, lambda, w, u );
  }

  return HMLP_ERROR_SUCCESS;
}; /** end SelfTesting() */


/** @brief Instantiate the splitters here. */ 
template<typename SPDMATRIX>
void LaunchHelper( SPDMATRIX &K, CommandLineHelper &cmd )
{
  using T = typename SPDMATRIX::T;

  const int N_CHILDREN = 2;
  /** Use geometric-oblivious splitters. */
  using SPLITTER     = gofmm::centersplit<SPDMATRIX, N_CHILDREN, T>;
  using RKDTSPLITTER = gofmm::randomsplit<SPDMATRIX, N_CHILDREN, T>;
  /** GOFMM tree splitter. */
  SPLITTER splitter( K );
  splitter.Kptr = &K;
  splitter.metric = cmd.metric;
  /** Randomized tree splitter. */
  RKDTSPLITTER rkdtsplitter( K );
  rkdtsplitter.Kptr = &K;
  rkdtsplitter.metric = cmd.metric;
	/** Create configuration for all user-define arguments. */
  gofmm::Configuration<T> config( cmd.metric, 
      cmd.n, cmd.m, cmd.k, cmd.s, cmd.stol, cmd.budget, cmd.secure_accuracy );
  /** (Optional) provide neighbors, leave uninitialized otherwise. */
  Data<pair<T, size_t>> NN;
  /** Compress K. */
  //auto *tree_ptr = gofmm::Compress( X, K, NN, splitter, rkdtsplitter, config );
  auto *tree_ptr = gofmm::Compress( K, NN, splitter, rkdtsplitter, config );
	auto &tree = *tree_ptr;
  /** Examine accuracies. */
  auto error = gofmm::SelfTesting( tree, 100, cmd.nrhs );


//  //#ifdef DUMP_ANALYSIS_DATA
//  gofmm::Summary<NODE> summary;
//  tree.Summary( summary );
//  summary.Print();

	/** delete tree_ptr */
  delete tree_ptr;
}; /** end LaunchHelper() */






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

    void Multiply( Data<T> &y, Data<T> &x )
    {
      //hmlp::Data<T> weights( x.col(), x.row() );

      //for ( size_t j = 0; j < x.col(); j ++ )
      //  for ( size_t i = 0; i < x.row(); i ++ )
      //    weights( j, i ) = x( i, j );


      y = gofmm::Evaluate( *tree_ptr, x );
      //auto potentials = hmlp::gofmm::Evaluate( *tree_ptr, weights );

      //for ( size_t j = 0; j < y.col(); j ++ )
      //  for ( size_t i = 0; i < y.row(); i ++ )
      //    y( i, j ) = potentials( j, i );

    };

  private:

    /** GOFMM tree */
    tree::Tree<
      gofmm::Setup<SPDMATRIX, centersplit<SPDMATRIX, 2, T>, T>, 
      gofmm::NodeData<T>> *tree_ptr = NULL; 

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

typedef tree::Tree<dSetup_t, gofmm::NodeData<double>> dTree_t;
typedef tree::Tree<sSetup_t, gofmm::NodeData<float >> sTree_t;





/** 
 *  PyCompress prototype. Notice that all pass-by-reference
 *  arguments are replaced by pass-by-pointer. There implementaion
 *  can be found at hmlp/package/$HMLP_ARCH/gofmm.gpp
 **/
Data<double> Evaluate( dTree_t *tree, Data<double> *weights );
Data<float>  Evaluate( dTree_t *tree, Data<float > *weights );

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
