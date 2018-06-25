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


#ifndef MPITREE_HPP
#define MPITREE_HPP

#include <algorithm>
#include <functional>
#include <type_traits>
#include <cstdint>


#include <hmlp_mpi.hpp>
#include <containers/DistData.hpp>

#include <tree.hpp>

using namespace std;
using namespace hmlp;


namespace hmlp
{
namespace mpitree
{


//template<size_t LEVELOFFSET=4>
//int Morton2Rank( size_t it, mpi::Comm comm )
//{
//  /**
//   *  MPI communicator size
//   */ 
//  int size = 0;
//  mpi::Comm_size( &size, comm );
//
//  size_t filter = ( 1 << LEVELOFFSET ) - 1;
//  size_t itlevel = it & filter;
//  size_t mpilevel = 0;
//
//  size_t tmp = size;
//  while ( tmp )
//  {
//    mpilevel ++;
//    tmp /= 2;
//  }
//
//  if ( ( 1 << itlevel ) > size ) itlevel = mpilevel;
//
//  return it >> ( ( 1 << LEVELOFFSET ) - itlevel + LEVELOFFSET );
//
//}; /** end Morton2rank() */







/**
 *  @brief This is the default ball tree splitter. Given coordinates,
 *         compute the direction from the two most far away points.
 *         Project all points to this line and split into two groups
 *         using a median select.
 *
 *  @para
 *
 *  @TODO  Need to explit the parallelism.
 */ 
template<int N_SPLIT, typename T>
struct centersplit
{
  // closure
  Data<T> *Coordinate;

  inline vector<vector<size_t> > operator()
  ( 
    vector<size_t>& gids
  ) const 
  {
    assert( N_SPLIT == 2 );

    Data<T> &X = *Coordinate;
    size_t d = X.row();
    size_t n = gids.size();

    T rcx0 = 0.0, rx01 = 0.0;
    size_t x0, x1;
    vector<vector<size_t> > split( N_SPLIT );


    vector<T> centroid = combinatorics::Mean( d, n, X, gids );
    vector<T> direction( d );
    vector<T> projection( n, 0.0 );

    //printf( "After Mean\n" );

    // Compute the farest x0 point from the centroid
    for ( int i = 0; i < n; i ++ )
    {
      T rcx = 0.0;
      for ( int p = 0; p < d; p ++ )
      {
        T tmp = X[ gids[ i ] * d + p ] - centroid[ p ];
        rcx += tmp * tmp;
      }
      //printf( "\n" );
      if ( rcx > rcx0 ) 
      {
        rcx0 = rcx;
        x0 = i;
      }
    }


    // Compute the farest point x1 from x0
    for ( int i = 0; i < n; i ++ )
    {
      T rxx = 0.0;
      for ( int p = 0; p < d; p ++ )
      {
        T tmp = X[ gids[ i ] * d + p ] - X[ gids[ x0 ] * d + p ];
        rxx += tmp * tmp;
      }
      if ( rxx > rx01 )
      {
        rx01 = rxx;
        x1 = i;
      }
    }

    // Compute direction
    for ( int p = 0; p < d; p ++ )
    {
      direction[ p ] = X[ gids[ x1 ] * d + p ] - X[ gids[ x0 ] * d + p ];
    }

    // Compute projection
    projection.resize( n, 0.0 );
    for ( int i = 0; i < n; i ++ )
      for ( int p = 0; p < d; p ++ )
        projection[ i ] += X[ gids[ i ] * d + p ] * direction[ p ];

    /** Parallel median search */
    T median;
    
    if ( 1 )
    {
      median = hmlp::combinatorics::Select( n, n / 2, projection );
    }
    else
    {
      auto proj_copy = projection;
      std::sort( proj_copy.begin(), proj_copy.end() );
      median = proj_copy[ n / 2 ];
    }

    split[ 0 ].reserve( n / 2 + 1 );
    split[ 1 ].reserve( n / 2 + 1 );


    /** TODO: Can be parallelized */
    std::vector<std::size_t> middle;
    for ( size_t i = 0; i < n; i ++ )
    {
      if      ( projection[ i ] < median ) split[ 0 ].push_back( i );
      else if ( projection[ i ] > median ) split[ 1 ].push_back( i );
      else                                 middle.push_back( i );
    }

    for ( size_t i = 0; i < middle.size(); i ++ )
    {
      if ( split[ 0 ].size() <= split[ 1 ].size() ) split[ 0 ].push_back( middle[ i ] );
      else                                          split[ 1 ].push_back( middle[ i ] );
    }


    return split;
  };


  inline std::vector<std::vector<size_t> > operator()
  ( 
    std::vector<size_t>& gids,
    hmlp::mpi::Comm comm
  ) const 
  {
    std::vector<std::vector<size_t> > split( N_SPLIT );

    return split;
  };

};





template<int N_SPLIT, typename T>
struct randomsplit
{
  Data<T> *Coordinate = NULL;

  inline vector<vector<size_t> > operator() ( vector<size_t>& gids ) const 
  {
    vector<vector<size_t> > split( N_SPLIT );
    return split;
  };

  inline vector<vector<size_t> > operator() ( vector<size_t>& gids, mpi::Comm comm ) const 
  {
    vector<vector<size_t> > split( N_SPLIT );
    return split;
  };
};



template<typename T>
bool less_first
(
  const pair<T, size_t> &a, 
  const pair<T, size_t> &b
)
{
  return ( a.first < b.first );
};

template<typename T>
bool less_second
( 
  const pair<T, size_t> &a, 
  const pair<T, size_t> &b 
)
{
  return ( a.second < b.second );
};


template<typename T>
bool equal_second 
( 
  const pair<T, size_t> &a, 
  const pair<T, size_t> &b 
)
{
  return ( a.second == b.second );
};


/**
 *  @biref Merge a single neighbor list B into A using auxulary space.
 */ 
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
void MergeNeighbors
( 
  size_t k,
  pair<T, size_t> *A, 
  pair<T, size_t> *B,
  vector<pair<T, size_t>, Allocator> &aux
)
{
  if ( aux.size() != 2 * k ) aux.resize( 2 * k );

  for ( size_t i = 0; i < k; i++ ) aux[     i ] = A[ i ];
  for ( size_t i = 0; i < k; i++ ) aux[ k + i ] = B[ i ];

  sort( aux.begin(), aux.end(), less_second<T> );
  auto it = unique( aux.begin(), aux.end(), equal_second<T> );
  sort( aux.begin(), it, less_first<T> );

  for ( size_t i = 0; i < k; i++ ) A[ i ] = aux[ i ];
};


/**
 *  @biref Merge neighbor lists B into A.
 */ 
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
void MergeNeighbors
( 
  size_t k, size_t n,
  vector<pair<T, size_t>, Allocator> &A, 
  vector<pair<T, size_t>, Allocator> &B
)
{
  assert( A.size() >= n * k && B.size() >= n * k );
	#pragma omp parallel
  {
    vector<pair<T, size_t> > aux( 2 * k );
    #pragma omp for
    for( size_t i = 0; i < n; i++ ) 
    {
      MergeNeighbors( k, &(A[ i * k ]), &(B[ i * k ]), aux );
    }
  }
}; /** */













template<typename NODE>
class DistSplitTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "DistSplit" );
      label = to_string( arg->treelist_id );

      double flops = 6.0 * arg->n;
      double  mops = 6.0 * arg->n;

      /** Setup the event */
      event.Set( label + name, flops, mops );
      /** Asuume computation bound */
      cost = mops / 1E+9;
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
					assert( arg->child );
          arg->child->DependencyAnalysis( RW, this );
        }
        else
        {
					assert( arg->lchild && arg->rchild );
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
      //printf( "rank %d level %lu DistSplit begin\n", global_rank, arg->l ); fflush( stdout );
			double beg = omp_get_wtime();
      arg->Split();
			double split_t = omp_get_wtime() - beg;;
      //printf( "rank %d level %lu DistSplit   end %lfs\n", 
			//		global_rank, arg->l, split_t ); fflush( stdout );
    };

}; /** end class DistSplitTask */




/**
 *  @brief Data and setup that are shared with all nodes.
 */ 
template<typename SPLITTER, typename DATATYPE>
class Setup
{
  public:

    typedef DATATYPE T;

    Setup() {};

    ~Setup() {};




    /**
     *  @brief Check if this node contain any query using morton.
		 *         Notice that queries[] contains gids; thus, morton[]
		 *         needs to be accessed using gids.
     *
     */ 
		std::vector<size_t> ContainAny( std::vector<size_t> &queries, size_t target )
    {
			std::vector<size_t> validation( queries.size(), 0 );

      if ( !morton.size() )
      {
        printf( "Morton id was not initialized.\n" );
        exit( 1 );
      }

      for ( size_t i = 0; i < queries.size(); i ++ )
      {
				/** notice that setup->morton only contains local morton ids */
        //auto it = this->setup->morton.find( queries[ i ] );

				//if ( it != this->setup->morton.end() )
				//{
        //  if ( tree::IsMyParent( *it, this->morton ) ) validation[ i ] = 1;
				//}


       if ( tree::IsMyParent( morton[ queries[ i ] ], target ) ) 
				 validation[ i ] = 1;

      }
      return validation;

    }; /** end ContainAny() */




    /** maximum leaf node size */
    size_t m;
    
    /** by default we use 4 bits = 0-15 levels */
    size_t max_depth = 15;

    /** coordinates (accessed with gids) */
    DistData<STAR, CBLK, T> *X_cblk = NULL;
    DistData<STAR, CIDS, T> *X      = NULL;

    /** neighbors<distance, gid> (accessed with gids) */
    DistData<STAR, CBLK, std::pair<T, std::size_t>> *NN_cblk = NULL;
    DistData<STAR, CIDS, std::pair<T, std::size_t>> *NN      = NULL;

    /** morton ids */
    std::vector<size_t> morton;

    /** tree splitter */
    SPLITTER splitter;

}; /** end class Setup */



template<typename NODE>
class DistIndexPermuteTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      name = std::string( "Permutation" );
      arg = user_arg;
      // Need an accurate cost model.
      cost = 1.0;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      if ( !arg->isleaf && !arg->child )
      {
        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      }
      this->TryEnqueue();
    };


    void Execute( Worker* user_worker )
    {
      if ( !arg->isleaf && !arg->child )
      {
        auto &gids = arg->gids; 
        auto &lgids = arg->lchild->gids;
        auto &rgids = arg->rchild->gids;
        gids = lgids;
        gids.insert( gids.end(), rgids.begin(), rgids.end() );
      }
    };

}; /** end class IndexPermuteTask */
















/**
 *
 */ 
//template<typename SETUP, int N_CHILDREN, typename NODEDATA>
template<typename SETUP, typename NODEDATA>
class Node : public tree::Node<SETUP, NODEDATA>
{
  public:

    /** Deduce data type from SETUP. */
    typedef typename SETUP::T T;

    static const int N_CHILDREN = 2;

    /** Inherit all parameters from tree::Node */


    /** Constructor for inner node (gids and n unassigned) */
    Node( SETUP *setup, size_t n, size_t l, 
        Node *parent,
        unordered_map<size_t, tree::Node<SETUP, NODEDATA>*> *morton2node,
        Lock *treelock, mpi::Comm comm ) 
    /** Inherits the constructor from tree::Node */
      : tree::Node<SETUP, NODEDATA>( setup, n, l, 
          parent, morton2node, treelock ) 
    {
      /** local communicator */
      this->comm = comm;
      
      /** get size and rank */
      mpi::Comm_size( comm, &size );
      mpi::Comm_rank( comm, &rank );
    };



    /** 
     *  Constructor for root 
     */
    Node( SETUP *setup, size_t n, size_t l, vector<size_t> &gids, 
        Node *parent, 
        unordered_map<size_t, tree::Node<SETUP, NODEDATA>*> *morton2node,
        Lock *treelock, mpi::Comm comm ) 
    /** 
     *  Inherits the constructor from tree::Node
     */
      : Node<SETUP, NODEDATA>( setup, n, l, parent, 
          morton2node, treelock, comm ) 
    {
      /** 
       *  Notice that "gids.size() < n" 
       */
      this->gids = gids;
    };


    Node( size_t morton ) : tree::Node<SETUP, NODEDATA>( morton )
    { 
    };


    void SetupChild( class Node *child )
    {
      this->kids[ 0 ] = child;
      this->child = child;
    };



    /**
     *  TODO: need to redistribute the parameters of K as well
     */ 
    void Split()
    {
      assert( N_CHILDREN == 2 );

      /** Reduce to get the total size of gids */
      int num_points_total = 0;
      int num_points_owned = (this->gids).size();

      /** n = sum( num_points_owned ) over all MPI processes in comm */
      mpi::Allreduce( &num_points_owned, &num_points_total, 1, MPI_SUM, comm );
      this->n = num_points_total;







      //printf( "rank %d n %lu gids.size() %lu --", 
      //    rank, this->n, this->gids.size() ); fflush( stdout );
      //for ( size_t i = 0; i < this->gids.size(); i ++ )
      //  printf( "%lu ", this->gids[ i ] );
      //printf( "\n" ); fflush( stdout );





      if ( child )
      {
				//printf( "Split(): n %lu  \n", this->n ); fflush( stdout );

        /** The local communicator of this node contains at least 2 processes. */
        assert( size > 1 );

        /** Distributed splitter */
        auto split = this->setup->splitter( this->gids, comm );

				//printf( "Finish Split(): n %lu  \n", this->n ); fflush( stdout );


        /** Get partner rank */
        int partner_rank = 0;
        int sent_size = 0; 
        int recv_size = 0;
        vector<size_t> &kept_gids = child->gids;
        vector<int>     sent_gids;
        vector<int>     recv_gids;

        if ( rank < size / 2 ) 
        {
          /** left child */
          partner_rank = rank + size / 2;
          /** MPI ranks 0:size/2-1 keep split[ 0 ] */
          kept_gids.resize( split[ 0 ].size() );
          for ( size_t i = 0; i < kept_gids.size(); i ++ )
            kept_gids[ i ] = this->gids[ split[ 0 ][ i ] ];
          /** MPI ranks 0:size/2-1 send split[ 1 ] */
          sent_gids.resize( split[ 1 ].size() );
          sent_size = sent_gids.size();
          for ( size_t i = 0; i < sent_gids.size(); i ++ )
            sent_gids[ i ] = this->gids[ split[ 1 ][ i ] ];
        }
        else       
        {
          /** right child */
          partner_rank = rank - size / 2;
          /** MPI ranks size/2:size-1 keep split[ 1 ] */
          kept_gids.resize( split[ 1 ].size() );
          for ( size_t i = 0; i < kept_gids.size(); i ++ )
            kept_gids[ i ] = this->gids[ split[ 1 ][ i ] ];
          /** MPI ranks size/2:size-1 send split[ 0 ] */
          sent_gids.resize( split[ 0 ].size() );
          sent_size = sent_gids.size();
          for ( size_t i = 0; i < sent_gids.size(); i ++ )
            sent_gids[ i ] = this->gids[ split[ 0 ][ i ] ];
        }
        assert( partner_rank >= 0 );

        //printf( "rank %d partner_arnk %d\n", rank, partner_rank ); fflush( stdout );


        /** exchange recv_gids.size() */
        mpi::Sendrecv( 
            &sent_size, 1, MPI_INT, partner_rank, 10,
            &recv_size, 1, MPI_INT, partner_rank, 10,
            comm, &status );

        printf( "rank %d kept_size %lu sent_size %d recv_size %d\n", 
            rank, kept_gids.size(), sent_size, recv_size ); fflush( stdout );

        /** resize recv_gids */
        recv_gids.resize( recv_size );

        /** Exchange recv_gids.size() */
        mpi::Sendrecv( 
            sent_gids.data(), sent_size, MPI_INT, partner_rank, 20,
            recv_gids.data(), recv_size, MPI_INT, partner_rank, 20,
            comm, &status );

        /** Enlarge kept_gids */
        kept_gids.reserve( kept_gids.size() + recv_gids.size() );
        for ( size_t i = 0; i < recv_gids.size(); i ++ )
          kept_gids.push_back( recv_gids[ i ] );
        

      }
			else
			{
				//printf( "Split(): n %lu  \n", this->n ); fflush( stdout );

        tree::Node<SETUP, NODEDATA>::Split<true>( 0 );

		  } /** end if ( child ) */
      
      /** Synchronize within local communicator */
      mpi::Barrier( comm );

    }; /** end Split() */



    mpi::Comm GetComm() { return comm; };

    int GetCommSize() { return size; };
    
    int GetCommRank() { return rank; };

    /** Support dependency analysis. */
    void DependOnChildren( Task *task )
    {
      this->DependencyAnalysis( RW, task );
      if ( size < 2 )
      {
        if ( this->lchild ) this->lchild->DependencyAnalysis( R, task );
        if ( this->rchild ) this->rchild->DependencyAnalysis( R, task );
      }
      else
      {
        if ( child ) child->DependencyAnalysis( R, task );
      }
      /** Try to enqueue if there is no dependency. */
      task->TryEnqueue();
    };

    void DependOnParent( Task *task )
    {
      this->DependencyAnalysis( R, task );
      if ( size < 2 )
      {
        if ( this->lchild ) this->lchild->DependencyAnalysis( RW, task );
        if ( this->rchild ) this->rchild->DependencyAnalysis( RW, task );
      }
      else
      {
        if ( child ) child->DependencyAnalysis( RW, task );
      }
      /** Try to enqueue if there is no dependency. */
      task->TryEnqueue();
    };

    void Print()
    {
      int global_rank = 0;
      mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
      //printf( "grank %d l %lu lrank %d offset %lu n %lu\n", 
      //    global_rank, this->l, this->rank, this->offset, this->n );
      //hmlp_print_binary( this->morton );
    };

    Node *child = NULL;

  private:

    /** initialize with all processes */
    mpi::Comm comm = MPI_COMM_WORLD;

    /** mpi status */
    mpi::Status status;

    /** subcommunicator size */
    int size = 1;

    /** subcommunicator rank */
    int rank = 0;

}; /** end class Node */






/**
 *
 */ 
template<typename SETUP, typename NODEDATA>
class LetNode : public tree::Node<SETUP, NODEDATA>
{
  public:

    /** inherit all parameters from hmlp::tree::Node */

    LetNode( SETUP *setup, size_t morton )
      : tree::Node<SETUP, NODEDATA>
        ( setup, (size_t)0, (size_t)1, NULL, NULL, NULL ) 
    {
      this->morton = morton;
    };

  private:


}; /** end class LetNode */













/**
 *  @brief This distributed tree inherits the shared memory tree
 *         with some additional MPI data structure and function call.
 */ 
template<class SETUP, class NODEDATA>
class Tree 
/**
 *  Inherits from tree::Tree
 */ 
: public tree::Tree<SETUP, NODEDATA>
{
  public:

    typedef typename SETUP::T T;

    /** 
     *  Inherit parameters n, m, and depth; local treelists and morton2node map.
     *
     *  Explanation for the morton2node map in the distributed tree:
     *
     *  morton2node has type map<size_t, tree::Node>, but it actually contains
     *  "three" different kinds of tree nodes.
     *
     *  1. Local tree nodes (exactly in type tree::Node)
     *
     *  2. Distributed tree nodes (in type mpitree::Node)
     *
     *  3. Local essential nodes (in type tree::Node with essential data)
     *
     */

    /** 
     *  Define local tree node type as NODE. Notice that all pointers in the
     *  interaction lists and morton2node map will be in this type.
     */
    typedef tree::Node<SETUP, NODEDATA> NODE;

    /** 
     *  Define distributed tree node type as MPINODE.
     */
    typedef Node<SETUP, NODEDATA> MPINODE;

    /** define our tree node type as NODE */
    //typedef LetNode<SETUP, NODEDATA> LETNODE;


    /** 
     *  Distribued tree nodes in the top-down order. Notice thay
     *  mpitreelist.back() is the root of the local tree.
     *
     *  i.e. mpitrelist.back() == treelist.front();
     */
    vector<MPINODE*> mpitreelists;

    /** local essential tree nodes (this is not thread-safe) */
    //std::map<size_t, LETNODE*> lettreelist;


    /** Default constructor */ 
    Tree() 
    /** Inherit constructor */
    : tree::Tree<SETUP, NODEDATA>::Tree()
    {
			/** Create a new comm_world for */
      mpi::Comm_dup( MPI_COMM_WORLD, &comm );
			/** Get size and rank */
      mpi::Comm_size( comm, &size );
      mpi::Comm_rank( comm, &rank );
      /** Create a ReadWrite object per rank */
      NearRecvFrom.resize( size );
      FarRecvFrom.resize( size );
    };

    /** 
     *  Destructor 
     */
    ~Tree()
    {
      //printf( "~Tree() distributed, mpitreelists.size() %lu\n",
      //    mpitreelists.size() ); fflush( stdout );
      /** 
       *  we do not free the last tree node, it will be deleted by 
       *  hmlp::tree::Tree()::~Tree() 
       */
      if ( mpitreelists.size() )
      {
        for ( size_t i = 0; i < mpitreelists.size() - 1; i ++ )
          if ( mpitreelists[ i ] ) delete mpitreelists[ i ];
        mpitreelists.clear();
      }
      //printf( "end ~Tree() distributed\n" ); fflush( stdout );
    };


    /**
     *  @brief free all tree nodes including local tree nodes,
     *         distributed tree nodes and let nodes
     */ 
    void CleanUp()
    {
      /** free all local tree nodes */
      if ( this->treelist.size() )
      {
        for ( size_t i = 0; i < this->treelist.size(); i ++ )
          if ( this->treelist[ i ] ) delete this->treelist[ i ];
      }
      this->treelist.clear();

      /** free all distributed tree nodes */
      if ( mpitreelists.size() )
      {
        for ( size_t i = 0; i < mpitreelists.size() - 1; i ++ )
          if ( mpitreelists[ i ] ) delete mpitreelists[ i ];
      }
      mpitreelists.clear();

      ///** free all local essential tree nodes */
      //if ( lettreelist.size() )
      //{
      //  for ( size_t i = 0; i < lettreelist.size(); i ++ )
      //    if ( lettreelist[ i ] ) delete lettreelist[ i ];
      //}
      //lettreelist.clear();

    }; /** end CleanUp() */

	







    /** 
     *  @breif allocate distributed tree node
     *
     * */
    void AllocateNodes( vector<size_t> &gids )
    {
      /** decide the number of distributed tree level according to mpi size */
      auto mycomm  = comm;
      int mysize  = size;
      int myrank  = rank;
      int mycolor = 0;
      size_t mylevel = 0;


			/** TODO: gids should be initialized as XBLK */



      /** root( setup, n = 0, l = 0, parent = NULL ) */
      auto *root = new MPINODE( &(this->setup), 
          this->n, mylevel, gids, NULL, 
          &(this->morton2node), &(this->lock), mycomm );

      /** push root to the mpi treelist */
      mpitreelists.push_back( root );

      while ( mysize > 1 )
      {
        mpi::Comm childcomm;

        /** increase level */
        mylevel += 1;

        /** left color = 0, right color = 1 */
        mycolor = ( myrank < mysize / 2 ) ? 0 : 1;

        //printf( "size %d rank %d color %d level %d\n",
        //    mysize, myrank, mycolor, mylevel ); fflush( stdout );

        /** get the subcommunicator for children */
        ierr = mpi::Comm_split( mycomm, mycolor, myrank, &(childcomm) );

        //printf( "size %d rank %d color %d level %d end\n",
        //    mysize, myrank, mycolor, mylevel ); fflush( stdout );

        /** update mycomm */
        mycomm = childcomm;

        /** Create the child */
        auto *parent = mpitreelists.back();
        auto *child  = new MPINODE( &(this->setup), 
            (size_t)0, mylevel, parent, 
            &(this->morton2node), &(this->lock), mycomm );

        /** Create the sibling in type NODE but not MPINODE */
        child->sibling = new NODE( (size_t)0 ); // Node morton is computed later.

        //printf( "size %d rank %d color %d level %d here\n",
        //    mysize, myrank, mycolor, mylevel ); fflush( stdout );

        /** setup parent's children */
        parent->SetupChild( child );
       
        //printf( "size %d rank %d color %d level %d setupchild\n",
        //    mysize, myrank, mycolor, mylevel ); fflush( stdout );

        /** push to the mpi treelist */
        mpitreelists.push_back( child );

        /** update communicator size */
        mpi::Comm_size( mycomm, &mysize );

        /** update myrank in the subcommunicator */
        mpi::Comm_rank( mycomm, &myrank );
      }
      /** synchronize */
      mpi::Barrier( comm );

			/** allocate local tree nodes */
      auto *local_tree_root = mpitreelists.back();

			//printf( "" );


      tree::Tree<SETUP, NODEDATA>::AllocateNodes( 
          local_tree_root );

    };




    vector<size_t> GetPermutation()
    {
      vector<size_t> perm_loc, perm_glb;
      perm_loc = tree::Tree<SETUP, NODEDATA>::GetPermutation();
      mpi::GatherVector( perm_loc, perm_glb, 0, comm );

      //if ( rank == 0 )
      //{
      //  /** Sanity check using an 0:N-1 table. */
      //  vector<bool> Table( this->n, false );
      //  for ( size_t i = 0; i < perm_glb.size(); i ++ )
      //    Table[ perm_glb[ i ] ] = true;
      //  for ( size_t i = 0; i < Table.size(); i ++ ) assert( Table[ i ] );
      //}

      return perm_glb;
    }; /** end GetTreePermutation() */





    /** */
    template<bool SORTED, typename KNNTASK>
    DistData<STAR, CBLK, pair<T, size_t>> AllNearestNeighbor
    (
      size_t n_tree, size_t n, size_t k, size_t max_depth,
      pair<T, size_t> initNN,
      KNNTASK &dummy
    )
    {
      /** Get the problem size from setup->K->row() */
      this->n = n;
      /** k-by-N, column major */
      DistData<STAR, CBLK, pair<T, size_t>> NN( k, n, initNN, comm );
      /** Use leaf size = 4 * k  */
      this->setup.m = 4 * k;
      if ( this->setup.m < 512 ) this->setup.m = 512;
      this->m = this->setup.m;
      /** Local problem size (assuming Round-Robin) */
      num_points_owned = ( n - 1 ) / size + 1;

      /** Edge case */
      if ( n % size )
      {
        if ( rank >= ( n % size ) ) num_points_owned -= 1;
      }

      /** Initial gids distribution (asssuming Round-Robin) */
      vector<size_t> gids( num_points_owned, 0 );
      for ( size_t i = 0; i < num_points_owned; i ++ )
        gids[ i ] = i * size + rank;

      /** Allocate distributed tree nodes in advance. */
      AllocateNodes( gids );

      /** Metric tree partitioning */
      DistSplitTask<MPINODE> mpisplittask;
      tree::SplitTask<NODE>  seqsplittask;
      DependencyCleanUp();
      DistTraverseDown<false>( mpisplittask );
      LocaTraverseDown( seqsplittask );
      hmlp_run();
      mpi::Barrier( comm );




      for ( size_t t = 0; t < n_tree; t ++ )
      {
        mpi::Barrier( comm );
        if ( rank == 0 ) printf( "Iteration #%lu\n", t );

        /** Query neighbors computed in CIDS distribution.  */
        DistData<STAR, CIDS, pair<T, size_t>> Q_cids( k, this->n, 
            this->treelist[ 0 ]->gids, initNN, comm );
        /** Pass in neighbor pointer. */
        this->setup.NN = &Q_cids;
        /** Overlap */
        if ( t != n_tree - 1 )
        {
          DependencyCleanUp();
          DistTraverseDown<false>( mpisplittask );
          hmlp_run();
          mpi::Barrier( comm );
        }
        DependencyCleanUp();
        LocaTraverseLeafs( dummy );
        LocaTraverseDown( seqsplittask );
        hmlp_run();
        mpi::Barrier( comm );

        if ( t == 0 )
        {
          /** Redistribute from CIDS to CBLK */
          NN = Q_cids; 
        }
        else
        {
          /** Queries computed in CBLK distribution */
          DistData<STAR, CBLK, pair<T, size_t>> Q_cblk( k, this->n, comm );
          /** Redistribute from CIDS to CBLK */
          Q_cblk = Q_cids;
          /** Merge Q_cblk into NN (sort and remove duplication) */
          assert( Q_cblk.col_owned() == NN.col_owned() );
          MergeNeighbors( k, NN.col_owned(), NN, Q_cblk );
        }

        //double mer_time = omp_get_wtime() - beg;

        //if ( rank == 0 )
        //printf( "%lfs %lfs %lfs\n", mpi_time, seq_time, mer_time ); fflush( stdout );
      }

      return NN;

    }; /** end AllNearestNeighbor() */

    


    /**
     *  @brief partition n points using a distributed binary tree
     *
     */ 
    void TreePartition( size_t n ) 
    {
      /** assertion */
      assert( n >= size );

      /** set up total problem size n and leaf node size m */
      this->n = n;
      this->m = this->setup.m;

      /** local problem size (assuming Round-Robin) */
      //num_points_owned = ( n - 1 ) / size + 1;

      /** local problem size (assuming Round-Robin) */
      num_points_owned = ( n - 1 ) / size + 1;

      /** edge case */
      if ( n % size )
      {
        if ( rank >= ( n % size ) ) num_points_owned -= 1;
      }




      /** initial gids distribution (asssuming Round-Robin) */
      std::vector<size_t> gids( num_points_owned, 0 );
      for ( size_t i = 0; i < num_points_owned; i ++ )
        //gids[ i ] = rank * num_points_owned + i;
        gids[ i ] = i * size + rank;

      /** edge case (happens in the last MPI process) */
      //if ( rank == size - 1 )
      //{
      //  num_points_owned = n - num_points_owned * ( size - 1 );
      //  gids.resize( num_points_owned );
      //}
    
      /** check initial gids */
      //printf( "rank %d gids[ 0 ] %lu gids[ -1 ] %lu\n",
      //    rank, gids[ 0 ], gids[ gids.size() - 1 ] ); fflush( stdout );

      /** allocate distributed tree nodes in advance */
      AllocateNodes( gids );

      DependencyCleanUp();
      //hmlp_redistribute_workers( 
      //    hmlp_read_nway_from_env( "HMLP_NORMAL_WORKER" ),
      //    hmlp_read_nway_from_env( "HMLP_SERVER_WORKER" ),
      //    hmlp_read_nway_from_env( "HMLP_NESTED_WORKER" ) );

      DistSplitTask<MPINODE> mpiSPLITtask;
      DistTraverseDown<false>( mpiSPLITtask );

      /** need to redistribute  */
      hmlp_run();
      mpi::Barrier( comm );
      this->setup.K->Redistribute( false, this->treelist[ 0 ]->gids );


      DependencyCleanUp();
      //hmlp_redistribute_workers( 
      //		omp_get_max_threads(), 
      //		omp_get_max_threads() / 4 + 1, 1 );

      tree::SplitTask<NODE> seqSPLITtask;
      LocaTraverseDown( seqSPLITtask );
      tree::IndexPermuteTask<NODE> seqINDXtask;
      LocaTraverseUp( seqINDXtask );
      DistIndexPermuteTask<MPINODE> mpiINDXtask;
      DistTraverseUp( mpiINDXtask );
      hmlp_run();

      //printf( "rank %d finish split\n", rank ); fflush( stdout );




      /** TODO: allocate space for point Morton ID */
      (this->setup).morton.resize( n );

 
      /** 
       *  Compute Morton ID for both distributed and local trees.
       *  Each rank will only have the MortonIDs it owns.
       */
      double beg = omp_get_wtime();
      Morton( mpitreelists[ 0 ], 0 );
      double morton_t = omp_get_wtime() - beg;
      //printf( "Morton takes %lfs\n", morton_t );
      mpi::Barrier( comm );

      Offset( mpitreelists[ 0 ], 0 );
      mpi::Barrier( comm );


      /**
       *  Construct morton2node map for local tree
       */
      this->morton2node.clear();
      for ( size_t i = 0; i < this->treelist.size(); i ++ )
      {
        this->morton2node[ this->treelist[ i ]->morton ] = this->treelist[ i ];
      }

      /**
       *  Construc morton2node map for distributed treee
       */ 
      for ( size_t i = 0; i < mpitreelists.size(); i ++ )
      {
        this->morton2node[ mpitreelists[ i ]->morton ] = mpitreelists[ i ];
      }



      /** now redistribute K */
      //this->setup.K->Redistribute( true, this->treelist[ 0 ]->gids );

    }; /** end TreePartition() */






    /**
     *  currently only used in DrawInteraction()
     */ 
    void Offset( MPINODE *node, size_t offset )
    {
      if ( node->GetCommSize() < 2 )
      {
        tree::Tree<SETUP, NODEDATA>::Offset( node, offset );
        return;
      }

      if ( node )
      {
        node->offset = offset;
        auto *child = node->child;
        if ( child )
        {
          if ( node->GetCommRank() < node->GetCommSize() / 2 )
            Offset( child, offset + 0 );
          else                   
            Offset( child, offset + node->n - child->n );
        }
      }
      
    }; /** end Offset() */




    /**
     *  @brief Distributed Morton ID
     */ 
    template<size_t LEVELOFFSET=5>
    void Morton( MPINODE *node, size_t morton )
    {
      /**
       *  MPI
       */ 
      int comm_size = 0;
      int comm_rank = 0;
      mpi::Comm_size( comm, &comm_size );
      mpi::Comm_rank( comm, &comm_rank );

      /** Compute the correct shift*/
      size_t shift = ( 1 << LEVELOFFSET ) - node->l + LEVELOFFSET;

      if ( node->parent )
      {
        /** Compute sibling's node morton */
        if ( morton % 2 )
          node->sibling->morton = ( ( morton - 1 ) << shift ) + node->l;
        else
          node->sibling->morton = ( ( morton + 1 ) << shift ) + node->l;

        /** Add sibling and its morton to the map */
        (*node->morton2node)[ node->sibling->morton ] = node->sibling;
      }

      /** 
       *  Call shared memory MortonID
       */
      if ( node->GetCommSize() < 2 )
      {
        tree::Tree<SETUP, NODEDATA>::Morton( node, morton );
        //printf( "level %lu rank %d size %d morton %8lu morton2rank %d\n", 
        //    node->l, comm_rank, comm_size, 
        //    node->morton, Morton2Rank( node->morton ) ); fflush( stdout );
        /**
         *  Exchange MortonIDs in 3-step:
         *
         *  1. Allgather leafnode MortonIDs, each rank sends num_of_leaves.
         *  
         *  2. Allgather leafnode sizes, each rank sends num_of_leaves.
         *
         *  3. Allgather gids
         */
        auto *local_root = this->treelist[ 0 ];
        int send_gids_size = local_root->gids.size();
        vector<int> recv_gids_size( comm_size );
        vector<int> recv_gids_disp( comm_size, 0 );
        vector<size_t> send_mortons( send_gids_size );
        vector<size_t> recv_mortons( this->n );
        vector<size_t> recv_gids( this->n );

        /**
         *  Gather MortonIDs I own
         */ 
        for ( size_t i = 0; i < send_gids_size; i ++ )
        {
          send_mortons[ i ] = this->setup.morton[ local_root->gids[ i ] ];
        }

        /**
         *  Recv gids_size and compute gids_disp
         */ 
        mpi::Allgather( &send_gids_size, 1, recv_gids_size.data(), 1, comm );
        for ( size_t p = 1; p < comm_size; p ++ )
        {
          recv_gids_disp[ p ] = recv_gids_disp[ p - 1 ] + recv_gids_size[ p - 1 ];
        }
        size_t total_gids = 0;
        for ( size_t p = 0; p < comm_size; p ++ ) total_gids += recv_gids_size[ p ];
        assert( total_gids == this->n );

        /**
         *  Recv gids and MortonIDs
         */ 
        mpi::Allgatherv( local_root->gids.data(), send_gids_size, 
            recv_gids.data(), 
            recv_gids_size.data(), recv_gids_disp.data(), comm );
        mpi::Allgatherv( send_mortons.data(), send_gids_size, 
            recv_mortons.data(), 
            recv_gids_size.data(), recv_gids_disp.data(), comm );

        /** 
         *  Update local MortonIDs
         */ 
        for ( size_t i = 0; i < recv_gids.size(); i ++ )
        {
          this->setup.morton[ recv_gids[ i ] ] = recv_mortons[ i ];
        }
      }
      else
      {
        /** set the node Morton ID */
        node->morton = ( morton << shift ) + node->l;
        /** child is the left child */
        if ( node->GetCommRank() < node->GetCommSize() / 2 )
          Morton( node->child, ( morton << 1 ) + 0 );
        /** child is the right child */
        else                   
          Morton( node->child, ( morton << 1 ) + 1 );
      }


      /** Print information */
      //if ( node->parent )
      //{
      //  printf( "level %lu rank %d size %d morton %8lu morton2rank %d sib_morton %8lu morton2rank %d\n", 
      //      node->l, comm_rank, comm_size, 
      //      node->morton, Morton2Rank( node->morton ),
      //      node->sibling->morton, Morton2Rank( node->sibling->morton ) ); fflush( stdout );
      //}
      //else
      //{
      //  printf( "level %lu rank %d size %d morton %8lu morton2rank %d\n", 
      //      node->l, comm_rank, comm_size, 
      //      node->morton, Morton2Rank( node->morton ) ); fflush( stdout );
      //}

    }; /** end Morton() */


    /**
     *
     *
     */ 
    template<size_t LEVELOFFSET=5>
    int Morton2Rank( size_t it )
    {
      size_t filter = ( 1 << LEVELOFFSET ) - 1;
      size_t itlevel = it & filter;
      size_t mpilevel = 0;

      size_t tmp = size;
      while ( tmp > 1 )
      {
        mpilevel ++;
        tmp /= 2;
      }

      if ( ( 1 << itlevel ) > size ) itlevel = mpilevel;

      return ( it >> ( ( 1 << LEVELOFFSET ) - itlevel + LEVELOFFSET ) ) << ( mpilevel - itlevel );

    }; /** end Morton2rank() */






    template<typename TASK, typename... Args>
    void LocaTraverseUp( TASK &dummy, Args&... args )
    {
      /** contain at lesat one tree node */
      assert( this->treelist.size() );

      /** 
       *  traverse the local tree without the root
       *
       *  IMPORTANT: local root alias of the distributed leaf node
       *  IMPORTANT: here l must be int, size_t will wrap over 
       *
       */

			//printf( "depth %lu\n", this->depth ); fflush( stdout );

      for ( int l = this->depth; l >= 1; l -- )
      {
        size_t n_nodes = 1 << l;
        auto level_beg = this->treelist.begin() + n_nodes - 1;

        /** loop over each node at level-l */
        for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          RecuTaskSubmit( node, dummy, args... );
        }
      }
    }; /** end LocaTraverseUp() */


    template<bool USE_RUN_TIME=true, typename TASK, typename... Args>
    void DistTraverseUp( TASK &dummy, Args&... args )
    {
      MPINODE *node = mpitreelists.back();
      while ( node )
      {
        if ( USE_RUN_TIME ) RecuTaskSubmit(  node, dummy, args... );
        else                RecuTaskExecute( node, dummy, args... );
        /** move to its parent */
        node = (MPINODE*)node->parent;
      }
    }; /** end DistTraverseUp() */


    template<typename TASK, typename... Args>
    void LocaTraverseDown( TASK &dummy, Args&... args )
    {
      /** contain at lesat one tree node */
      assert( this->treelist.size() );

      /** 
       *  traverse the local tree without the root
       *
       *  IMPORTANT: local root alias of the distributed leaf node
       *  IMPORTANT: here l must be int, size_t will wrap over 
       *
       */
      for ( int l = 1; l <= this->depth; l ++ )
      {
        size_t n_nodes = 1 << l;
        auto level_beg = this->treelist.begin() + n_nodes - 1;

        for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          RecuTaskSubmit( node, dummy, args... );
        }
      }
    }; /** end LocaTraverseDown() */


    template<bool USE_RUN_TIME=true, typename TASK, typename... Args>
    void DistTraverseDown( TASK &dummy, Args&... args )
    {
      auto *node = mpitreelists.front();
      while ( node )
      {
				//printf( "now at level %lu\n", node->l ); fflush( stdout );
        if ( USE_RUN_TIME ) RecuTaskSubmit(  node, dummy, args... );
        else                RecuTaskExecute( node, dummy, args... );
				//printf( "RecuTaskSubmit at level %lu\n", node->l ); fflush( stdout );

        /** 
         *  move to its child 
         *  IMPORTANT: here we need to cast the pointer back to mpitree::Node*
         */
        node = node->child;
      }
    }; /** end DistTraverseDown() */


    template<typename TASK, typename... Args>
    void LocaTraverseLeafs( TASK &dummy, Args&... args )
    {
      /** contain at lesat one tree node */
      assert( this->treelist.size() );

      int n_nodes = 1 << this->depth;
      auto level_beg = this->treelist.begin() + n_nodes - 1;

      for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
      {
        auto *node = *(level_beg + node_ind);
        RecuTaskSubmit( node, dummy, args... );
      }
    }; /** end LocaTraverseLeaf() */


    /**
     *  @brief For unordered traversal, we just call local
     *         downward traversal.
     */ 
    template<typename TASK, typename... Args>
    void LocaTraverseUnOrdered( TASK &dummy, Args&... args )
    {
      LocaTraverseDown( dummy, args... );
    }; /** end LocaTraverseUnOrdered() */


    /**
     *  @brief For unordered traversal, we just call distributed
     *         downward traversal.
     */ 
    template<typename TASK, typename... Args>
    void DistTraverseUnOrdered( TASK &dummy, Args&... args )
    {
      DistTraverseDown( dummy, args... );
    }; /** end DistTraverseUnOrdered() */





    void DependencyCleanUp()
    {
      for ( auto node : mpitreelists ) node->DependencyCleanUp();
      //for ( size_t i = 0; i < mpitreelists.size(); i ++ )
      //{
      //  mpitreelists[ i ]->DependencyCleanUp();
      //}

      tree::Tree<SETUP, NODEDATA>::DependencyCleanUp();






      for ( auto p : NearRecvFrom ) p.DependencyCleanUp();
      for ( auto p :  FarRecvFrom ) p.DependencyCleanUp();

      /** TODO also clean up the LET node */

    }; /** end DependencyCleanUp() */


    /** Global communicator, size, and rank */
    mpi::Comm comm = MPI_COMM_WORLD;
    int size = 1;
    int rank = 0;

    /**
     *  Interaction lists per rank
     *
     *  NearSentToRank[ p ]   contains all near node MortonIDs sent   to rank p.
     *  NearRecvFromRank[ p ] contains all near node MortonIDs recv from rank p.
     *  NearRecvFromRank[ p ][ morton ] = offset in the received vector.
     */
    vector<vector<size_t>>   NearSentToRank;
    vector<map<size_t, int>> NearRecvFromRank;
    vector<vector<size_t>>   FarSentToRank;
    vector<map<size_t, int>> FarRecvFromRank;

    vector<ReadWrite> NearRecvFrom;
    vector<ReadWrite> FarRecvFrom;
    
  private:

    /** global communicator error message */
    int ierr = 0;

    /** n = sum( num_points_owned ) from all MPI processes */
    size_t num_points_owned = 0;

}; /** end class Tree */


}; /** end namespace mpitree */
}; /** end namespace hmlp */

#endif /** define MPITREE_HPP */
