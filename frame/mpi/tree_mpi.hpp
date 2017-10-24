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


#include <mpi/hmlp_mpi.hpp>
#include <mpi/DistData.hpp>

#include <gofmm/tree.hpp>

using namespace hmlp;


namespace hmlp
{
namespace mpitree
{



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
  hmlp::Data<T> *Coordinate;

  inline std::vector<std::vector<std::size_t> > operator()
  ( 
    std::vector<std::size_t>& gids,
    std::vector<std::size_t>& lids
  ) const 
  {
    assert( N_SPLIT == 2 );

    hmlp::Data<T> &X = *Coordinate;
    size_t d = X.row();
    size_t n = lids.size();

    T rcx0 = 0.0, rx01 = 0.0;
    std::size_t x0, x1;
    std::vector<std::vector<std::size_t> > split( N_SPLIT );


    std::vector<T> centroid = hmlp::combinatorics::Mean( d, n, X, lids );
    std::vector<T> direction( d );
    std::vector<T> projection( n, 0.0 );

    //printf( "After Mean\n" );

    // Compute the farest x0 point from the centroid
    for ( int i = 0; i < n; i ++ )
    {
      T rcx = 0.0;
      for ( int p = 0; p < d; p ++ )
      {
        T tmp = X[ lids[ i ] * d + p ] - centroid[ p ];
        rcx += tmp * tmp;
        //printf( "%5.2lf ", X[ lids[ i ] * d + p  ] );
      }
      //printf( "\n" );
      //printf( "rcx %lf rcx0 %lf lids %d\n", rcx, rcx0, (int)lids[ i ] );
      if ( rcx > rcx0 ) 
      {
        rcx0 = rcx;
        x0 = i;
      }
    }

    //printf( "After Farest\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //  printf( "%5.2lf ", X[ lids[ x0 ] * d + p ] );
    //}
    //printf( "\n" );

    // Compute the farest point x1 from x0
    for ( int i = 0; i < n; i ++ )
    {
      T rxx = 0.0;
      for ( int p = 0; p < d; p ++ )
      {
        T tmp = X[ lids[ i ] * d + p ] - X[ lids[ x0 ] * d + p ];
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
      direction[ p ] = X[ lids[ x1 ] * d + p ] - X[ lids[ x0 ] * d + p ];
    }

    // Compute projection
    projection.resize( n, 0.0 );
    for ( int i = 0; i < n; i ++ )
      for ( int p = 0; p < d; p ++ )
        projection[ i ] += X[ lids[ i ] * d + p ] * direction[ p ];

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
  // closure
  hmlp::Data<T> *Coordinate;

  inline std::vector<std::vector<std::size_t> > operator()
  ( 
    std::vector<std::size_t>& gids,
    std::vector<std::size_t>& lids
  ) const 
  {
    std::vector<std::vector<std::size_t> > split( N_SPLIT );

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



template<typename T>
bool less_first
(
  const std::pair<T, size_t> &a, 
  const std::pair<T, size_t> &b
)
{
  return ( a.first < b.first );
};

template<typename T>
bool less_second
( 
  const std::pair<T, size_t> &a, 
  const std::pair<T, size_t> &b 
)
{
  return ( a.second < b.second );
};


template<typename T>
bool equal_second 
( 
  const std::pair<T, size_t> &a, 
  const std::pair<T, size_t> &b 
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
  std::pair<T, size_t> *A, 
  std::pair<T, size_t> *B,
  std::vector<std::pair<T, size_t>, Allocator> &aux
)
{
  if ( aux.size() != 2 * k ) aux.resize( 2 * k );

  for ( size_t i = 0; i < k; i++ ) aux[     i ] = A[ i ];
  for ( size_t i = 0; i < k; i++ ) aux[ k + i ] = B[ i ];

  std::sort( aux.begin(), aux.end(), less_second<T> );
  auto it = std::unique( aux.begin(), aux.end(), equal_second<T> );
  std::sort( aux.begin(), it, less_first<T> );

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
  std::vector<std::pair<T, size_t>, Allocator> &A, 
  std::vector<std::pair<T, size_t>, Allocator> &B
)
{
  assert( A.size() >= n * k && B.size() >= n * k );
	#pragma omp parallel
  {
    std::vector<std::pair<T, size_t> > aux( 2 * k );
    #pragma omp for
    for( size_t i = 0; i < n; i++ ) 
    {
      MergeNeighbors( k, &(A[ i * k ]), &(B[ i * k ]), aux );
    }
  }
}; /** */












template<typename BACKGROUND>
class BackGroundTask : public hmlp::Task
{
	public:

		BACKGROUND *bg = NULL;

		BackGroundTask( BACKGROUND *user_bg ) : hmlp::Task()
		{
			bg = user_bg;
			name = std::string( "BackGround" );
      /** asuume computation bound */
      cost = 9999.9;
			/** high priority */
      priority = true;
		};

    void Execute( Worker* user_worker )
    {
      bg->BackGroundProcess( &(user_worker->scheduler->do_terminate) );
    };

}; /** end class BusyGroundTask */


template<typename NODE>
class DistSplitTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = std::string( "DistSplit" );
      {
        //label = std::to_string( arg->treelist_id );
        std::ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      double flops = 6.0 * arg->n;
      double mops = 6.0 * arg->n;

      /** setup the event */
      event.Set( label + name, flops, mops );

      /** asuume computation bound */
      cost = mops / 1E+9;

      /** "HIGH" priority */
      priority = true;
    };


    void DependencyAnalysis()
    {
			//printf( "in DependencyAnalysis level %lu\n", arg->l );

      arg->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() > 1 )
        {
					assert( arg->child );
          arg->child->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        }
        else
        {
					assert( arg->lchild && arg->rchild );
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
template<typename SPLITTER, typename T>
class Setup
{
  public:

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
template<typename SETUP, int N_CHILDREN, typename NODEDATA, typename T>
class Node : public tree::Node<SETUP, N_CHILDREN, NODEDATA, T>
{
  public:

    /** inherit all parameters from hmlp::tree::Node */


    /** constructor for inner node (gids and n unassigned) */
    Node( SETUP *setup, size_t n, size_t l, 
        Node *parent, mpi::Comm comm ) 
    /** inherits from */
      : tree::Node<SETUP, N_CHILDREN, NODEDATA, T>
        ( setup, n, l, parent ) 
    {
      /** local communicator */
      this->comm = comm;
      
      /** get size and rank */
      mpi::Comm_size( comm, &size );
      mpi::Comm_rank( comm, &rank );
    };



    /** constructor for root */
    Node( SETUP *setup, size_t n, size_t l,
        std::vector<size_t> &gids,
        Node *parent, mpi::Comm comm ) 
    /** inherits from */
      : Node<SETUP, N_CHILDREN, NODEDATA, T>
        ( setup, n, l, parent, comm ) 
    {
      /** notice that "gids.size() < n" */
      this->gids = gids;
    };


    void SetupChild( class Node *child )
    {
      this->kids[ 0 ] = child;
      this->child = child;
    };





    void Split()
    {
      /** assertion */
      assert( N_CHILDREN == 2 );


      /** reduce to get the total size of gids */
      int num_points_total = 0;
      int num_points_owned = (this->gids).size();

      /** n = sum( num_points_owned ) over all MPI processes in comm */
      mpi::Allreduce( &num_points_owned, &num_points_total, 
          1, MPI_SUM, comm );
      this->n = num_points_total;







      //printf( "rank %d n %lu gids.size() %lu --", 
      //    rank, this->n, this->gids.size() ); fflush( stdout );
      //for ( size_t i = 0; i < this->gids.size(); i ++ )
      //  printf( "%lu ", this->gids[ i ] );
      //printf( "\n" ); fflush( stdout );





      if ( child )
      {
				//printf( "Split(): n %lu  \n", this->n ); fflush( stdout );

        /** the local communicator of this node contains at least 2 processes */
        assert( size > 1 );

        /** distributed split */
        auto split = this->setup->splitter( this->gids, comm );

				//printf( "Finish Split(): n %lu  \n", this->n ); fflush( stdout );


        /** get partner rank */
        int partner_rank = 0;
        int sent_size = 0; 
        int recv_size = 0;
        std::vector<size_t> &kept_gids = child->gids;
        std::vector<int>     sent_gids;
        std::vector<int>     recv_gids;

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

        //printf( "rank %d kept_size %lu sent_size %d recv_size %d\n", 
        //    rank, kept_gids.size(), sent_size, recv_size ); fflush( stdout );

        /** resize recv_gids */
        recv_gids.resize( recv_size );

        /** exchange recv_gids.size() */
        mpi::Sendrecv( 
            sent_gids.data(), sent_size, MPI_INT, partner_rank, 20,
            recv_gids.data(), recv_size, MPI_INT, partner_rank, 20,
            comm, &status );

        /** enlarge kept_gids */
        kept_gids.reserve( kept_gids.size() + recv_gids.size() );
        for ( size_t i = 0; i < recv_gids.size(); i ++ )
          kept_gids.push_back( recv_gids[ i ] );
        

      }
			else
			{
				//printf( "Split(): n %lu  \n", this->n ); fflush( stdout );

				/** TODO: deprecate lids */
			  this->lids = this->gids;
        tree::Node<SETUP, N_CHILDREN, NODEDATA, T>::Split<true>( 0 );

		  } /** end if ( child ) */
      
      /** synchronize within local communicator */
      hmlp::mpi::Barrier( comm );

    }; /** end Split() */



    hmlp::mpi::Comm GetComm() { return comm; };

    int GetCommSize() { return size; };
    
    int GetCommRank() { return rank; };

    void Print()
    {
      int global_rank = 0;
      hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );
      //printf( "grank %d l %lu lrank %d offset %lu n %lu\n", 
      //    global_rank, this->l, this->rank, this->offset, this->n );
      //hmlp_print_binary( this->morton );
    };

    Node *child = NULL;

  private:

    /** initialize with all processes */
    hmlp::mpi::Comm comm = MPI_COMM_WORLD;

    /** mpi status */
    hmlp::mpi::Status status;

    /** subcommunicator size */
    int size = 1;

    /** subcommunicator rank */
    int rank = 0;

}; /** end class Node */






/**
 *
 */ 
template<typename SETUP, int N_CHILDREN, typename NODEDATA, typename T>
class LetNode : public hmlp::tree::Node<SETUP, N_CHILDREN, NODEDATA, T>
{
  public:

    /** inherit all parameters from hmlp::tree::Node */

    LetNode( SETUP *setup, size_t morton )
      : hmlp::tree::Node<SETUP, N_CHILDREN, NODEDATA, T>
        ( setup, (size_t)0, (size_t)1, NULL ) 
    {
      this->morton = morton;
    };

  private:


}; /** end class LetNode */













/**
 *  @brief This distributed tree inherits the shared memory tree
 *         with some additional MPI data structure and function call.
 */ 
template<class SETUP, class NODEDATA, int N_CHILDREN, typename T>
class Tree : public hmlp::tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>
{
  public:

    /** inherit parameters n, m, and depth;
     *  local treelists and treequeue.
     **/

    /** define our tree node type as NODE */
    typedef tree::Node<SETUP, N_CHILDREN, NODEDATA, T> NODE;

    /** */
    typedef Node<SETUP, N_CHILDREN, NODEDATA, T> MPINODE;

    /** define our tree node type as NODE */
    typedef LetNode<SETUP, N_CHILDREN, NODEDATA, T> LETNODE;


    /** distribued tree (a list of tree nodes) */
    std::vector<MPINODE*> mpitreelists;

    /** local essential tree nodes (this is not thread-safe) */
    std::map<size_t, LETNODE*> lettreelist;

    /** inherit constructor */
    Tree() : hmlp::tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>::Tree()
    {
			/** create a new comm_world for */
      mpi::Comm_dup( MPI_COMM_WORLD, &comm );

			/** get size and rank */
      mpi::Comm_size( comm, &size );
      mpi::Comm_rank( comm, &rank );
    };

    /** inherit deconstructor */
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

      /** free all local essential tree nodes */
      if ( lettreelist.size() )
      {
        for ( size_t i = 0; i < lettreelist.size(); i ++ )
          if ( lettreelist[ i ] ) delete lettreelist[ i ];
      }
      lettreelist.clear();

    }; /** end CleanUp() */

	







    /** 
     *  @breif allocate distributed tree node
     *
     * */
    void AllocateNodes( std::vector<size_t> &gids )
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
          this->n, mylevel, gids, NULL, mycomm );

      /** push root to the mpi treelist */
      mpitreelists.push_back( root );

      while ( mysize > 1 )
      {
        hmlp::mpi::Comm childcomm;

        /** increase level */
        mylevel += 1;

        /** left color = 0, right color = 1 */
        mycolor = ( myrank < mysize / 2 ) ? 0 : 1;

        //printf( "size %d rank %d color %d level %d\n",
        //    mysize, myrank, mycolor, mylevel ); fflush( stdout );

        /** get the subcommunicator for children */
        ierr = hmlp::mpi::Comm_split( mycomm, mycolor, myrank, &(childcomm) );

        //printf( "size %d rank %d color %d level %d end\n",
        //    mysize, myrank, mycolor, mylevel ); fflush( stdout );

        /** update mycomm */
        mycomm = childcomm;

        /** create the child */
        auto *parent = mpitreelists.back();
        auto *child  = new MPINODE( &(this->setup), 
            (size_t)0, mylevel, parent, mycomm );

        //printf( "size %d rank %d color %d level %d here\n",
        //    mysize, myrank, mycolor, mylevel ); fflush( stdout );

        /** setup parent's children */
        parent->SetupChild( child );
       
        //printf( "size %d rank %d color %d level %d setupchild\n",
        //    mysize, myrank, mycolor, mylevel ); fflush( stdout );

        /** push to the mpi treelist */
        mpitreelists.push_back( child );

        /** update communicator size */
        hmlp::mpi::Comm_size( mycomm, &mysize );

        /** update myrank in the subcommunicator */
        hmlp::mpi::Comm_rank( mycomm, &myrank );
      }
      /** synchronize */
      hmlp::mpi::Barrier( comm );

			/** allocate local tree nodes */
      auto *local_tree_root = mpitreelists.back();

			//printf( "" );


      tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>::AllocateNodes( 
          local_tree_root );

    };




    /** */
    template<bool SORTED, typename KNNTASK>
    hmlp::DistData<STAR, CBLK, std::pair<T, size_t>> AllNearestNeighbor
    (
      size_t n_tree,
      size_t n,
      size_t k, 
      size_t max_depth,
      std::pair<T, std::size_t> initNN,
      KNNTASK &dummy
    )
    {
      /** get the problem size from setup->K->row() */
      this->n = n;

      /** k-by-N */
      hmlp::DistData<STAR, CBLK, std::pair<T, size_t>> NN( k, n, initNN, comm );

      /** use leaf size = 4 * k  */
      this->setup.m = 4 * k;
      if ( this->setup.m < 512 ) this->setup.m = 512;
      this->m = this->setup.m;

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





      //printf( "rank %d before AllocateNodes\n", rank ); fflush( stdout );

      /** allocate distributed tree nodes in advance */
      AllocateNodes( gids );

      //printf( "Finish allocate rkdt nodes\n" ); fflush( stdout );


      auto *bgtask = new BackGroundTask<SETUP>( &(this->setup) );
      bgtask->SetAsBackGround();

      //printf( "Finish bgtask\n" ); fflush( stdout );


      /** */
      DependencyCleanUp();
      //hmlp_redistribute_workers(  
      //    hmlp_read_nway_from_env( "HMLP_NORMAL_WORKER" ),
      //    hmlp_read_nway_from_env( "HMLP_SERVER_WORKER" ),
      //    hmlp_read_nway_from_env( "HMLP_NESTED_WORKER" ) );

      /** tree partitioning */
      DistSplitTask<MPINODE> mpisplittask;
      DistTraverseDown( mpisplittask );
      hmlp_run();
      this->setup.K->Redistribute( false, this->treelist[ 0 ]->gids );

      /** */
      DependencyCleanUp();
      //hmlp_redistribute_workers( 
      //		omp_get_max_threads(), 
      //		omp_get_max_threads() / 4 + 1, 1 );

      tree::SplitTask<NODE> splittask;
      LocaTraverseDown( splittask );
      //tree::IndexPermuteTask<NODE> seqINDXtask;
			//LocaTraverseUp( seqINDXtask );
      //DistIndexPermuteTask<MPINODE> mpiINDXtask;
			//DistTraverseUp( mpiINDXtask );
      hmlp_run();
      mpi::Barrier( comm );




      for ( size_t t = 0; t < n_tree; t ++ )
      {
        //printf( "t = %lu\n", t ); fflush( stdout );
        DependencyCleanUp();

//        /** tree partitioning */
//        DistSplitTask<MPINODE> mpisplittask;
//        DistTraverseDown( mpisplittask );
//        hmlp_run();
//	  	  MPI_Barrier( comm );
//			  this->setup.K->Redistribute( false, this->treelist[ 0 ]->gids );
//
//        /** queries computed in CIDS distribution  */
//        DistData<STAR, CIDS, std::pair<T, size_t>> Q_cids( k, this->n, 
//            this->treelist[ 0 ]->gids, initNN, comm );
//
//        /** 
//         *  notice that setup.NN has type Data<std::pair<T, size_t>>,
//         *  but that is fine because DistData inherits Data
//         */
//        this->setup.NN = &Q_cids;
//
//			  DependencyCleanUp();
//        tree::SplitTask<NODE> splittask;
//        LocaTraverseDown( splittask );
//        LocaTraverseLeafs( dummy );
//        hmlp_run();
//	  	  MPI_Barrier( comm );






        /** queries computed in CIDS distribution  */
        DistData<STAR, CIDS, std::pair<T, size_t>> Q_cids( k, this->n, 
            this->treelist[ 0 ]->gids, initNN, comm );

        /** 
         *  notice that setup.NN has type Data<std::pair<T, size_t>>,
         *  but that is fine because DistData inherits Data
         */
        this->setup.NN = &Q_cids;

        /**
         *  redistribute K to reduce communication
         */
        double beg = omp_get_wtime();
        //this->setup.K->Redistribute( this->treelist[ 0 ]->gids );
        //double redist_t = omp_get_wtime() - beg;
        //printf( "Redistribution time %lfs\n", redist_t ); fflush( stdout );

        beg = omp_get_wtime();
        /** neighbor search */
        LocaTraverseLeafs( dummy );
        if ( t + 1 < n_tree )
        {
          DistTraverseDown( mpisplittask );
          hmlp_run();
          mpi::Barrier( comm );
          this->setup.K->Redistribute( false, this->treelist[ 0 ]->gids );
          DependencyCleanUp();
          LocaTraverseDown( splittask );
        }
        hmlp_run();
        mpi::Barrier( comm );
        double nn_t = omp_get_wtime() - beg;
        //printf( "NN+tree time %lfs\n", nn_t ); fflush( stdout );

        if ( t == 0 )
        {
          /** redistribute from CIDS to CBLK */
          NN = Q_cids; 
        }
        else
        {
          /** queries computed in CBLK distribution */
          DistData<STAR, CBLK, std::pair<T, size_t>> Q_cblk( k, this->n, comm );

          /** redistribute from CIDS to CBLK */
          Q_cblk = Q_cids;

          /** merge Q_cblk into NN (sort and remove duplication) */
          assert( Q_cblk.col_owned() == NN.col_owned() );
          MergeNeighbors( k, NN.col_owned(), NN, Q_cblk );
        }
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
      printf( "rank %d gids[ 0 ] %lu gids[ -1 ] %lu\n",
          rank, gids[ 0 ], gids[ gids.size() - 1 ] ); fflush( stdout );

      /** allocate distributed tree nodes in advance */
      AllocateNodes( gids );

      auto *bgtask = new BackGroundTask<SETUP>( &(this->setup) );
      bgtask->SetAsBackGround();

      DependencyCleanUp();
      //hmlp_redistribute_workers( 
      //    hmlp_read_nway_from_env( "HMLP_NORMAL_WORKER" ),
      //    hmlp_read_nway_from_env( "HMLP_SERVER_WORKER" ),
      //    hmlp_read_nway_from_env( "HMLP_NESTED_WORKER" ) );

      DistSplitTask<MPINODE> mpiSPLITtask;
      DistTraverseDown( mpiSPLITtask );

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

 
      /** compute Morton ID for both distributed and local trees */
      double beg = omp_get_wtime();
      Morton( mpitreelists[ 0 ], 0 );
      double morton_t = omp_get_wtime() - beg;
      //printf( "Morton takes %lfs\n", morton_t );
      hmlp::mpi::Barrier( comm );

      Offset( mpitreelists[ 0 ], 0 );
      hmlp::mpi::Barrier( comm );

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
        tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>::Offset( node, offset );
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
    template<size_t LEVELOFFSET=4>
    void Morton( MPINODE *node, size_t morton )
    {
      /** call shared memory Morton ID */
      if ( node->GetCommSize() < 2 )
      {
        tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>::Morton( node, morton );
        return;
      }

      if ( node )
      {
        /** child is the left child */
        if ( node->GetCommRank() < node->GetCommSize() / 2 )
          Morton( node->child, ( morton << 1 ) + 0 );
        /** child is the right child */
        else                   
          Morton( node->child, ( morton << 1 ) + 1 );
        /** compute the correct shift*/
        size_t shift = ( 1 << LEVELOFFSET ) - node->l + LEVELOFFSET;
        /** set the node Morton ID */
        node->morton = ( morton << shift ) + node->l;
      }
    }; /** end Morton() */



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


    template<typename TASK, typename... Args>
    void DistTraverseUp( TASK &dummy, Args&... args )
    {
      MPINODE *node = mpitreelists.back();
      while ( node )
      {
        RecuTaskSubmit( node, dummy, args... );
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


    template<typename TASK, typename... Args>
    void DistTraverseDown( TASK &dummy, Args&... args )
    {
      auto *node = mpitreelists.front();
      while ( node )
      {
				//printf( "now at level %lu\n", node->l ); fflush( stdout );
        RecuTaskSubmit( node, dummy, args... );
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




    /** 
     *
     */
    template<bool USE_RUNTIME, 
      typename TASK1, typename MPITASK1,
      typename TASK2, typename MPITASK2>
    void ParallelTraverseUp( 
        TASK1 &dummy1, MPITASK1 &mpidummy1, 
        TASK2 &dummy2, MPITASK2 &mpidummy2 )
    {
      /** local and distributed treelists must be non-empty */
      assert( this->treelist.size() );
      assert( mpitreelists.size() );
     
      using NULLTASK = hmlp::NULLTask<hmlp::tree::Node<SETUP, N_CHILDREN, NODEDATA, T>>;
      using MPINULLTASK = hmlp::NULLTask<MPINODE>;


      /** 
       *  traverse the local tree without the root
       *
       *  IMPORTANT: local root alias of the distributed leaf node
       *  IMPORTANT: here l must be int, size_t will wrap over 
       *
       */
      for ( int l = this->depth; l >= 1; l -- )
      {
        std::size_t n_nodes = 1 << l;
        auto level_beg = this->treelist.begin() + n_nodes - 1;

        if ( !USE_RUNTIME )
        { 
          printf( "No implementation!!\n" ); fflush( stdout );
          exit( 1 );
        }
        else
        {
          for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            /** create the first normal task is it is not a NULLTask */
            if ( !std::is_same<NULLTASK, TASK1>::value )
            {
              auto *task1 = new TASK1();
              task1->Submit();
              //printf( "finish task1 submit\n" ); fflush( stdout );
              task1->Set( node );
              //printf( "finish task1 set\n" ); fflush( stdout );
              task1->DependencyAnalysis();
              //printf( "finish task1 create\n" ); fflush( stdout );
            }
            /** create the second task and perform dependency analysis */
            if ( !std::is_same<NULLTASK, TASK2>::value )
            {
              auto *task2 = new TASK2();
              task2->Submit();
              task2->Set( node );
              task2->DependencyAnalysis();
              //printf( "finish task2 create\n" ); fflush( stdout );
            }
          }
        }
        //printf( "finish level %d\n", l );
      }

      /** traverse the distributed tree upward */
      MPINODE *node = mpitreelists.back();
      while ( node )
      {
        /** create mpi tasks */
        if ( !std::is_same<MPINULLTASK, MPITASK1>::value )
        {
          auto *mpitask1 = new MPITASK1();
          mpitask1->Submit();
          mpitask1->Set( node );
          mpitask1->DependencyAnalysis();
        }
        /** create tasks and perform dependency analysis */
        if ( !std::is_same<MPINULLTASK, MPITASK2>::value )
        {
          auto *mpitask2 = new MPITASK2();
          mpitask2->Submit();
          mpitask2->Set( node );
          mpitask2->DependencyAnalysis();
        }

        /** 
         *  move to its parent 
         *  IMPORTANT: here we need to cast the pointer back to mpitree::Node*
         */
        node = (MPINODE*)node->parent;

      } /** end while( node ) */
    }; /** end ParallelTraverseUp() */







    template<bool USE_RUNTIME, 
      typename TASK1, typename MPITASK1,
      typename TASK2, typename MPITASK2>
    void ParallelTraverseDown( 
        TASK1 &dummy1, MPITASK1 &mpidummy1, 
        TASK2 &dummy2, MPITASK2 &mpidummy2 )
    {
      /** local and distributed treelists must be non-empty */
      assert( this->treelist.size() );
      assert( mpitreelists.size() );
     
      using NULLTASK = hmlp::NULLTask<hmlp::tree::Node<SETUP, N_CHILDREN, NODEDATA, T>>;
      using MPINULLTASK = hmlp::NULLTask<MPINODE>;


      /** traverse the distributed tree downward */
      MPINODE *node = mpitreelists.front();
      while ( node )
      {
        /** create mpi tasks */
        if ( !std::is_same<MPINULLTASK, MPITASK1>::value )
        {
          auto *mpitask1 = new MPITASK1();
          mpitask1->Submit();
          mpitask1->Set( node );
          mpitask1->DependencyAnalysis();
        }
        /** create tasks and perform dependency analysis */
        if ( !std::is_same<MPINULLTASK, MPITASK2>::value )
        {
          auto *mpitask2 = new MPITASK2();
          mpitask2->Submit();
          mpitask2->Set( node );
          mpitask2->DependencyAnalysis();
        }

        /** 
         *  move to its parent 
         *  IMPORTANT: here we need to cast the pointer back to mpitree::Node*
         */
        node = (MPINODE*)node->child;

      } /** end while( node ) */


      /** 
       *  traverse the local tree without the root
       *
       *  IMPORTANT: local root alias of the distributed leaf node
       *  IMPORTANT: here l must be int, size_t will wrap over 
       *
       */
      for ( int l = 1; l <= this->depth; l ++ )
      {
        std::size_t n_nodes = 1 << l;
        auto level_beg = this->treelist.begin() + n_nodes - 1;

        if ( !USE_RUNTIME )
        { 
          printf( "No implementation!!\n" ); fflush( stdout );
          exit( 1 );
        }
        else
        {
          for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            /** create the first normal task is it is not a NULLTask */
            if ( !std::is_same<NULLTASK, TASK1>::value )
            {
              auto *task1 = new TASK1();
              task1->Submit();
              //printf( "finish task1 submit\n" ); fflush( stdout );
              task1->Set( node );
              //printf( "finish task1 set\n" ); fflush( stdout );
              task1->DependencyAnalysis();
              //printf( "finish task1 create\n" ); fflush( stdout );
            }
            /** create the second task and perform dependency analysis */
            if ( !std::is_same<NULLTASK, TASK2>::value )
            {
              auto *task2 = new TASK2();
              task2->Submit();
              task2->Set( node );
              task2->DependencyAnalysis();
              //printf( "finish task2 create\n" ); fflush( stdout );
            }
          }
        }
      }
    }; /** end ParallelTraverseDown() */









    void DependencyCleanUp()
    {
      for ( size_t i = 0; i < mpitreelists.size(); i ++ )
      {
        mpitreelists[ i ]->DependencyCleanUp();
      }

      tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>::DependencyCleanUp();

      /** TODO also clean up the LET node */

    }; /** end DependencyCleanUp() */



    
  private:

    /** global communicator */
    mpi::Comm comm = MPI_COMM_WORLD;

    /** global communicator size */
    int size = 1;

    /** global communicator rank */
    int rank = 0;

    /** global communicator error message */
    int ierr = 0;

    /** n = sum( num_points_owned ) from all MPI processes */
    size_t num_points_owned = 0;

}; /** end class Tree */


}; /** end namespace mpitree */
}; /** end namespace hmlp */

#endif /** define MPITREE_HPP */
