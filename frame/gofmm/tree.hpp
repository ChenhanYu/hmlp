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




#ifndef TREE_HPP
#define TREE_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <set>
#include <vector>
#include <deque>
#include <iostream>
#include <random>
#include <hmlp.h>

#include <hmlp_runtime.hpp>
#include <containers/data.hpp>
#include <primitives/combinatorics.hpp>

#define REPORT_ANN_STATUS 0

//#define DEBUG_TREE 1


bool has_uneven_split = false;

namespace hmlp
{
namespace tree
{



/**
 *  @brief Check if ``it'' is ``me'''s ancestor by checking two facts.
 *         1) itlevel >= mylevel and
 *         2) morton above itlevel should be identical. For example,
 *
 *         me = 01110 100 (level 4)
 *         it = 01100 010 (level 2) is my parent
 *
 *         me = 00000 001 (level 1)
 *         it = 00000 011 (level 3) is not my parent
 */ 
template<size_t LEVELOFFSET=4>
bool IsMyParent( size_t me, size_t it )
{
  size_t filter = ( 1 << LEVELOFFSET ) - 1;
  size_t itlevel = it & filter;
  size_t mylevel = me & filter;
  size_t itshift = ( 1 << LEVELOFFSET ) - itlevel + LEVELOFFSET;
  bool ret = !( ( me ^ it ) >> itshift ) && ( itlevel <= mylevel );
#ifdef DEBUG_TREE
  hmlp_print_binary( me );
  hmlp_print_binary( it );
  hmlp_print_binary( ( me ^ it ) >> itshift );
  printf( "ismyparent %d itlevel %lu mylevel %lu shift %lu fixed shift %d\n",
      ret, itlevel, mylevel, itshift, 1 << LEVELOFFSET );
#endif
  return ret;

}; /** end IsMyParent() */


/**
 *
 */ 
bool ContainAnyMortonID( std::vector<size_t> &querys, size_t morton )
{
  for ( size_t i = 0; i < querys.size(); i ++ )
  {
    if ( IsMyParent( querys[ i ], morton ) ) return true;
  }
  return false;

}; /** end ContainAnyMortonID() */


/**
 *
 */ 
bool ContainAnyMortonID( std::set<size_t> &querys, size_t morton )
{
  for ( auto it = querys.begin(); it != querys.end(); it ++ )
  {
    if ( IsMyParent( (*it), morton ) ) return true;
  }
  return false;

}; /** end ContainAnyMortonID() */




/**
 *  @brief Permuate the order of gids and lids of each inner node
 *         to the order of leaf nodes.
 *         
 *  @para  The parallelism is exploited in the task level using a
 *         bottom up traversal.
 */ 
template<typename NODE>
class IndexPermuteTask : public hmlp::Task
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
      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      }
      this->TryEnqueue();
    };


    void Execute( Worker* user_worker )
    {
      auto &lids = arg->lids; 
      auto &gids = arg->gids; 
      auto *lchild = arg->lchild;
      auto *rchild = arg->rchild;
      
      if ( !arg->isleaf )
      {
        auto &llids = lchild->lids;
        auto &rlids = rchild->lids;
        auto &lgids = lchild->gids;
        auto &rgids = rchild->gids;
        lids = llids;
        lids.insert( lids.end(), rlids.begin(), rlids.end() );
        gids = lgids;
        gids.insert( gids.end(), rgids.begin(), rgids.end() );
      }
    };

}; /** end class IndexPermuteTask */


/**
 *  @brief
 *
 */ 
template<typename NODE>
class SplitTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      name = std::string( "Split" );
      arg = user_arg;
      // Need an accurate cost model.
      cost = 1.0;
    };

		void DependencyAnalysis()
		{
      arg->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      }
      this->TryEnqueue();
		};

    void Execute( Worker* user_worker )
    {
			//printf( "Local Split %lu level %lu begin\n", arg->treelist_id, arg->l ); fflush( stdout );
      arg->template Split<true>( 0 );
			//printf( "Local Split %lu level %lu end\n", arg->treelist_id, arg->l ); fflush( stdout );
    };
}; /** end class SplitTask */


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
    size_t n = gids.size();

    T rcx0 = 0.0, rx01 = 0.0;
    std::size_t x0, x1;
    std::vector<std::vector<std::size_t> > split( N_SPLIT );


    std::vector<T> centroid = hmlp::combinatorics::Mean( d, n, X, gids );
    std::vector<T> direction( d );
    std::vector<T> projection( n, 0.0 );

    //printf( "After Mean\n" );

    // Compute the farest x0 point from the centroid
    for ( size_t i = 0; i < n; i ++ )
    {
      T rcx = 0.0;
      for ( size_t p = 0; p < d; p ++ )
      {
        //T tmp = X[ lids[ i ] * d + p ] - centroid[ p ];
        T tmp = X( p, gids[ i ] ) - centroid[ p ];


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
    for ( size_t i = 0; i < n; i ++ )
    {
      T rxx = 0.0;
      for ( size_t p = 0; p < d; p ++ )
      {
        //T tmp = X[ lids[ i ] * d + p ] - X[ lids[ x0 ] * d + p ];
				T tmp = X( p, gids[ i ] ) - X( p, gids[ x0 ] );
        rxx += tmp * tmp;
      }
      if ( rxx > rx01 )
      {
        rx01 = rxx;
        x1 = i;
      }
    }

    //printf( "After Nearest\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //  printf( "%5.2lf ", X[ lids[ x1 ] * d + p ] );
    //}
    //printf( "\n" );


    // Compute direction
    for ( size_t p = 0; p < d; p ++ )
    {
      //direction[ p ] = X[ lids[ x1 ] * d + p ] - X[ lids[ x0 ] * d + p ];
      direction[ p ] = X( p, gids[ x1 ] ) - X( p, gids[ x0 ] );
    }

    //printf( "After Direction\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //  printf( "%5.2lf ", direction[ p ] );
    //}
    //printf( "\n" );
    //exit( 1 );



    // Compute projection
    projection.resize( n, 0.0 );
    for ( size_t i = 0; i < n; i ++ )
      for ( size_t p = 0; p < d; p ++ )
        //projection[ i ] += X[ lids[ i ] * d + p ] * direction[ p ];
        projection[ i ] += X( p, gids[ i ] ) * direction[ p ];

    //printf( "After Projetion\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //  printf( "%5.2lf ", projec[ p ] );
    //}
    //printf( "\n" );



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


    //printf( "split median %lf left %d right %d\n", 
    //    median,
    //    (int)split[ 0 ].size(), (int)split[ 1 ].size() );

    //if ( split[ 0 ].size() > 0.6 * n ||
    //     split[ 1 ].size() > 0.6 * n )
    //{
    //  for ( int i = 0; i < n; i ++ )
    //  {
    //    printf( "%E ", projection[ i ] );
    //  } 
    //  printf( "\n" );
    //}


    return split; 
  };
};


/**
 *  @brief This is the splitter used in the randomized tree. Given
 *         coordinates, project all points onto a random direction
 *         and split into two groups using a median select.
 *
 *  @para
 *
 *  @TODO  Need to explit the parallelism.
 */ 
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
    assert( N_SPLIT == 2 );

    hmlp::Data<T> &X = *Coordinate;
    size_t d = X.row();
    size_t n = gids.size();

    std::vector<std::vector<std::size_t> > split( N_SPLIT );

    std::vector<T> direction( d );
    std::vector<T> projection( n, 0.0 );

    // Compute random direction
    static std::default_random_engine generator;
    std::normal_distribution<T> distribution;
    for ( int p = 0; p < d; p ++ )
    {
      direction[ p ] = distribution( generator );
    }

    // Compute projection
    projection.resize( n, 0.0 );
    for ( size_t i = 0; i < n; i ++ )
      for ( size_t p = 0; p < d; p ++ )
        //projection[ i ] += X[ lids[ i ] * d + p ] * direction[ p ];
        projection[ i ] += X( p, gids[ i ] ) * direction[ p ];


    // Parallel median search
    // T median = Select( n, n / 2, projection );
    auto proj_copy = projection;
    std::sort( proj_copy.begin(), proj_copy.end() );
    T median = proj_copy[ n / 2 ];

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


    //printf( "split median %lf left %d right %d\n", 
    //    median,
    //    (int)split[ 0 ].size(), (int)split[ 1 ].size() );

    //if ( split[ 0 ].size() > 0.6 * n ||
    //     split[ 1 ].size() > 0.6 * n )
    //{
    //  for ( int i = 0; i < n; i ++ )
    //  {
    //    printf( "%E ", projection[ i ] );
    //  } 
    //  printf( "\n" );
    //}


    return split; 
  };
};


/**
 *  @brief 
 */ 
template<typename SETUP, int N_CHILDREN, typename NODEDATA, typename T>
class Node : public ReadWrite
{
  public:

    Node
    (
      SETUP *user_setup,
      size_t n, size_t l, 
      Node *parent 
    )
    {
      this->setup = user_setup;
      this->n = n;
      this->l = l;
      this->morton = 0;
      this->treelist_id = 0;
      this->gids.resize( n );
      this->lids.resize( n );
      this->isleaf = false;
      this->parent = parent;
      this->lchild = NULL;
      this->rchild = NULL;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };

    Node
    ( 
      SETUP *user_setup,
      int n, int l, 
      std::vector<std::size_t> gids,
      std::vector<std::size_t> lids,
      Node *parent 
    )
    {
      this->setup = user_setup;
      this->n = n;
      this->l = l;
      this->morton = 0;
      this->treelist_id = 0;
      this->gids = gids;
      this->lids = lids;
      this->isleaf = false;
      this->parent = parent;
      this->lchild = NULL;
      this->rchild = NULL;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };

    ~Node() {};

    void Resize( int n )
    {
      this->n = n;
      gids.resize( n );
      lids.resize( n );
    };


    template<bool PREALLOCATE>
    void Split( int dummy )
    {
      assert( N_CHILDREN == 2 );

      int m = setup->m;
      int max_depth = setup->max_depth;

      if ( n > m && l < max_depth || ( PREALLOCATE && kids[ 0 ] ) )
      {
        double beg = omp_get_wtime();
        auto split = setup->splitter( gids, lids );
        double splitter_time = omp_get_wtime() - beg;
        //printf( "splitter %5.3lfs\n", splitter_time );

        if ( std::abs( (int)split[ 0 ].size() - (int)split[ 1 ].size() ) > 1 )
        {
          if ( !has_uneven_split )
          {
            printf( "\n\nWARNING! uneven split. Using random split instead\n\n" );
            has_uneven_split = true;
          }
          //printf( "split[ 0 ].size() %lu split[ 1 ].size() %lu\n", 
          //  split[ 0 ].size(), split[ 1 ].size() );
          split[ 0 ].resize( gids.size() / 2 );
          split[ 1 ].resize( gids.size() - ( gids.size() / 2 ) );
          //#pragma omp parallel for
          for ( size_t i = 0; i < gids.size(); i ++ )
          {
            if ( i < gids.size() / 2 ) split[ 0 ][ i ] = i;
            else                       split[ 1 ][ i - ( gids.size() / 2 ) ] = i;
          }
        }

        for ( int i = 0; i < N_CHILDREN; i ++ )
        {
          int nchild = split[ i ].size();
     
          if ( PREALLOCATE )
          {
            assert( kids[ i ] );
            kids[ i ]->Resize( nchild );
          }
          else
          {
            kids[ i ] = new Node( setup, nchild, l + 1, this );
            printf( "bug here\n" );
          }

          //#pragma omp parallel for
          for ( int j = 0; j < nchild; j ++ )
          {
            kids[ i ]->gids[ j ] = gids[ split[ i ][ j ] ];
            kids[ i ]->lids[ j ] = lids[ split[ i ][ j ] ];
          }
        }

        /** facilitate binary tree */
        if ( N_CHILDREN > 1  )
        {
          lchild = kids[ 0 ];
          rchild = kids[ 1 ];
          if ( lchild ) lchild->sibling = rchild;
          if ( rchild ) rchild->sibling = lchild;
        }
      }
      else
      {
        if ( PREALLOCATE ) assert( kids[ 0 ] == NULL );
        isleaf = true;
      }
	  
    }; // end Split()


    /**
     *  @brief Check if this node contain any query using morton.
		 *         Notice that queries[] contains gids; thus, morton[]
		 *         needs to be accessed using gids.
     *
     */ 
    bool ContainAny( std::vector<size_t> &queries )
    {
      if ( !setup->morton.size() )
      {
        printf( "Morton id was not initialized.\n" );
        exit( 1 );
      }
      for ( size_t i = 0; i < queries.size(); i ++ )
      {
        if ( IsMyParent( setup->morton[ queries[ i ] ], morton ) ) 
        {
#ifdef DEBUG_TREE
          printf( "\n" );
          hmlp_print_binary( setup->morton[ queries[ i ] ] );
          hmlp_print_binary( morton );
          printf( "\n" );
#endif
          return true;
        }
      }
      return false;

    }; /** end ContainAny() */


    bool ContainAnyMortonID( std::vector<size_t> &querys )
    {
      return ContainAnyMortonID( querys, morton );
    }; /** end ContainAnyMortonID() */


    bool ContainAny( std::set<Node*> &querys )
    {
      if ( !setup->morton.size() )
      {
        printf( "Morton id was not initialized.\n" );
        exit( 1 );
      }
      for ( auto it = querys.begin(); it != querys.end(); it ++ )
      {
        if ( IsMyParent( (*it)->morton, morton ) ) 
        {
          return true;
        }
      }
      return false;

    }; /** end ContainAny() */


    void Print()
    {
      printf( "l %lu offset %lu n %lu\n", this->l, this->offset, this->n );
      hmlp_print_binary( this->morton );
    };


    // This is the call back pointer to the shared data.
    SETUP *setup;

    /** Per node private data */
    NODEDATA data;

    // number of points in this node.
    std::size_t n;

    // level in the tree
    std::size_t l;

    // Morton id
    std::size_t morton;

    std::size_t offset;



    // In top-down topology order. (-1 if not used)
    std::size_t treelist_id; 

    std::vector<std::size_t> gids;

    std::vector<std::size_t> lids;

    /** These two prunning lists are used when no NN pruning. */
    std::set<size_t> FarIDs;
    std::set<Node*>  FarNodes;
    std::set<size_t> FarNodeMortonIDs;

    /** Only leaf nodes will have this list. */
    std::set<size_t> NearIDs;
    std::set<Node*>  NearNodes;
    std::set<size_t> NearNodeMortonIDs;

    /** These two prunning lists are used when in NN pruning. */
    std::set<size_t> NNFarIDs;
    std::set<Node*>  NNFarNodes;
    std::set<size_t> NNFarNodeMortonIDs;

    /** Only leaf nodes will have this list. */
    std::set<size_t> NNNearIDs;
    std::set<Node*>  NNNearNodes;
    std::set<size_t> NNNearNodeMortonIDs;

    Node *parent = NULL;

    Node *kids[ N_CHILDREN ];

    /** make it easy */
    Node *lchild = NULL; 

    Node *rchild = NULL;

    Node *sibling = NULL;

    bool isleaf;

  private:

};


/**
 *  @brief Data and setup that are shared with all nodes.
 *
 *
 */ 
template<typename SPLITTER, typename T>
class Setup
{
  public:

    Setup() {};

    ~Setup() {};

    /** maximum leaf node size */
    size_t m;
    
    /** by default we use 4 bits = 0-15 levels */
    size_t max_depth = 15;

    /** coordinates (accessed with lids) */
    hmlp::Data<T> *X;

    /** neighbors<distance, gid> (accessed with lids) */
    hmlp::Data<std::pair<T, std::size_t>> *NN;

    /** morton ids */
    std::vector<size_t> morton;

    /** tree splitter */
    SPLITTER splitter;
};


/**
 *
 */ 
template<class SETUP, class NODEDATA, int N_CHILDREN, typename T>
class Tree
{
  public:

    /** define our tree node type as NODE */
    typedef Node<SETUP, N_CHILDREN, NODEDATA, T> NODE;

    /** data shared by all tree nodes */
    SETUP setup;

    /** number of points */
    size_t n;

    /** maximum leaf node size */
    size_t m;

    /** depth of local tree */
    size_t depth;

    std::vector<NODE*> treelist;

    std::deque<NODE*> treequeue;

    /** for omp dependent task */
    char omptasklist[ 1 << 16 ];

    /** constructor */
    Tree() : n( 0 ), m( 0 ), depth( 0 )
    {};

    /** deconstructor */
    ~Tree()
    {
      //printf( "~Tree() shared treelist.size() %lu treequeue.size() %lu\n",
      //    treelist.size(), treequeue.size() );
      for ( int i = 0; i < treelist.size(); i ++ )
      {
        if ( treelist[ i ] ) delete treelist[ i ];
      }
      for ( int i = 0; i < treequeue.size(); i ++ )
      {
        if ( treequeue[ i ] ) delete treequeue[ i ];
      }
      //printf( "end ~Tree() shared\n" );
    };

    /**
     *  @brief gid is the index of the lexicographic matrix order.
     *         lid is the index of the lexicographic storage order.
     *         These two indices are the same in non-distributed
     *         environment.
     */ 
    size_t Getlid( size_t gid ) 
    {
      return gid;
    }; /** end Getlid() */


    /**
     *  currently only used in DrawInteraction()
     */ 
    void Offset( NODE *node, size_t offset )
    {
      if ( node )
      {
        node->offset = offset;
        if ( node->lchild )
        {
          Offset( node->lchild, offset + 0 );
          Offset( node->rchild, offset + node->lchild->gids.size() );
        }
      }
      
    }; /** end Offset() */


    template<size_t LEVELOFFSET=4>
    void Morton( NODE *node, size_t morton )
    {
      if ( node )
      {
        Morton( node->lchild, ( morton << 1 ) + 0 );
        Morton( node->rchild, ( morton << 1 ) + 1 );
        size_t shift = ( 1 << LEVELOFFSET ) - node->l + LEVELOFFSET;
        node->morton = ( morton << shift ) + node->l;
        if ( node->lchild )
        {
#ifdef DEBUG_TREE
          std::cout << IsMyParent( node->lchild->morton, node->rchild->morton ) << std::endl;
          std::cout << IsMyParent( node->lchild->morton, node->morton         ) << std::endl;
#endif
        }
        else /** Setup morton id for all points in the leaf node */
        {
          auto &gids = node->gids;
          for ( size_t i = 0; i < gids.size(); i ++ )
          {
            setup.morton[ gids[ i ] ] = node->morton;
          }
        }
      }
    };


    /**
     *  @TODO: used in FindNearNode, problematic in distributed version
     *
     */ 
    template<size_t LEVELOFFSET=4>
    NODE *Morton2Node( size_t me )
    {
      assert( N_CHILDREN == 2 );
      // Get my level.
      size_t filter = ( 1 << LEVELOFFSET ) - 1;
      size_t mylevel = me & filter;
      auto level_beg = treelist.begin() + ( 1 << mylevel ) - 1;
      // Get my index in this level.
      size_t shift = ( 1 << LEVELOFFSET ) - mylevel + LEVELOFFSET;
      size_t index = me >> shift;
      if ( index >= ( 1 << mylevel ) )
      {
        printf( "level %lu index %lu\n", mylevel, index );
        hmlp::hmlp_print_binary( me );
      }
      return *(level_beg + index);
    };



    /**
     *  @brief Allocate the local tree using the local root
     *         with n points and depth l.
     *
     *         This routine will be called in two situations:
     *         1) called by Tree::TreePartition(), or
     *         2) called by MPITree::TreePartition().
     */ 
    void AllocateNodes( NODE *root )
    {
      /** all assertion */
      assert( N_CHILDREN == 2 );


      /** compute the global tree depth */
			int glb_depth = std::ceil( std::log2( n / m ) );
			if ( glb_depth > setup.max_depth ) glb_depth = setup.max_depth;

			/** compute the local tree depth */
			depth = glb_depth - root->l;

			printf( "local AllocateNodes n %lu m %lu glb_depth %d loc_depth %lu\n", 
					n, m, glb_depth, depth );



      /** clean up  */
      treelist.clear();
      treequeue.clear();

      /** reserve space for local tree nodes */
      treelist.reserve( 1 << ( depth + 1 ) );

      /** assume complete tree, compute the local tree depth first */
      //depth = 0;
      //size_t num_points_per_node = root->n;
      //while ( num_points_per_node > m && root->l + depth < max_depth )
      //{
      //  num_points_per_node = ( num_points_per_node + 1 ) / N_CHILDREN;
      //  depth ++;
      //}
      //size_t num_nodes_in_local_tree = 
      //  ( std::pow( (double)N_CHILDREN, depth + 1 ) - 1 ) / ( N_CHILDREN - 1 );

      /** push root into the treelist */
      treequeue.push_back( root );

      /** allocate children */
      while ( auto *node = treequeue.front() )
      {
        /** assign local tree node id */
        node->treelist_id = treelist.size();
        /** account for the depth of the distributed tree */
        if ( node->l < glb_depth )
        {
          for ( int i = 0; i < N_CHILDREN; i ++ )
          {
            node->kids[ i ] = new NODE( &setup, 0, node->l + 1, node );
            treequeue.push_back( node->kids[ i ] );
          }
					node->lchild = node->kids[ 0 ];
					node->rchild = node->kids[ 1 ];
        }
        else
        {
					/** leaf nodes */
					node->isleaf = true;
          treequeue.push_back( NULL );
        }
        treelist.push_back( node );
        treequeue.pop_front();
      }

    }; /** end AllocateNodes() */




    /**
     *  @brief parition points using a binary tree where the root is given
     *
     */
    void TreePartition( NODE *root )
    {
      assert( N_CHILDREN == 2 );
      
      /** declaration */
      std::deque<NODE*> treequeue;
      int max_depth = setup.max_depth;

      /** reset the warning flag and clean up the treelist */
      has_uneven_split = false;
      treelist.clear();
      treequeue.clear();
      treelist.reserve( ( root->n / m ) * N_CHILDREN );

      /** assume complete tree, compute the local tree depth first */
      depth = 0;
      size_t num_points_per_node = root->n;
      while ( num_points_per_node > m && root->l + depth < max_depth )
      {
        num_points_per_node = ( num_points_per_node + 1 ) / N_CHILDREN;
        depth ++;
      }
      size_t num_nodes_in_local_tree = 
        ( std::pow( (double)N_CHILDREN, depth + 1 ) - 1 ) / ( N_CHILDREN - 1 );

      /** TODO: remove lid in the future?? */
      root->lids = root->gids;


      /** push root into the treelist */
      treequeue.push_back( root );

      /** allocate children */
      while ( auto *node = treequeue.front() )
      {
        /** assign local tree node id */
        node->treelist_id = treelist.size();
        /** account for the depth of the distributed tree */
        if ( node->l < root->l + depth )
        {
          for ( int i = 0; i < N_CHILDREN; i ++ )
          {
            node->kids[ i ] = new NODE( &setup, node->n / N_CHILDREN, node->l + 1, node );
            treequeue.push_back( node->kids[ i ] );
          }
        }
        else
        {
          treequeue.push_back( NULL );
        }
        treelist.push_back( node );
        treequeue.pop_front();
      }

      //printf( "local treelist.size() %lu\n", treelist.size() ); fflush( stdout );







      /** */
      SplitTask<NODE> splittask;
      TraverseDown<false, false>( splittask );


    }; /** end TreePartition() */



    /**
     *  @brief shared memory version.
     *
     */ 
    void TreePartition
    (
      std::vector<std::size_t> &gids,
      std::vector<std::size_t> &lids
    )
    {
      assert( N_CHILDREN == 2 );

      double beg, alloc_time, split_time, morton_time, permute_time;

      std::deque<NODE*> treequeue;

      this->n = gids.size();
      this->m = setup.m;
      int max_depth = setup.max_depth;

      beg = omp_get_wtime();

      /** reset the warning flag and clean up the treelist */
      has_uneven_split = false;
      treelist.clear();
      treequeue.clear();
      treelist.reserve( ( n / m ) * N_CHILDREN );

      //auto *root = new NODE( &setup, n, 0, gids, lids, NULL );
  
//      /** root */
//      treequeue.push_back( new NODE( &setup, n, 0, gids, lids, NULL ) );
//    
//      // TODO: there is parallelism to be exploited here.
//      while ( auto *node = treequeue.front() )
//      {
//        node->treelist_id = treelist.size();
//        node->Split<false>( 0 );
//        for ( int i = 0; i < N_CHILDREN; i ++ )
//        {
//          treequeue.push_back( node->kids[ i ] );
//        }
//        treelist.push_back( node );
//        treequeue.pop_front();
//      }




      // Assume complete tree, compute the tree level first.
      depth = 0;
      size_t n_per_node = n;
      while ( n_per_node > m && depth < max_depth )
      {
        n_per_node = ( n_per_node + 1 ) / N_CHILDREN;
        depth ++;
      }
      size_t n_node = ( std::pow( (double)N_CHILDREN, depth + 1 ) - 1 ) / ( N_CHILDREN - 1 );
      //printf( "n %lu m %lu n_per_node %lu depth %lu n_nodes %lu\n", 
      //    n, m, n_per_node, depth, n_node );

      auto *root = new NODE( &setup, n, 0, gids, lids, NULL );
      treequeue.push_back( root );
      while ( auto *node = treequeue.front() )
      {
        node->treelist_id = treelist.size();
        if ( node->l < depth )
        {
          for ( int i = 0; i < N_CHILDREN; i ++ )
          {
            node->kids[ i ] = new NODE( &setup, node->n / N_CHILDREN, node->l + 1, node );
            treequeue.push_back( node->kids[ i ] );
          }
        }
        else
        {
          treequeue.push_back( NULL );
        }
        treelist.push_back( node );
        treequeue.pop_front();
      }

      alloc_time = omp_get_wtime() - beg;


      beg = omp_get_wtime();
      SplitTask<NODE> splittask;
      TraverseDown<false, false>( splittask );
      split_time = omp_get_wtime() - beg;







      /** all tree nodes were created. Now decide tree depth */
      if ( treelist.size() )
      {
        treelist.shrink_to_fit();
        depth = treelist.back()->l;
        assert( treelist.size() == (2 << depth) - 1 );
        //printf( "depth %lu number of tree nides %lu\n", depth, treelist.size() );
      }

      beg = omp_get_wtime();
      if ( N_CHILDREN == 2 )
      {
        setup.morton.resize( n );
        Morton( treelist[ 0 ], 0 );
        Offset( treelist[ 0 ], 0 );
      }
      else
      {
        printf( "No morton id available\n" );
      }
      morton_time = omp_get_wtime() - beg;


      /** adgust lids and gids to the appropriate order */
      beg = omp_get_wtime();
      IndexPermuteTask<NODE> indexpermutetask;
      TraverseUp<false, false>( indexpermutetask );
      permute_time = omp_get_wtime() - beg;

      //printf( "alloc %5.3lfs split %5.3lfs morton %5.3lfs permute %5.3lfs\n", 
      //  alloc_time, split_time, morton_time, permute_time );
    };


    /**
     *  @brief This routine can perform up to e different tree traversals
     *         together via omp task depend. However, currently gcc/g++
     *         fail to compile the following conext. Thus, this part of
     *         the code is only compiled with Intel compilers.
     *         [WARNING]
     */ 
    template<bool UPWARD = true, bool UNORDERED = true, bool DOWNWARD = true, 
      class TASK1, class TASK2, class TASK3>
    void UpDown( TASK1 &dummy1, TASK2 &dummy2, TASK3 &dummy3 )
    {
      
/** the following code is only compiled with Intel compilers */
#ifdef USE_INTEL
      if ( UPWARD )
      {
        #pragma omp parallel
        #pragma omp single
        {
          for ( int me = treelist.size(); me >= 1; me -- )
          {
            int lchild = me * 2;
            int rchild = lchild + 1;
            #pragma omp task depend(in:omptasklist[lchild],omptasklist[rchild]) depend(out:omptasklist[me])
            {
              auto *node = treelist[ me - 1 ];
              auto *task = new TASK1();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
          }
        } /** end pragma omp */
      }

      if ( UNORDERED )
      {
        #pragma omp parallel
        #pragma omp single
        {
          for ( int me = treelist.size(); me >= 1; me -- )
          {
            #pragma omp task depend(inout:omptasklist[me])
            {
              auto *node = treelist[ me - 1 ];
              auto *task = new TASK2();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
          }
        } /** end pragma omp */
      }

      if ( DOWNWARD )
      {
        #pragma omp parallel
        #pragma omp single
        {
          for ( int me = 1; me <= treelist.size(); me ++ )
          {
            int parent = me / 2;
            #pragma omp task depend(in:omptasklist[parent]) depend(out:omptasklist[me])
            {
              auto *node = treelist[ me - 1 ];
              auto *task = new TASK3();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
          }
        } /** end pragma omp */
      }

#else /** WARNING for other compilers*/
#warning hmlp::tree::UpDown() is only compiled with Intel compilers
      printf( "hmlp::tree::UpDown() is only compiled with Intel compilers\n" );
      exit( 1 );
#endif /** ifdef USE_INTEL */

    }; /** end UpDown() */


    /**
     *
     */ 
    template<class TASK>
    inline void OMPTraverseUp( TASK &dummy )
    {
      for ( int me = treelist.size(); me >= 1; me -- )
      {
        int lchild = me * 2;
        int rchild = lchild + 1;
        #pragma omp task depend(in:omptasklist[lchild],omptasklist[rchild]) depend(out:omptasklist[me])
        {
          auto *node = treelist[ me - 1 ];
          auto *task = new TASK();
          task->Set( node );
          task->Execute( NULL );
          delete task;
          //printf( "me %d\n", me );
        }
      }
    }; /** end OMPTraverseUp() */


    template<class TASK>
    void OMPTraverseDown( TASK &dummy )
    {
      for ( int me = 1; me <= treelist.size(); me ++ )
      {
        int parent = me / 2;
        #pragma omp task depend(in:omptasklist[parent]) depend(out:omptasklist[me])
        {
          auto *node = treelist[ me - 1 ];
          auto *task = new TASK();
          task->Set( node );
          task->Execute( NULL );
          delete task;
          //printf( "me %d\n", me );
        }
      }
    }; /** end OMPTraverseUp() */


    /**
     *
     */ 
    template<class TASK>
    void OMPUnordered( TASK &dummy )
    {
      for ( int me = treelist.size(); me >= 1; me -- )
      {
        #pragma omp task depend(inout:omptasklist[me])
        {
          auto *node = treelist[ me - 1 ];
          auto *task = new TASK();
          task->Set( node );
          task->Execute( NULL );
          delete task;
          //printf( "me %d\n", me );
        }
      }
    }; /** end OMPUnordered() */



    /**
     *  @brief This routine can perform up to e different tree traversals
     *         together via omp task depend. However, currently gcc/g++
     *         fail to compile the following conext. Thus, this part of
     *         the code is only compiled with Intel compilers.
     *         [WARNING]
     */ 
    template<bool RECURSIVE, class TASK>
    void PostOrder( NODE *node, TASK &dummy )
    {

#ifdef USE_INTEL
      if ( RECURSIVE )
      {
        #pragma omp parallel
        #pragma omp single nowait
        {
          for ( int i = 0; i < N_CHILDREN; i ++ )
          {
            if ( node->kids[ i ] )
            {
              #pragma omp task
              PostOrder<RECURSIVE>( node->kids[ i ], dummy );
            }
          }
          #pragma omp taskwait
          {
            auto *task = new TASK();
            task->Set( node );
            task->Execute( NULL );
            delete task;
          }
        }
      }
      else
      {
        #pragma omp parallel
        #pragma omp single
        {
          //int dep[ 1 << 16 ];
          for ( int me = treelist.size(); me >= 1; me -- )
          {
            int lchild = me * 2;
            int rchild = lchild + 1;
            #pragma omp task depend(in:omptasklist[lchild],omptasklist[rchild]) depend(out:omptasklist[me])
            {
              auto *node = treelist[ me - 1 ];
              auto *task = new TASK();
              task->Set( node );
              task->Execute( NULL );
              delete task;
              //printf( "me %d\n", me );
            }
          }
        }
        //printf( "finish omp parallel region\n" ); fflush( stdout );
      }
      //printf( "finish PostOrder\n" ); fflush( stdout );

#else /** WARNING for other compilers*/
#warning hmlp::tree::PostOrder() is only compiled with Intel compilers
      printf( "hmlp::tree::PostOrder() is only compiled with Intel compilers\n" );
      exit( 1 );
#endif
    };


    /**
     *
     */ 
    template<bool SORTED, typename KNNTASK>
    hmlp::Data<std::pair<T, std::size_t>> AllNearestNeighbor
    (
      std::size_t n_tree,
      std::size_t k, std::size_t max_depth,
      std::vector<std::size_t> &gids,
      std::vector<std::size_t> &lids,
      std::pair<T, std::size_t> initNN,
      KNNTASK &dummy
    )
    {
      /** k-by-N */
      hmlp::Data<std::pair<T, std::size_t>> NN( k, gids.size(), initNN );

      /** use leaf size = 4 * k  */
      setup.m = 4 * k;
      if ( setup.m < 32 ) setup.m = 32;
      setup.NN = &NN;

      /** Clean the treelist. */
      #pragma omp parallel for
      for ( int i = 0; i < treelist.size(); i ++ ) delete treelist[ i ];
      treelist.clear();

      double flops= 0.0; 
      double mops= 0.0;

      // This loop has to be sequential to prevent from race condiditon on NN.
      if ( REPORT_ANN_STATUS )
      {
        printf( "========================================================\n");
      }
      for ( int t = 0; t < n_tree; t ++ )      
      {
        //TreePartition( 2 * k, max_depth, gids, lids );

        //Flops/Mops for tree partitioning
        flops += std::log( gids.size() / setup.m ) * gids.size();
        mops  += std::log( gids.size() / setup.m ) * gids.size();

        TreePartition( gids, lids );
        TraverseLeafs<false, false>( dummy );

        /** TODO: need to redo the way without using dummy */
        flops += dummy.event.GetFlops();
        mops  += dummy.event.GetMops();

        /** Report accuracy */
        double knn_acc = 0.0;
        size_t num_acc = 0;

        std::size_t n_nodes = 1 << depth;
        auto level_beg = treelist.begin() + n_nodes - 1;
        for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          knn_acc += node->data.knn_acc;
          num_acc += node->data.num_acc;
        }
        if ( REPORT_ANN_STATUS )
        {
          printf( "ANN iter %2d, average accuracy %.2lf%% (over %4lu samples)\n", 
              t, knn_acc / num_acc, num_acc );
        }

        /** clean up for the new iteration */
        #pragma omp parallel for
        for ( int i = 0; i < treelist.size(); i ++ ) delete treelist[ i ];
        treelist.clear();

        /** increase leaf size */
        if ( knn_acc / num_acc < 0.8 )
        { 
          if ( 2.0 * setup.m < 2048 ) setup.m = 2.0 * setup.m;
        }
        else break;


#ifdef DEBUG_TREE
        printf( "Iter %2d NN 0 ", t );
        for ( size_t i = 0; i < NN.row(); i ++ )
        {
          printf( "%E(%lu) ", NN[ i ].first, NN[ i ].second );
        }
        printf( "\n" );
#endif
      }
      if ( REPORT_ANN_STATUS )
      {
        printf( "========================================================\n\n");
      }

      
      if ( SORTED )
      {
        struct 
        {
          bool operator () ( std::pair<T, size_t> a, std::pair<T, size_t> b )
          {   
            return a.first < b.first;
          }   
        } ANNLess;

        // Assuming O(N) sorting
        flops += NN.col() * NN.row();
        // Worst case (2* for swaps, 1* for loads)
        mops += 3* NN.col() * NN.row() ;

#pragma omp parallel for
        for ( size_t j = 0; j < NN.col(); j ++ )
        {
          std::sort( NN.data() + j * NN.row(), NN.data() + ( j + 1 ) * NN.row() );
        }
#ifdef DEBUG_TREE
        printf( "Sorted  NN 0 " );
        for ( size_t i = 0; i < NN.row(); i ++ )
        {
          printf( "%E(%lu) ", NN[ i ].first, NN[ i ].second );
        }
        printf( "\n" );
#endif
      } /** end if ( SORTED ) */

#ifdef DEBUG_TREE
      #pragma omp parallel for
      for ( size_t j = 0; j < NN.col(); j ++ )
      {
        for ( size_t i = 0; i < NN.row(); i ++ )
        {
          if ( NN( i, j ).second >= NN.col() )
          {
            printf( "NN bug ( %lu, %lu ) = %lf, %lu\n", i, j, NN( i, j ).first, NN( i, j ).second );
          }
        }
      }
      printf("flops: %E, mops: %E \n", flops, mops);
#endif

      return NN;
    }; // end AllNearestNeighbor



    template<bool AUTO_DEPENDENCY, bool USE_RUNTIME, class TASK>
    void TraverseUnOrdered( TASK &dummy )
    {
      std::vector<TASK*> tasklist;
      if ( USE_RUNTIME ) tasklist.resize( treelist.size() );

      for ( std::size_t l = 0; l <= depth; l ++ )
      {
        std::size_t n_nodes = 1 << l;
        auto level_beg = treelist.begin() + n_nodes - 1;
        if ( !USE_RUNTIME )
        {
          #pragma omp parallel for 
          for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            auto *task = new TASK();
            task->Set( node );
            task->Execute( NULL );
            delete task;
          }
        }
        else // using dynamic scheduling
        {
          for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            // Create tasks
            tasklist[ node->treelist_id ] = new TASK();
            auto *task = tasklist[ node->treelist_id ];
            task->Submit();
            task->Set( node );
            if ( AUTO_DEPENDENCY )
            {
              task->DependencyAnalysis();
            }
            else
            {
              task->Enqueue();
            }
          }
        }
      }

    }; /** end TraverseUnOrdered() */



    template<bool AUTO_DEPENDENCY, bool USE_RUNTIME, typename TASK>
    void TraverseLeafs( TASK &dummy )
    {
      assert( N_CHILDREN == 2 );

      std::vector<TASK*> tasklist;
      int n_nodes = 1 << depth;
      auto level_beg = treelist.begin() + n_nodes - 1;
      /** this will result in seg fault due to double free. */
      //dummy.event = Event();

      if ( USE_RUNTIME ) tasklist.resize( treelist.size() );

      if ( !USE_RUNTIME )
      {
         #pragma omp parallel for
         for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
         {
           auto *node = *(level_beg + node_ind);
           auto *task = new TASK();
           task->Set( node );
           task->Execute( NULL );
           //dummy.event.AddFlopsMops( task->event.GetFlops(), task->event.GetMops());
           delete task;
         }
      }
      else
      {
        tasklist.resize( n_nodes );
        for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          tasklist[ node->treelist_id ] = new TASK();
          auto *task = tasklist[ node->treelist_id ];
          task->Submit();
          task->Set( node );

		  //printf( "node->treelist_id %lu\n", node->treelist_id ); fflush( stdout );

          if ( node->kids[ 0 ] )
          {
            printf( "There should not be inner nodes in TraverseLeafs.\n" );
          }
          else
          {
            if ( AUTO_DEPENDENCY ) task->DependencyAnalysis();
            else                   task->Enqueue();
          }
		  
		  //printf( "end node->treelist_id %lu\n", node->treelist_id ); fflush( stdout );
        }
      }
    }; /** end TraverseLeafs() */


    template<bool AUTO_DEPENDENCY, bool USE_RUNTIME, typename TASK>
    void TraverseUp( TASK &dummy )
    {
#ifdef DEBUG_TREE
      printf( "TraverseUp()\n" );
#endif
      assert( N_CHILDREN == 2 );

      std::vector<TASK*> tasklist;
      if ( USE_RUNTIME ) tasklist.resize( treelist.size() );

      // IMPORTANT: here l must be int, use unsigned int will wrap over.
      for ( int l = depth; l >= 0; l -- )
      {
        std::size_t n_nodes = 1 << l;
        auto level_beg = treelist.begin() + n_nodes - 1;

        if ( !USE_RUNTIME )
        {
          int nthd_glb = omp_get_max_threads();
          
          if ( n_nodes >= nthd_glb || n_nodes == 1 )
          {
            #pragma omp parallel for if ( n_nodes > nthd_glb / 2 ) schedule( dynamic )
            for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
            {
              auto *node = *(level_beg + node_ind);
              auto *task = new TASK();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
          }
          else
          {
            #pragma omp parallel for schedule( dynamic )
            for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
            {
              auto *node = *(level_beg + node_ind);
              auto *task = new TASK();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
          }
        }
        else // using dynamic scheduling
        {
          for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            // Create tasks
            tasklist[ node->treelist_id ] = new TASK();
            auto *task = tasklist[ node->treelist_id ];
            task->Submit();
            task->Set( node );

            // Setup dependencies
            if ( AUTO_DEPENDENCY )
            {
              task->DependencyAnalysis();
            }
            else
            {
              if ( node->kids[ 0 ] )
              {
#ifdef DEBUG_TREE
                printf( "DependencyAdd %lu -> %lu\n", node->kids[ 0 ]->treelist_id, node->treelist_id );
                printf( "DependencyAdd %lu -> %lu\n", node->kids[ 1 ]->treelist_id, node->treelist_id );
#endif
                Scheduler::DependencyAdd( tasklist[ node->kids[ 0 ]->treelist_id ], task );
                Scheduler::DependencyAdd( tasklist[ node->kids[ 1 ]->treelist_id ], task );
              }
              else // leafnodes, directly enqueue if not depends on the preivous task
              {
                task->Enqueue();
              }
            }
          }
        }
      }
#ifdef DEBUG_TREE
      printf( "end TraverseUp()\n" );
#endif
    }; /** end TraverseUp() */


    /** 
     */
    template<bool AUTO_DEPENDENCY, bool USE_RUNTIME, typename TASK>
    void TraverseDown( TASK &dummy )
    {
      std::vector<TASK*> tasklist;
      if ( USE_RUNTIME ) tasklist.resize( treelist.size() );

      for ( std::size_t l = 0; l <= depth; l ++ )
      {
        std::size_t n_nodes = 1 << l;
        auto level_beg = treelist.begin() + n_nodes - 1;

        if ( !USE_RUNTIME )
        {
          int nthd_glb = omp_get_max_threads();
          
          if ( n_nodes >= nthd_glb || n_nodes == 1 )
		  {
            #pragma omp parallel for if ( n_nodes > nthd_glb / 2 )
            for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
            {
              auto *node = *(level_beg + node_ind);
              auto *task = new TASK();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
		  }
		  else
		  {
            #pragma omp parallel for schedule( dynamic )
            for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
            {
              //omp_set_num_threads( nthd_loc );
              auto *node = *(level_beg + node_ind);
              auto *task = new TASK();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
		  }
        }
        else // using dynamic scheduling
        {
          for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            // Create tasks
            tasklist[ node->treelist_id ] = new TASK();
            auto *task = tasklist[ node->treelist_id ];
            task->Submit();
            task->Set( node );

            if ( AUTO_DEPENDENCY )
            {
              task->DependencyAnalysis();
            }
            else
            {
              if ( node->parent )
              {
#ifdef DEBUG_TREE
                printf( "DependencyAdd %d -> %d\n", node->parent->treelist_id, node->treelist_id );
#endif
                Scheduler::DependencyAdd( tasklist[ node->parent->treelist_id ], task );
              }
              else // root, directly enqueue
              {
                task->Enqueue();
              }
            }
          }
        }
      }
    }; /** end TraverseDown() */

    void DependencyCleanUp()
    {
      for ( size_t i = 0; i < treelist.size(); i ++ )
      {
        treelist[ i ]->DependencyCleanUp();
      }
    }; /** end DependencyCleanUp() */


    /**
     *  @brief Summarize all events in each level. 
     *
     */ 
    template<typename SUMMARY>
    void Summary( SUMMARY &summary )
    {
      assert( N_CHILDREN == 2 );

      for ( std::size_t l = 0; l <= depth; l ++ )
      {
        std::size_t n_nodes = 1 << l;
        auto level_beg = treelist.begin() + n_nodes - 1;
        for ( std::size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          summary( node );
        }
      }
    }; // end void Summary()

};

}; // end namespace tree 
}; // end namespace hmlp

#endif // define TREE_HPP
