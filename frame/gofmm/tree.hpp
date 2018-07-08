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


using namespace std;
using namespace hmlp;


namespace hmlp
{

class MortonHelper
{
  public:

    typedef pair<size_t, size_t> Recursor;

    static Recursor Root() 
    { 
      return Recursor( 0, 0 ); 
    };

    static Recursor RecurLeft( Recursor r ) 
    { 
      return Recursor( ( r.first << 1 ) + 0, r.second + 1 ); 
    };

    static Recursor RecurRight( Recursor r ) 
    { 
      return Recursor( ( r.first << 1 ) + 1, r.second + 1 ); 
    };

    static size_t MortonID( Recursor r )
    {
      /** Compute the correct shift. */
      size_t shift = ( 1 << LEVELOFFSET ) - r.second + LEVELOFFSET;
      /** Append the depth of the tree. */
      return ( r.first << shift ) + r.second;
    };

    static size_t SiblingMortonID( Recursor r )
    {
      /** Compute the correct shift. */
      size_t shift = ( 1 << LEVELOFFSET ) - r.second + LEVELOFFSET;
      if ( r.first % 2 )
        return ( ( r.first - 1 ) << shift ) + r.second;
      else
        return ( ( r.first + 1 ) << shift ) + r.second;
    };

    //static size_t MortonID( size_t recursive_morton, size_t depth )
    //{
    //  /** Compute the correct shift. */
    //  size_t shift = ( 1 << LEVELOFFSET ) - depth + LEVELOFFSET;
    //  /** Append the depth of the tree. */
    //  return ( recursive_morton << shift ) + depth;
    //};

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
    static bool IsMyParent( size_t me, size_t it )
    {
      size_t filter = ( 1 << LEVELOFFSET ) - 1;
      size_t itlevel = it & filter;
      size_t mylevel = me & filter;
      size_t itshift = ( 1 << LEVELOFFSET ) - itlevel + LEVELOFFSET;
      bool is_my_parent = !( ( me ^ it ) >> itshift ) && ( itlevel <= mylevel );
    #ifdef DEBUG_TREE
      hmlp_print_binary( me );
      hmlp_print_binary( it );
      hmlp_print_binary( ( me ^ it ) >> itshift );
      printf( "ismyparent %d itlevel %lu mylevel %lu shift %lu fixed shift %d\n",
          is_my_parent, itlevel, mylevel, itshift, 1 << LEVELOFFSET );
    #endif
      return is_my_parent;
    }; /** end IsMyParent() */


    template<typename TQUERY>
    static bool ContainAny( size_t target, TQUERY &querys )
    {
      for ( auto & q : querys ) 
        if ( IsMyParent( q, target ) ) return true;
      return false;
    }; /** end ContainAny() */


  private:

    const static int LEVELOFFSET = 5;

}; /** end class MortonHelper */
  

template<typename T>
bool less_first( const pair<T, size_t> &a, const pair<T, size_t> &b )
{
  return ( a.first < b.first );
};
template<typename T>
bool less_second( const pair<T, size_t> &a, const pair<T, size_t> &b )
{
  return ( a.second < b.second );
};
template<typename T>
bool equal_second( const pair<T, size_t> &a, const pair<T, size_t> &b )
{
  return ( a.second == b.second );
};
  
  

template<typename T>
void MergeNeighbors( size_t k, pair<T, size_t> *A, 
    pair<T, size_t> *B, vector<pair<T, size_t>> &aux )
{
  /** Enlarge temporary buffer if it is too small. */
  if ( aux.size() != 2 * k ) aux.resize( 2 * k );

  for ( size_t i = 0; i < k; i++ ) aux[     i ] = A[ i ];
  for ( size_t i = 0; i < k; i++ ) aux[ k + i ] = B[ i ];

  sort( aux.begin(), aux.end(), less_second<T> );
  auto it = unique( aux.begin(), aux.end(), equal_second<T> );
  sort( aux.begin(), it, less_first<T> );

  for ( size_t i = 0; i < k; i++ ) A[ i ] = aux[ i ];
}; /** end MergeNeighbors() */


template<typename T>
void MergeNeighbors( size_t k, size_t n,
  vector<pair<T, size_t>> &A, vector<pair<T, size_t>> &B )
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
}; /** end MergeNeighbors() */









namespace tree
{
/**
 *  @brief Permuate the order of gids for each internal node
 *         to the order of leaf nodes.
 *         
 *  @para  The parallelism is exploited in the task level using a
 *         bottom up traversal.
 */ 
template<typename NODE>
class IndexPermuteTask : public Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      name = string( "Permutation" );
      arg = user_arg;
      // Need an accurate cost model.
      cost = 1.0;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( R, this );
        arg->rchild->DependencyAnalysis( R, this );
      }
      this->TryEnqueue();
    };


    void Execute( Worker* user_worker )
    {
      auto &gids = arg->gids; 
      auto *lchild = arg->lchild;
      auto *rchild = arg->rchild;
      
      if ( !arg->isleaf )
      {
        auto &lgids = lchild->gids;
        auto &rgids = rchild->gids;
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
class SplitTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      name = string( "Split" );
      arg = user_arg;
      // Need an accurate cost model.
      cost = 1.0;
    };

		void DependencyAnalysis() { arg->DependOnParent( this ); };
		//{
    //  arg->DependencyAnalysis( R, this );
    //  if ( !arg->isleaf )
    //  {
    //    arg->lchild->DependencyAnalysis( RW, this );
    //    arg->rchild->DependencyAnalysis( RW, this );
    //  }
    //  this->TryEnqueue();
		//};

    void Execute( Worker* user_worker ) { arg->Split(); };

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
  /** closure */
  Data<T> *Coordinate = NULL;

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
    for ( size_t i = 0; i < n; i ++ )
    {
      T rcx = 0.0;
      for ( size_t p = 0; p < d; p ++ )
      {
        T tmp = X( p, gids[ i ] ) - centroid[ p ];


        rcx += tmp * tmp;
      }
      if ( rcx > rcx0 ) 
      {
        rcx0 = rcx;
        x0 = i;
      }
    }

    //printf( "After Farest\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //}
    //printf( "\n" );

    // Compute the farest point x1 from x0
    for ( size_t i = 0; i < n; i ++ )
    {
      T rxx = 0.0;
      for ( size_t p = 0; p < d; p ++ )
      {
				T tmp = X( p, gids[ i ] ) - X( p, gids[ x0 ] );
        rxx += tmp * tmp;
      }
      if ( rxx > rx01 )
      {
        rx01 = rxx;
        x1 = i;
      }
    }



    // Compute direction
    for ( size_t p = 0; p < d; p ++ )
    {
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
      median = combinatorics::Select( n, n / 2, projection );
    }
    else
    {
      auto proj_copy = projection;
      sort( proj_copy.begin(), proj_copy.end() );
      median = proj_copy[ n / 2 ];
    }



    split[ 0 ].reserve( n / 2 + 1 );
    split[ 1 ].reserve( n / 2 + 1 );

    /** TODO: Can be parallelized */
    vector<size_t> middle;
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
  /** Closure */
  Data<T> *Coordinate = NULL;

  inline vector<vector<size_t> > operator()
  ( 
    vector<size_t>& gids
  ) const 
  {
    assert( N_SPLIT == 2 );

    Data<T> &X = *Coordinate;
    size_t d = X.row();
    size_t n = gids.size();

    vector<vector<size_t> > split( N_SPLIT );

    vector<T> direction( d );
    vector<T> projection( n, 0.0 );

    // Compute random direction
    static default_random_engine generator;
    normal_distribution<T> distribution;
    for ( int p = 0; p < d; p ++ )
    {
      direction[ p ] = distribution( generator );
    }

    // Compute projection
    projection.resize( n, 0.0 );
    for ( size_t i = 0; i < n; i ++ )
      for ( size_t p = 0; p < d; p ++ )
        projection[ i ] += X( p, gids[ i ] ) * direction[ p ];


    // Parallel median search
    // T median = Select( n, n / 2, projection );
    auto proj_copy = projection;
    sort( proj_copy.begin(), proj_copy.end() );
    T median = proj_copy[ n / 2 ];

    split[ 0 ].reserve( n / 2 + 1 );
    split[ 1 ].reserve( n / 2 + 1 );

    /** TODO: Can be parallelized */
    vector<size_t> middle;
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
//template<typename SETUP, int N_CHILDREN, typename NODEDATA>
template<typename SETUP, typename NODEDATA>
class Node : public ReadWrite
{
  public:

    /** Deduce data type from SETUP. */
    typedef typename SETUP::T T;
    /** Use binary trees. */
    static const int N_CHILDREN = 2;

    Node( SETUP* setup, size_t n, size_t l, 
        Node *parent, unordered_map<size_t, Node*> *morton2node, Lock *treelock )
    {
      this->setup = setup;
      this->n = n;
      this->l = l;
      this->morton = 0;
      this->treelist_id = 0;
      this->gids.resize( n );
      this->isleaf = false;
      this->parent = parent;
      this->lchild = NULL;
      this->rchild = NULL;
      this->morton2node = morton2node;
      this->treelock = treelock;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };

    Node( SETUP *setup, int n, int l, vector<size_t> gids,
      Node *parent, unordered_map<size_t, Node*> *morton2node, Lock *treelock )
    {
      this->setup = setup;
      this->n = n;
      this->l = l;
      this->morton = 0;
      this->treelist_id = 0;
      this->gids = gids;
      this->isleaf = false;
      this->parent = parent;
      this->lchild = NULL;
      this->rchild = NULL;
      this->morton2node = morton2node;
      this->treelock = treelock;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };
  

    /**
     *  Constructor of local essential tree (LET) node:
     *  This constructor will only be used in the distributed environment.
     */ 
    Node( size_t morton ) { this->morton = morton; };

    /** (Default) destructor */
    ~Node() {};

    void Resize( int n )
    {
      this->n = n;
      gids.resize( n );
    };


    void Split()
    {
      /** Early return if this is a leaf node. */
      if ( isleaf ) return;

      int m = setup->m;
      int max_depth = setup->max_depth;

      double beg = omp_get_wtime();
      auto split = setup->splitter( gids );
      double splitter_time = omp_get_wtime() - beg;
      //printf( "splitter %5.3lfs\n", splitter_time );

      if ( std::abs( (int)split[ 0 ].size() - (int)split[ 1 ].size() ) > 1 )
      {
        if ( !has_uneven_split )
        {
          printf( "\n\nWARNING! uneven split. Using random split instead %lu %lu\n\n",
              split[ 0 ].size(), split[ 1 ].size() );
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

      for ( size_t i = 0; i < N_CHILDREN; i ++ )
      {
        int nchild = split[ i ].size();

        /** TODO: need a better way */ 
        kids[ i ]->Resize( nchild );
        for ( int j = 0; j < nchild; j ++ )
        {
          kids[ i ]->gids[ j ] = gids[ split[ i ][ j ] ];
        }
      }
    }; /** end Split() */


    /**
     *  @brief Check if this node contain any query using morton.
		 *         Notice that queries[] contains gids; thus, morton[]
		 *         needs to be accessed using gids.
     *
     */ 
    bool ContainAny( vector<size_t> &queries )
    {
      if ( !setup->morton.size() )
      {
        printf( "Morton id was not initialized.\n" );
        exit( 1 );
      }
      for ( size_t i = 0; i < queries.size(); i ++ )
      {
        if ( MortonHelper::IsMyParent( setup->morton[ queries[ i ] ], morton ) ) 
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


    bool ContainAny( set<Node*> &querys )
    {
      if ( !setup->morton.size() )
      {
        printf( "Morton id was not initialized.\n" );
        exit( 1 );
      }
      for ( auto it = querys.begin(); it != querys.end(); it ++ )
      {
        if ( MortonHelper::IsMyParent( (*it)->morton, morton ) ) 
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


    /** Support dependency analysis. */
    void DependOnChildren( Task *task )
    {
      if ( this->lchild ) this->lchild->DependencyAnalysis( R, task );
      if ( this->rchild ) this->rchild->DependencyAnalysis( R, task );
      this->DependencyAnalysis( RW, task );
      /** Try to enqueue if there is no dependency. */
      task->TryEnqueue();
    };

    void DependOnParent( Task *task )
    {
      this->DependencyAnalysis( R, task );
      if ( this->lchild ) this->lchild->DependencyAnalysis( RW, task );
      if ( this->rchild ) this->rchild->DependencyAnalysis( RW, task );
      /** Try to enqueue if there is no dependency. */
      task->TryEnqueue();
    };

    void DependOnNoOne( Task *task )
    {
      this->DependencyAnalysis( RW, task );
      /** Try to enqueue if there is no dependency. */
      task->TryEnqueue();
    };


    /** This is the call back pointer to the shared setup. */
    SETUP *setup = NULL;

    /** Per node private data */
    NODEDATA data;

    /** Number of points in this node. */
    size_t n;

    /** Level in the tree */
    size_t l;

    /** Morton ID and offset. */
    size_t morton = 0;
    size_t offset = 0;

    /** ID in top-down topology order. */
    size_t treelist_id; 

    vector<size_t> gids;

    /** These two prunning lists are used when no NN pruning. */
    set<size_t> FarIDs;
    set<Node*>  FarNodes;
    set<size_t> FarNodeMortonIDs;

    /** Only leaf nodes will have this list. */
    set<size_t> NearIDs;
    set<Node*>  NearNodes;
    set<size_t> NearNodeMortonIDs;

    /** These two prunning lists are used when in NN pruning. */
    set<size_t> NNFarIDs;
    set<Node*>  NNFarNodes;
    set<Node*>  ProposedNNFarNodes;
    set<size_t> NNFarNodeMortonIDs;

    /** Only leaf nodes will have this list. */
    set<size_t> NNNearIDs;
    set<Node*>  NNNearNodes;
    set<Node*>  ProposedNNNearNodes;
    set<size_t> NNNearNodeMortonIDs;

    /** DistFar[ p ] contains a pair of gid and cached KIJ received from p. */
    vector<map<size_t, Data<T>>> DistFar;
    vector<map<size_t, Data<T>>> DistNear;




    /** Lock for exclusively modifying or accessing the tree.  */ 
    Lock *treelock = NULL;

    /** All points to other tree nodes.  */ 
    Node *kids[ N_CHILDREN ];
    Node *lchild  = NULL; 
    Node *rchild  = NULL;
    Node *sibling = NULL;
    Node *parent  = NULL;
    unordered_map<size_t, Node*> *morton2node = NULL;

    bool isleaf;

  private:

}; /** end class Node */


/**
 *  @brief Data and setup that are shared with all nodes.
 *
 *
 */ 
template<typename SPLITTER, typename DATATYPE>
class Setup
{
  public:

    typedef DATATYPE T;

    Setup() {};

    ~Setup() {};

    /** maximum leaf node size */
    size_t m = 0;
    
    /** by default we use 4 bits = 0-15 levels */
    size_t max_depth = 15;

    /** Coordinates (accessed with gids) */
    Data<T> *X = NULL;

    /** neighbors<distance, gid> (accessed with gids) */
    Data<pair<T, size_t>> *NN = NULL;

    /** MortonIDs of all indices. */
    vector<size_t> morton;

    /** Tree splitter */
    SPLITTER splitter;



    /**
     *  @brief Check if this node contain any query using morton.
		 *         Notice that queries[] contains gids; thus, morton[]
		 *         needs to be accessed using gids.
     *
     */ 
		vector<size_t> ContainAny( vector<size_t> &queries, size_t target )
    {
			vector<size_t> validation( queries.size(), 0 );

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


       //if ( tree::IsMyParent( morton[ queries[ i ] ], target ) ) 
       if ( MortonHelper::IsMyParent( morton[ queries[ i ] ], target ) ) 
				 validation[ i ] = 1;

      }
      return validation;

    }; /** end ContainAny() */









}; /** end class Setup */


/** */
template<class SETUP, class NODEDATA>
class Tree
{
  public:

    typedef typename SETUP::T T;
    /** Define our tree node type as NODE. */
    typedef Node<SETUP, NODEDATA> NODE;

    static const int N_CHILDREN = 2;



    /** data shared by all tree nodes */
    SETUP setup;

    /** number of points */
    size_t n = 0;

    /** maximum leaf node size */
    size_t m = 0;

    /** depth of local tree */
    size_t depth = 0;


    /** Mutex for exclusive right to modify treelist and morton2node. */ 
    Lock lock;

    /** Local tree nodes in the top-down order. */ 
    vector<NODE*> treelist;

    /** 
     *  Map MortonID to tree nodes. When distributed tree inherits Tree,
     *  morton2node will also contain distributed and LET node.
     */
    unordered_map<size_t, NODE*> morton2node;

    /** (Default) Tree constructor. */
    Tree() {};

    /** (Default) Tree destructor. */
    ~Tree()
    {
      //printf( "~Tree() shared treelist.size() %lu treequeue.size() %lu\n",
      //    treelist.size(), treequeue.size() );
      for ( int i = 0; i < treelist.size(); i ++ )
      {
        if ( treelist[ i ] ) delete treelist[ i ];
      }
      morton2node.clear();
      //printf( "end ~Tree() shared\n" );
    };

    /** Currently only used in DrawInteraction() */ 
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


    void RecursiveMorton( NODE *node, MortonHelper::Recursor r ) 
    {
      /** Return dirctly while no children exist. */
      if ( !node ) return;
      /** Set my MortonID. */
      node->morton = MortonHelper::MortonID( r );
      /** Recur to children. */
      RecursiveMorton( node->lchild, MortonHelper::RecurLeft( r ) );
      RecursiveMorton( node->rchild, MortonHelper::RecurRight( r ) );
      /** Fill the  */
      if ( !node->lchild )
      {
        for ( auto it : node->gids ) setup.morton[ it ] = node->morton;
      }
    };





//    template<size_t LEVELOFFSET=5>
//    void Morton( NODE *node, size_t morton )
//    {
//      if ( node )
//      {
//        /** Shift = 16 - l + 4. */ 
//        size_t shift = ( 1 << LEVELOFFSET ) - node->l + LEVELOFFSET;
//        node->morton = ( morton << shift ) + node->l;
//
//        /** Recurs. */
//        Morton( node->lchild, ( morton << 1 ) + 0 );
//        Morton( node->rchild, ( morton << 1 ) + 1 );
//
//        if ( node->lchild )
//        {
//#ifdef DEBUG_TREE
//          cout << IsMyParent( node->lchild->morton, node->rchild->morton ) << endl;
//          cout << IsMyParent( node->lchild->morton, node->morton         ) << endl;
//#endif
//        }
//        else /** Setup morton id for all points in the leaf node */
//        {
//          auto &gids = node->gids;
//          for ( size_t i = 0; i < gids.size(); i ++ )
//          {
//            setup.morton[ gids[ i ] ] = node->morton;
//          }
//        }
//      }
//    };


//    /**
//     *  @TODO: used in FindNearNode, problematic in distributed version
//     *
//     */ 
//    template<size_t LEVELOFFSET=5>
//    NODE *Morton2Node( size_t me )
//    {
//      assert( N_CHILDREN == 2 );
//      // Get my level.
//      size_t filter = ( 1 << LEVELOFFSET ) - 1;
//      size_t mylevel = me & filter;
//      auto level_beg = treelist.begin() + ( 1 << mylevel ) - 1;
//      // Get my index in this level.
//      size_t shift = ( 1 << LEVELOFFSET ) - mylevel + LEVELOFFSET;
//      size_t index = me >> shift;
//      if ( index >= ( 1 << mylevel ) )
//      {
//        printf( "level %lu index %lu\n", mylevel, index );
//        hmlp_print_binary( me );
//      }
//      return *(level_beg + index);
//    };



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
      /** Compute the global tree depth using std::log2(). */
			int glb_depth = std::ceil( std::log2( (double)n / m ) );
			if ( glb_depth > setup.max_depth ) glb_depth = setup.max_depth;
			/** Compute the local tree depth. */
			depth = glb_depth - root->l;

			//printf( "local AllocateNodes n %lu m %lu glb_depth %d loc_depth %lu\n", 
			//		n, m, glb_depth, depth );

      /** Clean up and reserve space for local tree nodes. */
      for ( auto node_ptr : treelist ) delete node_ptr;
      treelist.clear();
      morton2node.clear();
      treelist.reserve( 1 << ( depth + 1 ) );
      deque<NODE*> treequeue;
      /** Push root into the treelist. */
      treequeue.push_back( root );


      /** Allocate children with BFS (queue solution). */
      while ( auto *node = treequeue.front() )
      {
        /** Assign local treenode_id. */
        node->treelist_id = treelist.size();
        /** Account for the depth of the distributed tree. */
        if ( node->l < glb_depth )
        {
          for ( int i = 0; i < N_CHILDREN; i ++ )
          {
            node->kids[ i ] = new NODE( &setup, 0, node->l + 1, node, &morton2node, &lock );
            treequeue.push_back( node->kids[ i ] );
          }
					node->lchild = node->kids[ 0 ];
					node->rchild = node->kids[ 1 ];
          if ( node->lchild ) node->lchild->sibling = node->rchild;
          if ( node->rchild ) node->rchild->sibling = node->lchild;
        }
        else
        {
					/** Leaf nodes are annotated with this flag */
					node->isleaf = true;
          treequeue.push_back( NULL );
        }
        treelist.push_back( node );
        treequeue.pop_front();
      }

    }; /** end AllocateNodes() */



    /** @brief Shared-memory tree partition. */ 
    void TreePartition()
    {
      double beg, alloc_time, split_time, morton_time, permute_time;

      //this->n = gids.size();
      //this->m = setup.m;

      printf( "n %lu m %lu\n", n, m ); fflush( stdout );

      this->n = setup.ProblemSize();
      this->m = setup.LeafNodeSize();

      printf( "n %lu m %lu\n", n, m ); fflush( stdout );


      /** Reset and initialize global indices with lexicographical order. */
      global_indices.clear();
      for ( size_t i = 0; i < n; i ++ ) global_indices.push_back( i );


      /** Reset the warning flag and clean up the treelist */
      has_uneven_split = false;

      /** Allocate all tree nodes in advance. */
      beg = omp_get_wtime();
      AllocateNodes( new NODE( &setup, n, 0, global_indices, NULL, &morton2node, &lock ) );
      alloc_time = omp_get_wtime() - beg;

      /** Recursive spliting (topdown). */
      beg = omp_get_wtime();
      SplitTask<NODE> splittask;
      TraverseDown<false>( splittask );
      split_time = omp_get_wtime() - beg;


      /** Compute node and point MortonID. */ 
      setup.morton.resize( n );
      /** Compute MortonID recursively. */
      RecursiveMorton( treelist[ 0 ], MortonHelper::Root() );


      Offset( treelist[ 0 ], 0 );

      /** Construct morton2node map for the local tree. */
      morton2node.clear();
      for ( size_t i = 0; i < treelist.size(); i ++ )
      {
        morton2node[ treelist[ i ]->morton ] = treelist[ i ];
      }


      /** Adgust gids to the appropriate order.  */
      IndexPermuteTask<NODE> indexpermutetask;
      TraverseUp<false>( indexpermutetask );

    }; /** end TreePartition() */



    vector<size_t> GetPermutation()
    {
      int n_nodes = 1 << this->depth;
      auto level_beg = this->treelist.begin() + n_nodes - 1;

      vector<size_t> perm;

      for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
      {
        auto *node = *(level_beg + node_ind);
        auto gids = node->gids;
        perm.insert( perm.end(), gids.begin(), gids.end() );
      }

      return perm;
    }; /** end GetTreePermutation() */





    /**
     *
     */ 
    template<typename KNNTASK>
    Data<pair<T, size_t>> AllNearestNeighbor( size_t n_tree, size_t k, 
      size_t max_depth, pair<T, size_t> initNN,
      KNNTASK &dummy )
    {
      /** k-by-N, neighbor pairs. */
      Data<pair<T, size_t>> NN( k, setup.ProblemSize(), initNN );

      /** Use leaf_size = 4 * k. */
      setup.m = 4 * k;
      if ( setup.m < 32 ) setup.m = 32;
      setup.NN = &NN;

      if ( REPORT_ANN_STATUS )
      {
        printf( "========================================================\n");
      }

      /** This loop has to be sequential to avoid race condiditon on NN. */
      for ( int t = 0; t < n_tree; t ++ )      
      {
        /** Report accuracy */
        double knn_acc = 0.0;
        size_t num_acc = 0;
        /** Randomize metric tree and exhausted search for each leaf node. */
        TreePartition();
        TraverseLeafs<false>( dummy );

        size_t n_nodes = 1 << depth;
        auto level_beg = treelist.begin() + n_nodes - 1;
        for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
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

        /** Increase leaf_size with less than 80% accuracy. */
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
      } /** end for each tree. */

      if ( REPORT_ANN_STATUS )
      {
        printf( "========================================================\n\n");
      }

      /** Sort neighbor pairs in ascending order. */
      #pragma omp parallel for
      for ( size_t j = 0; j < NN.col(); j ++ )
        sort( NN.data() + j * NN.row(), NN.data() + ( j + 1 ) * NN.row() );

      /** Check for illegle values. */
      for ( auto &neig : NN )
      {
        if ( neig.second < 0 || neig.second >= NN.col() )
        {
          printf( "Illegle neighbor gid %lu\n", neig.second );
          break;
        }
      }

      /** Return a matrix of neighbor pairs. */
      return NN;

    }; /** end AllNearestNeighbor() */











    template<bool USE_RUNTIME=true, typename TASK, typename... Args>
    void TraverseLeafs( TASK &dummy, Args&... args )
    {
      /** contain at lesat one tree node */
      assert( this->treelist.size() );

      int n_nodes = 1 << this->depth;
      auto level_beg = this->treelist.begin() + n_nodes - 1;

      if ( USE_RUNTIME )
      {
        for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          RecuTaskSubmit( node, dummy, args... );
        }
      }
      else
      {
        int nthd_glb = omp_get_max_threads();
        /** do not parallelize if there is less nodes than threads */
        #pragma omp parallel for if ( n_nodes > nthd_glb / 2 ) schedule( dynamic )
        for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          RecuTaskExecute( node, dummy, args... );
        }
      }
    }; /** end TraverseLeafs() */





    template<bool USE_RUNTIME=true, typename TASK, typename... Args>
    void TraverseUp( TASK &dummy, Args&... args )
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

      int local_begin_level = ( treelist[ 0 ]->l ) ? 1 : 0;

      /** traverse level-by-level in sequential */
      for ( int l = this->depth; l >= local_begin_level; l -- )
      {
        size_t n_nodes = 1 << l;
        auto level_beg = this->treelist.begin() + n_nodes - 1;


        if ( USE_RUNTIME )
        {
          /** loop over each node at level-l */
          for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            RecuTaskSubmit( node, dummy, args... );
          }
        }
        else
        {
          int nthd_glb = omp_get_max_threads();
          /** do not parallelize if there is less nodes than threads */
          #pragma omp parallel for if ( n_nodes > nthd_glb / 2 ) schedule( dynamic )
          for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            RecuTaskExecute( node, dummy, args... );
          }
        }
      }
    }; /** end TraverseUp() */






    template<bool USE_RUNTIME=true, typename TASK, typename... Args>
    void TraverseDown( TASK &dummy, Args&... args )
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
      int local_begin_level = ( treelist[ 0 ]->l ) ? 1 : 0;

      for ( int l = local_begin_level; l <= this->depth; l ++ )
      {
        size_t n_nodes = 1 << l;
        auto level_beg = this->treelist.begin() + n_nodes - 1;

        if ( USE_RUNTIME )
        {
          /** loop over each node at level-l */
          for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            RecuTaskSubmit( node, dummy, args... );
          }
        }
        else
        {
          int nthd_glb = omp_get_max_threads();
          /** do not parallelize if there is less nodes than threads */
          #pragma omp parallel for if ( n_nodes > nthd_glb / 2 ) schedule( dynamic )
          for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            RecuTaskExecute( node, dummy, args... );
          }
        }
      }
    }; /** end TraverseDown() */



    /**
     *  @brief For unordered traversal, we just call local
     *         downward traversal.
     */ 
    template<bool USE_RUNTIME=true, typename TASK, typename... Args>
    void TraverseUnOrdered( TASK &dummy, Args&... args )
    {
      TraverseDown<USE_RUNTIME>( dummy, args... );
    }; /** end TraverseUnOrdered() */


    void DependencyCleanUp()
    {
      //for ( size_t i = 0; i < treelist.size(); i ++ )
      //{
      //  treelist[ i ]->DependencyCleanUp();
      //}
      for ( auto node : treelist ) node->DependencyCleanUp();

      for ( auto it : morton2node )
      {
        auto *node = it.second;
        if ( node ) node->DependencyCleanUp();
      }
    }; /** end DependencyCleanUp() */

    void ExecuteAllTasks()
    {
      hmlp_run();
      DependencyCleanUp();
    }; /** end ExecuteAllTasks() */



    /** @brief Summarize all events in each level. */ 
    template<typename SUMMARY>
    void Summary( SUMMARY &summary )
    {
      assert( N_CHILDREN == 2 );

      for ( std::size_t l = 0; l <= depth; l ++ )
      {
        size_t n_nodes = 1 << l;
        auto level_beg = treelist.begin() + n_nodes - 1;
        for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          summary( node );
        }
      }
    }; /** end void Summary() */


  private:

  protected:

    vector<size_t> global_indices;

}; /** end class Tree */
}; /** end namespace tree */
}; /** end namespace hmlp */

#endif /** define TREE_HPP */
