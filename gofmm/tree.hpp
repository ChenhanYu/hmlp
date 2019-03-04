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


#include <cassert>
#include <typeinfo>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <set>
#include <vector>
#include <deque>
#include <iostream>
#include <random>
#include <cmath>
#include <cstdint>




/** Use HMLP related support. */
#include <hmlp.h>
#include <hmlp_base.hpp>
/** Use HMLP primitives. */
#include <primitives/combinatorics.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;

#define REPORT_ANN_STATUS 0

//#define DEBUG_TREE 1


bool has_uneven_split = false;




namespace hmlp
{

typedef uint64_t mortonType;
typedef uint64_t sizeType;
typedef uint64_t indexType;
typedef uint32_t depthType;


class MortonHelper
{
  public:

    /** The first value is the MortonId without the depth. */
    typedef pair<mortonType, depthType> Recursor;

    /** The root node's Morton ID is 0. */
    static Recursor Root() noexcept
    { 
      return Recursor( 0, 0 ); 
    };

    /** \brief Move the recursor forward to the left. */
    static Recursor RecurLeft( Recursor r ) noexcept 
    { 
      return Recursor( ( r.first << 1 ) + 0, r.second + 1 ); 
    };

    /** \brief Move the recursor forward to the right. */
    static Recursor RecurRight( Recursor r ) noexcept
    { 
      return Recursor( ( r.first << 1 ) + 1, r.second + 1 ); 
    };

    /** \brief Compute the MortonId from the recursor. */
    static mortonType MortonID( Recursor r ) noexcept
    {
      /* Compute the correct shift. */
      auto shift = getShiftFromDepth( r.second );
      /* Append the depth of the tree. */
      return ( r.first << shift ) + r.second;
    };

    /** \brief Compute the sibling MortonId from the recursor. */
    static mortonType SiblingMortonID( Recursor r )
    {
      /* Compute the correct shift. */
      auto shift = getShiftFromDepth( r.second );
      /* Append the depth of the tree. */
      if ( r.first % 2 )
        return ( ( r.first - 1 ) << shift ) + r.second;
      else
        return ( ( r.first + 1 ) << shift ) + r.second;
    };

    /** 
     *  \brief return the MPI rank that owns it. 
     *  \param [in] it the MortonId it the target node
     *  \param [in] comm_size the global communicator size
     *  \return the mpi rank that owns it
     */
    static int Morton2Rank( mortonType it, int comm_size )
    {
      auto it_depth = getDepthFromMorton( it );
      depthType mpi_depth = 0;
      while ( comm_size >>= 1 ) mpi_depth ++;
      if ( it_depth > mpi_depth ) it_depth = mpi_depth;
      auto it_shift = getShiftFromDepth( it_depth );
      return ( it >> it_shift ) << ( mpi_depth - it_depth );
    }; /* end Morton2rank() */

    //static void Morton2Offsets( Recursor r, depthType depth, vector<size_t> &offsets )
    static void Morton2Offsets( Recursor r, depthType depth, vector<indexType> &offsets )
    {
      if ( r.second == depth ) 
      {
        offsets.push_back( r.first );
      }
      else
      {
        Morton2Offsets( RecurLeft( r ), depth, offsets );
        Morton2Offsets( RecurRight( r ), depth, offsets );
      }
    }; /** end Morton2Offsets() */


    //static vector<size_t> Morton2Offsets( mortonType me, mortonType depth )
    static vector<indexType> Morton2Offsets( mortonType me, mortonType depth )
    {
      //vector<size_t> offsets;
      vector<indexType> offsets;
      auto mydepth = getDepthFromMorton( me );
      assert( mydepth <= depth );
      Recursor r( me >> getShiftFromDepth( mydepth ), mydepth );
      Morton2Offsets( r, depth, offsets );
      return offsets;
    }; /** end Morton2Offsets() */


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
    static bool IsMyParent( mortonType me, mortonType it )
    {
      auto itlevel = getDepthFromMorton( it );
      auto mylevel = getDepthFromMorton( me );
      auto itshift = getShiftFromDepth( itlevel );
      bool is_my_parent = !( ( me ^ it ) >> itshift ) && ( itlevel <= mylevel );
#ifdef DEBUG_TREE
      hmlp_print_binary( me );
      hmlp_print_binary( it );
      hmlp_print_binary( ( me ^ it ) >> itshift );
      printf( "ismyparent %d itlevel %lu mylevel %lu shift %lu fixed shift %d\n",
          is_my_parent, itlevel, mylevel, itshift, 1 << level_offset_ );
#endif
      return is_my_parent;
    }; /** end IsMyParent() */


    /**
     *  \brief Whether the querys contain taraget?
     *  \param [in] target
     *  \param [in] querys
     *  \return true if at lease one query contains the target.
     */ 
    template<typename TQUERY>
    static bool ContainAny( mortonType target, TQUERY &querys )
    {
      for ( auto & q : querys )
      {
        if ( IsMyParent( q, target ) ) return true;
      }
      return false;
    }; /* end ContainAny() */


  private:

    static depthType getDepthFromMorton( mortonType it ) 
    {
      mortonType filter = ( 1 << level_offset_ ) - 1;
      /* Convert mortonType to depthType. */
      return (depthType)(it & filter);
    }; /* end getDepthFromMorton() */

    static depthType getShiftFromDepth( depthType depth )
    {
      return ( 1 << level_offset_ ) - depth + level_offset_;
    }; /* end getShiftFromDepth() */

    /** Reserve 5-bit for the depth. */
    const static int level_offset_ = 5;

}; /* end class MortonHelper */
  

template<typename TKEY, typename TVALUE>
bool less_key( const pair<TKEY, TVALUE> &a, const pair<TKEY, TVALUE> &b )
{
  return ( a.first < b.first );
};

template<typename TKEY, typename TVALUE>
bool less_value( const pair<TKEY, TVALUE> &a, const pair<TKEY, TVALUE> &b )
{
  return ( a.second < b.second );
};

template<typename TKEY, typename TVALUE>
bool equal_value( const pair<TKEY, TVALUE> &a, const pair<TKEY, TVALUE> &b )
{
  return ( a.second == b.second );
};
  
  

/**
 *  \brief
 */ 
template<typename T, typename TINDEX>
void MergeNeighbors( sizeType num_neighbors, std::pair<T, TINDEX> *A, std::pair<T, TINDEX> *B, 
    std::vector<std::pair<T, TINDEX>> &aux )
{
  /* Enlarge temporary buffer if it is too small. */
  aux.resize( 2 * num_neighbors );
  /* Merge two lists into one. */
  for ( sizeType i = 0; i < num_neighbors; i++ ) 
  {
    aux[                 i ] = A[ i ];
    aux[ num_neighbors + i ] = B[ i ];
  }
  /* First sort according to the index. */
  std::sort( aux.begin(), aux.end(), less_value<T, TINDEX> );
  auto last = std::unique( aux.begin(), aux.end(), equal_value<T, TINDEX> );
  std::sort( aux.begin(), last, less_key<T, TINDEX> );
  /* Copy the results back from aux. */
  for ( sizeType i = 0; i < num_neighbors; i++ ) 
  {
    A[ i ] = aux[ i ];
  }
}; /* end MergeNeighbors() */


template<typename T, typename TINDEX>
hmlpError_t MergeNeighbors( sizeType k, sizeType n, 
    std::vector<std::pair<T, TINDEX>> &A, std::vector<std::pair<T, TINDEX>> &B )
{
  /* Check whether A and B have enough elements. */
  if ( A.size() < n * k || B.size() < n * k )
  {
    return HMLP_ERROR_INVALID_VALUE;
  }
  #pragma omp parallel
  {
    /* Allocate 2k auxilary workspace per thread. */
    std::vector<std::pair<T, TINDEX> > aux( 2 * k );
    #pragma omp for
    for( size_t i = 0; i < n; i++ ) 
    {
      MergeNeighbors( k, &(A[ i * k ]), &(B[ i * k ]), aux );
    }
  }
  /* Return with no error. */
  return HMLP_ERROR_SUCCESS;
}; /* end MergeNeighbors() */









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

    NODE *arg = nullptr;

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
      if ( !arg->isLeaf() )
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
      
      if ( !arg->isLeaf() )
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

    NODE *arg = nullptr;

    void Set( NODE *user_arg )
    {
      name = string( "Split" );
      arg = user_arg;
      // Need an accurate cost model.
      cost = 1.0;
    };

    void DependencyAnalysis() { arg->DependOnParent( this ); };

    void Execute( Worker* user_worker ) { arg->Split(); };

}; /** end class SplitTask */


///**
// *  @brief This is the default ball tree splitter. Given coordinates,
// *         compute the direction from the two most far away points.
// *         Project all points to this line and split into two groups
// *         using a median select.
// *
// *  @para
// *
// *  @TODO  Need to explit the parallelism.
// */ 
//template<int N_SPLIT, typename T>
//struct centersplit
//{
//  /** closure */
//  Data<T> *Coordinate = NULL;
//
//  inline vector<vector<size_t> > operator()
//  ( 
//    vector<size_t>& gids
//  ) const 
//  {
//    assert( N_SPLIT == 2 );
//
//    Data<T> &X = *Coordinate;
//    size_t d = X.row();
//    size_t n = gids.size();
//
//    T rcx0 = 0.0, rx01 = 0.0;
//    size_t x0, x1;
//    vector<vector<size_t> > split( N_SPLIT );
//
//
//    vector<T> centroid = combinatorics::Mean( d, n, X, gids );
//    vector<T> direction( d );
//    vector<T> projection( n, 0.0 );
//
//    //printf( "After Mean\n" );
//
//    // Compute the farest x0 point from the centroid
//    for ( size_t i = 0; i < n; i ++ )
//    {
//      T rcx = 0.0;
//      for ( size_t p = 0; p < d; p ++ )
//      {
//        T tmp = X( p, gids[ i ] ) - centroid[ p ];
//
//
//        rcx += tmp * tmp;
//      }
//      if ( rcx > rcx0 ) 
//      {
//        rcx0 = rcx;
//        x0 = i;
//      }
//    }
//
//    //printf( "After Farest\n" );
//    //for ( int p = 0; p < d; p ++ )
//    //{
//    //}
//    //printf( "\n" );
//
//    // Compute the farest point x1 from x0
//    for ( size_t i = 0; i < n; i ++ )
//    {
//      T rxx = 0.0;
//      for ( size_t p = 0; p < d; p ++ )
//      {
//				T tmp = X( p, gids[ i ] ) - X( p, gids[ x0 ] );
//        rxx += tmp * tmp;
//      }
//      if ( rxx > rx01 )
//      {
//        rx01 = rxx;
//        x1 = i;
//      }
//    }
//
//
//
//    // Compute direction
//    for ( size_t p = 0; p < d; p ++ )
//    {
//      direction[ p ] = X( p, gids[ x1 ] ) - X( p, gids[ x0 ] );
//    }
//
//    //printf( "After Direction\n" );
//    //for ( int p = 0; p < d; p ++ )
//    //{
//    //  printf( "%5.2lf ", direction[ p ] );
//    //}
//    //printf( "\n" );
//    //exit( 1 );
//
//
//
//    // Compute projection
//    projection.resize( n, 0.0 );
//    for ( size_t i = 0; i < n; i ++ )
//      for ( size_t p = 0; p < d; p ++ )
//        projection[ i ] += X( p, gids[ i ] ) * direction[ p ];
//
//    //printf( "After Projetion\n" );
//    //for ( int p = 0; p < d; p ++ )
//    //{
//    //  printf( "%5.2lf ", projec[ p ] );
//    //}
//    //printf( "\n" );
//
//
//
//    /** Parallel median search */
//    T median;
//    
//    if ( 1 )
//    {
//      median = combinatorics::Select( n, n / 2, projection );
//    }
//    else
//    {
//      auto proj_copy = projection;
//      sort( proj_copy.begin(), proj_copy.end() );
//      median = proj_copy[ n / 2 ];
//    }
//
//
//
//    split[ 0 ].reserve( n / 2 + 1 );
//    split[ 1 ].reserve( n / 2 + 1 );
//
//    /** TODO: Can be parallelized */
//    vector<size_t> middle;
//    for ( size_t i = 0; i < n; i ++ )
//    {
//      if      ( projection[ i ] < median ) split[ 0 ].push_back( i );
//      else if ( projection[ i ] > median ) split[ 1 ].push_back( i );
//      else                                 middle.push_back( i );
//    }
//
//    for ( size_t i = 0; i < middle.size(); i ++ )
//    {
//      if ( split[ 0 ].size() <= split[ 1 ].size() ) split[ 0 ].push_back( middle[ i ] );
//      else                                          split[ 1 ].push_back( middle[ i ] );
//    }
//
//
//    //printf( "split median %lf left %d right %d\n", 
//    //    median,
//    //    (int)split[ 0 ].size(), (int)split[ 1 ].size() );
//
//    //if ( split[ 0 ].size() > 0.6 * n ||
//    //     split[ 1 ].size() > 0.6 * n )
//    //{
//    //  for ( int i = 0; i < n; i ++ )
//    //  {
//    //    printf( "%E ", projection[ i ] );
//    //  } 
//    //  printf( "\n" );
//    //}
//
//
//    return split; 
//  };
//};
//
//
///**
// *  @brief This is the splitter used in the randomized tree. Given
// *         coordinates, project all points onto a random direction
// *         and split into two groups using a median select.
// *
// *  @para
// *
// *  @TODO  Need to explit the parallelism.
// */ 
//template<int N_SPLIT, typename T>
//struct randomsplit
//{
//  /** Closure */
//  Data<T> *Coordinate = NULL;
//
//  inline vector<vector<size_t> > operator()
//  ( 
//    vector<size_t>& gids
//  ) const 
//  {
//    assert( N_SPLIT == 2 );
//
//    Data<T> &X = *Coordinate;
//    size_t d = X.row();
//    size_t n = gids.size();
//
//    vector<vector<size_t> > split( N_SPLIT );
//
//    vector<T> direction( d );
//    vector<T> projection( n, 0.0 );
//
//    // Compute random direction
//    static default_random_engine generator;
//    normal_distribution<T> distribution;
//    for ( int p = 0; p < d; p ++ )
//    {
//      direction[ p ] = distribution( generator );
//    }
//
//    // Compute projection
//    projection.resize( n, 0.0 );
//    for ( size_t i = 0; i < n; i ++ )
//      for ( size_t p = 0; p < d; p ++ )
//        projection[ i ] += X( p, gids[ i ] ) * direction[ p ];
//
//
//    // Parallel median search
//    // T median = Select( n, n / 2, projection );
//    auto proj_copy = projection;
//    sort( proj_copy.begin(), proj_copy.end() );
//    T median = proj_copy[ n / 2 ];
//
//    split[ 0 ].reserve( n / 2 + 1 );
//    split[ 1 ].reserve( n / 2 + 1 );
//
//    /** TODO: Can be parallelized */
//    vector<size_t> middle;
//    for ( size_t i = 0; i < n; i ++ )
//    {
//      if      ( projection[ i ] < median ) split[ 0 ].push_back( i );
//      else if ( projection[ i ] > median ) split[ 1 ].push_back( i );
//      else                                 middle.push_back( i );
//    }
//
//    for ( size_t i = 0; i < middle.size(); i ++ )
//    {
//      if ( split[ 0 ].size() <= split[ 1 ].size() ) split[ 0 ].push_back( middle[ i ] );
//      else                                          split[ 1 ].push_back( middle[ i ] );
//    }
//
//
//    //printf( "split median %lf left %d right %d\n", 
//    //    median,
//    //    (int)split[ 0 ].size(), (int)split[ 1 ].size() );
//
//    //if ( split[ 0 ].size() > 0.6 * n ||
//    //     split[ 1 ].size() > 0.6 * n )
//    //{
//    //  for ( int i = 0; i < n; i ++ )
//    //  {
//    //    printf( "%E ", projection[ i ] );
//    //  } 
//    //  printf( "\n" );
//    //}
//
//
//    return split; 
//  };
//};


/**
 *  @brief 
 */ 
template<typename SETUP, typename NODEDATA>
class Node : public ReadWrite
{
  public:

    /** Deduce data type from SETUP. */
    typedef typename SETUP::T T;
    /** Use binary trees. */
    static const int N_CHILDREN = 2;

    Node( SETUP* setup, sizeType n, depthType l, 
        Node *parent, unordered_map<mortonType, Node*> *morton2node, Lock *treelock )
    {
      this->setup = setup;
      this->n = n;
      this->l = l;
      this->treelist_id = 0;
      this->gids.resize( n );
      this->parent = parent;
      this->morton2node = morton2node;
      this->treelock = treelock;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };

    Node( SETUP *setup, sizeType n, depthType l, vector<size_t> gids,
      Node *parent, unordered_map<mortonType, Node*> *morton2node, Lock *treelock )
    {
      this->setup = setup;
      this->n = n;
      this->l = l;
      this->treelist_id = 0;
      this->gids = gids;
      this->parent = parent;
      this->morton2node = morton2node;
      this->treelock = treelock;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };
  

    /**
     *  Constructor of local essential tree (LET) node:
     *  This constructor will only be used in the distributed environment.
     */ 
    Node( mortonType morton ) 
    { 
      this->morton_ = morton; 
    };

    /** (Default) destructor */
    ~Node() {};

    void Resize( sizeType n )
    {
      this->n = n;
      gids.resize( n );
    };


    void Split()
    {
      try
      {
        /** Early return if this is a leaf node. */
        if ( isLeaf() ) 
        {
          return;
        }

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
      }
      catch ( const exception & e )
      {
        cout << e.what() << endl;
      }
    }; /** end Split() */


    /**
     *  @brief Check if this node contain any query using morton.
     *         Notice that queries[] contains gids; thus, morton[]
     *         needs to be accessed using gids.
     *
     */ 
    bool ContainAny( const std::vector<size_t> & queries )
    {
      if ( !setup->morton.size() )
      {
        printf( "Morton id was not initialized.\n" );
        exit( 1 );
      }
      for ( size_t i = 0; i < queries.size(); i ++ )
      {
        if ( MortonHelper::IsMyParent( setup->morton[ queries[ i ] ], getMortonID() ) ) 
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
        if ( MortonHelper::IsMyParent( (*it)->getMortonID(), getMortonID() ) ) 
        {
          return true;
        }
      }
      return false;

    }; /** end ContainAny() */


    void Print()
    {
      printf( "l %lu offset %lu n %lu\n", this->getGlobalDepth(), this->offset, this->n );
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
    sizeType n = 0;

    /** Level in the tree */
    //depthType l = 0;

    
    hmlpError_t setMortonID( mortonType morton )
    {
      morton_ = morton;
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }


    /** Morton ID and offset. */
    mortonType getMortonID() const noexcept
    {
      return morton_;
    }
    size_t offset = 0;

    /** ID in top-down topology order. */
    size_t treelist_id; 

    vector<size_t> gids;

    /** These two prunning lists are used when no NN pruning. */
    set<size_t> FarIDs;
    set<Node*>  FarNodes;
    set<mortonType> FarNodeMortonIDs;

    /** Only leaf nodes will have this list. */
    set<size_t> NearIDs;
    set<Node*>  NearNodes;
    set<mortonType> NearNodeMortonIDs;

    /** These two prunning lists are used when in NN pruning. */
    set<size_t> NNFarIDs;
    set<Node*>  NNFarNodes;
    set<Node*>  ProposedNNFarNodes;
    set<mortonType> NNFarNodeMortonIDs;

    /** Only leaf nodes will have this list. */
    set<size_t> NNNearIDs;
    set<Node*>  NNNearNodes;
    set<Node*>  ProposedNNNearNodes;
    set<mortonType> NNNearNodeMortonIDs;

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
    unordered_map<mortonType, Node*> *morton2node = NULL;



    depthType getGlobalDepth() const noexcept
    {
      return l;
    };

    //bool isLeaf();
    bool isLeaf() const noexcept
    {
      return is_leaf_;
    }

    hmlpError_t setLeaf() noexcept
    {
      is_leaf_ = true;
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    };

    bool isCompressionFailureFrontier() const noexcept 
    { 
      return is_compression_failure_frontier_;
    };

    hmlpError_t setCompressionFailureFrontier() noexcept
    {
      is_compression_failure_frontier_ = true;
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    };

  protected:

    /** Level in the tree */
    depthType l = 0;

    /** Node MortonID. */
    mortonType morton_ = 0;

    bool is_leaf_ = false;

    bool is_compression_failure_frontier_ = false;

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

    /** neighbors<distance, gid> (accessed with gids) */
    Data<pair<T, size_t>> *NN = NULL;

    /** MortonIDs of all indices. */
    vector<mortonType> morton;

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
    typedef typename std::pair<T, size_t> neigType;
    /** Define our tree node type as NODE. */
    typedef Node<SETUP, NODEDATA> NODE;

    static const int N_CHILDREN = 2;

    /* Data shared by all tree nodes. */
    SETUP setup;

    /** number of points */
    size_t n = 0;

    /** depth of local tree */
    //size_t depth = 0;


    /** Mutex for exclusive right to modify treelist and morton2node. */ 
    Lock lock;

    /** 
     *  Map MortonID to tree nodes. When distributed tree inherits Tree,
     *  morton2node will also contain distributed and LET node.
     */
    unordered_map<mortonType, NODE*> morton2node;

    /** (Default) Tree constructor. */
    Tree() {};

    /** (Default) Tree destructor. */
    ~Tree()
    {
      //printf( "~Tree() shared treelist.size() %lu treequeue.size() %lu\n",
      //    treelist.size(), treequeue.size() );
      for ( int i = 0; i < treelist_.size(); i ++ )
      {
        if ( treelist_[ i ] ) delete treelist_[ i ];
      }
      morton2node.clear();
      //printf( "end ~Tree() shared\n" );
    };

    /**
     *  \return the local tree height
     */ 
    depthType getLocalHeight() const noexcept 
    { 
      return loc_height_; 
    };

    /**
     *  \return the global tree height
     */ 
    depthType getGlobalHeight() const noexcept 
    { 
      return glb_height_; 
    };


    sizeType getLocalNodeSize() const noexcept
    {
      return treelist_.size();
    }

    NODE* getLocalNodeAt( sizeType i )
    {
      if ( i >= getLocalNodeSize() )
      {
        throw std::out_of_range( "accessing invalid local tree node" );
      }
      return treelist_[ i ];
    }

    NODE* getLocalRoot() 
    {
      if ( getLocalNodeSize() == 0 ) 
      {
        throw std::out_of_range( "accessing the local root while there is no tree node" );
      }
      return treelist_.front();
    }

    NODE* getLocalNodeAt( depthType depth, sizeType i )
    {
      /* Compute number of nodes at this depth. */
      int n_nodes = 1 << depth;
      if ( depth > getLocalHeight() )
      {
        throw std::out_of_range( "accessing invalid local tree depth" );
      }
      if ( i >= n_nodes )
      {
        throw std::out_of_range( "accessing invalid local tree node" );
      }
      /* Compute the iterator.*/
      return *(treelist_.begin() + n_nodes - 1 + i);
    }


    //NODE* getFirstNodeAtLocalDepth( depthType depth ) noexcept
    //{
    //  if ( depth > getLocalHeight() )
    //  {
    //    throw std::out_of_range( "accessing invalid local tree depth" );
    //  }
    //  int n_nodes = 1 << depth;
    //  //auto level_beg = this->treelist_.begin() + n_nodes - 1;

    //  //return getLocalRoot() + ( n_nodes - 1 );
    //  return *(treelist_.begin() + n_nodes - 1);
    //};

    //NODE *getFirstLeafNode() noexcept
    //{
    //  return getFirstNodeAtLocalDepth( getLocalHeight() );
    //};

    std::vector<indexType> & getOwnedIndices()
    {
      if ( getLocalNodeSize() == 0 )
      {
        return no_index_exist_;
      }
      return treelist_[ 0 ]->gids;
    };


    /** Currently only used in DrawInteraction() */ 
    hmlpError_t RecursiveOffset( NODE *node, size_t offset )
    {
      if ( node )
      {
        node->offset = offset;
        if ( node->lchild )
        {
          RETURN_IF_ERROR( RecursiveOffset( node->lchild, offset + 0 ) );
          RETURN_IF_ERROR( RecursiveOffset( node->rchild, offset + node->lchild->gids.size() ) );
        }
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /** end RecursiveOffset() */


    hmlpError_t RecursiveMorton( NODE *node, MortonHelper::Recursor r ) 
    {
      /** Return dirctly while no children exist. */
      if ( !node ) return HMLP_ERROR_SUCCESS;
      /** Set my MortonID. */
      RETURN_IF_ERROR( node->setMortonID( MortonHelper::MortonID( r ) ) );
      /** Recur to children. */
      RETURN_IF_ERROR( RecursiveMorton( node->lchild, MortonHelper::RecurLeft( r ) ) );
      RETURN_IF_ERROR( RecursiveMorton( node->rchild, MortonHelper::RecurRight( r ) ) );
      /** Fill the  */
      if ( !node->lchild )
      {
        for ( auto it : node->gids ) 
        {
          setup.morton[ it ] = node->getMortonID();
        }
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    };


    /** @brief Shared-memory tree partition. */ 
    hmlpError_t TreePartition()
    {
      double beg, alloc_time, split_time, morton_time, permute_time;

      this->n = setup.ProblemSize();
      //this->m = setup.LeafNodeSize();

      /** Reset and initialize global indices with lexicographical order. */
      global_indices.clear();
      for ( size_t i = 0; i < n; i ++ ) global_indices.push_back( i );

      /** Reset the warning flag and clean up the treelist */
      has_uneven_split = false;

      /** Allocate all tree nodes in advance. */
      beg = omp_get_wtime();
      HANDLE_ERROR( allocateNodes( new NODE( &setup, n, 0, global_indices, NULL, &morton2node, &lock ) ) );
      alloc_time = omp_get_wtime() - beg;

      /** Recursive spliting (topdown). */
      beg = omp_get_wtime();
      SplitTask<NODE> splittask;
      RETURN_IF_ERROR( TraverseDown( splittask ) );
      RETURN_IF_ERROR( ExecuteAllTasks() );
      split_time = omp_get_wtime() - beg;


      /** Compute node and point MortonID. */ 
      setup.morton.resize( n );
      /* Compute MortonID (for nodes and indices) recursively. */
      RETURN_IF_ERROR( RecursiveMorton( getLocalRoot(), MortonHelper::Root() ) );
      /* Compute the offset (related to the left most) for drawing interaction. */
      RETURN_IF_ERROR( RecursiveOffset( getLocalRoot(), 0 ) );

      /** Construct morton2node map for the local tree. */
      morton2node.clear();
      for ( size_t i = 0; i < treelist_.size(); i ++ )
      {
        morton2node[ treelist_[ i ]->getMortonID() ] = treelist_[ i ];
      }

      /** Adgust gids to the appropriate order.  */
      IndexPermuteTask<NODE> indexpermutetask;
      RETURN_IF_ERROR( TraverseUp( indexpermutetask ) );
      RETURN_IF_ERROR( ExecuteAllTasks() );

      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /* end TreePartition() */



    vector<size_t> GetPermutation()
    {
      int n_nodes = 1 << this->getLocalHeight();
      auto level_beg = this->treelist_.begin() + n_nodes - 1;

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
     *  \brief Compute approximated kappa-nearest neighbors using 
     *         randomized spatial trees.
     *  \param [in] KNNTASK the type of the knn kernel
     *  \param [in] n_tree the number of maximum iterations
     *  \param [in] kappa the number of neighbors to compute
     *  \param [in] max_depth the maximum depth 
     *  \param [inout] neighbors all neighbors in Data<neigType>
     *  \param [in] initNN the initial value
     *  \param [in] dummy
     *  \return error code
     */ 
    template<typename KNNTASK>
    hmlpError_t AllNearestNeighbor( sizeType n_tree, sizeType kappa, 
      depthType max_depth, Data<neigType>& neighbors, const neigType initNN, KNNTASK &dummy )
    {
      /* Check if arguments are valid. */
      if ( n_tree < 0 || kappa < 0 || max_depth < 1 )
      {
        return HMLP_ERROR_INVALID_VALUE;
      }

      /* k-by-N, neighbor pairs in neigType. */
      neighbors.clear();
      neighbors.resize( kappa, setup.ProblemSize(), initNN );

      /* Use leaf_size = 4 * k. */
      RETURN_IF_ERROR( this->setup.setLeafNodeSize( ( 4 * kappa < 512 ) ? 512 : 4 * kappa ) );

      /* We need to assign the buffer to the setup structure. */
      setup.NN = &neighbors;

      /** This loop has to be sequential to avoid race condiditon on NN. */
      for ( sizeType t = 0; t < n_tree; t ++ )      
      {
        /** Randomize metric tree and exhausted search for each leaf node. */
        RETURN_IF_ERROR( TreePartition() );
        printf( "TreePartition\n" ); fflush( stdout );
        RETURN_IF_ERROR( TraverseLeafs( dummy ) );
        printf( "TraverseLeaf\n" ); fflush( stdout );
        RETURN_IF_ERROR( ExecuteAllTasks() );
      } 

      /** Sort neighbor pairs in ascending order. */
      #pragma omp parallel for
      for ( size_t j = 0; j < neighbors.col(); j ++ )
      {
        sort( neighbors.data() + j * kappa, neighbors.data() + ( j + 1 ) * kappa );
      }

      /** Check for illegle values. */
      for ( size_t j = 0; j < neighbors.col(); j ++ )
      {
        for ( size_t i = 0; i < kappa; i ++ )
        {
          auto & neig = neighbors[ j * kappa + i ];
          if ( neig.second < 0 || neig.second >= neighbors.col() )
          {
            fprintf( stderr, "\n\x1B[31m[ERROR] \x1B Illegle neighbor (%lu,%lu) with gid %lu\n\n", 
                i, j, neig.second );
            return HMLP_ERROR_INTERNAL_ERROR;
          }
        }
      }

      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /* end AllNearestNeighbor() */


    Data<int> CheckAllInteractions()
    {
      /** Get the total depth of the tree. */
      int total_depth = treelist_.back()->getGlobalDepth();
      /** Number of total leaf nodes. */
      int num_leafs = 1 << total_depth;
      /** Create a 2^l-by-2^l table to check all interactions. */
      Data<int> A( num_leafs, num_leafs, 0 );
      /** Now traverse all tree nodes (excluding the root). */
      for ( int t = 1; t < treelist_.size(); t ++ )
      {
        auto *node = treelist_[ t ];
        /** Loop over all near interactions. */
        for ( auto *it : node->NNNearNodes )
        {
          auto I = MortonHelper::Morton2Offsets( node->morton, total_depth );
          auto J = MortonHelper::Morton2Offsets(   it->morton, total_depth );
          for ( auto i : I ) for ( auto j : J ) A( i, j ) += 1; 
        }
        /** Loop over all far interactions. */
        for ( auto *it : node->NNFarNodes )
        {
          auto I = MortonHelper::Morton2Offsets( node->morton, total_depth );
          auto J = MortonHelper::Morton2Offsets(   it->morton, total_depth );
          for ( auto i : I ) for ( auto j : J ) A( i, j ) += 1; 
        }
      }

      for ( size_t i = 0; i < num_leafs; i ++ )
      {
        for ( size_t j = 0; j < num_leafs; j ++ ) printf( "%d", A( i, j ) );
        printf( "\n" );
      }


      return A;
    }; /** end CheckAllInteractions() */









    template<typename TASK, typename... Args>
    hmlpError_t TraverseLeafs( TASK &dummy, Args&... args )
    {
      /* Return with no error. */
      if ( getLocalNodeSize() == 0 )
      {
        return HMLP_ERROR_SUCCESS;
      }

      int n_nodes = 1 << this->getLocalHeight();
      auto level_beg = this->treelist_.begin() + n_nodes - 1;


      if ( out_of_order_traversal )
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
          //auto *node = *(level_beg + node_ind);
          auto *node = *(level_beg + node_ind);
          //auto *node = getFirstLeafNode() + node_ind;
          RecuTaskExecute( node, dummy, args... );
        }
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /** end TraverseLeafs() */





    template<typename TASK, typename... Args>
    hmlpError_t TraverseUp( TASK &dummy, Args&... args )
    {
      /* Return with no error. */
      if ( getLocalNodeSize() == 0 )
      {
        return HMLP_ERROR_SUCCESS;
      }

      /** 
       *  traverse the local tree without the root
       *
       *  IMPORTANT: local root alias of the distributed leaf node
       *  IMPORTANT: here l must be int, size_t will wrap over 
       *
       */

      int local_begin_level = ( treelist_[ 0 ]->getGlobalDepth() ) ? 1 : 0;

      /** traverse level-by-level in sequential */
      for ( int l = this->getLocalHeight(); l >= local_begin_level; l -- )
      {
        size_t n_nodes = 1 << l;
        auto level_beg = this->treelist_.begin() + n_nodes - 1;


        if ( out_of_order_traversal )
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
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /** end TraverseUp() */






    template<typename TASK, typename... Args>
    hmlpError_t TraverseDown( TASK &dummy, Args&... args )
    {
      /* Return with no error. */
      if ( getLocalNodeSize() == 0 )
      {
        return HMLP_ERROR_SUCCESS;
      }

      /** 
       *  traverse the local tree without the root
       *
       *  IMPORTANT: local root alias of the distributed leaf node
       *  IMPORTANT: here l must be int, size_t will wrap over 
       *
       */
      int local_begin_level = ( treelist_[ 0 ]->getGlobalDepth() ) ? 1 : 0;

      for ( int l = local_begin_level; l <= this->getLocalHeight(); l ++ )
      {
        size_t n_nodes = 1 << l;
        auto level_beg = this->treelist_.begin() + n_nodes - 1;

        if ( out_of_order_traversal )
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
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /** end TraverseDown() */



    /**
     *  @brief For unordered traversal, we just call local
     *         downward traversal.
     */ 
    template<typename TASK, typename... Args>
    hmlpError_t TraverseUnOrdered( TASK &dummy, Args&... args )
    {
      return TraverseDown( dummy, args... );
    }; /** end TraverseUnOrdered() */


    hmlpError_t DependencyCleanUp()
    {
      for ( auto node : treelist_ ) 
      {
        node->DependencyCleanUp();
      }

      for ( auto it : morton2node )
      {
        auto *node = it.second;
        if ( node ) node->DependencyCleanUp();
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /* end DependencyCleanUp() */

    hmlpError_t ExecuteAllTasks()
    {
      /* Invoke the runtime scheduler. */
      RETURN_IF_ERROR( hmlp_run() );
      /* Clean up all the in and out set of each read/write object. */
      RETURN_IF_ERROR( DependencyCleanUp() );
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /* end ExecuteAllTasks() */


    bool DoOutOfOrder() { return out_of_order_traversal; };


    /** @brief Summarize all events in each level. */ 
    template<typename SUMMARY>
    void Summary( SUMMARY &summary )
    {
      assert( N_CHILDREN == 2 );

      for ( std::size_t l = 0; l <= getLocalHeight(); l ++ )
      {
        size_t n_nodes = 1 << l;
        auto level_beg = treelist_.begin() + n_nodes - 1;
        for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          summary( node );
        }
      }
    }; /** end void Summary() */


  private:

    bool out_of_order_traversal = true;

  protected:

    /** Depth of the local tree. */
    depthType loc_height_ = 0;
    /** Depth of the global tree. */
    depthType glb_height_ = 0;

    /** Local tree nodes in the top-down order. */ 
    std::vector<NODE*> treelist_;


    /** TODO: what is this? */
    vector<size_t> global_indices;
    /** */
    vector<indexType> no_index_exist_;


    /**
     *  @brief Allocate the local tree using the local root
     *         with n points and depth l.
     *
     *         This routine will be called in two situations:
     *         1) called by Tree::TreePartition(), or
     *         2) called by MPITree::TreePartition().
     */ 
    hmlpError_t allocateNodes( NODE *root )
    {
      /* Compute the global tree depth using std::log2(). */
      glb_height_ = std::ceil( std::log2( (double)n / setup.getLeafNodeSize() ) );
      /* If the global depth exceeds the limit, then set it to the maximum depth. */
      if ( glb_height_ > setup.getMaximumDepth() ) glb_height_ = setup.getMaximumDepth();
      /** Compute the local tree depth. */
      loc_height_ = glb_height_ - root->getGlobalDepth();


      /* Clean up and reserve space for local tree nodes. */
      for ( auto node_ptr : treelist_ ) delete node_ptr;
      treelist_.clear();
      morton2node.clear();
      treelist_.reserve( 1 << ( getLocalHeight() + 1 ) );
      deque<NODE*> treequeue;
      /** Push root into the treelist. */
      treequeue.push_back( root );


      /** Allocate children with BFS (queue solution). */
      while ( auto *node = treequeue.front() )
      {
        /** Assign local treenode_id. */
        node->treelist_id = getLocalNodeSize();
        /** Account for the depth of the distributed tree. */
        if ( node->getGlobalDepth() < glb_height_ )
        {
          for ( int i = 0; i < N_CHILDREN; i ++ )
          {
            node->kids[ i ] = new NODE( &setup, 0, node->getGlobalDepth() + 1, node, &morton2node, &lock );
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
          RETURN_IF_ERROR( node->setLeaf() );
          treequeue.push_back( NULL );
        }
        treelist_.push_back( node );
        treequeue.pop_front();
      }

      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /* end allocateNodes() */








}; /* end class Tree */
}; /* end namespace tree */
}; /* end namespace hmlp */

#endif /* define TREE_HPP */
