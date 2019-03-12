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
    static bool containAny( mortonType target, const TQUERY& querys )
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

}; /* end class SplitTask */


template<typename NODE>
class Info
{
  public:

    Info( std::vector<mortonType>& gid_to_morton, 
        std::unordered_map<mortonType, NODE*>& morton_to_node )
      : gid_to_morton_( gid_to_morton ), morton_to_node_( morton_to_node )
    {
    }

    mortonType globalIndexToMortonID( indexType gid ) const
    {
      return gid_to_morton_[ gid ];
    };

    NODE* mortonToNodePointer( mortonType morton )
    {
      if ( morton_to_node_.count( morton ) == 0 )
      {
        return nullptr;
      }
      return morton_to_node_[ morton ];
    }

    /**
     *  @brief Check if this node contain any query using morton.
     *         Notice that queries[] contains gids; thus, morton[]
     *         needs to be accessed using gids.
     *
     */ 
    vector<size_t> ContainAny( vector<indexType> &queries, mortonType target )
    {
      vector<size_t> validation( queries.size(), 0 );

      if ( !gid_to_morton_.size() )
      {
        printf( "Morton id was not initialized.\n" );
        exit( 1 );
      }

      for ( size_t i = 0; i < queries.size(); i ++ )
      {
        if ( MortonHelper::IsMyParent( gid_to_morton_[ queries[ i ] ], target ) )
        {
          validation[ i ] = 1;
        }
      }
      return validation;

    }; /** end ContainAny() */








  protected:

      std::vector<mortonType>& gid_to_morton_;

      std::unordered_map<mortonType, NODE*>& morton_to_node_;

}; /* end class Info */


/**
 *  @brief 
 */ 
template<typename ARGUMENT, typename NODEDATA>
class Node : public ReadWrite
{
  public:

    /** Deduce data type from ARGUMENT. */
    typedef typename ARGUMENT::T T;
    /** Use binary trees. */
    static const int N_CHILDREN = 2;

    /**
     *  Constructor of local essential tree (LET) node:
     *  This constructor will only be used in the distributed environment.
     */ 
    Node( mortonType morton ) 
    { 
      this->morton_ = morton; 
    };

    Node( ARGUMENT* setup, sizeType n, depthType l, Node *parent, Info<Node>* info )
      : info( info )
    {
      this->setup = setup;
      this->n = n;
      this->l = l;
      this->treelist_id = 0;
      this->gids.resize( n );
      this->parent = parent;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };

    Node( ARGUMENT *setup, sizeType n, depthType l, vector<size_t> gids, Node *parent, Info<Node>* info )
      : info( info )
    {
      this->setup = setup;
      this->n = n;
      this->l = l;
      this->treelist_id = 0;
      this->gids = gids;
      this->parent = parent;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };
  
    /** (Default) destructor */
    ~Node() {};


    Info<Node>* info = nullptr;



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
    bool containAnyGlobalIndex( const std::vector<indexType> & queries )
    {
      //if ( !setup->morton.size() )
      //{
      //  throw std::out_of_range( "MortonID was not initialized" );
      //}
      for ( auto gid : queries )
      {
        //if ( MortonHelper::IsMyParent( setup->morton[ gid ], getMortonID() ) ) 
        if ( MortonHelper::IsMyParent( info->globalIndexToMortonID( gid ), getMortonID() ) ) 
        {
#ifdef DEBUG_TREE
          printf( "\n" );
          //hmlp_print_binary( setup->morton[ queries[ gid ] ] );
          hmlp_print_binary( info->globalIndexToMortonID( queries[ gid ] ) );
          hmlp_print_binary( morton_ );
          printf( "\n" );
#endif
          return true;
        }
      }
      /* Other return false as not containing any index in queries. */
      return false;
    }; /* end containAnyGlobalIndex() */


    bool containAnyNodePointer( set<Node*> &querys )
    {
      //if ( !setup->morton.size() )
      //{
      //  printf( "Morton id was not initialized.\n" );
      //  exit( 1 );
      //}
      for ( auto it = querys.begin(); it != querys.end(); it ++ )
      {
        if ( MortonHelper::IsMyParent( (*it)->getMortonID(), getMortonID() ) ) 
        {
          return true;
        }
      }
      return false;

    }; /** end ContainAnyNodePointer() */


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
    ARGUMENT *setup = NULL;

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




    /** All points to other tree nodes.  */ 
    Node *kids[ N_CHILDREN ];
    Node *lchild  = NULL; 
    Node *rchild  = NULL;
    Node *sibling = NULL;
    Node *parent  = NULL;



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
 *  \brief Data and setup that are shared with all nodes.
 *  \param [in] SPLITTER: the type of the tree splitter
 *  \param [in] DATATYPE: the datatype
 */ 
template<typename SPLITTER, typename DATATYPE>
class ArgumentBase
{
  public:

    typedef DATATYPE T;

    ArgumentBase() {};

    ~ArgumentBase() {};

    /** neighbors<distance, gid> (accessed with gids) */
    Data<pair<T, size_t>> *NN = NULL;
    /** Tree splitter */
    SPLITTER splitter;

}; /* end class ArgumentBase */


/** */
template<class ARGUMENT, class NODEDATA>
class Tree
{
  public:

    typedef typename ARGUMENT::T T;
    typedef typename std::pair<T, indexType> neigType;
    /** Define our tree node type as NODE. */
    typedef Node<ARGUMENT, NODEDATA> NODE;
    /** Number of children or each tree node is always 2. */
    static const int N_CHILDREN = 2;

    /** Data shared by all tree nodes. */
    ARGUMENT setup;
    /** Tree information: maps between global indices and MortonID */
    Info<NODE> info;

    /** 
     *  \brief Tree constructor 
     */
    Tree() 
      : info( gid_to_morton_, morton_to_node_ )
    {
    };
    /** 
     *  \breif Tree destructor 
     */
    ~Tree()
    {
      //HANDLE_ERROR( clean_() );
      //printf( "~Tree() shared treelist.size() %lu treequeue.size() %lu\n",
      //    treelist.size(), treequeue.size() );
      for ( int i = 0; i < treelist_.size(); i ++ )
      {
        if ( treelist_[ i ] ) delete treelist_[ i ];
      }
      morton_to_node_.clear();
      //printf( "end ~Tree() shared\n" );
    };
    /** 
     *  \returns number of total indices 
     */
    sizeType getGlobalProblemSize() const noexcept
    {
      return glb_num_of_indices_;
    };
    /**
     *  \returns the local tree height
     */ 
    depthType getLocalHeight() const noexcept 
    { 
      return loc_height_; 
    };
    /**
     *  \returns the global tree height
     */ 
    depthType getGlobalHeight() const noexcept 
    { 
      return glb_height_; 
    };
    /**
     *  \returns the number of local tree nodes
     */ 
    sizeType getLocalNodeSize() const noexcept
    {
      return treelist_.size();
    };

    /**
     *  \param [in] i: the local index
     *  \return the ith (top-down order) node pointer in the local tree
     */ 
    NODE* getLocalNodeAt( sizeType i )
    {
      if ( i >= getLocalNodeSize() )
      {
        throw std::out_of_range( "accessing invalid local tree node" );
      }
      return treelist_[ i ];
    }

    /**
     *  \returns the local root node pointer 
     */ 
    NODE* getLocalRoot() 
    {
      if ( getLocalNodeSize() == 0 ) 
      {
        throw std::out_of_range( "accessing the local root while there is no tree node" );
      }
      return treelist_.front();
    }

    /**
     *  \brief Get the ith local tree node pointer at certain depth
     *  \param [in] loc_depth: the local depth
     *  \param [in] i: the left-to-right index at loc_depth
     *  \returns the local root node pointer 
     */ 
    NODE* getLocalNodeAt( depthType loc_depth, sizeType i )
    {
      /* Compute number of nodes at this depth. */
      int n_nodes = 1 << loc_depth;
      if ( loc_depth > getLocalHeight() )
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
          gid_to_morton_[ it ] = node->getMortonID();
        }
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    };


    /** 
     *  \brief Parition and create a complete binary tree in shared memory tree. 
     *  \return the error code
     */
    hmlpError_t partition()
    {
      double beg, alloc_time, split_time, morton_time, permute_time;

      /* Clean up and reserve space for local tree nodes. */
      RETURN_IF_ERROR( clean_() );
      /* Set the global problem size from the setup. */
      this->glb_num_of_indices_ = setup.ProblemSize();
      /* Reset and initialize global indices with lexicographical order. */
      global_index_distribution_.resize( getGlobalProblemSize() );
      /* Round-Robin over MPI ranks (in shared memory == lexicographical order). */
      for ( sizeType i = 0; i < getGlobalProblemSize(); i ++ ) 
      {
        global_index_distribution_[ i ] = i;
      }

      /* Reset the warning flag and clean up the treelist */
      has_uneven_split = false;

      /** Allocate all tree nodes in advance. */
      beg = omp_get_wtime();
      HANDLE_ERROR( allocateNodes_( new NODE( &setup, getGlobalProblemSize(), 
              0, global_index_distribution_, nullptr, &info ) ) );
      alloc_time = omp_get_wtime() - beg;

      /** Recursive spliting (topdown). */
      beg = omp_get_wtime();
      SplitTask<NODE> splittask;
      RETURN_IF_ERROR( traverseDown( splittask ) );
      RETURN_IF_ERROR( ExecuteAllTasks() );
      split_time = omp_get_wtime() - beg;


      /** Compute node and point MortonID. */ 
      gid_to_morton_.resize( getGlobalProblemSize() );
      /* Compute MortonID (for nodes and indices) recursively. */
      RETURN_IF_ERROR( RecursiveMorton( getLocalRoot(), MortonHelper::Root() ) );
      /* Compute the offset (related to the left most) for drawing interaction. */
      RETURN_IF_ERROR( RecursiveOffset( getLocalRoot(), 0 ) );

      /** Construct morton_to_node_ map for the local tree. */
      morton_to_node_.clear();
      for ( size_t i = 0; i < treelist_.size(); i ++ )
      {
        morton_to_node_[ treelist_[ i ]->getMortonID() ] = treelist_[ i ];
      }

      /** Adgust gids to the appropriate order.  */
      IndexPermuteTask<NODE> indexpermutetask;
      RETURN_IF_ERROR( traverseUp( indexpermutetask ) );
      RETURN_IF_ERROR( ExecuteAllTasks() );

      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /* end partition() */



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
     *  \param [in] KNNTASK: the type of the knn kernel
     *  \param [in] n_tree: the number of maximum iterations
     *  \param [in] kappa: the number of neighbors to compute
     *  \param [in] max_depth: the maximum depth 
     *  \param [inout] neighbors: all neighbors in Data<neigType>
     *  \param [in] initNN: the initial value
     *  \param [in] dummy:
     *  \returns the error code
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
        RETURN_IF_ERROR( partition() );
        RETURN_IF_ERROR( traverseLeafs( dummy ) );
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
    }; /* end CheckAllInteractions() */









    template<typename TASK, typename... Args>
    hmlpError_t traverseLeafs( TASK &dummy, Args&... args )
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
        /* Do not parallelize if there is less nodes than threads */
        #pragma omp parallel for if ( n_nodes > nthd_glb / 2 ) schedule( dynamic )
        for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
        {
          auto *node = *(level_beg + node_ind);
          RecuTaskExecute( node, dummy, args... );
        }
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /** end traverseLeafs() */





    template<typename TASK, typename... Args>
    hmlpError_t traverseUp( TASK &dummy, Args&... args )
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
    }; /** end traverseUp() */






    template<typename TASK, typename... Args>
    hmlpError_t traverseDown( TASK &dummy, Args&... args )
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
    }; /** end traverseDown() */



    /**
     *  @brief For unordered traversal, we just call local
     *         downward traversal.
     */ 
    template<typename TASK, typename... Args>
    hmlpError_t traverseUnOrdered( TASK &dummy, Args&... args )
    {
      return traverseDown( dummy, args... );
    }; /** end traverseUnOrdered() */


    /**
     *  \brief Remove all dependencies that were previously post to the owned tree nodes
     *         and local essential tree (LET) nodes.
     *  \returns the error code.
     */ 
    hmlpError_t dependencyClean()
    {
      /* Remove dependencies of each owned nodes. */
      for ( auto node : treelist_ ) 
      {
        node->DependencyCleanUp();
      }
      /* Remove dependencies of each LET nodes. */
      for ( auto it : morton_to_node_ )
      {
        auto *node = it.second;
        if ( node ) node->DependencyCleanUp();
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /* end dependencyClean() */

    /**
     *  \brief Enter a runtime epoch to consume all tasks starting from all
     *         sources (tasks without incoming dependencies),
     *  \return the error code.
     */ 
    hmlpError_t ExecuteAllTasks()
    {
      /* Invoke the runtime scheduler. */
      RETURN_IF_ERROR( hmlp_run() );
      /* Clean up all the in and out set of each read/write object. */
      RETURN_IF_ERROR( dependencyClean() );
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
    /** Number of indices owned locally. */
    sizeType loc_num_of_indices_ = 0;
    /** Total number of indices. */
    sizeType glb_num_of_indices_ = 0;
    /** Local tree nodes in the top-down order. */ 
    std::vector<NODE*> treelist_;
    /** The index distribution (default: round-robing over MPI ranks). */
    std::vector<indexType> global_index_distribution_;
    /** This is the empty list. */
    std::vector<indexType> no_index_exist_;
    /** */
    std::vector<mortonType> gid_to_morton_;
    /** The map from MortonID to the tree node pointer.  */
    unordered_map<mortonType, NODE*> morton_to_node_;
    /** Mutex for exclusive right to modify treelist and morton_to_node_. */ 
    Lock lock_;
    /**
     *  \brief Clean up the tree.
     *  \returns the error code
     *  \retval HMLP_ERROR_SUCCESS if no error is reported
     *  \retval
     */ 
    hmlpError_t clean_()
    {
      /* Delete all local and LET tree nodes. */
      for ( auto morton_and_node_ptr : morton_to_node_ ) 
      {
        if ( morton_and_node_ptr.second != nullptr )
        {
          delete morton_and_node_ptr.second;
        }
      }
      /* Clear morton_to_node__, treelist_, and global_index_distribution_. */
      morton_to_node_.clear();
      treelist_.clear();
      global_index_distribution_.clear();
      /* Reset loc_height_ and glb_height_. */
      loc_height_ = 0;
      glb_height_ = 0;
      /* Reset loc_num_of_indices_ and glb_num_of_indices. */
      loc_num_of_indices_ = 0;
      glb_num_of_indices_ = 0;
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    };
    /**
     *  \brief Allocate the local tree using the local root with n points and depth l.
     *  \details This routine will be called in two situations: 1) called by Tree::partition(), 
     *  or 2) called by MPITree::TreePartition().
     *  \param [in,out] root: the root of the local tree
     *  \returns the error code
     *  \retval HMLP_ERROR_SUCCESS if no error is reported
     *  \retval
     */ 
    hmlpError_t allocateNodes_( NODE *root )
    {
      /* The root node cannot be nulltr. */
      if ( root == nullptr )
      {
        return HMLP_ERROR_INVALID_VALUE;
      }
      /* Compute the global tree depth using std::log2(). */
      glb_height_ = std::ceil( std::log2( (double)getGlobalProblemSize() / setup.getLeafNodeSize() ) );
      /* If the global depth exceeds the limit, then set it to the maximum depth. */
      if ( glb_height_ > setup.getMaximumDepth() ) 
      {
        glb_height_ = setup.getMaximumDepth();
      }
      /** Compute the local tree depth. */
      loc_height_ = glb_height_ - root->getGlobalDepth();
      /* Use a queue to perform BFS. */
      deque<NODE*> treequeue;
      /* Push root into the treelist. */
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
            node->kids[ i ] = new NODE( &setup, 0, node->getGlobalDepth() + 1, node, &info );
            /* Fail to allocate memory. Return with error. */
            if ( node->kids[ i ] == nullptr )
            {
              return HMLP_ERROR_ALLOC_FAILED;
            }
            /* Push children to the queue. */
            treequeue.push_back( node->kids[ i ] );
          }
          /* Set lchild and rchild to facilitate binary tree operations. */
          node->lchild = node->kids[ 0 ];
          node->rchild = node->kids[ 1 ];
          /* Set siblings to facilitate binary tree operations. */
          if ( node->lchild != nullptr ) 
          {
            node->lchild->sibling = node->rchild;
          }
          if ( node->rchild != nullptr ) 
          {
            node->rchild->sibling = node->lchild;
          }
        }
        else
        {
          /* Leaf nodes are annotated with this flag. */
          RETURN_IF_ERROR( node->setLeaf() );
          /* Push nullptr to the queue, which will result in termination of the while loop. */
          treequeue.push_back( nullptr );
        }
        /* Insert the node to the treelist. */
        treelist_.push_back( node );
        /* Pop the node from the queue. */
        treequeue.pop_front();
      }
      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    }; /* end allocateNodes_() */
}; /* end class Tree */
}; /* end namespace tree */
}; /* end namespace hmlp */

#endif /* define TREE_HPP */
