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

#include <mpi/tree_mpi.hpp>
#include <gofmm/gofmm.hpp>
#include <primitives/combinatorics.hpp>


using namespace hmlp::gofmm;

namespace hmlp
{
namespace mpigofmm
{



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



  /** compute all2c (distance to the approximate centroid) */
  std::vector<T> AllToCentroid( std::vector<size_t> &gids ) const
  {
    /** declaration */
    SPDMATRIX &K = *Kptr;
    std::vector<T> temp( gids.size(), 0.0 );

    /** loop over each point in gids */
    #pragma omp parallel for
    for ( size_t i = 0; i < gids.size(); i ++ )
    {
      switch ( metric )
      {
        case KERNEL_DISTANCE:
        {
          temp[ i ] = K( gids[ i ], gids[ i ] );
          for ( size_t j = 0; j < n_centroid_samples; j ++ )
          {
            /** important sample ( Kij, j ) if provided */
            std::pair<T, size_t> sample = K.ImportantSample( gids[ i ] );
            temp[ i ] -= ( 2.0 / n_centroid_samples ) * sample.first;
          }
          break;
        }
        case ANGLE_DISTANCE:
        {
          temp[ i ] = 0.0;
          for ( size_t j = 0; j < n_centroid_samples; j ++ )
          {
            /** important sample ( Kij, j ) if provided */
            std::pair<T, size_t> sample = K.ImportantSample( gids[ i ] );
            T kij = sample.first;
            T kii = K( gids[ i ], gids[ i ] );
            T kjj = K( sample.second, sample.second );
            temp[ i ] += ( 1.0 - ( kij * kij ) / ( kii * kjj ) );
          }
          temp[ i ] /= n_centroid_samples;
          break;
        }
        default:
        {
          printf( "centersplit() invalid splitting scheme\n" ); fflush( stdout );
          exit( 1 );
        }
      } /** end switch ( metric ) */
    }

    return temp;

  }; /** end AllToCentroid() */



  /** compute all2f (distance to the farthest point) */
  std::vector<T> AllToFarthest( std::vector<size_t> &gids, size_t gidf2c ) const
  {
    /** declaration */
    SPDMATRIX &K = *Kptr;
    std::vector<T> temp( gids.size(), 0.0 );

    /** loop over each point in gids */
    #pragma omp parallel for
    for ( size_t i = 0; i < gids.size(); i ++ )
    {
      switch ( metric )
      {
        case KERNEL_DISTANCE:
        {
          T kij = K( gids[ i ],    gidf2c );
          T kii = K( gids[ i ], gids[ i ] );
          temp[ i ] = kii - 2.0 * kij;
          break;
        }
        case ANGLE_DISTANCE:
        {
          T kij = K( gids[ i ],    gidf2c );
          T kii = K( gids[ i ], gids[ i ] );
          T kjj = K(    gidf2c,    gidf2c );
          temp[ i ] = ( 1.0 - ( kij * kij ) / ( kii * kjj ) );
          break;
        }
        default:
        {
          printf( "centersplit() invalid splitting scheme\n" ); fflush( stdout );
          exit( 1 );
        }
      } /** end switch ( metric ) */
    }

    return temp;

  }; /** end AllToFarthest() */



  /** compute all projection i.e. dip - diq for all i */
  std::vector<T> AllToLeftRight( std::vector<size_t> &gids,
      size_t gidf2c, size_t gidf2f ) const
  {
    /** declaration */
    SPDMATRIX &K = *Kptr;
    std::vector<T> temp( gids.size(), 0.0 );

    /** loop over each point in gids */
    #pragma omp parallel for
    for ( size_t i = 0; i < gids.size(); i ++ )
    {
      switch ( metric )
      {
        case KERNEL_DISTANCE:
        {
          T kip = K( gids[ i ], gidf2f );
          T kiq = K( gids[ i ], gidf2c );
          temp[ i ] = kip - kiq;
          break;
        }
        case ANGLE_DISTANCE:
        {
          T kip = K( gids[ i ],    gidf2f );
          T kiq = K( gids[ i ],    gidf2c );
          T kii = K( gids[ i ], gids[ i ] );
          T kpp = K(    gidf2f,    gidf2f );
          T kqq = K(    gidf2c,    gidf2c );
          /** ingore 1 from both terms */
          temp[ i ] = ( kip * kip ) / ( kii * kpp ) - ( kiq * kiq ) / ( kii * kqq );
          break;
        }
        default:
        {
          printf( "centersplit() invalid splitting scheme\n" ); fflush( stdout );
          exit( 1 );
        }
      }
    }

    return temp;

  }; /** end AllToLeftRight() */






	/** overload the operator */
  inline std::vector<std::vector<std::size_t> > operator()
  ( 
    std::vector<std::size_t>& gids,
    std::vector<std::size_t>& lids
  ) const 
  {
    /** all assertions */
    assert( N_SPLIT == 2 );
    assert( Kptr );

    /** declaration */
    SPDMATRIX &K = *Kptr;
    std::vector<std::vector<std::size_t>> split( N_SPLIT );
    size_t n = gids.size();
    std::vector<T> temp( n, 0.0 );

    /** all timers */
    double beg, d2c_time, d2f_time, projection_time, max_time;

    /** compute all2c (distance to the approximate centroid) */
    beg = omp_get_wtime();
    temp = AllToCentroid( gids );
    d2c_time = omp_get_wtime() - beg;

    /** find f2c (farthest from center) */
    auto itf2c = std::max_element( temp.begin(), temp.end() );
    auto idf2c = std::distance( temp.begin(), itf2c );
    auto gidf2c = gids[ idf2c ];

    /** compute the all2f (distance to farthest point) */
    beg = omp_get_wtime();
    temp = AllToFarthest( gids, gidf2c );
    d2f_time = omp_get_wtime() - beg;

    /** find f2f (far most to far most) */
    beg = omp_get_wtime();
    auto itf2f = std::max_element( temp.begin(), temp.end() );
    auto idf2f = std::distance( temp.begin(), itf2f );
    auto gidf2f = gids[ idf2f ];
    max_time = omp_get_wtime() - beg;

    /** compute all2leftright (projection i.e. dip - diq) */
    beg = omp_get_wtime();
    temp = AllToLeftRight( gids, gidf2c, gidf2f );
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
      if      ( temp[ i ] < median ) split[ 0 ].push_back( i );
      else if ( temp[ i ] > median ) split[ 1 ].push_back( i );
      else                               middle.push_back( i );
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






	/** distributed operator */
  inline std::vector<std::vector<size_t> > operator()
  ( 
    std::vector<size_t>& gids,
    hmlp::mpi::Comm comm
  ) const 
  {
    /** all assertions */
    assert( N_SPLIT == 2 );
    assert( Kptr );

    /** declaration */
    int size, rank;
    hmlp::mpi::Comm_size( comm, &size );
    hmlp::mpi::Comm_rank( comm, &rank );
    SPDMATRIX &K = *Kptr;
    std::vector<std::vector<size_t>> split( N_SPLIT );

    /** reduce to get the total size of gids */
    int n = 0;
    int num_points_owned = gids.size();
    std::vector<T> temp( gids.size(), 0.0 );

    /** n = sum( num_points_owned ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &num_points_owned, &n, 1, 
        MPI_INT, MPI_SUM, comm );

    /** early return */
    if ( n == 0 ) return split;

    /** compute d2c (distance to the approx centroid) for each owned point */
    temp = AllToCentroid( gids );

    /** find the f2c (far most to center) from points owned */
    auto itf2c = std::max_element( temp.begin(), temp.end() );
    auto idf2c = std::distance( temp.begin(), itf2c );

    /** create a pair for MPI Allreduce */
    hmlp::mpi::NumberIntPair<T> local_max_pair, max_pair; 
    local_max_pair.val = temp[ idf2c ];
    local_max_pair.key = rank;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** boardcast gidf2c from the MPI process which has the max_pair */
    int gidf2c = gids[ idf2c ];
    hmlp::mpi::Bcast( &gidf2c, 1, MPI_INT, max_pair.key, comm );

    //printf( "rank %d val %E key %d; global val %E key %d\n", 
    //    rank, local_max_pair.val, local_max_pair.key,
    //    max_pair.val, max_pair.key ); fflush( stdout );
    //printf( "rank %d gidf2c %d\n", rank, gidf2c  ); fflush( stdout );



    /** compute all2f (distance to farthest point) */
    temp = AllToFarthest( gids, gidf2c );




    /** find f2f (far most to far most) from owned points */
    auto itf2f = std::max_element( temp.begin(), temp.end() );
    auto idf2f = std::distance( temp.begin(), itf2f );

    /** create a pair for MPI Allreduce */
    local_max_pair.val = temp[ idf2f ];
    local_max_pair.key = rank;

    /** max_pair = max( local_max_pairs ) over all MPI processes in comm */
    hmlp::mpi::Allreduce( &local_max_pair, &max_pair, 1, MPI_MAXLOC, comm );

    /** boardcast gidf2f from the MPI process which has the max_pair */
    int gidf2f = gids[ idf2f ];
    hmlp::mpi::Bcast( &gidf2f, 1, MPI_INT, max_pair.key, comm );

    //printf( "rank %d val %E key %d; global val %E key %d\n", 
    //    rank, local_max_pair.val, local_max_pair.key,
    //    max_pair.val, max_pair.key ); fflush( stdout );
    //printf( "rank %d gidf2f %d\n", rank, gidf2f  ); fflush( stdout );


    /** compute all2leftright (projection i.e. dip - diq) */
    temp = AllToLeftRight( gids, gidf2c, gidf2f );


    /** parallel median select */
    T  median = hmlp::combinatorics::Select( n / 2, temp, comm );

    //printf( "rank %d median %E\n", rank, median ); fflush( stdout );

    std::vector<size_t> middle;
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
    hmlp::mpi::Allreduce( &num_mid_owned, &nmid, 1, MPI_SUM, comm );
    hmlp::mpi::Allreduce( &num_lhs_owned, &nlhs, 1, MPI_SUM, comm );
    hmlp::mpi::Allreduce( &num_rhs_owned, &nrhs, 1, MPI_SUM, comm );

    printf( "rank %d [ %d %d %d ] global [ %d %d %d ]\n",
        rank, num_lhs_owned, num_mid_owned, num_rhs_owned,
        nlhs, nmid, nrhs ); fflush( stdout );

    /** assign points in the middle to left or right */
    if ( nmid )
    {
      int nlhs_required = ( n - 1 ) / 2 + 1 - nlhs;
      int nrhs_required = nmid - nlhs_required;
      assert( nlhs_required >= 0 );
      assert( nrhs_required >= 0 );

      /** now decide the portion */
      int nlhs_required_owned = num_mid_owned * nlhs_required / nmid;
      int nrhs_required_owned = num_mid_owned - nlhs_required_owned;
      assert( nlhs_required_owned >= 0 );
      assert( nrhs_required_owned >= 0 );

      for ( size_t i = 0; i < middle.size(); i ++ )
      {
        if ( i < nlhs_required_owned ) split[ 0 ].push_back( middle[ i ] );
        else                           split[ 1 ].push_back( middle[ i ] );
      }
    };

    return split;
  };


}; /** end struct centersplit */



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
  /** 
   *  The type here is tree::Node but not mpitree::Node.
   *  NearNodes and NNNearNodes also take tree::Node.
   *  This is ok, because they will only contain leaf nodes,
   *  which will never be distributed.
   *  However, FarNodes and NNFarNodes may contain distributed
   *  tree nodes. In this case, we have to do type casting.
   */
  auto level_beg = tree.treelist.begin() + n_nodes - 1;


  /** 
   * traverse all leaf nodes. 
   *
   * TODO: the following omp parallel for will result in segfault on KNL.
   *       Not sure why.
   **/
  for ( size_t node_ind = 0; node_ind < n_nodes; node_ind ++ )
  {
    auto *node = *(level_beg + node_ind);
    auto &data = node->data;

    /** if no skeletons, then add every leaf nodes */
    /** TODO: this is not affected by the BUDGET */
    if ( !node->data.isskel )
    {
      for ( size_t i = 0; i < n_nodes; i ++ )
      {
        node->NearNodes.insert(   *(level_beg + i) );
        node->NNNearNodes.insert( *(level_beg + i) );
        node->NearNodeMortonIDs.insert(   (*(level_beg + i))->morton );
        node->NNNearNodeMortonIDs.insert( (*(level_beg + i))->morton );
      }
    }

    /** add myself to the list. */
    node->NearNodes.insert( node );
    node->NNNearNodes.insert( node );
    node->NearNodeMortonIDs.insert( node->morton );
    node->NNNearNodeMortonIDs.insert( node->morton );
 
    /** TODO: ballot table */
    std::vector<std::pair<size_t, size_t>> ballot( n_nodes );
    for ( size_t i = 0; i < n_nodes; i ++ )
    {
      ballot[ i ].first  = 0;
      ballot[ i ].second = i;
    }


    /** 
     *  TODO: traverse all points and their neighbors. NN is stored as k-by-N 
     *       
     *  LET required     
     *
     */


    /** sort the ballot list */
    struct 
    {
      bool operator () ( std::pair<size_t, size_t> a, std::pair<size_t, size_t> b )
      {   
        return a.first > b.first;
      }   
    } BallotMore;


  } /** end for each leaf owned leaf node in the local tree */


  /** TODO: symmetrinize Near( node ) */
  if ( SYMMETRIC )
  {
    /** TODO: LET required */
  }

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
template<bool SYMMETRIC, typename NODE, size_t LEVELOFFSET = 4>
void FindFarNodes(
  size_t level, size_t max_depth,
  size_t recurs_morton, NODE *target )
{
  size_t shift = ( 1 << LEVELOFFSET ) - level + LEVELOFFSET;
  size_t source_morton = ( recurs_morton << shift ) + level;

  if ( hmlp::tree::ContainAnyMortonID( target->NearNodeMortonIDs, source_morton ) )
  {
    if ( level < max_depth )
    {
      /** recurse to two children */
      FindFarNodes<SYMMETRIC>( 
          level + 1, max_depth,
          ( recurs_morton << 1 ) + 0, target );
      FindFarNodes( 
          level + 1, max_depth,
          ( recurs_morton << 1 ) + 1, target );
    }
  }
  else
  {
    /** create LETNode, insert ``source'' to Far( target ) */
    //auto *letnode = new hmlp::mpitree::LETNode( level, source_morton );
    target->FarNodeMortonIDs.insert( source_morton );
  }


  /**
   *  case: NNPRUNE
   *
   *  Build NNNearNodes for the FMM approximation.
   *  Near( target ) contains other leaf nodes
   *  
   **/
  if ( hmlp::tree::ContainAnyMortonID( target->NNNearNodeMortonIDs, source_morton ) )
  {
    if ( level < max_depth )
    {
      /** recurse to two children */
      FindFarNodes<SYMMETRIC>( 
          level + 1, max_depth,
          ( recurs_morton << 1 ) + 0, target );
      FindFarNodes( 
          level + 1, max_depth,
          ( recurs_morton << 1 ) + 1, target );
    }
  }
  else
  {
    if ( SYMMETRIC && ( source_morton < target->morton ) )
    {
      /** since target->morton is larger than the visiting node,
       * the interaction between the target and this node has
       * been computed. 
       */ 
    }
    else
    {
      /** create LETNode, insert ``source'' to Far( target ) */
      //auto *letnode = new hmlp::mpitree::LETNode( level, source_morton );
      target->NNFarNodeMortonIDs.insert( source_morton );
    }
  }

}; /** end FindFarNodes() */



template<bool SYMMETRIC, typename NODE>
void MergeFarNodes( NODE *node )
{
  /** if I don't have any skeleton, then I'm nobody's far field */
  if ( !node->data.isskel ) return;

  if ( node->isleaf )
  {
    FindFarNodes<SYMMETRIC>( 0, /** max_depth */, 0, node  );
  }
  else
  {
    /** merge Far( lchild ) and Far( rchild ) from children */
    auto *lchild = node->lchild;
    auto *rchild = node->rchild;

    /** case: !NNPRUNE (HSS specific) */ 
    auto &pFarNodes =   node->FarNodeMortonIDs;
    auto &lFarNodes = lchild->FarNodeMortonIDs;
    auto &rFarNodes = rchild->FarNodeMortonIDs;

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
    auto &pNNFarNodes =   node->NNFarNodeMortonIDs;
    auto &lNNFarNodes = lchild->NNFarNodeMortonIDs;
    auto &rNNFarNodes = rchild->NNFarNodeMortonIDs;

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
  }

  /** TODO: deal with symmetricness for NN case*/
  if ( SYMMETRIC )
  {
  }

}; /** end MergeFarNodes() */




template<bool SYMMETRIC, typename NODE>
void ParallelMergeFarNodes( NODE *node )
{
  /** if I don't have any skeleton, then I'm nobody's far field */
  if ( !node->data.isskel ) return;

  /** distributed treenode */
  if ( node->GetCommSize() < 2 )
  {
    MergeFarNodes( node );
  }
  else
  {
    /** merge Far( lchild ) and Far( rchild ) from children */
    auto *child = node->child;

    /** recv  */

  }


}; /** end ParallelMergeFarNodes() */


template<typename LETNODE, typename NODE>
void SimpleFarNodes( NODE *node )
{
  if ( !node->isleaf )
  {
    auto *lchild = node->lchild;
    auto *rchild = node->rchild;

    lchild->FarNodes.insert( rchild );
    lchild->NNFarNodes.insert( rchild );
    rchild->FarNodes.insert( lchild );
    rchild->NNFarNodes.insert( lchild );

    SimpleFarNodes<LETNODE>( lchild );
    SimpleFarNodes<LETNODE>( rchild );
  }

  printf( "%lu NNFarNodes: ", node->treelist_id ); 
  for ( auto it  = node->NNFarNodes.begin();
             it != node->NNFarNodes.end(); it ++ )
  {
    printf( "%lu ", (*it)->treelist_id );
  };
  printf( "\n" );
  
}; /** end SimpleFarNodes() */


template<typename LETNODE, typename NODE>
void ParallelSimpleFarNodes( NODE *node )
{
  printf( "level %lu beg\n", node->l ); fflush( stdout );
  hmlp::mpi::Barrier( MPI_COMM_WORLD );

  hmlp::mpi::Status status;
  hmlp::mpi::Comm comm = node->GetComm();
  int size = node->GetCommSize();
  int rank = node->GetCommRank();


  if ( rank == 0 )
  {
    printf( "level %lu here\n", node->l ); fflush( stdout );
  }
  hmlp::mpi::Barrier( MPI_COMM_WORLD );



  if ( size < 2 )
  {
    SimpleFarNodes<LETNODE>( node );
  }
  else
  {
    auto *child = node->child;
    size_t send_morton = child->morton;
    size_t recv_morton = 0;
    LETNODE *letnode = NULL;

    if ( rank == 0 )
    {
      hmlp::mpi::Sendrecv( 
          &send_morton, 1, size / 2, 0, 
          &recv_morton, 1, size / 2, 0, comm, &status );

      /** initialize a LetNode with sibling's MortonID */
      letnode = new LETNODE( recv_morton );
      child->FarNodes.insert( letnode );

      if ( child->isleaf )
      {
        /** exchange gids */
        auto &send_gids = child->gids;
        auto &recv_gids = letnode->gids;
        hmlp::mpi::ExchangeVector(
            send_gids, size / 2, 0, 
            recv_gids, size / 2, 0, comm, &status );
      }
      else
      {
        /** exchange skels */
        auto &send_skels = child->data.skels;
        auto &recv_skels = letnode->data.skels;
        hmlp::mpi::ExchangeVector(
            send_skels, size / 2, 0, 
            recv_skels, size / 2, 0, comm, &status );
        /** exchange interpolative coefficients */
        auto &send_proj = child->data.proj;
        auto &recv_proj = letnode->data.proj;
        hmlp::mpi::ExchangeVector(
            send_proj, size / 2, 0, 
            recv_proj, size / 2, 0, comm, &status );
      }
    }

    if ( rank == size / 2 )
    {
      hmlp::mpi::Sendrecv( 
          &send_morton, 1, 0, 0, 
          &recv_morton, 1, 0, 0, comm, &status );

      /** initialize a LetNode with sibling's MortonID */
      letnode = new LETNODE( recv_morton );
      child->FarNodes.insert( letnode );
      if ( child->isleaf )
      {
        /** exchange gids */
        auto &send_gids = child->gids;
        auto &recv_gids = letnode->gids;
        hmlp::mpi::ExchangeVector(
            send_gids, 0, 0, 
            recv_gids, 0, 0, comm, &status );
      }
      else
      {
        /** exchange skels */
        auto &send_skels = child->data.skels;
        auto &recv_skels = letnode->data.skels;
        hmlp::mpi::ExchangeVector(
            send_skels, 0, 0, 
            recv_skels, 0, 0, comm, &status );
        /** exchange interpolative coefficients */
        auto &send_proj = child->data.proj;
        auto &recv_proj = letnode->data.proj;
        hmlp::mpi::ExchangeVector(
            send_proj, 0, 0, 
            recv_proj, 0, 0, comm, &status );
      }
    }


    ParallelSimpleFarNodes<LETNODE>( child );
  }

}; /** end ParallelSimpleFarNodes() */














template<typename NODE>
void RowSamples( NODE *node, size_t nsamples )
{
  /** gather shared data and create reference */
  auto &K = *node->setup->K;

  /** amap contains nsamples of row gids of K */
  std::vector<size_t> &amap = node->data.candidate_rows;

  /** clean up candidates from previous iteration */
  amap.clear();

  if ( nsamples < K.col() - node->n )
  {
    amap.reserve( nsamples );

    //for ( auto cur = ordered_snids.begin(); cur != ordered_snids.end(); cur++ )
    //{
    //  amap.push_back( cur->second );
    //}

    /** uniform samples */
    if ( amap.size() < nsamples )
    {
      while ( amap.size() < nsamples )
      {
        size_t sample = rand() % K.col();

        /** create a single query */
        std::vector<size_t> sample_query( 1, sample );

        /**
         *  check duplication using std::find, but check whether the sample
         *  belongs to the diagonal block using Morton ID.
         */ 
        if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() &&
             !node->ContainAny( sample_query ) )
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
      if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() )
      {
        amap.push_back( sample );
      }
    }
  }

}; /** end RowSamples() */




/**
 *  @brief Involve MPI routins
 */ 
template<typename NODE>
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
    /** concatinate [ lskels, rskels ] */
    auto &lskels = lchild->data.skels;
    auto &rskels = rchild->data.skels;
    candidate_cols = lskels;
    candidate_cols.insert( candidate_cols.end(), 
        rskels.begin(), rskels.end() );
  }

  /** sample off-diagonal rows */
  RowSamples( node, 2 * candidate_cols.size() );
  /** get KIJ for skeletonization */
  KIJ = K( candidate_rows, candidate_cols );

}; /** end GetSkeletonMatrix() */


/**
 *
 */ 
template<typename NODE>
class GetSkeletonMatrixTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      /** this task contains MPI routines */
      this->has_mpi_routines = true;

      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "par-gskm" );
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
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      }

      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      printf( "%lu GetSkeletonMatrix beg\n", arg->treelist_id );
      GetSkeletonMatrix<NODE>( arg );
      printf( "%lu GetSkeletonMatrix end\n", arg->treelist_id );
    };

}; /** end class GetSkeletonMatrixTask */









/**
 *  @brief Involve MPI routins
 */ 
template<typename NODE>
void ParallelGetSkeletonMatrix( NODE *node )
{
  /** gather shared data and create reference */
  auto &K = *(node->setup->K);

  /** gather per node data and create reference */
  auto &data = node->data;
  auto &candidate_rows = data.candidate_rows;
  auto &candidate_cols = data.candidate_cols;
  auto &KIJ = data.KIJ;

  /** MPI */
  int size = node->GetCommSize();
  int rank = node->GetCommRank();
  hmlp::mpi::Comm comm = node->GetComm();

  if ( size > 1 )
  {
    /** 
     *  this node (mpitree::Node) belongs to the distributed tree 
     *  only executed by 0th and size/2 th rank of 
     *  the local communicator
     *  
     */
    NODE *child = node->child;
    MPI_Status status;

    if ( rank == 0 )
    {
      int recv_size = 0;
      std::vector<int> recv_buff;
      hmlp::mpi::Recv( &recv_size, 1, MPI_INT, 
          size / 2, 10, comm, &status );
      recv_buff.resize( recv_size, 0 );
      hmlp::mpi::Recv( recv_buff.data(), recv_size, MPI_INT, 
          size / 2, 10, comm, &status );
      /** concatinate [ lskels, rskels ] */
      candidate_cols= child->data.skels;
      candidate_cols.reserve( candidate_cols.size() + recv_size );
      for ( size_t i = 0; i < recv_size; i ++ )
        candidate_cols.push_back( recv_buff[ i ] );
      /** sample off-diagonal rows */
      RowSamples( node, 2 * candidate_cols.size() );
      /** get KIJ for skeletonization */
      KIJ = K( candidate_rows, candidate_cols );
    }

    if ( rank == size / 2 )
    {
      int send_size = child->data.skels.size();
      std::vector<int> send_buff( send_size, 0 );
      for ( size_t i = 0; i < send_size; i ++ )
        send_buff[ i ] = child->data.skels[ i ];
      hmlp::mpi::Send( &send_size, 1, MPI_INT, 
          0, 10, comm );
      hmlp::mpi::Send( send_buff.data(), send_size, MPI_INT, 
          0, 10, comm );
    }
  }
  else
  {
    /** this node is the root of the local tree */
    GetSkeletonMatrix( node );
  };

}; /** end ParallelGetSkeletonMatrix() */


/**
 *
 */ 
template<typename NODE>
class ParallelGetSkeletonMatrixTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      /** this task contains MPI routines */
      this->has_mpi_routines = true;

      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "par-gskm" );
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
    };


    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      if ( !arg->isleaf )
      {
        if ( arg->GetCommSize() < 2 )
        {
          //printf( "shared case\n" ); fflush( stdout );
          arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
          arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        }
        else
        {
          //printf( "distributed case size %d\n", arg->GetCommSize() ); fflush( stdout );
          arg->child->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        }
      }

      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      printf( "%lu Par-GetSkeletonMatrix beg\n", arg->treelist_id );
      ParallelGetSkeletonMatrix<NODE>( arg );
      printf( "%lu Par-GetSkeletonMatrix end\n", arg->treelist_id );
    };

}; /** end class ParallelGetSkeletonMatrixTask */












/**
 *  @brief Skeletonization with interpolative decomposition.
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
void ParallelSkeletonize( NODE *node )
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

  /** Bill's l2 norm scaling factor */
  T scaled_stol = std::sqrt( (T)n / q ) * std::sqrt( (T)m / (N - q) ) * stol;

  /** account for uniform sampling */
  scaled_stol *= std::sqrt( (T)q / N );

  hmlp::lowrank::id<ADAPTIVE, LEVELRESTRICTION>
  ( 
    KIJ.row(), KIJ.col(), maxs, scaled_stol, /** ignore if !ADAPTIVE */
    KIJ, skels, proj, jpvt
  );

  /** depending on the flag, decide isskel or not */
  if ( LEVELRESTRICTION )
  {
    data.isskel = (skels.size() != 0);
  }
  else
  {
    assert( skels.size() );
    assert( proj.size() );
    assert( jpvt.size() );
    data.isskel = true;
  }
  
  /** relabel skeletions with the real lids */
  for ( size_t i = 0; i < skels.size(); i ++ )
  {
    skels[ i ] = candidate_cols[ skels[ i ] ];
  }

}; /** end ParallelSkeletonize() */




template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
class SkeletonizeTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "sk" );
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

      event.Set( label + name, flops, mops );
      arg->data.skeletonize = event;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
        //printf( "%d Par-Skel beg\n", global_rank );
        ParallelSkeletonize<ADAPTIVE, LEVELRESTRICTION, NODE, T>( arg );
        //printf( "%d Par-Skel end\n", global_rank );
    };

}; /** end class SkeletonTask */




/**
 *
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename NODE, typename T>
class ParallelSkeletonizeTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "par-sk" );
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

      event.Set( label + name, flops, mops );
      arg->data.skeletonize = event;
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
      /** try to enqueue if there is no dependency */
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker )
    {
      int global_rank = 0;
      hmlp::mpi::Comm_rank( MPI_COMM_WORLD, &global_rank );

      if ( arg->GetCommRank() == 0 )
      {
        printf( "%d Par-Skel beg\n", global_rank );
        ParallelSkeletonize<ADAPTIVE, LEVELRESTRICTION, NODE, T>( arg );
        printf( "%d Par-Skel end\n", global_rank );
      }
    };

}; /** end class ParallelSkeletonTask */




/**
 *  @brief
 */ 
template<typename NODE, typename T>
class InterpolateTask : public hmlp::Task
{
  public:

    NODE *arg;

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
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );
    }

    void Execute( Worker* user_worker )
    {
      /** only executed by rank 0 */
      if ( arg->GetCommRank() == 0  ) Interpolate<NODE, T>( arg );
    };

}; /** end class InterpolateTask */























/**
 *  @brief ComputeAll
 */ 
template<
bool USE_RUNTIME, 
bool USE_OMP_TASK, 
bool SYMMETRIC_PRUNE, 
bool NNPRUNE, 
bool CACHE, 
typename NODE, 
typename TREE, 
typename T>
hmlp::Data<T> Evaluate
( 
  TREE &tree,
  hmlp::Data<T> &weights
)
{
  const bool AUTO_DEPENDENCY = true;

  /** all timers */
  double beg, time_ratio, evaluation_time = 0.0;
  double allocate_time, computeall_time;
  double forward_permute_time, backward_permute_time;

  /** nrhs-by-n initialize potentials */
  beg = omp_get_wtime();
  hmlp::Data<T> potentials( weights.row(), weights.col(), 0.0 );
  tree.setup.w = &weights;
  tree.setup.u = &potentials;
  allocate_time = omp_get_wtime() - beg;


  /** not yet implemented */
  printf( "\nnot yet implemented\n" ); fflush( stdout );


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
hmlp::mpitree::Tree<
  hmlp::gofmm::Setup<SPDMATRIX, SPLITTER, T>, 
  hmlp::gofmm::Data<T>,
  N_CHILDREN,
  T> 
*Compress
( 
  hmlp::Data<T> *X,
  SPDMATRIX &K, 
  hmlp::Data<std::pair<T, std::size_t>> &NN,
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

  /** instantiation for the GOFMM tree */
  using SETUP              = hmlp::gofmm::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA               = hmlp::gofmm::Data<T>;
  using NODE               = hmlp::tree::Node<SETUP, N_CHILDREN, DATA, T>;
  using MPINODE            = hmlp::mpitree::Node<SETUP, N_CHILDREN, DATA, T>;
  using LETNODE            = hmlp::mpitree::LetNode<SETUP, N_CHILDREN, DATA, T>;
  using TREE               = hmlp::mpitree::Tree<SETUP, DATA, N_CHILDREN, T>;

  using NULLTASK           = hmlp::NULLTask<NODE>;
  using MPINULLTASK        = hmlp::NULLTask<MPINODE>;

  using GETMATRIXTASK      = hmlp::mpigofmm::GetSkeletonMatrixTask<NODE>;
  using MPIGETMATRIXTASK   = hmlp::mpigofmm::ParallelGetSkeletonMatrixTask<MPINODE>;

  using SKELTASK           = hmlp::mpigofmm::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T>;
  using MPISKELTASK        = hmlp::mpigofmm::ParallelSkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, MPINODE, T>;

  using PROJTASK           = hmlp::gofmm::InterpolateTask<NODE, T>;
  using MPIPROJTASK        = hmlp::mpigofmm::InterpolateTask<MPINODE, T>;

  using FINDNEARNODESTASK  = hmlp::mpigofmm::FindNearNodesTask<SYMMETRIC, TREE>;
  using CACHENEARNODESTASK = hmlp::gofmm::CacheNearNodesTask<NNPRUNE, NODE>;

  /** instantiation for the randomisze Spd-Askit tree */
  using RKDTSETUP          = hmlp::gofmm::Setup<SPDMATRIX, RKDTSPLITTER, T>;
  using RKDTNODE           = hmlp::tree::Node<RKDTSETUP, N_CHILDREN, DATA, T>;
  using KNNTASK            = hmlp::gofmm::KNNTask<3, RKDTNODE, T>;

  /** all timers */
  double beg, omptask45_time, omptask_time, ref_time;
  double time_ratio, compress_time = 0.0, other_time = 0.0;
  double ann_time, tree_time, skel_time, mergefarnodes_time, cachefarnodes_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

  /** dummy instances for each task */
  GETMATRIXTASK      getmatrixtask;
  MPIGETMATRIXTASK   mpigetmatrixtask;
  SKELTASK           skeltask;
  MPISKELTASK        mpiskeltask;
  PROJTASK           projtask;
  MPIPROJTASK        mpiprojtask;
  KNNTASK            knntask;
  CACHENEARNODESTASK cachenearnodestask;
  NULLTASK           nulltask;
  MPINULLTASK        mpinulltask;


  /** original order of the matrix */
  beg = omp_get_wtime();
  std::vector<std::size_t> gids( n ), lids( n );
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
  hmlp::mpitree::Tree<RKDTSETUP, DATA, N_CHILDREN, T> rkdt;
  rkdt.setup.X = X;
  rkdt.setup.K = &K;
	rkdt.setup.metric = metric; 
  rkdt.setup.splitter = rkdtsplitter;
  std::pair<T, std::size_t> initNN( std::numeric_limits<T>::max(), n );
  printf( "NeighborSearch ...\n" ); fflush( stdout );
  beg = omp_get_wtime();
  if ( NN.size() != n * k )
  {
    NN = rkdt.template AllNearestNeighbor<SORTED>
         ( n_iter, k, 10, gids, lids, initNN, knntask );
  }
  else
  {
    printf( "not performed (precomputed or k=0) ...\n" ); fflush( stdout );
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
  auto *tree_ptr = new hmlp::mpitree::Tree<SETUP, DATA, N_CHILDREN, T>();
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
  printf( "TreePartitioning ...\n" ); fflush( stdout );
  beg = omp_get_wtime();
  tree.TreePartition( n );
  tree_time = omp_get_wtime() - beg;
  printf( "end TreePartitioning ...\n" ); fflush( stdout );
 

  /** Skeletonization */
  printf( "Skeletonization (HMLP Runtime) ...\n" ); fflush( stdout );
  const bool AUTODEPENDENCY = true;
  beg = omp_get_wtime();
  tree.template ParallelTraverseUp<true>( 
      getmatrixtask, mpigetmatrixtask, skeltask, mpiskeltask );
  /** FindNearNodes (executed with skeletonization) */
  auto *findnearnodestask = new FINDNEARNODESTASK();
  findnearnodestask->Set( &tree, budget );
  findnearnodestask->Submit();
  findnearnodestask->DependencyAnalysis();
  
  /** (Any order) compute the interpolative coefficients */
  tree.template ParallelTraverseUp<true>( 
      projtask, mpiprojtask, nulltask, mpinulltask );
  //if ( CACHE )
  //  tree.template TraverseLeafs  <AUTODEPENDENCY, true>( cachenearnodestask );
  printf( "before run\n" ); fflush( stdout );
  other_time += omp_get_wtime() - beg;
  hmlp_run();
  skel_time = omp_get_wtime() - beg;
  printf( "Done\n" ); fflush( stdout );


  //ParallelSimpleFarNodes<LETNODE>( tree.mpitreelists.front() ); 





  return tree_ptr;

}; /** end Compress() */




















}; /** end namespace gofmm */
}; /** end namespace hmlp */

#endif /** define GOFMM_MPI_HPP */
