#ifndef SPDASKIT_HPP
#define SPDASKIT_HPP

#include <vector>
#include <deque>
#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <random>
#include <numeric>


#include <hmlp.h>
#include <hmlp_blas_lapack.h>
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>
#include <tree.hpp>
#include <skel.hpp>
#include <data.hpp>

//#define DEBUG_SPDASKIT 1


namespace hmlp
{
namespace spdaskit
{



template<typename SPDMATRIX, typename SPLITTER, typename T>
class Setup : public hmlp::tree::Setup<SPLITTER, T>
{
  public:

    // Number of neighbors
    size_t k;

    // Maximum rank 
    size_t s;

    // Relative error for rank-revealed QR
    T stol;

    // The SPDMATRIX
    SPDMATRIX *K;

    // Weights
    hmlp::Data<T> w;

}; // end class Setup


/**
 *  @brief This class contains all iaskit related data.
 */ 
template<typename T>
class Data
{
  public:

    Lock lock;

    bool isskel = false;

    std::vector<size_t> skels;

    hmlp::Data<T> proj;

    // Weights
    hmlp::Data<T> w_skel;

    // Potential
    T u;

    // These two prunning lists are used when no NN pruning.
    std::vector<size_t> prune;

    std::vector<size_t> noprune;

    // These two prunning lists are used when in NN pruning.
    std::vector<size_t> NNprune;

    std::vector<size_t> NNnoprune;

    // Events (from HMLP Runtime)
    hmlp::Event skeletonize;

    hmlp::Event updateweight;

}; // end class Data


/**
 *  @brief This class does not need to inherit hmlp::Data<T>, but it should
 *         support two interfaces for data fetching.
 */ 
template<typename T>
class SPDMatrix : public hmlp::Data<T>
{
  public:

    //template<typename TINDEX>
    //std::vector<T> operator()( std::vector<TINDEX> &imap, std::vector<TINDEX> &jmap )
    //{
    //  std::vector<T> submatrix( imap.size() * jmap.size() );
    //  for ( TINDEX j = 0; j < jmap.size(); j ++ )
    //  {
    //    for ( TINDEX i = 0; i < imap.size(); i ++ )
    //    {
    //      submatrix[ j * imap.size() + i ] = (*this)[ d * jmap[ j ] + imap[ i ] ];
    //    }
    //  }
    //  return submatrix;
    //}; 

}; // end class SPDMatrix


template<typename NODE>
class Summary
{

  public:

    Summary() {};

    std::deque<hmlp::Statistic> rank;

    std::deque<hmlp::Statistic> skeletonize;

    std::deque<hmlp::Statistic> updateweight;

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
    };

    void Print()
    {
      for ( size_t l = 0; l < rank.size(); l ++ )
      {
        printf( "@SUMMARY\n" );
        //rank[ l ].Print();
        skeletonize[ l ].Print();
      }
    };

}; // end class Summary


template<int N_SPLIT, typename T>
struct centersplit
{
  // closure
  SPDMatrix<T> *Kptr;

  inline std::vector<std::vector<std::size_t> > operator()
  ( 
    std::vector<std::size_t>& gids,
    std::vector<std::size_t>& lids
  ) const 
  {
    assert( N_SPLIT == 2 );

    SPDMatrix<T> &K = *Kptr;
    size_t N = K.dim();
    size_t n = lids.size();
    std::vector<std::vector<std::size_t> > split( N_SPLIT );

    std::vector<T> temp( n, 0.0 );


    // Compute d2c (distance to center)
    #pragma omp parallel for
    for ( size_t i = 0; i < n; i ++ )
    {
      temp[ i ] = K( lids[ i ], lids[ i ] );
      for ( size_t j = 0; j < std::log( n ); j ++ )
      {
        size_t sample = rand() % n;
        temp[ i ] -= 2.0 * K( lids[ i ], lids[ sample ] );
      }
    }

    // Find the f2c (far most to center)
    auto itf2c = std::max_element( temp.begin(), temp.end() );
    size_t idf2c = std::distance( temp.begin(), itf2c );

    // Compute the d2f (distance to far most)
    #pragma omp parallel for
    for ( size_t i = 0; i < n; i ++ )
    {
      temp[ i ] = K( lids[ i ], lids[ i ] ) - 2.0 * K( lids[ i ], lids[ idf2c ] );
    }

    // Find the f2f (far most to far most)
    auto itf2f = std::max_element( temp.begin(), temp.end() );
    size_t idf2f = std::distance( temp.begin(), itf2f );

#ifdef DEBUG_SPDASKIT
    printf( "idf2c %lu idf2f %lu\n", idf2c, idf2f );
#endif

    // Compute projection
    #pragma omp parallel for
    for ( size_t i = 0; i < n; i ++ )
    {
      temp[ i ] = K( lids[ i ], lids[ idf2f ] ) - K( lids[ i ], lids[ idf2c ] );
    }

    // Parallel median search
    T median = hmlp::tree::Select( n, n / 2, temp );

    split[ 0 ].reserve( n / 2 + 1 );
    split[ 1 ].reserve( n / 2 + 1 );

    for ( size_t i = 0; i < n; i ++ )
    {
      if ( temp[ i ] > median ) split[ 1 ].push_back( i );
      else                      split[ 0 ].push_back( i );
    }

    return split;
  };
}; // end struct 


template<int N_SPLIT, typename T>
struct randomsplit
{
  // closure
  SPDMatrix<T> *Kptr;

  inline std::vector<std::vector<std::size_t> > operator()
  ( 
    std::vector<std::size_t>& gids,
    std::vector<std::size_t>& lids
  ) const 
  {
    assert( N_SPLIT == 2 );

    SPDMatrix<T> &K = *Kptr;
    size_t N = K.dim();
    size_t n = lids.size();
    std::vector<std::vector<std::size_t> > split( N_SPLIT );
    std::vector<T> temp( n, 0.0 );

    size_t idf2c = std::rand() % n;
    size_t idf2f = std::rand() % n;

    while ( idf2c == idf2f ) idf2f = std::rand() % n;

#ifdef DEBUG_SPDASKIT
    printf( "randomsplit idf2c %lu idf2f %lu\n", idf2c, idf2f );
#endif

    // Compute projection
    #pragma omp parallel for
    for ( size_t i = 0; i < n; i ++ )
    {
      temp[ i ] = K( lids[ i ], lids[ idf2f ] ) - K( lids[ i ], lids[ idf2c ] );
    }

    // Parallel median search
    T median = hmlp::tree::Select( n, n / 2, temp );

    split[ 0 ].reserve( n / 2 + 1 );
    split[ 1 ].reserve( n / 2 + 1 );

    for ( size_t i = 0; i < n; i ++ )
    {
      if ( temp[ i ] > median ) split[ 1 ].push_back( i );
      else                      split[ 0 ].push_back( i );
    }

    return split;
  };
}; // end randomsplit


template<class NODE, typename T>
class KNNTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      name = std::string( "neighbor search" );
      arg = user_arg;
      // Need an accurate cost model.
      cost = 1.0;

      //--------------------------------------
      double flops, mops;
      auto &lids = arg->lids;
      auto &NN = *arg->setup->NN;
      flops = lids.size();
      flops *= 4.0 * lids.size();
      // Heap select worst case
      mops = std::log( NN.dim() ) * lids.size();
      mops *= lids.size();
      // Access K
      mops += flops;
      event.Set( flops, mops );
      //--------------------------------------
    };

    void Execute( Worker* user_worker )
    {
      auto &K = *arg->setup->K;
      auto &NN = *arg->setup->NN;
      auto &lids = arg->lids;
     
      // Can be parallelized
      for ( size_t j = 0; j < lids.size(); j ++ )
      {
        for ( size_t i = 0; i < lids.size(); i ++ )
        {
          size_t ilid = lids[ i ];
          size_t jlid = lids[ j ];
          T dist = K( ilid, ilid ) + K( jlid, jlid ) - 2.0 * K( ilid, jlid );
          std::pair<T, size_t> query( dist, ilid );
          hmlp::HeapSelect( 1, NN.dim(), &query, NN.data() + jlid * NN.dim() );
        }
      }
    };
}; // end class SkeletonizeTask


template<typename NODE>
void Skeletonize( NODE *node )
{
  // Early return if we do not need to skeletonize
  if ( !node->parent ) return;

  // random sampling or important sampling for rows.
  std::vector<size_t> amap;
  std::vector<size_t> bmap;
  std::vector<size_t> &lids = node->lids;


  // Gather shared data and create reference
  auto &K = *node->setup->K;
  auto maxs = node->setup->s;
  auto nsamples = 4 * maxs;

  // Gather per node data and create reference
  auto &data = node->data;
  auto &skels = data.skels;
  auto &proj = data.proj;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  if ( node->isleaf )
  {
    bmap = node->lids;
  }
  else
  {
    auto &lskels = lchild->data.skels;
    auto &rskels = rchild->data.skels;
    bmap = lskels;
    bmap.insert( bmap.end(), rskels.begin(), rskels.end() );
  }


  // TODO: random sampling or important sampling for rows. (Severin)
  // Create nsamples.
  // amap = .....
    
   
    // My uniform sample.
  if ( bmap.size() > maxs )
  {
    if ( nsamples < K.num() - node->n )
    {
      amap.reserve( nsamples );
      while ( amap.size() < nsamples )
      {
        size_t sample = rand() % K.num();
        if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() &&
             std::find( lids.begin(), lids.end(), sample ) == lids.end() )
        {
          amap.push_back( sample );
        }
      }
    }
    else // Use all off-diagonal blocks without samples.
    {
      for ( int sample = 0; sample < K.num(); sample ++ )
      {
        if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() )
        {
          amap.push_back( sample );
        }
      }
    }
  }

  // TODO: apdative id. Need to return skels and proj. (James)
  {
    // auto Kab = K( amap, bmap );
    // id( amap.size, bmap.size(), maxs, stol, skels, proj );

    // My fix rank id.
    if ( bmap.size() > maxs )
    {
      auto Kab = K( amap, bmap );
      hmlp::skel::id( amap.size(), bmap.size(), maxs, Kab, skels, proj );
    }
    else
    {
      skels = bmap;
      proj.resize( bmap.size(), bmap.size(), 0.0 );
      for ( int i = 0; i < bmap.size(); i++ )
      {
        proj[ i * proj.dim() + i ] = 1.0;
      }
    }
  }

  data.isskel = true;

}; // end void Skeletonize()


/**
 *
 */ 
template<typename NODE>
class SkeletonizeTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      name = std::string( "Skeletonization" );
      arg = user_arg;
      // Need an accurate cost model.
      cost = 1.0;
    };

    void GetEventRecord()
    {
      arg->data.skeletonize = event;
    };

    void Execute( Worker* user_worker )
    {
      Skeletonize( arg );
    };
}; // end class SkeletonizeTask



template<typename NODE>
void UpdateWeights( NODE *node )
{
  // This function computes the skeleton weights.
  if ( !node->parent ) return;

  // Gather shared data and create reference
  auto &w = node->setup->w;

  // Gather per node data and create reference
  auto &data = node->data;
  auto &proj = data.proj;
  auto &skels = data.skels;
  auto &w_skel = data.w_skel;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  // w_skel is s-by-nrhs
  w_skel.resize( skels.size(), w.dim() );

  if ( node->isleaf )
  {
    auto w_leaf = w( node->lids );

#ifdef DEBUG_SPDASKIT
    printf( "m %lu n %lu k %lu\n", 
      w_skel.dim(), w_skel.num(), w_leaf.num());
    printf( "proj.dim() %lu w_leaf.dim() %lu w_skel.dim() %lu\n", 
        proj.dim(), w_leaf.dim(), w_skel.dim() );
#endif

    xgemm
    (
      "N", "T",
      w_skel.dim(), w_skel.num(), w_leaf.num(),
      1.0, proj.data(),   proj.dim(),
           w_leaf.data(), w_leaf.dim(),
      0.0, w_skel.data(), w_skel.dim()
    );
  }
  else
  {
    auto &w_lskel = lchild->data.w_skel;
    auto &w_rskel = rchild->data.w_skel;
    auto &lskel = lchild->data.skels;
    auto &rskel = rchild->data.skels;
    xgemm
    (
      "N", "N",
      w_skel.dim(), w_skel.num(), lskel.size(),
      1.0, proj.data(),    proj.dim(),
           w_lskel.data(), w_lskel.dim(),
      0.0, w_skel.data(),  w_skel.dim()
    );
    xgemm
    (
      "N", "N",
      w_skel.dim(), w_skel.num(), rskel.size(),
      1.0, proj.data() + proj.dim() * lskel.size(), proj.dim(),
           w_rskel.data(), w_rskel.dim(),
      1.0, w_skel.data(),  w_skel.dim()
    );
  }

}; // end void SetWeights()


/**
 *
 */ 
template<typename NODE>
class UpdateWeightsTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      name = std::string( "SetWeights" );
      arg = user_arg;
      // Need an accurate cost model.
      cost = 1.0;
    };

    void GetEventRecord()
    {
      arg->data.updateweight = event;
    };

    void Execute( Worker* user_worker )
    {
      UpdateWeights( arg );
    };
}; // end class SetWeights


template<typename NODE, typename T>
void Evaluate( NODE *node, NODE *target, hmlp::Data<T> &potentials )
{
  auto &w = node->setup->w;
  auto &lids = node->lids;
  auto &K = *node->setup->K;
  auto &data = node->data;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  auto &amap = target->lids;

  if ( potentials.size() != amap.size() * w.dim() )
  {
    potentials.resize( amap.size(), w.dim(), 0.0 );
  }

  assert( target->isleaf );

  if ( node->isleaf ) // direct evaluation
  {
#ifdef DEBUG_SPDASKIT
    printf( "level %lu direct evaluation\n", node->l );
#endif
    auto Kab = K( amap, lids ); // amap.size()-by-lids.size()
    auto wb  = w( lids ); // nrhs-by-lids.size()
    xgemm
    (
      "N", "T",
      Kab.dim(), wb.dim(), wb.num(),
      1.0, Kab.data(),        Kab.dim(),
           wb.data(),         wb.dim(),
      1.0, potentials.data(), potentials.dim()
    );
  }
  else
  {
    if ( !data.isskel || IsMyParent( target->morton, node->morton ) )
    {
#ifdef DEBUG_SPDASKIT
      printf( "level %lu is not prunable\n", node->l );
#endif
      Evaluate( lchild, target, potentials );      
      Evaluate( rchild, target, potentials );
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
        Kab.dim(), w_skel.num(), w_skel.dim(),
        1.0, Kab.data(),        Kab.dim(),
             w_skel.data(),     w_skel.dim(),
        1.0, potentials.data(), potentials.dim()
      );          
    }
  }

}; // end void Evaluate()


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
  size_t lid, 
  std::vector<size_t> &nnandi, // k + 1 non-prunable lists
  hmlp::Data<T> &potentials 
)
{
  auto &w = node->setup->w;
  auto &lids = node->lids;
  auto &K = *node->setup->K;
  auto &data = node->data;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  auto amap = std::vector<size_t>( 1 );

  amap[ 0 ] = lid;

  if ( !SYMBOLIC ) // No potential evaluation.
  {
    assert( potentials.size() == amap.size() * w.dim() );
  }

  if ( !data.isskel || node->ContainAny( nnandi ) )
  {
    //printf( "level %lu is not prunable\n", node->l );
    if ( node->isleaf )
    {
      if ( SYMBOLIC )
      {
        data.lock.Acquire();
        {
          // Add lid to notprune list. We use a lock.
          if ( NNPRUNE ) data.noprune.push_back( lid );
          else           data.NNnoprune.push_back( lid );
        }
        data.lock.Release();
      }
      else
      {
#ifdef DEBUG_SPDASKIT
        printf( "level %lu direct evaluation\n", node->l );
#endif
        auto Kab = K( amap, lids ); // amap.size()-by-lids.size()
        auto wb  = w( lids ); // nrhs-by-lids.size()
        xgemm
        (
          "N", "T",
          Kab.dim(), wb.dim(), wb.num(),
          1.0, Kab.data(),        Kab.dim(),
          wb.data(),         wb.dim(),
          1.0, potentials.data(), potentials.dim()
        );
      }
    }
    else
    {
      Evaluate<SYMBOLIC, NNPRUNE>( lchild, lid, nnandi, potentials );      
      Evaluate<SYMBOLIC, NNPRUNE>( rchild, lid, nnandi, potentials );
    }
  }
  else // need lid's morton and neighbors' mortons
  {
    //printf( "level %lu is prunable\n", node->l );
    if ( SYMBOLIC )
    {
      data.lock.Acquire();
      {
        // Add lid to prunable list.
        if ( NNPRUNE ) data.prune.push_back( lid );
        else           data.NNprune.push_back( lid );
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
        Kab.dim(), w_skel.num(), w_skel.dim(),
        1.0, Kab.data(),        Kab.dim(),
        w_skel.data(),     w_skel.dim(),
        1.0, potentials.data(), potentials.dim()
      );          
    }
  }



}; // end T Evaluate()


template<bool SYMBOLIC, bool NNPRUNE, typename TREE, typename T>
void Evaluate
( 
  TREE &tree, 
  size_t gid, 
  hmlp::Data<T> &potentials
)
{
  std::vector<size_t> nnandi;
  auto &w = tree.setup.w;
  size_t lid = tree.Getlid( gid );

  potentials.clear();
  potentials.resize( 1, w.dim(), 0.0 );

  if ( NNPRUNE )
  {
    auto &NN = *tree.setup.NN;
    nnandi.reserve( NN.dim() + 1 );
    nnandi.push_back( lid );
    for ( size_t i = 0; i < NN.dim(); i ++ )
    {
      nnandi.push_back( tree.Getlid( NN[ lid * NN.dim() + i ].second ) );
    }
#ifdef DEBUG_SPDASKIT
    printf( "nnandi.size() %lu\n", nnandi.size() );
#endif
  }
  else
  {
    nnandi.reserve( 1 );
    nnandi.push_back( lid );
  }

  Evaluate<SYMBOLIC, NNPRUNE>( tree.treelist[ 0 ], lid, nnandi, potentials );

}; // end void Evaluate()




template<typename NODE, typename T>
void ComputeError( NODE *node, hmlp::Data<T> potentials )
{
  auto &K = *node->setup->K;
  auto &w = node->setup->w;
  auto &lids = node->lids;

  auto &amap = node->lids;
  std::vector<size_t> bmap = std::vector<size_t>( K.num() );

  for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

  auto Kab = K( amap, bmap );

  auto nrm2 = hmlp_norm( potentials.dim(), potentials.num(), 
                         potentials.data(), potentials.dim() ); 

  xgemm
  (
    "N", "T",
    Kab.dim(), w.dim(), w.num(),
    -1.0, Kab.data(),        Kab.dim(),
          w.data(),          w.dim(),
     1.0, potentials.data(), potentials.dim()
  );          

  auto err = hmlp_norm( potentials.dim(), potentials.num(), 
                        potentials.data(), potentials.dim() ); 
  
  printf( "node relative error %E, nrm2 %E\n", err / nrm2, nrm2 );


}; // end void ComputeError()


/**
 *  @brief
 */ 
template<typename TREE, typename T>
T ComputeError( TREE &tree, size_t gid, hmlp::Data<T> potentials )
{
  auto &K = *tree.setup.K;
  auto &w = tree.setup.w;
  auto lid = tree.Getlid( gid );

  auto amap = std::vector<size_t>( 1 );
  auto bmap = std::vector<size_t>( K.num() );
  amap[ 0 ] = lid;
  for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

  auto Kab = K( amap, bmap );

  auto nrm2 = hmlp_norm( potentials.dim(), potentials.num(), 
                         potentials.data(), potentials.dim() ); 

  xgemm
  (
    "N", "T",
    Kab.dim(), w.dim(), w.num(),
    -1.0, Kab.data(),        Kab.dim(),
          w.data(),          w.dim(),
     1.0, potentials.data(), potentials.dim()
  );          

  auto err = hmlp_norm( potentials.dim(), potentials.num(), 
                        potentials.data(), potentials.dim() ); 

  return err / nrm2;
}; // end T ComputeError()



}; // end namespace spdaskit
}; // end namespace hmlp


#endif // define SPDASKIT_HPP
