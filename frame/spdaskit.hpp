#ifndef SPDASKIT_HPP
#define SPDASKIT_HPP

#include <vector>
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

    // Maximum rank 
    size_t s;

    // Relative error for rank-revealed QR
    T stol;

    // The SPDMATRIX
    SPDMATRIX K;

    // Neighbors
    hmlp::Data<std::pair<T,size_t>> NN;

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

    bool isskel = false;

    std::vector<size_t> skels;

    hmlp::Data<T> proj;

    // Weights
    hmlp::Data<T> w_skel;

    // Potential
    T u;

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
  auto &K = node->setup->K;
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
  auto &K = node->setup->K;
  auto &data = node->data;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  auto &amap = target->lids;

  if ( potentials.size() != amap.size() * w.dim() )
  {
    potentials.resize( amap.size(), w.dim(), 0.0 );
  }

  if ( node->isleaf ) // direct evaluation
  {
    printf( "level %lu direct evaluation\n", node->l );
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
    if ( !data.isskel && node->parent )
    {
      printf( "level %lu not skel\n", node->l );
      Evaluate( lchild, target, potentials );      
      Evaluate( rchild, target, potentials );
    }
    else
    {
      // Check if target is the child of node.
      auto *parent = target;
      
      while ( parent )
      {
        if ( parent == lchild )
        {
          printf( "level %lu recur left prun right\n", node->l );
          Evaluate( lchild, target, potentials );
          auto Kab = K( amap, rchild->data.skels );
          auto &w_skel = rchild->data.w_skel;
          xgemm
          (
            "N", "N",
            Kab.dim(), w_skel.num(), w_skel.dim(),
            1.0, Kab.data(),        Kab.dim(),
                 w_skel.data(),     w_skel.dim(),
            1.0, potentials.data(), potentials.dim()
          );          
          break;
        }
        if ( parent == rchild )
        {
          printf( "level %lu recur right prun left\n", node->l );
          Evaluate( rchild, target, potentials );
          auto Kab = K( amap, lchild->data.skels );
          auto &w_skel = lchild->data.w_skel;
          xgemm
          (
            "N", "N",
            Kab.dim(), w_skel.num(), w_skel.dim(),
            1.0, Kab.data(),        Kab.dim(),
                 w_skel.data(),     w_skel.dim(),
            1.0, potentials.data(), potentials.dim()
          );          
          break;
        }
        parent = parent->parent;
      }
    }
  }

}; // end void Evaluate()


template<typename NODE, typename T>
void ComputeError( NODE *node, hmlp::Data<T> potentials )
{
  auto &K = node->setup->K;
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




}; // end namespace spdaskit
}; // end namespace hmlp


#endif // define SPDASKIT_HPP
