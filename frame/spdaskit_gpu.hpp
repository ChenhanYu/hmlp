#ifndef SPDASKIT_GPU_HPP
#define SPDASKIT_GPU_HPP

#define DEBUG_SPDASKIT_GPU 1

namespace hmlp
{
namespace spdaskit
{
namespace gpu
{
 

template<typename DEVICE, typename NODE>
void UpdateWeights( DEVICE *dev, NODE *node )
{
  assert( dev && node );

#ifdef DEBUG_SPDASKIT_GPU
  printf( "%lu UpdateWeight\n", node->treelist_id );
#endif
  /** early return */
  if ( !node->parent || !node->data.isskel ) return;

  /** gather shared data and create reference */
  auto &w = *node->setup->w;

  /** gather per node data and create reference */
  auto &data = node->data;
  auto &proj = data.proj;
  auto &skels = data.skels;
  auto &w_skel = data.w_skel;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  /** s-by-nrhs, TODO: need a way to only allocate GPU memory */
  w_skel.clear();
  w_skel.resize( skels.size(), w.row() );
  auto m = w_skel.row();
  auto n = w_skel.col();

  proj.FetchH2D( dev );
  //proj.PrefetchH2D( dev );
  //proj.WaitPrefetch( dev );
  w_skel.FetchH2D( dev );
  //w_skel.PrefetchH2D( dev );
  //w_skel.WaitPrefetch( dev );

  printf( "proj and w_skel H2D\n" );

  assert( proj.device_data( dev ) );
  assert( w_skel.device_data( dev ) );

  /** */
  if ( node->isleaf )
  {
    /** need a memcpy */
    auto w_leaf = w( node->lids );
    auto k = w_leaf.col();
    w_leaf.FetchH2D( dev );
    //w_leaf.PrefetchH2D( dev );
    //w_leaf.WaitPrefetch( dev );
    assert( w_leaf.device_data( dev ) );
    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle(),
      CUBLAS_OP_N, CUBLAS_OP_T,
      m, n, k,
      1.0, proj.device_data( dev ),     proj.row(),
           w_leaf.device_data( dev ), w_leaf.row(),
      0.0, w_skel.device_data( dev ), w_skel.row()
    );
  }
  else
  {
    auto &w_lskel = lchild->data.w_skel;
    auto &w_rskel = rchild->data.w_skel;

    w_lskel.FetchH2D( dev );
    w_rskel.FetchH2D( dev );
    assert( w_lskel.device_data( dev ) );
    assert( w_rskel.device_data( dev ) );

    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle(),
      CUBLAS_OP_N, CUBLAS_OP_T,
      m, n, w_lskel.row(),
      1.0, proj.device_data( dev ),       proj.row(),
           w_lskel.device_data( dev ), w_lskel.row(),
      0.0, w_skel.device_data( dev ),   w_skel.row()
    );
    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle(),
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, w_rskel.row(),
      1.0, proj.device_data( dev ) + m * w_lskel.row(),    proj.row(),
           w_rskel.device_data( dev ),                  w_rskel.row(),
      1.0, w_skel.device_data( dev ),                    w_skel.row()
    );
  }

  /** cliam redistribution, now only dev has the latest copy */
  w_skel.Redistribute<true>( dev );

  /** store back */
  w_skel.FetchD2H( dev );

  printf( "finish GPU task\n" );
};





};
};
};

#endif // define SPDASKIT_GPU_HPP
