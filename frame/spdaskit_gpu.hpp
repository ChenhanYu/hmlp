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
  printf( "\n%lu UpdateWeight on GPU\n", node->treelist_id );
#endif
  /** early return */
  if ( !node->parent || !node->data.isskel ) return;

  double beg;

  /** gather shared data and create reference */
  auto &w = *node->setup->w;

  /** gather per node data and create reference */
  auto &data = node->data;
  auto &proj = data.proj;
  auto &skels = data.skels;
  auto &w_skel = data.w_skel;
  auto &w_leaf = data.w_leaf;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  /** s-by-nrhs, TODO: need a way to only allocate GPU memory */
  w_skel.clear();
  w_skel.resize( skels.size(), w.row() );
  auto m = w_skel.row();
  auto n = w_skel.col();

  printf( "-->w_proj\n" );
  proj.FetchH2D( dev );
  printf( "-->w_skel\n" );
  //w_skel.FetchH2D( dev );
  w_skel.AllocateD( dev );

  assert( proj.device_data( dev ) );
  assert( w_skel.device_data( dev ) );

  /** */
  if ( node->isleaf )
  {
    /** need a memcpy */
    auto k = w_leaf.row();
    printf( "-->w_leaf\n" );
    w_leaf.FetchH2D( dev );
    assert( w_leaf.device_data( dev ) );

    beg = omp_get_wtime();
    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle(),
      CUBLAS_OP_N, CUBLAS_OP_N,
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

    printf( "-->w_lskel\n" );
    w_lskel.FetchH2D( dev );
    printf( "-->w_rskel\n" );
    w_rskel.FetchH2D( dev );
    assert( w_lskel.device_data( dev ) );
    assert( w_rskel.device_data( dev ) );

    beg = omp_get_wtime();
    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle(),
      CUBLAS_OP_N, CUBLAS_OP_N,
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

  /** make sure that cublas is fully stopped */
  dev->waitexecute();

  double gemm_time = omp_get_wtime() - beg;
  printf( "cublas %5.2lf\n", gemm_time );


  /** cliam redistribution, now only dev has the latest copy */
  w_skel.Redistribute<true>( dev );

  /** store back */
  printf( "-->w_skel\n" );
  w_skel.FetchD2H( dev );

  printf( "finish GPU task\n\n" );
};


template<int SUBTASKID, bool NNPRUNE, typename NODE, typename T, typename DEVICE>
void LeavesToLeaves( DEVICE *dev, NODE *node, size_t itbeg, size_t itend )
{
  assert( node->isleaf );

  /** gather shared data and create reference */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;
  auto &u = *node->setup->u;

  auto &lids = node->lids;
  auto &data = node->data;
  auto &amap = node->lids;
  auto &NearKab = data.NearKab;

  std::set<NODE*> *NearNodes;
  if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
  else           NearNodes = &node->NearNodes;

  auto &u_leaf = data.u_leaf[ SUBTASKID ];
  u_leaf.clear();

  /** early return if nothing to do */
  if ( itbeg == itend ) 
  {
    u_leaf.resize( 0, 0 );
    return;
  }
  else
  {
    u_leaf.clear();
    u_leaf.resize( lids.size(), w.row(), 0.0 );
  }

  //if ( NearKab.size() ) /** Kab is cached */
  //{
  //  size_t itptr = 0;
  //  size_t offset = 0;

  //  for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
  //  {
  //    if ( itbeg <= itptr && itptr < itend )
  //    {
  //      auto &bmap = (*it)->lids;
  //      //auto wb = w( bmap );
  //      auto wb = (*it)->data.w_leaf;

  //      /** Kab * wb */
  //      xgemm
  //      (
  //        "N", "N",
  //        u_leaf.row(), u_leaf.col(), wb.row(),
  //        1.0, NearKab.data() + offset * NearKab.row(), NearKab.row(),
  //                  wb.data(),                               wb.row(),
  //        1.0,  u_leaf.data(),                           u_leaf.row()
  //      );
  //    }
  //  }
  //}



};



};
};
};

#endif // define SPDASKIT_GPU_HPP
