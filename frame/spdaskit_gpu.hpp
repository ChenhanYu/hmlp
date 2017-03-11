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

  double beg, flops;

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
    flops = 2.0 * m * n * k;
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
    flops = 2.0 * m * n * ( w_lskel.row() + w_rskel.row() );

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
  printf( "cublas %5.2lf (%5.2lf GFLOPS)\n", 
      gemm_time, flops / ( gemm_time * 1E+9 ) );


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
  assert( dev && node && node->isleaf );

#ifdef DEBUG_SPDASKIT_GPU
  //printf( "\n%lu LeavesToLeaves on GPU\n", node->treelist_id );
#endif

  double beg, gemm_time = 0.0, flops = 0.0;

  /** use cuda stream for asynchronous execution */
  int stream_id = node->treelist_id % 8;

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

  /** early return if nothing to do */
  if ( SUBTASKID != 1 ) 
  {
    u_leaf.resize( 0, 0 );
    return;
  }
  else
  {
    //u_leaf.clear();
    //u_leaf.resize( lids.size(), w.row(), 0.0 );
    //u_leaf.AllocateD( dev );
  }

  //printf( "NearKab\n" );
  //NearKab.FetchH2D( dev );

  size_t m = u_leaf.row();
  size_t n = u_leaf.col();
  size_t offset = 0;

  beg = omp_get_wtime();
  for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
  {
    auto wb = (*it)->data.w_leaf;
    size_t k = wb.row();

    /** Kab * wb */
    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( stream_id ),
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      1.0, NearKab.device_data( dev ) + offset * m, m,
                wb.device_data( dev ),              k,
      1.0,  u_leaf.device_data( dev ),              m
    );
    //flops += 2.0 * m * n * k;
    offset += (*it)->lids.size();
  }
  
  /** make sure that cublas is fully stopped */
  dev->wait( stream_id );

  gemm_time = omp_get_wtime() - beg;
  printf( "cublas m %lu n %lu %5.2lf (%5.2lf GFLOPS)\n", 
      u_leaf.row(), u_leaf.col(), 
      gemm_time, flops / ( gemm_time * 1E+9 ) );

  
  /** cliam redistribution, now only dev has the latest copy */
  u_leaf.Redistribute<true>( dev );

  /** store back */
  //u_leaf.FetchD2H( dev );
  u_leaf.PrefetchD2H( dev, stream_id );

  //printf( "Finish GPU LeavesToLeaves\n\n" );
};



};
};
};

#endif // define SPDASKIT_GPU_HPP
