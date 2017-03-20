#ifndef SPDASKIT_GPU_HPP
#define SPDASKIT_GPU_HPP

#include <omp.h>

#define DEBUG_SPDASKIT_GPU 1

namespace hmlp
{
namespace spdaskit
{
namespace gpu
{

void assemble
( cudaStream_t stream, int m, int n, const double *a, const size_t *amap, double *A );
void assemble
( cudaStream_t stream, int m, int n, const float *a, const size_t *amap, float *A );




template<typename DEVICE, typename NODE>
void UpdateWeights( DEVICE *dev, NODE *node )
{
  assert( dev && node );

#ifdef DEBUG_SPDASKIT_GPU
  //printf( "\n%lu UpdateWeight on GPU\n", node->treelist_id );
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
  //printf( "-->w_skel\n" );
  w_skel.CacheD( dev );


  auto m = w_skel.row();
  auto n = w_skel.col();

  //printf( "-->w_proj\n" );
  proj.FetchH2D( dev );

  assert( proj.device_data( dev ) );
  assert( w_skel.device_data( dev ) );

  /** */
  if ( node->isleaf )
  {
    /** need a memcpy */
    auto k = w_leaf.row();
    flops = 2.0 * m * n * k;
    //printf( "-->w_leaf\n" );
    w_leaf.FetchH2D( dev );
    assert( w_leaf.device_data( dev ) );

    beg = omp_get_wtime();
    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( 0 ),
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      1.0, proj.device_data( dev ),     proj.row(),
           w_leaf.device_data( dev ), w_leaf.row(),
      0.0, w_skel.device_data( dev ), w_skel.row()
    );
    dev->wait( 0 );
  }
  else
  {
    auto &w_lskel = lchild->data.w_skel;
    auto &w_rskel = rchild->data.w_skel;
    flops = 2.0 * m * n * ( w_lskel.row() + w_rskel.row() );

    //printf( "-->w_lskel\n" );
    w_lskel.FetchH2D( dev );
    //printf( "-->w_rskel\n" );
    w_rskel.FetchH2D( dev );
    assert( w_lskel.device_data( dev ) );
    assert( w_rskel.device_data( dev ) );

    beg = omp_get_wtime();
    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( 0 ),
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, w_lskel.row(),
      1.0, proj.device_data( dev ),       proj.row(),
           w_lskel.device_data( dev ), w_lskel.row(),
      0.0, w_skel.device_data( dev ),   w_skel.row()
    );
    dev->wait( 0 );
    hmlp::xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( 0 ),
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, w_rskel.row(),
      1.0, proj.device_data( dev ) + m * w_lskel.row(),    proj.row(),
           w_rskel.device_data( dev ),                  w_rskel.row(),
      1.0, w_skel.device_data( dev ),                    w_skel.row()
    );
    dev->wait( 0 );
  }

  double gemm_time = omp_get_wtime() - beg;
  printf( "cublas %5.2lf (%5.2lf GFLOPS)\n", 
      gemm_time, flops / ( gemm_time * 1E+9 ) );

  /** cliam redistribution, now only dev has the latest copy */
  w_skel.Redistribute<true>( dev );
  w_skel.FetchD2H( dev );

  //printf( "finish GPU task\n\n" );
};


template<bool NNPRUNE, typename NODE, typename T, typename DEVICE>
void SkeletonsToNodes( DEVICE *dev, NODE *node )
{
#ifdef DEBUG_SPDASKIT_GPU
  //printf( "%lu GPU Skel2Node u_skel.row() %lu\n", node->treelist_id, node->data.u_skel.row() ); fflush( stdout );
#endif

  double beg, kij_s2n_time = 0.0, u_leaf_time, before_writeback_time, after_writeback_time;

  /** gather shared data and create reference */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;
  auto &u = *node->setup->u;

  /** Gather per node data and create reference */
  auto &lids = node->lids;
  auto &data = node->data;
  auto &proj = data.proj;
  auto &skels = data.skels;
  auto &u_skel = data.u_skel;
  auto *lchild = node->lchild;
  auto *rchild = node->rchild;

  if ( node->isleaf )
  {
    std::set<NODE*> *NearNodes;
    if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
    else           NearNodes = &node->NearNodes;
    auto &amap = node->lids;
    auto &u_leaf = node->data.u_leaf[ 0 ];
    u_leaf.clear();
    u_leaf.resize( lids.size(), w.row(), 0.0 );

    assert( u_leaf.size() == w.row() * lids.size() );

    /** accumulate far interactions */
    if ( data.isskel )
    {
      //xgemm
      //(
      //  "T", "N",
      //  u_leaf.row(), u_leaf.col(), u_skel.row(),
      //  1.0,   proj.data(),   proj.row(),
      //       u_skel.data(), u_skel.row(),
      //  1.0, u_leaf.data(), u_leaf.row()
      //);

      proj.CacheD( dev );
      proj.FetchH2D( dev );

      u_skel.CacheD( dev );
      u_skel.FetchH2D( dev );

      u_leaf.CacheD( dev );

      xgemm
      (
        reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( 0 ),
        CUBLAS_OP_T, CUBLAS_OP_N,
        u_leaf.row(), u_leaf.col(), u_skel.row(),
        1.0,   proj.device_data( dev ),   proj.row(),
             u_skel.device_data( dev ), u_skel.row(),
        0.0, u_leaf.device_data( dev ), u_leaf.row()
      );
      dev->wait( 0 );

      u_leaf.Redistribute<true>( dev );
      u_leaf.FetchD2H( dev );
    }
  }
  else
  {
    if ( !node->parent || !node->data.isskel ) return;

    auto &u_lskel = lchild->data.u_skel;
    auto &u_rskel = rchild->data.u_skel;
    auto &lskel = lchild->data.skels;
    auto &rskel = rchild->data.skels;

    proj.CacheD( dev );
    proj.FetchH2D( dev );

    u_skel.CacheD( dev );
    u_skel.FetchH2D( dev );

    u_lskel.CacheD( dev );
    u_lskel.FetchH2D( dev );

    u_rskel.CacheD( dev );
    u_rskel.FetchH2D( dev );

    xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( 0 ),
      CUBLAS_OP_T, CUBLAS_OP_N,
      u_lskel.row(), u_lskel.col(), proj.row(),
      1.0, proj.device_data( dev ),    proj.row(),
           u_skel.device_data( dev ),  u_skel.row(),
      1.0, u_lskel.device_data( dev ), u_lskel.row()
    );
    dev->wait( 0 );
    xgemm
    (
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( 0 ),
      CUBLAS_OP_T, CUBLAS_OP_N,
      u_rskel.row(), u_rskel.col(), proj.row(),
      1.0, proj.device_data( dev ) + proj.row() * lskel.size(), proj.row(),
           u_skel.device_data( dev ), u_skel.row(),
      1.0, u_rskel.device_data( dev ), u_rskel.row()
    );
    dev->wait( 0 );
    
    u_lskel.Redistribute<true>( dev );
    u_rskel.Redistribute<true>( dev );
    u_lskel.FetchD2H( dev );
    u_rskel.FetchD2H( dev );
  }
};









template<bool CACHE, bool NNPRUNE, typename NODE, typename T, typename DEVICE>
void LeavesToLeaves( DEVICE *dev, NODE *node )
{
  assert( node && node->isleaf );

#ifdef DEBUG_SPDASKIT_GPU
  //printf( "\n%lu LeavesToLeaves on GPU\n", node->treelist_id );
#endif

  double beg, gemm_time = 0.0, flops = 0.0;

  /** use cuda stream for asynchronous execution */
  int stream_id = 0;

  /** gather shared data and create reference */
  auto &K = *node->setup->K;
  auto &w = *node->setup->w;
  auto &u = *node->setup->u;

  auto &lids = node->lids;
  auto &data = node->data;
  auto &amap = node->lids;
  auto &Nearbmap = data.Nearbmap;
  auto &NearKab = data.NearKab;
  auto &w_leaf = data.w_leaf;


  /** if NearKab has not yet been computed */
  if ( NearKab.size() != lids.size() * Nearbmap.size() )
  {
    std::vector<size_t> bmap( Nearbmap.size() );
    for ( size_t i = 0; i < bmap.size(); i ++ )
      bmap[ i ] = Nearbmap[ i ];
    assert( !CACHE );
    NearKab = K( lids, bmap );
  }

  if ( dev )
  {
    size_t m = w_leaf.col();
    size_t n = NearKab.col();
    size_t k = NearKab.row();

    flops += 2.0 * m * n * k;

    //dev->wait( node->treelist_id % 8 );
    w_leaf.FetchH2D( dev );
    NearKab.FetchH2D( dev );
    Nearbmap.FetchH2D( dev );

    //printf( "prepare cublas gemm\n" ); fflush( stdout );
    assert( m * n * sizeof(T) < 1200000000 );

    cublasHandle_t &handle = 
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( stream_id );
    T *A = w_leaf.device_data( dev );
    T *B = NearKab.device_data( dev );
    T *C = (T*)reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->workspace();

    //dev->wait( stream_id );

    beg = omp_get_wtime();
    /** w_leaf' * Kab  */
    hmlp::xgemm
    (
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      m, n, k,
      1.0, A, w_leaf.row(),  
           B, NearKab.row(),
      0.0, C, m
    );

    //dev->wait( stream_id );
    //gemm_time = omp_get_wtime() - beg;

    //printf( "assemble\n" );
    assemble
    ( 
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->getstream( stream_id ),
      m, n,
      C,
      Nearbmap.device_data( dev ),
      u.device_data( dev )
    );
  
    /** make sure that cublas is fully stopped */
    dev->wait( stream_id );

    gemm_time = omp_get_wtime() - beg;
    printf( "cublas m %lu n %lu k %lu, %5.3lf (%5.2lf GFLOPS)\n", 
      m, n, k, 
      gemm_time, flops / ( gemm_time * 1E+9 ) );

    /** now only this device has the latest copy */
    u.Redistribute<true>( dev );


    /** free device memory */
    //w_leaf.FreeD( dev );
    //NearKab.FreeD( dev );
    //Nearbmap.FreeD( dev );
  }
  else
  {
    //printf( "cpu gemm begin\n" ); fflush( stdout );
    std::set<NODE*> *NearNodes;
    if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
    else           NearNodes = &node->NearNodes;

    size_t n = w_leaf.col();
    size_t k = NearKab.row();
    size_t offset = 0;

    assert( NearKab.size() );
    assert( k == w_leaf.row() );

    for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
    {
      assert( omp_get_thread_num() > 0 && omp_get_thread_num() < 20 );
      auto &u_leaf = (*it)->data.u_leaf[ omp_get_thread_num() ];
      size_t m = (*it)->lids.size();

      assert( offset < NearKab.col() );
      assert( w_leaf.size() == k * n );


      if ( u_leaf.size() != m * n )
      {
        //printf( "u_leaf.size() %lu m %lu n %lu w.row() %lu\n", 
        //    u_leaf.size(), m, n, w.row() );
        u_leaf.clear();
        u_leaf.resize( m, n, 0.0 );
      }

      //printf( "NearKab.col() %lu m %lu offset %lu\n",
      //    NearKab.col(), m, offset ); fflush( stdout );

      hmlp::xgemm
      (
        "T", "N",
        m, n, k,
        1.0, NearKab.data() + offset * k, k,
              w_leaf.data(),              w_leaf.row(),
        1.0,  u_leaf.data(),              u_leaf.row()
      );
      offset += m;
    }
    //printf( "cpu gemm finishd\n" ); fflush( stdout );
  }

  /** free Kab if not cached */
  if ( !CACHE ) NearKab.resize( 0, 0 );

  //printf( "Finish GPU LeavesToLeaves\n\n" );
};


template<bool CACHE, bool NNPRUNE, typename NODE, typename T>
class LeavesToLeavesVer2Task : public hmlp::Task
{
  public:

    NODE *arg;

    int stream_id;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      stream_id = ( arg->treelist_id % 8 ) + 1;
      name = std::string( "l2l" );
      {
        //label = std::to_string( arg->treelist_id );
        std::ostringstream ss;
        ss << arg->treelist_id;
        label = ss.str();
      }

      assert( arg->isleaf );


      /** TODO: fill in flops and mops */
      //--------------------------------------
      double flops = 0.0, mops = 0.0;
      auto *node = arg;
      auto &data = node->data;
      auto &w_leaf = data.w_leaf;
      auto *NearNodes = &node->NearNodes;
      if ( NNPRUNE ) NearNodes = &node->NNNearNodes;
      auto &amap = node->lids;

      if ( !CACHE )
      {
        std::vector<size_t> bmap;
        for ( auto it = NearNodes->begin(); it != NearNodes->end(); it ++ )
        {
          bmap.insert( bmap.end(), (*it)->lids.begin(), (*it)->lids.end() );
        }
        data.Nearbmap.resize( bmap.size(), 1 );
        for ( size_t i = 0; i < bmap.size(); i ++ ) 
          data.Nearbmap[ i ] = bmap[ i ];
      }

      size_t m = w_leaf.col();
      size_t n = arg->data.Nearbmap.size();
      size_t k = arg->lids.size();

      flops += 2.0 * m * n * k;
      mops += 2.0 * ( m * n + m * k + k * n );

      /** setup the event */
      event.Set( name + label, flops, mops );

      /** assume computation bound */
      cost = flops / 1E+9;

      //printf( "cost %5.2lf\n", cost );

      /** high priority */
      priority = false;
    };

    void Prefetch( Worker* user_worker )
    {
      hmlp::Device *device = NULL;
      if ( user_worker ) device = user_worker->GetDevice();

      //if ( !arg->data.NearKab.is_up_to_date( hmlp_get_device( 0 ) ) )
      //  device = NULL;

      if ( device )
      {
        if ( !CACHE )
        {
          assert( !arg->data.NearKab.size() );
          auto &K = *arg->setup->K;
          std::vector<size_t> bmap( arg->data.Nearbmap.size() );
          for ( size_t i = 0; i < bmap.size(); i ++ )
            bmap[ i ] = arg->data.Nearbmap[ i ];
          arg->data.NearKab = K( arg->lids, bmap );
          
          printf( "not cache\n" );
        }
        if ( arg->data.NearKab.size() < 4096 * 4096 * 4 )
          arg->data.NearKab.CacheD( device );
        arg->data.NearKab.PrefetchH2D( device, stream_id );

        /** use device cache (512MB) */
        arg->data.Nearbmap.CacheD( device );
        arg->data.Nearbmap.PrefetchH2D( device, stream_id );
        //arg->data.Nearbmap.PrefetchH2D( device, 2 );
        arg->data.w_leaf.CacheD( device );
        arg->data.w_leaf.PrefetchH2D( device, stream_id );
        //arg->data.w_leaf.PrefetchH2D( device, 3 );
      }
    };

    void DependencyAnalysis()
    {
      assert( arg->isleaf );
      //if ( arg->data.NearKab.is_up_to_date( hmlp_get_device( 0 ) ) )
        this->ForceEnqueue( 0 );
      //else
      //  this->Enqueue();
    };

    void Execute( Worker* user_worker )
    {
      hmlp::Device *device = NULL;
      if ( user_worker ) device = user_worker->GetDevice();
      gpu::LeavesToLeaves<CACHE, NNPRUNE, NODE, T>( device, arg );
    };

};













};
};
};

#endif // define SPDASKIT_GPU_HPP
