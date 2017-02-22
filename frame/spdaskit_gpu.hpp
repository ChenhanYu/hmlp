#ifndef SPDASKIT_GPU_HPP
#define SPDASKIT_GPU_HPP

#define DEBUG_SPDASKIT_GPU 1

namespace hmlp
{
namespace spdaskit
{
namespace gpu
{
 

template<typename NODE>
void UpdateWeights( NODE *node )
{
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

  /** get the device id */
  // cuda_set_device();


  /** s-by-nrhs, TODO: need a way to only allocate GPU memory */
  w_skel.clear();
  w_skel.resize( skels.size(), w.row() );


};





};
};
};

#endif // define SPDASKIT_GPU_HPP
