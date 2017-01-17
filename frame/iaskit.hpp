#ifndef IASKIT_HPP
#define IASKIT_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <random>


#include <hmlp.h>
#include <hmlp_blas_lapack.h>
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>
#include <tree.hpp>

#define DEBUG_IASKIT 1

namespace hmlp
{
namespace iaskit
{

template<typename T>
class Data
{
  public:

    std::vector<size_t> skels;

    std::vector<T> proj;

};


template<typename NODE>
void skeletonize( NODE *node )
{
  auto lchild = node->lchild;
  auto rchild = node->rchild;

  // random sampling or important sampling for rows.
  std::vector<size_t> amap;

  std::vector<size_t> bmap;

  //bmap = lchild

  auto &data = node->data;

  printf( "id %d l %d n %d isleaf %d\n", node->treelist_id, node->l, node->n, node->isleaf );
  printf( "skels.size() %lu\n", node->data.skels.size() );

}; // end skeletonize()


  

template<typename NODE>
class Task : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      name = std::string( "Skeletonization" );
      arg = user_arg;
    };

    void Execute( Worker* user_worker )
    {
      //printf( "SkeletonizeTask Execute 2\n" );
      skeletonize( arg );
    };

  private:
};







}; // end namespace iaskit
}; // end namespace hmlp

#endif // define IASKIT_HPP
