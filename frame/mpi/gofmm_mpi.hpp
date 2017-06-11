#ifndef GOFMM_MPI_HPP
#define GOFMM_MPI_HPP

#include <mpi.h>
#include <tree_mpi.hpp>
#include <gofmm.hpp>

namespace hmlp
{
namespace gofmm
{
namespace mpi
{






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
 *  @brielf template of the mpi compression routine
 */ 
template<
  bool        ADAPTIVE, 
  bool        LEVELRESTRICTION, 
  SplitScheme SPLIT,
  typename    SPLITTER, 
  typename    RKDTSPLITTER, 
  typename    T, 
  typename    SPDMATRIX>
hmlp::tree::mpi::Tree<
  hmlp::gofmm::Setup<SPDMATRIX, SPLITTER, T>, 
  hmlp::tree::mpi::Node<
    hmlp::gofmm::Setup<SPDMATRIX, SPLITTER, T>, 
    N_CHILDREN,
    hmlp::gofmm::Data<T>,
    T
    >,
  N_CHILDREN,
  T
  > 
Compress
( 
  hmlp::Data<T> *X,
  SPDMATRIX &K, 
  hmlp::Data<std::pair<T, std::size_t>> &NN,
  SPLITTER splitter, 
  RKDTSPLITTER rkdtsplitter,
  size_t n, size_t m, size_t k, size_t s, 
  double stol, double budget 
)
{
  /** options */
  const bool SYMMETRIC = true;
  const bool NNPRUNE   = true;
  const bool CACHE     = true;

  /** instantiation for the GOFMM tree */
  using SETUP              = hmlp::gofmm::Setup<SPDMATRIX, SPLITTER, T>;
  using DATA               = hmlp::gofmm::Data<T>;
  using NODE               = hmlp::tree::mpi::Node<SETUP, N_CHILDREN, DATA, T>;
  using TREE               = hmlp::tree::mpi::Tree<SETUP, NODE, N_CHILDREN, T>;
  using SKELTASK           = hmlp::gofmm::SkeletonizeTask<ADAPTIVE, LEVELRESTRICTION, NODE, T>;
  using PROJTASK           = hmlp::gofmm::InterpolateTask<NODE, T>;
  using NEARNODESTASK      = hmlp::gofmm::NearNodesTask<SYMMETRIC, TREE>;
  using CACHENEARNODESTASK = hmlp::gofmm::CacheNearNodesTask<NNPRUNE, NODE>;

  /** instantiation for the randomisze GOFMM tree */
  using RKDTSETUP          = hmlp::gofmm::Setup<SPDMATRIX, RKDTSPLITTER, T>;
  using RKDTNODE           = hmlp::tree::mpi::Node<RKDTSETUP, N_CHILDREN, DATA, T>;
  using KNNTASK            = hmlp::gofmm::KNNTask<3, SPLIT, RKDTNODE, T>;

  /** all timers */
  double beg, omptask45_time, omptask_time, ref_time;
  double time_ratio, compress_time = 0.0, other_time = 0.0;
  double ann_time, tree_time, skel_time, mergefarnodes_time, cachefarnodes_time;
  double nneval_time, nonneval_time, fmm_evaluation_time, symbolic_evaluation_time;

  /** dummy instances for each task */
  SKELTASK           skeltask;
  PROJTASK           projtask;
  KNNTASK            knntask;
  CACHENEARNODESTASK cachenearnodestask;

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
  hmlp::tree::mpi::Tree<RKDTSETUP, RKDTNODE, N_CHILDREN, T> rkdt;
  rkdt.setup.X = X;
  rkdt.setup.K = &K;
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

  /** initialize metric ball tree using approximate center split */
  hmlp::tree::mpi::Tree<SETUP, NODE, N_CHILDREN, T> tree;
  //tree.setup.X = X;
  //tree.setup.K = &K;
  //tree.setup.splitter = splitter;
  //tree.setup.NN = &NN;
  //tree.setup.m = m;
  //tree.setup.k = k;
  //tree.setup.s = s;
  //tree.setup.stol = stol;
  //printf( "TreePartitioning ...\n" ); fflush( stdout );
  //beg = omp_get_wtime();
  //tree.TreePartition( gids, lids );
  //tree_time = omp_get_wtime() - beg;


  


  /** not yet implemented */
  printf( "\nnot yet implemented\n" ); fflush( stdout );



  return tree;

}; /** end Compress() */



}; /** end namespace mpi */
}; /** end namespace gofmm */
}; /** end namespace hmlp */

#endif /** define GOFMM_MPI_HPP */
