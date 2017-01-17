#include <tree.hpp>

#define N_CHILDREN 2

using namespace hmlp::tree;

using T = double;
using DATA = hmlp::Data<T>;
using SPLITTER = centersplit<N_CHILDREN, T>;
using NODE = Node<SPLITTER, N_CHILDREN, DATA, T>;

//std::vector<Node<centersplit<2, double>, 2, double>*> TreePartition
std::vector<NODE*> TreePartition
(
  int d, int n, int m, int lmax,
  std::vector<T> &X,
  std::vector<long> &gids,
  std::vector<long> &lids
)
{
  return TreePartition( d, n, m, lmax, X, gids, lids );
}
