#include <tree.hpp>

using namespace hmlp::tree;

std::vector<Node<centersplit<2, double>, 2, double>*> TreePartition
(
  int d, int n, int m, int lmax,
  std::vector<double> &X,
  std::vector<long> &gids,
  std::vector<long> &lids
)
{
  return TreePartition( d, n, m, lmax, X, gids, lids );
}
