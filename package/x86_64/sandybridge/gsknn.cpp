// GSKNN templates
#include <gsknn.hpp>

// Sandy-bridge
// #include <rnn_rank_k_asm_d8x4.hpp>
#include <rank_k_d8x4.hpp>
#include <rnn_r_int_d8x4_row.hpp>

using namespace hmlp::gsknn;

void dgsknn
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
)
{
  // Sandy-bridge
  rank_k_int_d8x4 semiringkernel;
  rnn_r_int_d8x4_row microkernel;
  gsknn<
    104, 2048, 256, 8, 4, 104, 2048, 8, 4, 32,
    true,
    rank_k_int_d8x4,
    rnn_r_int_d8x4_row,
    double, double, double, double>
  (
    m, n, k, r,
    A, A2, amap,
    B, B2, bmap,
    D,     I,
    semiringkernel, microkernel
  );
}

void dgsknn_ref
(
  int m, int n, int k, int r,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *D,             int *I
)
{
  gsknn_ref<double>
  (
    m, n, k, r,
    A, A2, amap,
    B, B2, bmap,
    D,     I
  );
}
