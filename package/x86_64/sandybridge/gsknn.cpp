// GSKNN templates
#include <gsknn.hpp>

// Sandy-bridge
#include <rank_k_d8x4.hpp>
#include <rnn_r_int_d8x4_row.hpp>

// Haswell
// #include <rank_k_asm_d8x6.hpp>
// #include <gaussian_d8x6.hpp>

// Knights Landing
// #include <rank_k_int_d24x8.hpp>
// #include <gaussian_int_d24x8.hpp>

using namespace hmlp::gsknn;

void dgsknn
(
  ks_t *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
)
{
  switch ( kernel->type )
  {
    case KS_GAUSSIAN:

      // Sandy-bridge
      //rank_k_int_d8x4 semiringkernel;
      rank_k_asm_d8x4 semiringkernel;
      //gaussian_ref_d8x4 microkernel;
      rnn_r_int_d8x4_row microkernel;
      gsknn<
        104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
        true, false, false,
        //rank_k_int_d8x4,
        rank_k_asm_d8x4,
        rnn_r_int_d8x4_row,
        double, double, double, double>
      (
        kernel,
        m, n, k,
        u,     umap,
        A, A2, amap,
        B, B2, bmap,
        w,     wmap,
        semiringkernel, microkernel
      );

      // Haswell
      // rank_k_asm_d8x6 semiringkernel;
      // gaussian_int_d8x6 microkernel;
      // gsknn<
      //   // MC, NC, KC, MR, NR, PACK_MC, PACK_NC, PACK_MR, PACK_NR, SIMD_SIZE
      //   72, 960, 256, 8, 6, 72, 960, 8, 6, 32,
      //   // USE_L2NORM, USE_VAR_BANDWIDTH, USE_STRASSEN
      //   true, false, false,
      //   rank_k_asm_d8x6, gaussian_int_d8x6,
      //   double, double, double, double>
      // (
      //   kernel,
      //   m, n, k,
      //   u,     umap,
      //   A, A2, amap,
      //   B, B2, bmap,
      //   w,     wmap,
      //   semiringkernel, microkernel
      // );

      // Knights Landing
      // rank_k_int_d24x8 semiringkernel;
      // gaussian_int_d24x8 microkernel;
      // gsknn<
      //   120, 14400, 336, 24, 8, 120, 14400, 24, 8, 64,
      //   true, false, false,
      //   rank_k_int_d24x8, gaussian_int_d24x8,
      //   double, double, double, double>
      // (
      //   kernel,
      //   m, n, k,
      //   u,     umap,
      //   A, A2, amap,
      //   B, B2, bmap,
      //   w,     wmap,
      //   semiringkernel, microkernel
      // );

      break;
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      break;
    case KS_POLYNOMIAL:
      break;
    case KS_LAPLACE:
      break;
    case KS_TANH:
      break;
    case KS_QUARTIC:
      break;
    case KS_MULTIQUADRATIC:
      break;
    case KS_EPANECHNIKOV:
      break;
    default:
      exit( 1 );
  }
}

void dgsknn_ref
(
  ks_t *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
)
{
  gsknn_ref<double>
  (
    kernel,
    m, n, k,
    u,     umap,
    A, A2, amap,
    B, B2, bmap,
    w,     wmap
  );
}
