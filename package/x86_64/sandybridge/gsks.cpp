// GSKS templates
#include <gsks.hpp>

// Sandy-bridge
#include <rank_k_d8x4.hpp>
#include <gaussian_d8x4.hpp>
#include <variable_bandwidth_gaussian_d8x4.hpp>

// Haswell
// #include <rank_k_asm_d8x6.hpp>
// #include <gaussian_d8x6.hpp>

// Knights Landing
// #include <rank_k_int_d24x8.hpp>
// #include <gaussian_int_d24x8.hpp>

using namespace hmlp::gsks;

void dgsks
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
      {
        rank_k_asm_d8x4 semiringkernel;
        gaussian_int_d8x4 fusedkernel;

        gsks<
          104, 
          4096, 
          256, 
          8, 
          4, 
          104, 
          4096, 
          8, 
          4, 
          32,
          true,  // USE_L2NORM
          false, // USE_VAR_BANDWIDTH
          false, // USE_STRASSEN
          rank_k_asm_d8x4,
          gaussian_int_d8x4,
          double, double, double, double>
            ( 
             kernel,
             m, n, k,
             u,     umap,
             A, A2, amap,
             B, B2, bmap,
             w,     wmap,
             semiringkernel, fusedkernel 
            );

        // Haswell
        // rank_k_asm_d8x6 semiringkernel;
        // gaussian_int_d8x6 microkernel;
        // gsks<
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
        // gsks<
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
      }
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      {
        rank_k_asm_d8x4 semiringkernel;
        variable_bandwidth_gaussian_int_d8x4 fusedkernel;

        gsks<
          104, 
          4096, 
          256, 
          8, 
          4, 
          104, 
          4096, 
          8, 
          4, 
          32,
          true,  // USE_L2NORM
          true,  // USE_VAR_BANDWIDTH
          false, // USE_STRASSEN
          rank_k_asm_d8x4, variable_bandwidth_gaussian_int_d8x4,
          double, double, double, double>
        ( 
          kernel,
          m, n, k,
          u,     umap,
          A, A2, amap,
          B, B2, bmap,
          w,     wmap,
          semiringkernel, fusedkernel 
        );

        break;
      }
    case KS_POLYNOMIAL:
      {
        rank_k_asm_d8x4 semiringkernel;


        break;
      }
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

void dgsks_ref
(
  ks_t *kernel,
  int m, int n, int k,
  double *u,             int *umap,
  double *A, double *A2, int *amap,
  double *B, double *B2, int *bmap,
  double *w,             int *wmap
)
{
  gsks_ref<double>
  (
    kernel,
    m, n, k,
    u,     umap,
    A, A2, amap,
    B, B2, bmap,
    w,     wmap
  );
}
