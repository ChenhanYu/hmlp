#include <gsks.hxx>
#include <rankk_int_d8x4.hxx>
#include <gaussian_d8x4.hpp>

void dgsks(
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

      rankk_int_d8x4 semiringkernel;
      gaussian_ref_d8x4 microkernel;
      gsks<104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
        true, false,
        rankk_int_d8x4,
        gaussian_ref_d8x4,
        double, double, double, double>
          ( kernel,
            m, n, k,
            u,     umap,
            A, A2, amap,
            B, B2, bmap,
            w,     wmap,
            semiringkernel, microkernel );

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

void dgsks_ref(
    ks_t *kernel,
    int m, int n, int k,
    double *u,             int *umap,
    double *A, double *A2, int *amap,
    double *B, double *B2, int *bmap,
    double *w,             int *wmap
    )
{
  gsks_ref<double>(
      kernel,
      m, n, k,
      u,     umap,
      A, A2, amap,
      B, B2, bmap,
      w,     wmap
      );
}
