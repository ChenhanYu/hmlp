#include <math.h>
#include <immintrin.h> // AVX
#include <ks.h>
#include <gsks_internal.h>
#include <avx_type.h>

void laplace_int_s16x6(
    int    k,
    int    rhs,
    //float  *h,
    float  *u,
    float  *aa,
    float  *a,
    float  *bb,
    float  *b,
    float  *w,
    float  *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  printf( "laplace_int_s16x6 not yet implemented.\n" );
}

void laplace_int_d8x6(
    int    k,
    int    rhs,
    //double *h,
    double *u,
    double *aa,
    double *a,
    double *bb,
    double *b,
    double *w,
    double *c,
    ks_t   *ker,
    aux_t  *aux
    )
{
  printf( "laplace_int_d8x6 not yet implemented.\n" );
}
