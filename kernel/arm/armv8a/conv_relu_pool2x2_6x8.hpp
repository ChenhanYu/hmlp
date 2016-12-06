#include <stdio.h>
#include <hmlp_internal.hpp>

// #define DEBUG_MICRO 1

void conv_relu_pool2x2_d6x8
(
  dim_t               k,
  double*    restrict alpha,
  double*    restrict a,
  double*    restrict b,
  double*    restrict beta,
  double*    restrict c, inc_t rs_c, inc_t cs_c,
  aux_s<double, double, double, double> *aux
);



struct conv_relu_pool2x2_asm_d6x8 
{
  inline void operator()
  ( 
    int k, 
    double *a, 
    double *b, 
    double *c, int ldc, 
    aux_s<double, double, double, double> *aux 
  ) const 
  {
    double alpha = 1.0;
    double beta  = aux->pc ? 1.0 : 0.0;
    conv_relu_pool2x2_d6x8
    (
      k,
      &alpha,
      a,
      b,
      &beta,
      c, 1, ldc,
      aux
    );
  }; // end inline void operator()
}; // end struct rank_k_asm_d8x4
