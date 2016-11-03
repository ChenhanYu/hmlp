// GKMX templates
#include <gkmx.hpp>

// Sandy-bridge
#include <rank_k_d8x4.hpp>

using namespace hmlp::gkmx;


template <typename T>
struct identity 
{
  //inline T operator()( const T& x, int i, int j, int b ) const 
  inline T operator()( const T& x ) const 
  {
    return x; 
  }
  T** A2;
  T** B2;
};


void gkmx_dfma
(
  hmlpOperation_t transA, hmlpOperation_t transB,
	int m, int n, int k,
	double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
)
{
  // Sandy-bridge
  //rank_k_asm_d8x4 semiringkernel;
  //rank_k_asm_d8x4 microkernel;

  rank_k_int_d8x4 semiringkernel;
  rank_k_int_d8x4 microkernel;

  gkmx
  <104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
  false,
  //rank_k_asm_d8x4, rank_k_asm_d8x4,
  rank_k_int_d8x4, rank_k_int_d8x4,
  double, double, double, double>
  (
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    semiringkernel,
    microkernel
  );
};


void gkmx_dfma_simple
(
  hmlpOperation_t transA, hmlpOperation_t transB,
	int m, int n, int k,
	double *A, int lda,
  double *B, int ldb,
  double *C, int ldc
)
{
  std::plus<double> op1;
  std::multiplies<double> op2;

  identity<double> opkernel;

  double initV = 0.0;

  gkmm
  <104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
  false>
  (
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    opkernel, op1, op2, initV
  );
}
