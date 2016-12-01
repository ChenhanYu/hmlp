// GKMX templates
#include <gkmx.hpp>

// Armv8a
#include <rank_k_d6x8.hpp>

using namespace hmlp::gkmx;

template<typename T>
struct identity 
{
  inline T operator()( const T& x, int i, int j, int b ) const 
  //inline T operator()( const T& x ) const 
  {
    return x; 
  }
  T** A2;
  T** B2;
};

template<typename TC, typename TV>
struct downcast
{
  //inline TC operator()( const TV& x ) const 
  inline TC operator()( const TV& x, int i, int j, int b ) const 
  {
    return (TC)x;
  }
  TV** A2;
  TV** B2;
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
  // Armv8a
  rank_k_asm_d6x8 semiringkernel;
  rank_k_asm_d6x8 microkernel;

  gkmx
  <120, 3072, 240, 6, 8, 120, 3072, 6, 8, 16,
  false, true,
  rank_k_asm_d6x8, rank_k_asm_d6x8,
  double, double, double, double>
  (
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    0, // batchId
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
  false, true>
  (
    transA, transB,
    m, n, k,
    A, lda,
    B, ldb,
    C, ldc,
    0, // batchId
    opkernel, op1, op2, initV
  );
};

void gkmx_mixfma_simple
(
  hmlpOperation_t transA, hmlpOperation_t transB,
	int m, int n, int k,
	double *A, int lda,
  double *B, int ldb,
  float  *C, int ldc
)
{
  std::plus<double> op1;
  std::multiplies<double> op2;

  //identity<double> opkernel;
  downcast<float, double> opkernel;

  double initV = 0.0;

  //gkmm
  //<104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
  //false>
  //(
  //  transA, transB,
  //  m, n, k,
  //  A, lda,
  //  B, ldb,
  //  C, ldc,
  //  opkernel, op1, op2, initV
  //);
};

