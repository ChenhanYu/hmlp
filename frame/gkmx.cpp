#include <gkmx.hxx>

// Sandy-bridge
#include <rankk_int_d8x4.hxx>

//using namespace gkmx;

void dgemm_tn(
	int m, int n, int k,
	double *A, int lda,
	double *B, int ldb,
	double *C, int ldc
	)
{
  // Sandy-bridge
  rankk_int_d8x4 semiringkernel;
  rankk_int_d8x4 microkernel;
  gkmx
  <104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
  false,
	rankk_int_d8x4, rankk_int_d8x4,
	double, double, double, double>
	(
    HMLP_OP_T, HMLP_OP_N,
	  m, n, k,
	  A, lda,
	  B, ldb,
	  C, ldc,
	  semiringkernel,
	  microkernel
	);
}
