#include <gkmx.hxx>
#include <rankk_int_d8x4.hxx>

using namespace gkmx;

void dgemm_tn(
	int m, int n, int k,
	double *A, int lda, int *amap,
	double *B, int ldb, int *bmap,
	double *C, int ldc
	)
{
  rankk_int_d8x4 semiringkernel;
  rankk_int_d8x4 microkernel;
  //rankk_ref_d8x4 semiringkernel;
  //rankk_ref_d8x4 microkernel;

  // Type TA, TB, TC, TV
  gkmx_new<104, 4096, 256, 8, 4, 104, 4096, 8, 4, 32,
	rankk_int_d8x4,
	rankk_int_d8x4,
	//rankk_ref_d8x4,
	//rankk_ref_d8x4,
	double, double, double, double>
	  (
	   m, n, k,
	   A, lda, amap,
	   B, ldb, bmap,
	   C, ldc,
	   semiringkernel,
	   microkernel
	  );
}
