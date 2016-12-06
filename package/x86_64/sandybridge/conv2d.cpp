// GKMX templates
#include <conv2d.hpp>

// Sandy-bridge
#include <rank_k_d8x4.hpp>

using namespace hmlp::cnn;

void dconv2d
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  double *B,
  int w1, int h1, int d1,
	double *A,
	double *C
)
{
  // Sandy-bridge
  rank_k_asm_d8x4 semiringkernel;
  rank_k_asm_d8x4 microkernel;

  conv2d
  <104, 1024, 256, 8, 4, 104, 1024, 8, 4, 32,
  false,
	rank_k_asm_d8x4, rank_k_asm_d8x4,
	double, double, double, double>
	(
    w0, h0, d0, s, p, batchSize,
    B,
    w1, h1, d1,
    A,
    C,
	  semiringkernel,
	  microkernel
	);
};

void dconv2d_ref
(
  int w0, int h0, int d0, int s, int p, int batchSize,
  double *B,
  int w1, int h1, int d1,
	double *A,
	double *C
)
{
  conv2d_ref
  (
    w0, h0, d0, s, p, batchSize,
    B,
    w1, h1, d1,
    A,
    C
 );
};
