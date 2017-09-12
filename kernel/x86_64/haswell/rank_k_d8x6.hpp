#include <stdio.h>
#include <hmlp_internal.hpp>


/** BLIS kernel prototype declaration */ 
BLIS_GEMM_KERNEL(bli_sgemm_asm_16x6,float);
BLIS_GEMM_KERNEL(bli_dgemm_asm_8x6,double);


struct rank_k_asm_s16x6 
{

  inline STRA_OPERATOR(float) const
  {
    printf( "no STRA_OPERATOR implementation\n" );
    exit( 1 );
  };

  inline GEMM_OPERATOR(float) const
  {
    float alpha = 1.0;
    /** if this is the first kc iteration then beta = 1.0 */
    float beta = aux->pc ? 1.0 : 0.0;
    /** invoke blis kernel */
    bli_sgemm_asm_16x6
    (
      k,
      &alpha,
      a,
      b,
      &beta,
      c, 1, ldc,
      aux
    );
  };

}; /** end struct rank_k_asm_s_16x6 */


struct rank_k_asm_d8x6 
{

  inline STRA_OPERATOR(double) const
  {
    printf( "no STRA_OPERATOR implementation\n" );
    exit( 1 );
  };

  inline GEMM_OPERATOR(double) const
  {
    double alpha = 1.0;
    /** if this is the first kc iteration then beta = 1.0 */
    double beta = aux->pc ? 1.0 : 0.0;
    /** invoke blis kernel */
    bli_dgemm_asm_8x6
    (
      k,
      &alpha,
      a,
      b,
      &beta,
      c, 1, ldc,
      aux
    );
  };

}; /** ebd struct rank_k_asm_d8x6 */
