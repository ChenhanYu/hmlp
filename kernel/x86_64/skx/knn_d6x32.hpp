#include <stdlib.h>
#include <stdio.h>
#include <hmlp_internal.hpp>

/** BLIS kernel prototype declaration */ 
BLIS_GEMM_KERNEL(bli_sgemm_opt_12x32_l2,float);
BLIS_GEMM_KERNEL(bli_dgemm_opt_6x32_l2,double);

struct knn_int_s12x32
{
  const size_t mr         = 12;
  const size_t nr         = 32;
  const size_t pack_mr    = 12;
  const size_t pack_nr    = 32;
  const size_t align_size = 64;
  const bool   row_major  = true;

  inline void operator()
  (
    int k,
    int r,
    float *a, float *aa,
    float *b, float *bb, int *bmap,
    float *c,
    float *Keys, int *Values, int ldr,
    aux_s<float, float, float, float> *aux
  ) const
  {
    float ctmp[ mr * nr ];
    float alpha = 1.0;
    /** If this is not the first kc iteration then beta = 1.0 */
    float beta = aux->pc ? 1.0 : 0.0;
    /** If pc, then c == NULL. We use ctmp. */
    if ( !pc ) c = ctmp;

    /** invoke blis kernel */
    bli_sgemm_opt_12x32_l2
    (
      k,
      &alpha,
      a,
      b,
      &beta,
      c, rs_c, cs_c,
      aux
    );

    for ( size_t i = 0; i < aux->ib; i ++ )
    {
      for ( size_t j = 0; j < aux->jb; j ++ )
      {
        c[ i * nr + j ] = aa[ i ] - 2.0 * c[ i * nr + j ] + bb[ j ];
        c[ i * nr + j ] = std::max( c[ i * nr + j ], 0 );
      }
      hmlp::heap_select<float>
      ( 
        aux->jb, r, 
        c      + i * nr, bmap, 
        Keys   + i * ldr, 
        Values + i * ldr 
      );
    }

  };

}; /** end struct knn_int_s12x32 */

struct knn_int_d6x32
{
  const size_t mr         =  8;
  const size_t nr         =  4;
  const size_t pack_mr    =  8;
  const size_t pack_nr    =  4;
  const size_t align_size = 32;
  const bool   row_major  = true;

  inline void operator()
  (
    int k,
    int r,
    double *a, double *aa,
    double *b, double *bb, int *bmap,
    double *c,
    double *Keys, int *Values, int ldr,
    aux_s<double, double, double, double> *aux
  ) const
  {
  };
}; 
