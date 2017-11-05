#include <stdio.h>
#include <hmlp_internal.hpp>
#include <packing.hpp>

/** BLIS kernel prototype declaration */ 
BLIS_GEMM_KERNEL(bli_sgemm_asm_16x6,float);
BLIS_GEMM_KERNEL(bli_dgemm_asm_8x6,double);


struct rank_k_asm_s16x6 
{
  const static size_t mr         = 16;
  const static size_t nr         =  6;
  const static size_t pack_mr    = 16;
  const static size_t pack_nr    =  6;
  const static size_t align_size = 32;
  const static bool   row_major  = false;

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
      c, rs_c, cs_c,
      aux
    );
  };

}; /** end struct rank_k_asm_s_16x6 */


struct rank_k_asm_d8x6 
{
  const static size_t mr         =  8;
  const static size_t nr         =  6;
  const static size_t pack_mr    =  8;
  const static size_t pack_nr    =  6;
  const static size_t align_size = 32;
  const static bool   row_major  = false;

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
      c, rs_c, cs_c,
      aux
    );
  };

  template<typename TC>
  inline void operator()                                                
  (                                                              
    dim_t k,                                                     
    double *a,                                                    
    double *b,                                                    
    TC     *c,                                                    
    double *v, inc_t rs_v, inc_t cs_v,                            
    aux_s<double, double, TC, double> *aux                       
  )
  {
    double alpha = 1.0;
    /** if this is the first kc iteration then beta = 1.0 */
    double beta = aux->pc ? 1.0 : 0.0;

    /** allocate temporary buffer */
    double vtmp[ mr * nr ];

    if ( !is_same<TC, hmlp::MatrixLike<pack_mr, double, double>>::value )
    {
      if ( aux->pc )
      {
        for ( size_t j = 0; j < aux->jb; j ++ )
          for ( size_t i = 0; i < aux->ib; i ++ )
            vtmp[ j * mr + i ] = v[ j * cs_v + i * rs_v ];
      }

      v = vtmp;
      rs_v = 1;
      cs_v = mr;
    }

    /** invoke blis kernel */
    bli_dgemm_asm_8x6
    (
      k,
      &alpha,
      a,
      b,
      &beta,
      v, rs_v, cs_v,
      reinterpret_cast<aux_s<double, double, double, double>*>( aux )
    );

    /**
     *  If TC is not MatrixLike<PACK_MR,double,double>, we treat this
     *  the same as the edge case.
     */ 
    if ( !is_same<TC, hmlp::MatrixLike<pack_mr, double, double>>::value ||
         aux->ib != mr || aux->jb != nr )
    {
      //printf( "bug %d %d %d %d %d %d\n", aux->m, aux->i, aux->ib, aux->n, aux->j, aux->jb );
      c->Unpack( aux->m, aux->i, aux->ib, aux->n, aux->j, aux->jb, v );
    }

  };

}; /** ebd struct rank_k_asm_d8x6 */
