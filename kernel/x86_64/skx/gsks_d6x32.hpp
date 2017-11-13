#include <stdio.h>
#include <math.h>

/** HMLP */
#include <hmlp.h>
#include <hmlp_internal.hpp>

/** BLIS kernel prototype declaration */ 
BLIS_GEMM_KERNEL(bli_sgemm_opt_12x32_l2,float);
BLIS_GEMM_KERNEL(bli_dgemm_opt_6x32_l2,double);


struct gsks_gaussian_int_s12x32
{
  inline GSKS_OPERATOR(float) const
  {
    printf( "not implemented yet\n" );
    exit( 1 );
  };
};


struct gsks_gaussian_int_d6x32
{
  const size_t mr         = 32;
  const size_t nr         =  6;
  const size_t pack_mr    = 32;
  const size_t pack_nr    =  6;
  const size_t align_size = 64;
  const bool   row_major  = false;


  //inline void operator()
  //(
  //  kernel_s<double> *ker,
  //  int k,
  //  int rhs,
  //  double *u,
  //  double *a, double *aa, 
  //  double *b, double *bb,
  //  double *w,
  //  double *c, int ldc,
  //  aux_s<double, double, double, double> *aux 
  //) const 

  inline GSKS_OPERATOR(double) const
  {
    double ctmp[ mr * nr ];
    double alpha = 1.0;
    /** If this is not the first kc iteration then beta = 1.0 */
    double beta = aux->pc ? 1.0 : 0.0;
    /** If pc, then c != NULL. We copy c to ctmp. */
    if ( aux->pc ) 
    {
      for ( size_t j = 0; j < aux->jb; j ++ )
        for ( size_t i = 0; i < aux->ib; i ++ )
          //ctmp[ j * mr + i ] = c[ ( aux->j + j ) * ldc + ( aux->i + i ) ];
          ctmp[ j * mr + i ] = c[ j * ldc + i ];
    }

    /** invoke blis kernel */
    bli_dgemm_opt_6x32_l2
    (
      k,
      &alpha,
      b,
      a,
      &beta,
      ctmp, mr, 1,
      aux
    );

    for ( size_t j = 0; j < aux->jb; j ++ )
    {
      for ( size_t i = 0; i < aux->ib; i ++ )
      {
        ctmp[ j * mr + i ] *= -2.0;
        ctmp[ j * mr + i ] += aa[ i ] + bb[ j ];
        ctmp[ j * mr + i ] = std::max( ctmp[ j * mr + i ], (double)0 );
        /**
         *  Accumulate K * w to u 
         */ 
        u[ i ] += std::exp( ker->scal * ctmp[ j * mr + i ] ) * w[ j ];
      }
    }

  }; /** end inline void operator */

}; /** end struct gaussian_int_d6x32 */
