#include <stdio.h>
#include <math.h>

#include <hmlp.h>
#include <hmlp_internal.hpp>

/** self-defined vector type */
#include <avx_type.h> 


#define MM256_ADD8x6(beta,regis);              \
  regis.v = _mm256_broadcast_sd( &beta );      \
  c03_0.v = _mm256_add_pd( regis.v, c03_0.v ); \
  c03_1.v = _mm256_add_pd( regis.v, c03_1.v ); \
  c03_2.v = _mm256_add_pd( regis.v, c03_2.v ); \
  c03_3.v = _mm256_add_pd( regis.v, c03_3.v ); \
  c03_4.v = _mm256_add_pd( regis.v, c03_4.v ); \
  c03_5.v = _mm256_add_pd( regis.v, c03_5.v ); \
  c47_0.v = _mm256_add_pd( regis.v, c47_0.v ); \
  c47_1.v = _mm256_add_pd( regis.v, c47_1.v ); \
  c47_2.v = _mm256_add_pd( regis.v, c47_2.v ); \
  c47_3.v = _mm256_add_pd( regis.v, c47_3.v ); \
  c47_4.v = _mm256_add_pd( regis.v, c47_4.v ); \
  c47_5.v = _mm256_add_pd( regis.v, c47_5.v ); \

#define MM256_MUL8x6(alpha,regis);             \
  regis.v = _mm256_broadcast_sd( &alpha );     \
  c03_0.v = _mm256_mul_pd( regis.v, c03_0.v ); \
  c03_1.v = _mm256_mul_pd( regis.v, c03_1.v ); \
  c03_2.v = _mm256_mul_pd( regis.v, c03_2.v ); \
  c03_3.v = _mm256_mul_pd( regis.v, c03_3.v ); \
  c03_4.v = _mm256_mul_pd( regis.v, c03_4.v ); \
  c03_5.v = _mm256_mul_pd( regis.v, c03_5.v ); \
  c47_0.v = _mm256_mul_pd( regis.v, c47_0.v ); \
  c47_1.v = _mm256_mul_pd( regis.v, c47_1.v ); \
  c47_2.v = _mm256_mul_pd( regis.v, c47_2.v ); \
  c47_3.v = _mm256_mul_pd( regis.v, c47_3.v ); \
  c47_4.v = _mm256_mul_pd( regis.v, c47_4.v ); \
  c47_5.v = _mm256_mul_pd( regis.v, c47_5.v ); \

#define MM256_SQUARE8x6();                     \
  c03_0.v = _mm256_mul_pd( c03_0.v, c03_0.v ); \
  c03_1.v = _mm256_mul_pd( c03_1.v, c03_1.v ); \
  c03_2.v = _mm256_mul_pd( c03_2.v, c03_2.v ); \
  c03_3.v = _mm256_mul_pd( c03_3.v, c03_3.v ); \
  c03_4.v = _mm256_mul_pd( c03_4.v, c03_4.v ); \
  c03_5.v = _mm256_mul_pd( c03_5.v, c03_5.v ); \
  c47_0.v = _mm256_mul_pd( c47_0.v, c47_0.v ); \
  c47_1.v = _mm256_mul_pd( c47_1.v, c47_1.v ); \
  c47_2.v = _mm256_mul_pd( c47_2.v, c47_2.v ); \
  c47_3.v = _mm256_mul_pd( c47_3.v, c47_3.v ); \
  c47_4.v = _mm256_mul_pd( c47_4.v, c47_4.v ); \
  c47_5.v = _mm256_mul_pd( c47_5.v, c47_5.v ); \

#define MM256_EXP8x6();               \
  c03_0.v = _mm256_exp_pd( c03_0.v ); \
  c03_1.v = _mm256_exp_pd( c03_1.v ); \
  c03_2.v = _mm256_exp_pd( c03_2.v ); \
  c03_3.v = _mm256_exp_pd( c03_3.v ); \
  c03_4.v = _mm256_exp_pd( c03_4.v ); \
  c03_5.v = _mm256_exp_pd( c03_5.v ); \
  c47_0.v = _mm256_exp_pd( c47_0.v ); \
  c47_1.v = _mm256_exp_pd( c47_1.v ); \
  c47_2.v = _mm256_exp_pd( c47_2.v ); \
  c47_3.v = _mm256_exp_pd( c47_3.v ); \
  c47_4.v = _mm256_exp_pd( c47_4.v ); \
  c47_5.v = _mm256_exp_pd( c47_5.v ); \

#define MM256_POW8x6(power,regis);             \
  regis.v = _mm256_broadcast_sd( &power );     \
  c03_0.v = _mm256_pow_pd( c03_0.v, regis.v ); \
  c03_1.v = _mm256_pow_pd( c03_1.v, regis.v ); \
  c03_2.v = _mm256_pow_pd( c03_2.v, regis.v ); \
  c03_3.v = _mm256_pow_pd( c03_3.v, regis.v ); \
  c03_4.v = _mm256_pow_pd( c03_4.v, regis.v ); \
  c03_5.v = _mm256_pow_pd( c03_5.v, regis.v ); \
  c47_0.v = _mm256_pow_pd( c47_0.v, regis.v ); \
  c47_1.v = _mm256_pow_pd( c47_1.v, regis.v ); \
  c47_2.v = _mm256_pow_pd( c47_2.v, regis.v ); \
  c47_3.v = _mm256_pow_pd( c47_3.v, regis.v ); \
  c47_4.v = _mm256_pow_pd( c47_4.v, regis.v ); \
  c47_5.v = _mm256_pow_pd( c47_5.v, regis.v ); \
  



struct gsks_gaussian_int_s16x6
{

  inline GSKS_OPERATOR(float) const
  {
    printf( "no GSKS_OPERATOR implementation\n" );
    exit( 1 );
  };

}; /** struct gsks_gaussian_int_s16x6 */



struct gsks_gaussian_int_d8x6
{

  inline GSKS_OPERATOR(double) const
  {
    double alpha = ker->scal;

    /** rank-k update */
    #include <component/rank_k_int_d8x6.hpp>

    /** compute a^2 + b^2 - 2ab */
    #include <component/sq2nrm_int_d8x6.hpp>

    /** scale before the kernel evaluation */
    MM256_MUL8x6(alpha,a03);

    /** prefetch u, w */
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

    /** c = exp( c ) */
    MM256_EXP8x6();

    /** multiple rhs weighted sum */
    #include <component/weighted_sum_int_d8x6.hpp>
  };

}; /** end struct gsks_gaussian_int_d8x6 */



struct gsks_polynomial_int_s16x6
{

  inline GSKS_OPERATOR(float) const
  {
    printf( "no GSKS_OPERATOR implementation\n" );
    exit( 1 );
  };

}; /** struct gsks_polynomial_int_s16x6 */


struct gsks_polynomial_int_d8x6
{

  inline GSKS_OPERATOR(double) const
  {
    double powe  = ker->powe;
    double scal  = ker->scal;
    double cons  = ker->cons;

    /** rank-k update */
    #include <component/rank_k_int_d8x6.hpp>

    /** prefetch u, w */
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w ) );

    /** c = c * scal */
    MM256_MUL8x6(scal,a03);

    /** c = c + cons */
    MM256_ADD8x6(cons,a03);

    /** c = pow( c, powe ) */
    if ( powe == 2.0 ) 
    {
      MM256_SQUARE8x6();
    }
    else if ( powe == 4.0 )
    {
      MM256_SQUARE8x6();
      MM256_SQUARE8x6();
    }
    else
    {
      MM256_POW8x6(powe,a03);
    }

    /** multiple rhs weighted sum */
    #include <component/weighted_sum_int_d8x6.hpp>
  };

}; /** struct gsks_polynomial_int_d8x6 */







