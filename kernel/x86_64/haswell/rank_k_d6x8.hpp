#include <stdio.h>
#include <hmlp_internal.hpp>


/** BLIS kernel prototype declaration */ 
BLIS_GEMM_KERNEL(bli_sgemm_asm_6x16,float);
BLIS_GEMM_KERNEL(bli_dgemm_asm_6x8,double);


struct rank_k_asm_s6x16
{
};


struct rank_k_asm_d6x8 
{
};
