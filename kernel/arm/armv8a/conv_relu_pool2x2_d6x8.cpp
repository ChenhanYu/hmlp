    /* 

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


*/

//#include "blis.h"
#include <hmlp_internal.hpp>

/*
   o 4x4 Single precision micro-kernel fully functional.
   o Runnable on ARMv8, compiled with aarch64 GCC.
   o Use it together with the armv8 BLIS configuration.
   o Tested on Juno board. Around 7.3 GFLOPS @ 1.1 GHz. 

   December 2014.
 
 * UPDATE NOVEMBER 2015
 * Micro-kernel changed to 8x12
 * Tested on Juno Board. Around  8.1 GFLOPS, 1 x A57 core  @ 1.1 GHz.
 * Tested on Juno Board. Around 15.9 GFLOPS, 2 x A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around  3.1 GFLOPS, 1 x A53 core  @ 850 MHz. 
 * Tested on Juno board. Around 12   GFLOPS, 4 x A53 cores @ 850 MHz.
*/
void conv_relu_pool2x2_s8x12(
                        dim_t              k,
                        float*    restrict alpha,
                        float*    restrict a,
                        float*    restrict b,
                        float*    restrict beta,
                        float*    restrict c, inc_t rs_c, inc_t cs_c,
                        //auxinfo_t*         data
                        aux_s<float, float, float, float> *aux
                      )
{
	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );
	void* a_next = aux->a_next;
	void* b_next = aux->b_next;

	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;

__asm__ volatile 
(
"                                            \n\t"
"                                            \n\t"
" ldr x0,%[aaddr]                            \n\t" // Load address of A. 
" ldr x1,%[baddr]                            \n\t" // Load address of B.
" ldr x2,%[caddr]                            \n\t" // Load address of C.
"                                            \n\t"
" ldr x3,%[a_next]                           \n\t" // Pointer to next block of A.
" ldr x4,%[b_next]                           \n\t" // Pointer to next pointer of B.
"                                            \n\t"
" ldr x5,%[k_iter]                           \n\t" // Number of unrolled iterations (k_iter).
" ldr x6,%[k_left]                           \n\t" // Number of remaining iterations (k_left).
"                                            \n\t" 
" ldr x7,%[alpha]                            \n\t" // Alpha address.      
" ldr x8,%[beta]                             \n\t" // Beta address.     
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c.
" lsl x10,x9,#2                              \n\t" // cs_c * sizeof(float) -- AUX.
"                                            \n\t" 
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
" lsl x14,x13,#2                             \n\t" // rs_c * sizeof(float).
"                                            \n\t"
" add x16,x2,x10                             \n\t" //Load address Column 1 of C
" add x17,x16,x10                            \n\t" //Load address Column 2 of C
" add x18,x17,x10                            \n\t" //Load address Column 3 of C
" add x19,x18,x10                            \n\t" //Load address Column 4 of C
" add x20,x19,x10                            \n\t" //Load address Column 5 of C
" add x21,x20,x10                            \n\t" //Load address Column 6 of C
" add x22,x21,x10                            \n\t" //Load address Column 7 of C
" add x23,x22,x10                            \n\t" //Load address Column 8 of C
" add x24,x23,x10                            \n\t" //Load address Column 9 of C
" add x25,x24,x10                            \n\t" //Load address Column 10 of C
" add x26,x25,x10                            \n\t" //Load address Column 11 of C
"                                            \n\t"
" ldr q0, [x0]                               \n\t"
" ldr q1, [x0, #16]                          \n\t" // Load a
"                                            \n\t"
" ldr q2, [x1]                               \n\t" // Load b
" ldr q3, [x1, #16]                          \n\t"
" ldr q4, [x1, #32]                          \n\t"
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x16]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x17]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x18]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x19]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x24]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x25]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x26]                       \n\t" // Prefetch c.
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t" // Vector for accummulating column 0
" prfm    PLDL1KEEP, [x1, #192]              \n\t" 
" dup  v9.4s, wzr                            \n\t" // Vector for accummulating column 0
" prfm    PLDL1KEEP, [x1, #256]              \n\t"
" dup  v10.4s, wzr                           \n\t" // Vector for accummulating column 1
" prfm    PLDL1KEEP, [x1, #320]              \n\t"
" dup  v11.4s, wzr                           \n\t" // Vector for accummulating column 1
" dup  v12.4s, wzr                           \n\t" // Vector for accummulating column 2 
" dup  v13.4s, wzr                           \n\t" // Vector for accummulating column 2
"                                            \n\t"
" dup  v14.4s, wzr                           \n\t" // Vector for accummulating column 3
" prfm    PLDL1KEEP, [x0, #128]              \n\t"
" dup  v15.4s, wzr                           \n\t" // Vector for accummulating column 3
" prfm    PLDL1KEEP, [x0, #192]              \n\t"
" dup  v16.4s, wzr                           \n\t" // Vector for accummulating column 4
" dup  v17.4s, wzr                           \n\t" // Vector for accummulating column 4
" dup  v18.4s, wzr                           \n\t" // Vector for accummulating column 5 
" dup  v19.4s, wzr                           \n\t" // Vector for accummulating column 5
"                                            \n\t"
" dup  v20.4s, wzr                           \n\t" // Vector for accummulating column 6 
" dup  v21.4s, wzr                           \n\t" // Vector for accummulating column 6
" dup  v22.4s, wzr                           \n\t" // Vector for accummulating column 7
" dup  v23.4s, wzr                           \n\t" // Vector for accummulating column 7
" dup  v24.4s, wzr                           \n\t" // Vector for accummulating column 8 
" dup  v25.4s, wzr                           \n\t" // Vector for accummulating column 8
"                                            \n\t"
" dup  v26.4s, wzr                           \n\t" // Vector for accummulating column 9 
" dup  v27.4s, wzr                           \n\t" // Vector for accummulating column 9
" dup  v28.4s, wzr                           \n\t" // Vector for accummulating column 10
" dup  v29.4s, wzr                           \n\t" // Vector for accummulating column 10
" dup  v30.4s, wzr                           \n\t" // Vector for accummulating column 11 
" dup  v31.4s, wzr                           \n\t" // Vector for accummulating column 11
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .SCONSIDERKLEFT                        \n\t"
"                                            \n\t"
"add x0, x0, #32                             \n\t" //update address of A
"add x1, x1, #48                             \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .SLASTITER                             \n\t" // (as loop is do-while-like).
"                                            \n\t"
" .SLOOPKITER:                               \n\t" // Body of the k_iter loop.
"                                            \n\t"
" ldr q5, [x0]                               \n\t"
" fmla v8.4s, v0.4s,v2.s[0]                  \n\t" // Accummulate.
" fmla v9.4s, v1.4s,v2.s[0]                  \n\t" // Accummulate.
" ldr q6, [x0, #16]                          \n\t"
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
" ldr q2, [x1]                               \n\t"
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #336]              \n\t" 
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #400]              \n\t" 
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #464]              \n\t" 
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q3, [x1, #16]                          \n\t"
"                                            \n\t"
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q4, [x1, #32]                          \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
" ldr q0, [x0, #32]                          \n\t"
" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
" ldr q1, [x0, #48]                          \n\t"
" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
" ldr q2, [x1, #48]                          \n\t"
"                                            \n\t"
" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x0, #224]              \n\t"
" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x0, #288]              \n\t"
" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q3, [x1, #64]                          \n\t"
"                                            \n\t"
" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q4, [x1, #80]                          \n\t"
"                                            \n\t" //End It 2
"                                            \n\t"
" ldr q5, [x0, #64]                          \n\t"
" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
" ldr q6, [x0, #80]                          \n\t"
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
" ldr q2, [x1, #96]                          \n\t"
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q3, [x1, #112]                         \n\t"
"                                            \n\t"
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q4, [x1, #128]                         \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" ldr q0, [x0, #96]                          \n\t"
" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
" ldr q1, [x0, #112]                         \n\t"
" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
" ldr q2, [x1, #144]                         \n\t"
"                                            \n\t"
" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q3, [x1, #160]                         \n\t"
"                                            \n\t"
" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q4, [x1, #176]                         \n\t"
" add x1, x1, #192                           \n\t"
" add x0, x0, #128                           \n\t"
"                                            \n\t" //End It 4
" sub x5,x5,1                                \n\t" // i-=1.
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .SLOOPKITER                            \n\t"
"                                            \n\t" 
" .SLASTITER:                                \n\t" // Last iteration of k_iter loop.
"                                            \n\t" 
"                                            \n\t"
" ldr q5, [x0]                               \n\t"
" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
" ldr q6, [x0, #16]                          \n\t"
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
" ldr q2, [x1]                               \n\t"
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q3, [x1, #16]                          \n\t"
"                                            \n\t"
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q4, [x1, #32]                          \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
" ldr q0, [x0, #32]                          \n\t"
" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
" ldr q1, [x0, #48]                          \n\t"
" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
" ldr q2, [x1, #48]                          \n\t"
"                                            \n\t"
" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q3, [x1, #64]                          \n\t"
"                                            \n\t"
" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q4, [x1, #80]                          \n\t"
"                                            \n\t" //End It 2
"                                            \n\t"
" ldr q5, [x0, #64]                          \n\t"
" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
" ldr q6, [x0, #80]                          \n\t"
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
" ldr q2, [x1, #96]                          \n\t"
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q3, [x1, #112]                         \n\t"
"                                            \n\t"
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
" ldr q4, [x1, #128]                         \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
" add x1, x1, #144                           \n\t"
" add x0, x0, #96                            \n\t"
"                                            \n\t" //End It 4
"                                            \n\t"
" .SCONSIDERKLEFT:                           \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .SPOSTACCUM                            \n\t" // else, we enter the k_left loop.
"                                            \n\t"
" .SLOOPKLEFT:                               \n\t" // Body of the left iterations
"                                            \n\t"
" ldr q0, [x0],#16                           \n\t"
" ldr q1, [x0],#16                           \n\t" // Load a
"                                            \n\t"
" ldr q2, [x1],#16                           \n\t" // Load b
" ldr q3, [x1],#16                           \n\t"
" ldr q4, [x1],#16                           \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t" // i = i-1.
"                                            \n\t"
" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .SLOOPKLEFT                            \n\t" // if i!=0.
"                                            \n\t"
" .SPOSTACCUM:                               \n\t"
"                                            \n\t"
" ld1r {v6.4s},[x7]                          \n\t" // Load alpha.
" ld1r {v7.4s},[x8]                          \n\t" // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .SGENSTORED                            \n\t"
"                                            \n\t"
" .SCOLSTORED:                               \n\t" // C is column-major.
"                                            \n\t"
" dup  v0.4s, wzr                            \n\t"
" dup  v1.4s, wzr                            \n\t"
" dup  v2.4s, wzr                            \n\t"
" dup  v3.4s, wzr                            \n\t"
" dup  v4.4s, wzr                            \n\t"
" dup  v5.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ldr q0, [x2]                               \n\t" //Load column 0 of C
" ldr q1, [x2, #16]                          \n\t"
" ldr q2, [x16]                              \n\t" //Load column 1 of C
" ldr q3, [x16, #16]                         \n\t"
" ldr q4, [x17]                              \n\t" //Load column 2 of C
" ldr q5, [x17, #16]                         \n\t"
"                                            \n\t"
" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS1:                     \n\t"
"                                            \n\t"
" fmla v0.4s,v8.4s,v6.s[0]                   \n\t" // Scale by alpha
" fmla v1.4s,v9.4s,v6.s[0]                   \n\t" // Scale by alpha
" fmla v2.4s,v10.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v3.4s,v11.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v4.4s,v12.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v5.4s,v13.4s,v6.s[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" str q0, [x2]                               \n\t" //Store column 0 of C
" str q1, [x2, #16]                          \n\t"
" str q2, [x16]                              \n\t" //Store column 1 of C
" str q3, [x16, #16]                         \n\t"
" str q4, [x17]                              \n\t" //Store column 2 of C
" str q5, [x17, #16]                         \n\t"
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t"
" dup  v9.4s, wzr                            \n\t"
" dup  v10.4s, wzr                           \n\t"
" dup  v11.4s, wzr                           \n\t"
" dup  v12.4s, wzr                           \n\t"
" dup  v13.4s, wzr                           \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ldr q8, [x18]                              \n\t" //Load column 3 of C
" ldr q9, [x18, #16]                         \n\t"
" ldr q10, [x19]                             \n\t" //Load column 4 of C
" ldr q11, [x19, #16]                        \n\t"
" ldr q12, [x20]                             \n\t" //Load column 5 of C
" ldr q13, [x20, #16]                        \n\t"
"                                            \n\t"
" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS2:                     \n\t"
"                                            \n\t"
" fmla v8.4s, v14.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v9.4s, v15.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v10.4s,v16.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v11.4s,v17.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v12.4s,v18.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v13.4s,v19.4s,v6.s[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" str q8, [x18]                              \n\t" //Store column 3 of C
" str q9, [x18, #16]                         \n\t"
" str q10, [x19]                             \n\t" //Store column 4 of C
" str q11, [x19, #16]                        \n\t"
" str q12, [x20]                             \n\t" //Store column 5 of C
" str q13, [x20, #16]                        \n\t"
"                                            \n\t"
" dup  v0.4s, wzr                            \n\t"
" dup  v1.4s, wzr                            \n\t"
" dup  v2.4s, wzr                            \n\t"
" dup  v3.4s, wzr                            \n\t"
" dup  v4.4s, wzr                            \n\t"
" dup  v5.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ldr q0, [x21]                              \n\t" //Load column 6 of C
" ldr q1, [x21, #16]                         \n\t"
" ldr q2, [x22]                              \n\t" //Load column 7 of C
" ldr q3, [x22, #16]                         \n\t"
" ldr q4, [x23]                              \n\t" //Load column 8 of C
" ldr q5, [x23, #16]                         \n\t"
"                                            \n\t"
" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS3:                     \n\t"
"                                            \n\t"
" fmla v0.4s,v20.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v1.4s,v21.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v2.4s,v22.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v3.4s,v23.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v4.4s,v24.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v5.4s,v25.4s,v6.s[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" str q0, [x21]                              \n\t" //Store column 6 of C
" str q1, [x21, #16]                         \n\t"
" str q2, [x22]                              \n\t" //Store column 7 of C
" str q3, [x22, #16]                         \n\t"
" str q4, [x23]                              \n\t" //Store column 8 of C
" str q5, [x23, #16]                         \n\t"
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t"
" dup  v9.4s, wzr                            \n\t"
" dup  v10.4s, wzr                            \n\t"
" dup  v11.4s, wzr                            \n\t"
" dup  v12.4s, wzr                            \n\t"
" dup  v13.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS4                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ldr q8, [x24]                              \n\t" //Load column 9 of C
" ldr q9, [x24, #16]                         \n\t"
" ldr q10, [x25]                             \n\t" //Load column 10 of C
" ldr q11, [x25, #16]                        \n\t"
" ldr q12, [x26]                             \n\t" //Load column 11 of C
" ldr q13, [x26, #16]                        \n\t"
"                                            \n\t"
" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS4:                     \n\t"
"                                            \n\t"
" prfm pldl2keep,[x3]                        \n\t"
" prfm pldl2keep,[x4]                        \n\t"
"                                            \n\t"
" fmla v8.4s, v26.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v9.4s, v27.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v10.4s,v28.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v11.4s,v29.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v12.4s,v30.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v13.4s,v31.4s,v6.s[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" str q8, [x24]                              \n\t" //Store column 9 of C
" str q9, [x24, #16]                         \n\t"
" str q10, [x25]                             \n\t" //Store column 10 of C
" str q11, [x25, #16]                        \n\t"
" str q12, [x26]                             \n\t" //Store column 11 of C
" str q13, [x26, #16]                        \n\t"
"                                            \n\t"
"                                            \n\t"
" b .SEND                                    \n\t" // Done (TODO: this obviously needs to be moved down to remove jump).
"                                            \n\t"
"                                            \n\t"
" .SGENSTORED:                               \n\t" // C is general-stride stored.
"                                            \n\t"
"                                            \n\t"
" dup  v0.4s, wzr                            \n\t"
" dup  v1.4s, wzr                            \n\t"
" dup  v2.4s, wzr                            \n\t"
" dup  v3.4s, wzr                            \n\t"
" dup  v4.4s, wzr                            \n\t"
" dup  v5.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROGENSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x2                                \n\t"
"                                            \n\t"
" ld1 {v0.s}[0],[x27],x14                    \n\t" // Load c00  into quad and increment by rs_c.
" ld1 {v0.s}[1],[x27],x14                    \n\t" // Load c01  into quad and increment by rs_c.
" ld1 {v0.s}[2],[x27],x14                    \n\t" // Load c02  into quad and increment by rs_c.
" ld1 {v0.s}[3],[x27],x14                    \n\t" // Load c03  into quad and increment by rs_c.
" ld1 {v1.s}[0],[x27],x14                    \n\t" // Load c04  into quad and increment by rs_c.
" ld1 {v1.s}[1],[x27],x14                    \n\t" // Load c05  into quad and increment by rs_c.
" ld1 {v1.s}[2],[x27],x14                    \n\t" // Load c06  into quad and increment by rs_c.
" ld1 {v1.s}[3],[x27],x14                    \n\t" // Load c07  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x16                               \n\t"
"                                            \n\t"
" ld1 {v2.s}[0],[x27],x14                    \n\t" // Load c10  into quad and increment by rs_c.
" ld1 {v2.s}[1],[x27],x14                    \n\t" // Load c11  into quad and increment by rs_c.
" ld1 {v2.s}[2],[x27],x14                    \n\t" // Load c12  into quad and increment by rs_c.
" ld1 {v2.s}[3],[x27],x14                    \n\t" // Load c13  into quad and increment by rs_c.
" ld1 {v3.s}[0],[x27],x14                    \n\t" // Load c14  into quad and increment by rs_c.
" ld1 {v3.s}[1],[x27],x14                    \n\t" // Load c15  into quad and increment by rs_c.
" ld1 {v3.s}[2],[x27],x14                    \n\t" // Load c16  into quad and increment by rs_c.
" ld1 {v3.s}[3],[x27],x14                    \n\t" // Load c17  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x17                               \n\t"
"                                            \n\t"
" ld1 {v4.s}[0],[x27],x14                    \n\t" // Load c20  into quad and increment by rs_c.
" ld1 {v4.s}[1],[x27],x14                    \n\t" // Load c21  into quad and increment by rs_c.
" ld1 {v4.s}[2],[x27],x14                    \n\t" // Load c22  into quad and increment by rs_c.
" ld1 {v4.s}[3],[x27],x14                    \n\t" // Load c23  into quad and increment by rs_c.
" ld1 {v5.s}[0],[x27],x14                    \n\t" // Load c24  into quad and increment by rs_c.
" ld1 {v5.s}[1],[x27],x14                    \n\t" // Load c25  into quad and increment by rs_c.
" ld1 {v5.s}[2],[x27],x14                    \n\t" // Load c26  into quad and increment by rs_c.
" ld1 {v5.s}[3],[x27],x14                    \n\t" // Load c27  into quad and increment by rs_c.
"                                            \n\t"
" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROGENSTOREDS1:                     \n\t"
"                                            \n\t"
" fmla v0.4s, v8.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v1.4s, v9.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v2.4s,v10.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v3.4s,v11.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v4.4s,v12.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v5.4s,v13.4s,v6.s[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x2                                \n\t"
"                                            \n\t"
" st1 {v0.s}[0],[x27],x14                    \n\t" // Store c00  into quad and increment by rs_c.
" st1 {v0.s}[1],[x27],x14                    \n\t" // Store c01  into quad and increment by rs_c.
" st1 {v0.s}[2],[x27],x14                    \n\t" // Store c02  into quad and increment by rs_c.
" st1 {v0.s}[3],[x27],x14                    \n\t" // Store c03  into quad and increment by rs_c.
" st1 {v1.s}[0],[x27],x14                    \n\t" // Store c04  into quad and increment by rs_c.
" st1 {v1.s}[1],[x27],x14                    \n\t" // Store c05  into quad and increment by rs_c.
" st1 {v1.s}[2],[x27],x14                    \n\t" // Store c06  into quad and increment by rs_c.
" st1 {v1.s}[3],[x27],x14                    \n\t" // Store c07  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x16                               \n\t"
"                                            \n\t"
" st1 {v2.s}[0],[x27],x14                    \n\t" // Store c10  into quad and increment by rs_c.
" st1 {v2.s}[1],[x27],x14                    \n\t" // Store c11  into quad and increment by rs_c.
" st1 {v2.s}[2],[x27],x14                    \n\t" // Store c12  into quad and increment by rs_c.
" st1 {v2.s}[3],[x27],x14                    \n\t" // Store c13  into quad and increment by rs_c.
" st1 {v3.s}[0],[x27],x14                    \n\t" // Store c14  into quad and increment by rs_c.
" st1 {v3.s}[1],[x27],x14                    \n\t" // Store c15  into quad and increment by rs_c.
" st1 {v3.s}[2],[x27],x14                    \n\t" // Store c16  into quad and increment by rs_c.
" st1 {v3.s}[3],[x27],x14                    \n\t" // Store c17  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x17                               \n\t"
"                                            \n\t"
" st1 {v4.s}[0],[x27],x14                    \n\t" // Store c20  into quad and increment by rs_c.
" st1 {v4.s}[1],[x27],x14                    \n\t" // Store c21  into quad and increment by rs_c.
" st1 {v4.s}[2],[x27],x14                    \n\t" // Store c22  into quad and increment by rs_c.
" st1 {v4.s}[3],[x27],x14                    \n\t" // Store c23  into quad and increment by rs_c.
" st1 {v5.s}[0],[x27],x14                    \n\t" // Store c24  into quad and increment by rs_c.
" st1 {v5.s}[1],[x27],x14                    \n\t" // Store c25  into quad and increment by rs_c.
" st1 {v5.s}[2],[x27],x14                    \n\t" // Store c26  into quad and increment by rs_c.
" st1 {v5.s}[3],[x27],x14                    \n\t" // Store c27  into quad and increment by rs_c.
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t"
" dup  v9.4s, wzr                            \n\t"
" dup  v10.4s, wzr                           \n\t"
" dup  v11.4s, wzr                           \n\t"
" dup  v12.4s, wzr                           \n\t"
" dup  v13.4s, wzr                           \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROGENSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x18                               \n\t"
"                                            \n\t"
" ld1 {v8.s}[0],[x27],x14                    \n\t" // Load c30  into quad and increment by rs_c.
" ld1 {v8.s}[1],[x27],x14                    \n\t" // Load c31  into quad and increment by rs_c.
" ld1 {v8.s}[2],[x27],x14                    \n\t" // Load c32  into quad and increment by rs_c.
" ld1 {v8.s}[3],[x27],x14                    \n\t" // Load c33  into quad and increment by rs_c.
" ld1 {v9.s}[0],[x27],x14                    \n\t" // Load c34  into quad and increment by rs_c.
" ld1 {v9.s}[1],[x27],x14                    \n\t" // Load c35  into quad and increment by rs_c.
" ld1 {v9.s}[2],[x27],x14                    \n\t" // Load c36  into quad and increment by rs_c.
" ld1 {v9.s}[3],[x27],x14                    \n\t" // Load c37  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x19                               \n\t"
"                                            \n\t"
" ld1 {v10.s}[0],[x27],x14                   \n\t" // Load c40  into quad and increment by rs_c.
" ld1 {v10.s}[1],[x27],x14                   \n\t" // Load c41  into quad and increment by rs_c.
" ld1 {v10.s}[2],[x27],x14                   \n\t" // Load c42  into quad and increment by rs_c.
" ld1 {v10.s}[3],[x27],x14                   \n\t" // Load c43  into quad and increment by rs_c.
" ld1 {v11.s}[0],[x27],x14                   \n\t" // Load c44  into quad and increment by rs_c.
" ld1 {v11.s}[1],[x27],x14                   \n\t" // Load c45  into quad and increment by rs_c.
" ld1 {v11.s}[2],[x27],x14                   \n\t" // Load c46  into quad and increment by rs_c.
" ld1 {v11.s}[3],[x27],x14                   \n\t" // Load c47  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x20                               \n\t"
"                                            \n\t"
" ld1 {v12.s}[0],[x27],x14                   \n\t" // Load c50  into quad and increment by rs_c.
" ld1 {v12.s}[1],[x27],x14                   \n\t" // Load c51  into quad and increment by rs_c.
" ld1 {v12.s}[2],[x27],x14                   \n\t" // Load c52  into quad and increment by rs_c.
" ld1 {v12.s}[3],[x27],x14                   \n\t" // Load c53  into quad and increment by rs_c.
" ld1 {v13.s}[0],[x27],x14                   \n\t" // Load c54  into quad and increment by rs_c.
" ld1 {v13.s}[1],[x27],x14                   \n\t" // Load c55  into quad and increment by rs_c.
" ld1 {v13.s}[2],[x27],x14                   \n\t" // Load c56  into quad and increment by rs_c.
" ld1 {v13.s}[3],[x27],x14                   \n\t" // Load c57  into quad and increment by rs_c.
"                                            \n\t"
" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROGENSTOREDS2:                     \n\t"
"                                            \n\t"
" fmla v8.4s, v14.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v9.4s, v15.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v10.4s,v16.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v11.4s,v17.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v12.4s,v18.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v13.4s,v19.4s,v6.s[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x18                               \n\t"
"                                            \n\t"
" st1 {v8.s}[0],[x27],x14                    \n\t" // Store c30  into quad and increment by rs_c.
" st1 {v8.s}[1],[x27],x14                    \n\t" // Store c31  into quad and increment by rs_c.
" st1 {v8.s}[2],[x27],x14                    \n\t" // Store c32  into quad and increment by rs_c.
" st1 {v8.s}[3],[x27],x14                    \n\t" // Store c33  into quad and increment by rs_c.
" st1 {v9.s}[0],[x27],x14                    \n\t" // Store c34  into quad and increment by rs_c.
" st1 {v9.s}[1],[x27],x14                    \n\t" // Store c35  into quad and increment by rs_c.
" st1 {v9.s}[2],[x27],x14                    \n\t" // Store c36  into quad and increment by rs_c.
" st1 {v9.s}[3],[x27],x14                    \n\t" // Store c37  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x19                               \n\t"
"                                            \n\t"
" st1 {v10.s}[0],[x27],x14                   \n\t" // Store c40  into quad and increment by rs_c.
" st1 {v10.s}[1],[x27],x14                   \n\t" // Store c41  into quad and increment by rs_c.
" st1 {v10.s}[2],[x27],x14                   \n\t" // Store c42  into quad and increment by rs_c.
" st1 {v10.s}[3],[x27],x14                   \n\t" // Store c43  into quad and increment by rs_c.
" st1 {v11.s}[0],[x27],x14                   \n\t" // Store c44  into quad and increment by rs_c.
" st1 {v11.s}[1],[x27],x14                   \n\t" // Store c45  into quad and increment by rs_c.
" st1 {v11.s}[2],[x27],x14                   \n\t" // Store c46  into quad and increment by rs_c.
" st1 {v11.s}[3],[x27],x14                   \n\t" // Store c47  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x20                               \n\t"
"                                            \n\t"
" st1 {v12.s}[0],[x27],x14                   \n\t" // Store c50  into quad and increment by rs_c.
" st1 {v12.s}[1],[x27],x14                   \n\t" // Store c51  into quad and increment by rs_c.
" st1 {v12.s}[2],[x27],x14                   \n\t" // Store c52  into quad and increment by rs_c.
" st1 {v12.s}[3],[x27],x14                   \n\t" // Store c53  into quad and increment by rs_c.
" st1 {v13.s}[0],[x27],x14                   \n\t" // Store c54  into quad and increment by rs_c.
" st1 {v13.s}[1],[x27],x14                   \n\t" // Store c55  into quad and increment by rs_c.
" st1 {v13.s}[2],[x27],x14                   \n\t" // Store c56  into quad and increment by rs_c.
" st1 {v13.s}[3],[x27],x14                   \n\t" // Store c57  into quad and increment by rs_c.
"                                            \n\t"
" dup  v0.4s, wzr                            \n\t"
" dup  v1.4s, wzr                            \n\t"
" dup  v2.4s, wzr                            \n\t"
" dup  v3.4s, wzr                            \n\t"
" dup  v4.4s, wzr                            \n\t"
" dup  v5.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROGENSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x21                               \n\t"
"                                            \n\t"
" ld1 {v0.s}[0],[x27],x14                    \n\t" // Load c60  into quad and increment by rs_c.
" ld1 {v0.s}[1],[x27],x14                    \n\t" // Load c61  into quad and increment by rs_c.
" ld1 {v0.s}[2],[x27],x14                    \n\t" // Load c62  into quad and increment by rs_c.
" ld1 {v0.s}[3],[x27],x14                    \n\t" // Load c63  into quad and increment by rs_c.
" ld1 {v1.s}[0],[x27],x14                    \n\t" // Load c64  into quad and increment by rs_c.
" ld1 {v1.s}[1],[x27],x14                    \n\t" // Load c65  into quad and increment by rs_c.
" ld1 {v1.s}[2],[x27],x14                    \n\t" // Load c66  into quad and increment by rs_c.
" ld1 {v1.s}[3],[x27],x14                    \n\t" // Load c67  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x22                               \n\t"
"                                            \n\t"
" ld1 {v2.s}[0],[x27],x14                    \n\t" // Load c70  into quad and increment by rs_c.
" ld1 {v2.s}[1],[x27],x14                    \n\t" // Load c71  into quad and increment by rs_c.
" ld1 {v2.s}[2],[x27],x14                    \n\t" // Load c72  into quad and increment by rs_c.
" ld1 {v2.s}[3],[x27],x14                    \n\t" // Load c73  into quad and increment by rs_c.
" ld1 {v3.s}[0],[x27],x14                    \n\t" // Load c74  into quad and increment by rs_c.
" ld1 {v3.s}[1],[x27],x14                    \n\t" // Load c75  into quad and increment by rs_c.
" ld1 {v3.s}[2],[x27],x14                    \n\t" // Load c76  into quad and increment by rs_c.
" ld1 {v3.s}[3],[x27],x14                    \n\t" // Load c77  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x23                               \n\t"
"                                            \n\t"
" ld1 {v4.s}[0],[x27],x14                    \n\t" // Load c80  into quad and increment by rs_c.
" ld1 {v4.s}[1],[x27],x14                    \n\t" // Load c81  into quad and increment by rs_c.
" ld1 {v4.s}[2],[x27],x14                    \n\t" // Load c82  into quad and increment by rs_c.
" ld1 {v4.s}[3],[x27],x14                    \n\t" // Load c83  into quad and increment by rs_c.
" ld1 {v5.s}[0],[x27],x14                    \n\t" // Load c84  into quad and increment by rs_c.
" ld1 {v5.s}[1],[x27],x14                    \n\t" // Load c85  into quad and increment by rs_c.
" ld1 {v5.s}[2],[x27],x14                    \n\t" // Load c86  into quad and increment by rs_c.
" ld1 {v5.s}[3],[x27],x14                    \n\t" // Load c87  into quad and increment by rs_c.
"                                            \n\t"
" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROGENSTOREDS3:                     \n\t"
"                                            \n\t"
" fmla v0.4s,v20.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v1.4s,v21.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v2.4s,v22.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v3.4s,v23.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v4.4s,v24.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v5.4s,v25.4s,v6.s[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x21                               \n\t"
"                                            \n\t"
" st1 {v0.s}[0],[x27],x14                    \n\t" // Store c60  into quad and increment by rs_c.
" st1 {v0.s}[1],[x27],x14                    \n\t" // Store c61  into quad and increment by rs_c.
" st1 {v0.s}[2],[x27],x14                    \n\t" // Store c62  into quad and increment by rs_c.
" st1 {v0.s}[3],[x27],x14                    \n\t" // Store c63  into quad and increment by rs_c.
" st1 {v1.s}[0],[x27],x14                    \n\t" // Store c64  into quad and increment by rs_c.
" st1 {v1.s}[1],[x27],x14                    \n\t" // Store c65  into quad and increment by rs_c.
" st1 {v1.s}[2],[x27],x14                    \n\t" // Store c66  into quad and increment by rs_c.
" st1 {v1.s}[3],[x27],x14                    \n\t" // Store c67  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x22                               \n\t"
"                                            \n\t"
" st1 {v2.s}[0],[x27],x14                    \n\t" // Store c70  into quad and increment by rs_c.
" st1 {v2.s}[1],[x27],x14                    \n\t" // Store c71  into quad and increment by rs_c.
" st1 {v2.s}[2],[x27],x14                    \n\t" // Store c72  into quad and increment by rs_c.
" st1 {v2.s}[3],[x27],x14                    \n\t" // Store c73  into quad and increment by rs_c.
" st1 {v3.s}[0],[x27],x14                    \n\t" // Store c74  into quad and increment by rs_c.
" st1 {v3.s}[1],[x27],x14                    \n\t" // Store c75  into quad and increment by rs_c.
" st1 {v3.s}[2],[x27],x14                    \n\t" // Store c76  into quad and increment by rs_c.
" st1 {v3.s}[3],[x27],x14                    \n\t" // Store c77  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x23                               \n\t"
"                                            \n\t"
" st1 {v4.s}[0],[x27],x14                    \n\t" // Store c80  into quad and increment by rs_c.
" st1 {v4.s}[1],[x27],x14                    \n\t" // Store c81  into quad and increment by rs_c.
" st1 {v4.s}[2],[x27],x14                    \n\t" // Store c82  into quad and increment by rs_c.
" st1 {v4.s}[3],[x27],x14                    \n\t" // Store c83  into quad and increment by rs_c.
" st1 {v5.s}[0],[x27],x14                    \n\t" // Store c84  into quad and increment by rs_c.
" st1 {v5.s}[1],[x27],x14                    \n\t" // Store c85  into quad and increment by rs_c.
" st1 {v5.s}[2],[x27],x14                    \n\t" // Store c86  into quad and increment by rs_c.
" st1 {v5.s}[3],[x27],x14                    \n\t" // Store c87  into quad and increment by rs_c.
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t"
" dup  v9.4s, wzr                            \n\t"
" dup  v10.4s, wzr                           \n\t"
" dup  v11.4s, wzr                           \n\t"
" dup  v12.4s, wzr                           \n\t"
" dup  v13.4s, wzr                           \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROGENSTOREDS4                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x24                               \n\t"
"                                            \n\t"
" ld1 {v8.s}[0],[x27],x14                    \n\t" // Load c90  into quad and increment by rs_c.
" ld1 {v8.s}[1],[x27],x14                    \n\t" // Load c91  into quad and increment by rs_c.
" ld1 {v8.s}[2],[x27],x14                    \n\t" // Load c92  into quad and increment by rs_c.
" ld1 {v8.s}[3],[x27],x14                    \n\t" // Load c93  into quad and increment by rs_c.
" ld1 {v9.s}[0],[x27],x14                    \n\t" // Load c94  into quad and increment by rs_c.
" ld1 {v9.s}[1],[x27],x14                    \n\t" // Load c95  into quad and increment by rs_c.
" ld1 {v9.s}[2],[x27],x14                    \n\t" // Load c96  into quad and increment by rs_c.
" ld1 {v9.s}[3],[x27],x14                    \n\t" // Load c97  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x25                               \n\t"
"                                            \n\t"
" ld1 {v10.s}[0],[x27],x14                   \n\t" // Load c100  into quad and increment by rs_c.
" ld1 {v10.s}[1],[x27],x14                   \n\t" // Load c101  into quad and increment by rs_c.
" ld1 {v10.s}[2],[x27],x14                   \n\t" // Load c102  into quad and increment by rs_c.
" ld1 {v10.s}[3],[x27],x14                   \n\t" // Load c103  into quad and increment by rs_c.
" ld1 {v11.s}[0],[x27],x14                   \n\t" // Load c104  into quad and increment by rs_c.
" ld1 {v11.s}[1],[x27],x14                   \n\t" // Load c105  into quad and increment by rs_c.
" ld1 {v11.s}[2],[x27],x14                   \n\t" // Load c106  into quad and increment by rs_c.
" ld1 {v11.s}[3],[x27],x14                   \n\t" // Load c107  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x26                               \n\t"
"                                            \n\t"
" ld1 {v12.s}[0],[x27],x14                   \n\t" // Load c110  into quad and increment by rs_c.
" ld1 {v12.s}[1],[x27],x14                   \n\t" // Load c111  into quad and increment by rs_c.
" ld1 {v12.s}[2],[x27],x14                   \n\t" // Load c112  into quad and increment by rs_c.
" ld1 {v12.s}[3],[x27],x14                   \n\t" // Load c113  into quad and increment by rs_c.
" ld1 {v13.s}[0],[x27],x14                   \n\t" // Load c114  into quad and increment by rs_c.
" ld1 {v13.s}[1],[x27],x14                   \n\t" // Load c115  into quad and increment by rs_c.
" ld1 {v13.s}[2],[x27],x14                   \n\t" // Load c116  into quad and increment by rs_c.
" ld1 {v13.s}[3],[x27],x14                   \n\t" // Load c117  into quad and increment by rs_c.
"                                            \n\t"
" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROGENSTOREDS4:                     \n\t"
"                                            \n\t"
" prfm pldl2keep,[x3]                        \n\t"
" prfm pldl2keep,[x4]                        \n\t"
"                                            \n\t"
" fmla v8.4s, v26.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v9.4s, v27.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v10.4s,v28.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v11.4s,v29.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v12.4s,v30.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v13.4s,v31.4s,v6.s[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x24                               \n\t"
"                                            \n\t"
" st1 {v8.s}[0],[x27],x14                    \n\t" // Store c90  into quad and increment by rs_c.
" st1 {v8.s}[1],[x27],x14                    \n\t" // Store c91  into quad and increment by rs_c.
" st1 {v8.s}[2],[x27],x14                    \n\t" // Store c92  into quad and increment by rs_c.
" st1 {v8.s}[3],[x27],x14                    \n\t" // Store c93  into quad and increment by rs_c.
" st1 {v9.s}[0],[x27],x14                    \n\t" // Store c94  into quad and increment by rs_c.
" st1 {v9.s}[1],[x27],x14                    \n\t" // Store c95  into quad and increment by rs_c.
" st1 {v9.s}[2],[x27],x14                    \n\t" // Store c96  into quad and increment by rs_c.
" st1 {v9.s}[3],[x27],x14                    \n\t" // Store c97  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x25                               \n\t"
"                                            \n\t"
" st1 {v10.s}[0],[x27],x14                   \n\t" // Store c100  into quad and increment by rs_c.
" st1 {v10.s}[1],[x27],x14                   \n\t" // Store c101  into quad and increment by rs_c.
" st1 {v10.s}[2],[x27],x14                   \n\t" // Store c102  into quad and increment by rs_c.
" st1 {v10.s}[3],[x27],x14                   \n\t" // Store c103  into quad and increment by rs_c.
" st1 {v11.s}[0],[x27],x14                   \n\t" // Store c104  into quad and increment by rs_c.
" st1 {v11.s}[1],[x27],x14                   \n\t" // Store c105  into quad and increment by rs_c.
" st1 {v11.s}[2],[x27],x14                   \n\t" // Store c106  into quad and increment by rs_c.
" st1 {v11.s}[3],[x27],x14                   \n\t" // Store c107  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x26                               \n\t"
"                                            \n\t"
" st1 {v12.s}[0],[x27],x14                   \n\t" // Store c110  into quad and increment by rs_c.
" st1 {v12.s}[1],[x27],x14                   \n\t" // Store c111  into quad and increment by rs_c.
" st1 {v12.s}[2],[x27],x14                   \n\t" // Store c112  into quad and increment by rs_c.
" st1 {v12.s}[3],[x27],x14                   \n\t" // Store c113  into quad and increment by rs_c.
" st1 {v13.s}[0],[x27],x14                   \n\t" // Store c114  into quad and increment by rs_c.
" st1 {v13.s}[1],[x27],x14                   \n\t" // Store c115  into quad and increment by rs_c.
" st1 {v13.s}[2],[x27],x14                   \n\t" // Store c116  into quad and increment by rs_c.
" st1 {v13.s}[3],[x27],x14                   \n\t" // Store c147  into quad and increment by rs_c.
"                                            \n\t"
" .SEND:                                     \n\t" // Done!
"                                            \n\t"
:// output operands (none)
:// input operands
 [aaddr]  "m" (a),      // 0
 [baddr]  "m" (b),      // 1
 [caddr]  "m" (c),      // 2
 [k_iter] "m" (k_iter), // 3
 [k_left] "m" (k_left), // 4
 [alpha]  "m" (alpha),  // 5
 [beta]   "m" (beta),   // 6
 [rs_c]   "m" (rs_c),   // 7
 [cs_c]   "m" (cs_c),   // 8
 [a_next] "m" (a_next), // 9
 [b_next] "m" (b_next), // 10
 [k]      "m" (k)       // 11
:// Register clobber list
 "x0", "x1", "x2","x3","x4",
 "x5", "x6", "x7", "x8",
 "x9", "x10","x11","x12",
 "x13","x14","x15",
 "x16","x17","x18","x19",       
 "x20","x21","x22","x23",
 "x24","x25","x26","x27",
 "v0", "v1", "v2", "v3",
 "v4", "v5", "v6", "v7",
 "v8", "v9", "v10","v11",
 "v12","v13","v14","v15",
 "v16","v17","v18","v19",
 "v20","v21","v22","v23",
 "v24","v25","v26","v27",
 "v28","v29","v30","v31"
);

}


/*
   o 4x4 Double precision micro-kernel NOT fully functional yet.
   o Runnable on ARMv8, compiled with aarch64 GCC.
   o Use it together with the armv8 BLIS configuration.
   o Tested on Juno board. Around 3 GFLOPS @ 1.1 GHz. 

   December 2014.
  
 * UPDATE OCTOBER 2015: Now is fully functional.
 * Tested on Juno board. Around 5.6 GFLOPS, 2 A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around 4 GFLOPS, 4 A53 cores @ 850 MHz.
 
 * UPDATE NOVEMBER 2015
 * Micro-kernel changed to 6x8
 * Tested on Juno Board. Around 4   GFLOPS, 1 x A57 core  @ 1.1 GHz.
 * Tested on Juno Board. Around 7.6 GFLOPS, 2 x A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around 1.5 GFLOPS, 1 x A53 core  @ 850 MHz. 
 * Tested on Juno board. Around 5.5 GFLOPS, 4 x A53 cores @ 850 MHz.
*/
void conv_relu_pool2x2_d6x8(
                        dim_t              k,
                        double*   restrict alpha,
                        double*   restrict a,
                        double*   restrict b,
                        double*   restrict beta,
                        double*   restrict c, inc_t rs_c, inc_t cs_c,
                        //auxinfo_t*         data
                        aux_s<double, double, double, double> *aux
                      )
{
	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );
	void* a_next = aux->a_next;
	void* b_next = aux->b_next;

	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;

__asm__ volatile
(
"                                            \n\t" 
" ldr x0,%[aaddr]                            \n\t" // Load address of A 
" ldr x1,%[baddr]                            \n\t" // Load address of B
" ldr x2,%[caddr]                            \n\t" // Load address of C
"                                            \n\t"
" ldr x3,%[a_next]                           \n\t" // Move pointer
" ldr x4,%[b_next]                           \n\t" // Move pointer
"                                            \n\t"
" ldr x5,%[k_iter]                           \n\t" // Init guard (k_iter)
" ldr x6,%[k_left]                           \n\t" // Init guard (k_iter)
"                                            \n\t" 
" ldr x7,%[alpha]                            \n\t" // Alpha address      
" ldr x8,%[beta]                             \n\t" // Beta address      
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x9,#3                              \n\t" // cs_c * sizeof(double)
"                                            \n\t"
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
" lsl x14,x13,#3                             \n\t" // rs_c * sizeof(double). 
"                                            \n\t"
" add x20,x2,x10                             \n\t" //Load address Column 1 of C
" add x21,x20,x10                            \n\t" //Load address Column 2 of C
" add x22,x21,x10                            \n\t" //Load address Column 3 of C
" add x23,x22,x10                            \n\t" //Load address Column 4 of C
" add x24,x23,x10                            \n\t" //Load address Column 5 of C
" add x25,x24,x10                            \n\t" //Load address Column 6 of C
" add x26,x25,x10                            \n\t" //Load address Column 7 of C
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x24]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x25]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x26]                       \n\t" // Prefetch c.
"                                            \n\t"
" ldr q0, [x0]                               \n\t"
" ldr q1, [x0, #16]                          \n\t" // Load a
" ldr q2, [x0, #32]                          \n\t"
"                                            \n\t"
" ldr q3, [x1]                               \n\t" // Load b
" ldr q4, [x1, #16]                          \n\t"
" ldr q5, [x1, #32]                          \n\t"
" ldr q6, [x1, #48]                          \n\t"
"                                            \n\t"
" dup  v8.2d, xzr                            \n\t" // Vector for accummulating column 0
" prfm    PLDL1KEEP, [x1, #256]              \n\t" 
" dup  v9.2d, xzr                            \n\t" // Vector for accummulating column 0
" prfm    PLDL1KEEP, [x1, #320]              \n\t"
" dup  v10.2d, xzr                           \n\t" // Vector for accummulating column 0
" prfm    PLDL1KEEP, [x1, #384]              \n\t"
" dup  v11.2d, xzr                           \n\t" // Vector for accummulating column 1
" prfm    PLDL1KEEP, [x1, #448]              \n\t"
" dup  v12.2d, xzr                           \n\t" // Vector for accummulating column 1 
" dup  v13.2d, xzr                           \n\t" // Vector for accummulating column 1
"                                            \n\t"
" dup  v14.2d, xzr                           \n\t" // Vector for accummulating column 2
" prfm    PLDL1KEEP, [x0, #192]              \n\t"
" dup  v15.2d, xzr                           \n\t" // Vector for accummulating column 2
" prfm    PLDL1KEEP, [x0, #256]              \n\t"
" dup  v16.2d, xzr                           \n\t" // Vector for accummulating column 2
" prfm    PLDL1KEEP, [x0, #320]              \n\t"
" dup  v17.2d, xzr                           \n\t" // Vector for accummulating column 3
" dup  v18.2d, xzr                           \n\t" // Vector for accummulating column 3 
" dup  v19.2d, xzr                           \n\t" // Vector for accummulating column 3
"                                            \n\t"
" dup  v20.2d, xzr                           \n\t" // Vector for accummulating column 4 
" dup  v21.2d, xzr                           \n\t" // Vector for accummulating column 4
" dup  v22.2d, xzr                           \n\t" // Vector for accummulating column 4
" dup  v23.2d, xzr                           \n\t" // Vector for accummulating column 5
" dup  v24.2d, xzr                           \n\t" // Vector for accummulating column 5 
" dup  v25.2d, xzr                           \n\t" // Vector for accummulating column 5
"                                            \n\t"
" dup  v26.2d, xzr                           \n\t" // Vector for accummulating column 6 
" dup  v27.2d, xzr                           \n\t" // Vector for accummulating column 6
" dup  v28.2d, xzr                           \n\t" // Vector for accummulating column 6
" dup  v29.2d, xzr                           \n\t" // Vector for accummulating column 7
" dup  v30.2d, xzr                           \n\t" // Vector for accummulating column 7 
" dup  v31.2d, xzr                           \n\t" // Vector for accummulating column 7
"                                            \n\t"
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .DCONSIDERKLEFT                        \n\t"
"                                            \n\t"
"add x0, x0, #48                             \n\t" //update address of A
"add x1, x1, #64                             \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .DLASTITER                             \n\t" // (as loop is do-while-like).
"                                            \n\t"
" DLOOP:                                     \n\t" // Body
"                                            \n\t"
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" prfm    PLDL1KEEP, [x1, #448]              \n\t" //512-64=448
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" prfm    PLDL1KEEP, [x1, #512]              \n\t"
" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
" prfm    PLDL1KEEP, [x1, #576]              \n\t"
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
" ldr q3, [x1]                               \n\t"
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
" ldr q7, [x0, #32]                          \n\t"
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
" ldr q4, [x1, #16]                          \n\t"
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
" ldr q5, [x1, #32]                          \n\t"
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q0, [x0]                               \n\t"
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q1, [x0, #16]                          \n\t"
"                                            \n\t"
" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q6, [x1, #48]                          \n\t"
"                                            \n\t"                  // End it 1
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" prfm    PLDL1KEEP, [x1, #640]              \n\t"
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" prfm    PLDL1KEEP, [x0, #336]              \n\t"
" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
" prfm    PLDL1KEEP, [x0, #400]              \n\t"
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
" ldr q3, [x1, #64]                          \n\t"
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
" ldr q2, [x0, #80]                          \n\t"
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
" ldr q4, [x1, #80]                          \n\t"
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
" ldr q5, [x1, #96]                          \n\t"
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q0, [x0, #48]                          \n\t"
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q1, [x0, #64]                          \n\t"
"                                            \n\t"
" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q6, [x1, #112]                         \n\t"
"                                            \n\t"                  //End it 2
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" prfm    PLDL1KEEP, [x0, #464]              \n\t"
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
" ldr q3, [x1, #128]                         \n\t"
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
" ldr q7, [x0, #128]                         \n\t"
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
" ldr q4, [x1, #144]                         \n\t"
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
" ldr q5, [x1, #160]                         \n\t"
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q0, [x0, #96]                          \n\t"
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q1, [x0, #112]                         \n\t"
"                                            \n\t"
" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q6, [x1, #176]                         \n\t"
"                                            \n\t"                  // End it 3
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
" ldr q3, [x1, #192]                         \n\t"
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
" ldr q2, [x0, #176]                         \n\t"
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
" ldr q4, [x1, #208]                         \n\t"
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
" ldr q5, [x1, #224]                         \n\t"
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q0, [x0, #144]                         \n\t"
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q1, [x0, #160]                         \n\t"
"                                            \n\t"
" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q6, [x1, #240]                         \n\t"
"                                            \n\t"                  //End it 4
" add x0, x0, #192                           \n\t"
" add x1, x1, #256                           \n\t"
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne DLOOP                                  \n\t"
"                                            \n\t"
".DLASTITER:                                 \n\t"
"                                            \n\t"
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
" ldr q3, [x1]                               \n\t"
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
" ldr q7, [x0, #32]                          \n\t"
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
" ldr q4, [x1, #16]                          \n\t"
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
" ldr q5, [x1, #32]                          \n\t"
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q0, [x0]                               \n\t"
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q1, [x0, #16]                          \n\t"
"                                            \n\t"
" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q6, [x1, #48]                          \n\t"
"                                            \n\t"                  // End it 1
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
" ldr q3, [x1, #64]                          \n\t"
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
" ldr q2, [x0, #80]                          \n\t"
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
" ldr q4, [x1, #80]                          \n\t"
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
" ldr q5, [x1, #96]                          \n\t"
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q0, [x0, #48]                          \n\t"
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q1, [x0, #64]                          \n\t"
"                                            \n\t"
" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q6, [x1, #112]                         \n\t"
"                                            \n\t"                  //End it 2
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
" ldr q3, [x1, #128]                         \n\t"
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
" ldr q7, [x0, #128]                         \n\t"
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
" ldr q4, [x1, #144]                         \n\t"
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
" ldr q5, [x1, #160]                         \n\t"
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q0, [x0, #96]                          \n\t"
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q1, [x0, #112]                         \n\t"
"                                            \n\t"
" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
" ldr q6, [x1, #176]                         \n\t"
"                                            \n\t"                  // End it 3
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" add x1, x1, #192                           \n\t"
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
"                                            \n\t"                  //End it 4
" add x0, x0, #144                           \n\t"
"                                            \n\t"
" .DCONSIDERKLEFT:                           \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .DPOSTACCUM                            \n\t" // else, we enter the k_left loop.
"                                            \n\t"
".DLOOPKLEFT:                                \n\t"
"                                            \n\t"
" ldr q0, [x0],#16                           \n\t"
" ldr q1, [x0],#16                           \n\t" // Load a
" ldr q2, [x0],#16                           \n\t"
"                                            \n\t"
" ldr q3, [x1],#16                           \n\t" // Load b
" ldr q4, [x1],#16                           \n\t"
" ldr q5, [x1],#16                           \n\t"
" ldr q6, [x1],#16                           \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t"
"                                            \n\t"
" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .DLOOPKLEFT                            \n\t" // if i!=0.
"                                            \n\t"
" .DPOSTACCUM:                               \n\t"
"                                            \n\t"
" ld1r {v6.2d},[x7]                          \n\t" // Load alpha.
" ld1r {v7.2d},[x8]                          \n\t" // Load beta
"                                            \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .DGENSTORED                            \n\t"
"                                            \n\t"
" .DCOLSTORED:                               \n\t" // C is column-major.
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" dup  v0.2d, xzr                            \n\t"
" dup  v1.2d, xzr                            \n\t"
" dup  v2.2d, xzr                            \n\t"
" dup  v3.2d, xzr                            \n\t"
" dup  v4.2d, xzr                            \n\t"
" dup  v5.2d, xzr                            \n\t"
"                                            \n\t"
" ldr q0, [x2]                               \n\t" //Load column 0 of C
" ldr q1, [x2, #16]                          \n\t"
" ldr q2, [x2, #32]                          \n\t"
"                                            \n\t"
" ldr q3, [x20]                              \n\t" //Load column 1 of C
" ldr q4, [x20, #16]                         \n\t"
" ldr q5, [x20, #32]                         \n\t"
"                                            \n\t"
" fadd  v8.2d, v8.2d,v0.2d                   \n\t"  
" fadd  v9.2d, v9.2d,v1.2d                   \n\t"  
" fadd v10.2d,v10.2d,v2.2d                   \n\t"  
" fadd v11.2d,v11.2d,v3.2d                   \n\t"  
" fadd v12.2d,v12.2d,v4.2d                   \n\t"  
" fadd v13.2d,v13.2d,v5.2d                   \n\t"  
"                                            \n\t"
" .DBETAZEROCOLSTOREDS1:                     \n\t"
"                                            \n\t"
" fmax  v8.2d, v8.2d,v11.2d                  \n\t" //Max column 0 and 1 of C 
" fmax  v9.2d, v9.2d,v12.2d                  \n\t" //Max column 0 and 1 of C 
" fmax v10.2d,v10.2d,v13.2d                  \n\t" //Max column 0 and 1 of C 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" dup  v0.2d, xzr                            \n\t"
" dup  v1.2d, xzr                            \n\t"
" dup  v2.2d, xzr                            \n\t"
" dup  v3.2d, xzr                            \n\t"
" dup  v4.2d, xzr                            \n\t"
" dup  v5.2d, xzr                            \n\t"
"                                            \n\t"
" ldr q0, [x21]                              \n\t" //Load column 2 of C
" ldr q1, [x21, #16]                         \n\t"
" ldr q2, [x21, #32]                         \n\t"
"                                            \n\t"
" ldr q3, [x22]                              \n\t" //Load column 3 of C
" ldr q4, [x22, #16]                         \n\t"
" ldr q5, [x22, #32]                         \n\t"
"                                            \n\t"
" fadd v14.2d,v14.2d,v0.2d                   \n\t"  
" fadd v15.2d,v15.2d,v1.2d                   \n\t"  
" fadd v16.2d,v16.2d,v2.2d                   \n\t"  
" fadd v17.2d,v17.2d,v3.2d                   \n\t"  
" fadd v18.2d,v18.2d,v4.2d                   \n\t"  
" fadd v19.2d,v19.2d,v5.2d                   \n\t"  
"                                            \n\t"
" .DBETAZEROCOLSTOREDS2:                     \n\t"
"                                            \n\t"
" fmax  v8.2d, v8.2d,v14.2d                  \n\t" //Max column 0 and 2 of C 
" fmax  v9.2d, v9.2d,v15.2d                  \n\t" //Max column 0 and 2 of C 
" fmax v10.2d,v10.2d,v16.2d                  \n\t" //Max column 0 and 2 of C 
" fmax  v8.2d, v8.2d,v17.2d                  \n\t" //Max column 0 and 3 of C 
" fmax  v9.2d, v9.2d,v18.2d                  \n\t" //Max column 0 and 3 of C 
" fmax v10.2d,v10.2d,v19.2d                  \n\t" //Max column 0 and 3 of C 
"                                            \n\t"
" str  q8, [x2]                              \n\t" //Store column 0 of C
" str  q9, [x2, #16]                         \n\t"
" str q10, [x2, #32]                         \n\t"
"                                            \n\t"
" str q3, [x20]                              \n\t" //Store column 1 of C
" str q4, [x20, #16]                         \n\t"
" str q5, [x20, #32]                         \n\t"
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" dup  v0.2d, xzr                            \n\t"
" dup  v1.2d, xzr                            \n\t"
" dup  v2.2d, xzr                            \n\t"
" dup  v3.2d, xzr                            \n\t"
" dup  v4.2d, xzr                            \n\t"
" dup  v5.2d, xzr                            \n\t"
"                                            \n\t"
" ldr q0, [x23]                              \n\t" //Load column 4 of C
" ldr q1, [x23, #16]                         \n\t"
" ldr q2, [x23, #32]                         \n\t"
"                                            \n\t"
" ldr q3, [x24]                              \n\t" //Load column 5 of C
" ldr q4, [x24, #16]                         \n\t"
" ldr q5, [x24, #32]                         \n\t"
"                                            \n\t"
" fadd v20.2d,v20.2d,v0.2d                   \n\t"  
" fadd v21.2d,v21.2d,v1.2d                   \n\t"  
" fadd v22.2d,v22.2d,v2.2d                   \n\t"  
" fadd v23.2d,v23.2d,v3.2d                   \n\t"  
" fadd v24.2d,v24.2d,v4.2d                   \n\t"  
" fadd v25.2d,v25.2d,v5.2d                   \n\t"  
"                                            \n\t"
" .DBETAZEROCOLSTOREDS3:                     \n\t"
"                                            \n\t"
" fmax v20.2d,v20.2d,v0.2d                   \n\t" //Max column 4 and 5 of C 
" fmax v21.2d,v21.2d,v1.2d                   \n\t" //Max column 4 and 5 of C 
" fmax v22.2d,v22.2d,v2.2d                   \n\t" //Max column 4 and 5 of C 
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS4                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" dup  v0.2d, xzr                            \n\t"
" dup  v1.2d, xzr                            \n\t"
" dup  v2.2d, xzr                            \n\t"
" dup  v3.2d, xzr                            \n\t"
" dup  v4.2d, xzr                            \n\t"
" dup  v5.2d, xzr                            \n\t"
"                                            \n\t"
" ldr q0, [x25]                              \n\t" //Load column 6 of C
" ldr q1, [x25, #16]                         \n\t"
" ldr q2, [x25, #32]                         \n\t"
"                                            \n\t"
" ldr q3, [x26]                              \n\t" //Load column 7 of C
" ldr q4, [x26, #16]                         \n\t"
" ldr q5, [x26, #32]                         \n\t"
"                                            \n\t"
" fadd v26.2d,v26.2d,v0.2d                   \n\t"  
" fadd v27.2d,v27.2d,v1.2d                   \n\t"  
" fadd v28.2d,v28.2d,v2.2d                   \n\t"  
" fadd v29.2d,v29.2d,v3.2d                   \n\t"  
" fadd v30.2d,v30.2d,v4.2d                   \n\t"  
" fadd v31.2d,v31.2d,v5.2d                   \n\t"  
"                                            \n\t"
" .DBETAZEROCOLSTOREDS4:                     \n\t"
"                                            \n\t"
" fmax v20.2d,v20.2d,v26.2d                  \n\t" //Max column 4 and 6 of C 
" fmax v21.2d,v21.2d,v27.2d                  \n\t" //Max column 4 and 6 of C 
" fmax v22.2d,v22.2d,v28.2d                  \n\t" //Max column 4 and 6 of C 
" fmax v20.2d,v20.2d,v29.2d                  \n\t" //Max column 4 and 7 of C 
" fmax v21.2d,v21.2d,v30.2d                  \n\t" //Max column 4 and 7 of C 
" fmax v22.2d,v22.2d,v31.2d                  \n\t" //Max column 4 and 7 of C 
"                                            \n\t"
" str q26, [x20]                             \n\t" //Store column 1 of C
" str q27, [x20, #16]                        \n\t"
" str q28, [x20, #32]                        \n\t"
"                                            \n\t"
" b .DEND                                    \n\t"
"                                            \n\t"
" .DGENSTORED:                               \n\t" // C is general-stride stored.
"                                            \n\t"
" dup  v0.2d, xzr                            \n\t"
" dup  v1.2d, xzr                            \n\t"
" dup  v2.2d, xzr                            \n\t"
" dup  v3.2d, xzr                            \n\t"
" dup  v4.2d, xzr                            \n\t"
" dup  v5.2d, xzr                            \n\t"
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x2                                \n\t"
"                                            \n\t" // Load address of C.
" ld1 {v0.d}[0],[x27],x14                    \n\t" // Load c00  into quad and increment by rs_c.
" ld1 {v0.d}[1],[x27],x14                    \n\t" // Load c01  into quad and increment by rs_c.
" ld1 {v1.d}[0],[x27],x14                    \n\t" // Load c02  into quad and increment by rs_c.
" ld1 {v1.d}[1],[x27],x14                    \n\t" // Load c03  into quad and increment by rs_c.
" ld1 {v2.d}[0],[x27],x14                    \n\t" // Load c04  into quad and increment by rs_c.
" ld1 {v2.d}[1],[x27],x14                    \n\t" // Load c05  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x20                               \n\t" // Load address of C.
"                                            \n\t"
" ld1 {v3.d}[0],[x27],x14                    \n\t" // Load c10  into quad and increment by rs_c.
" ld1 {v3.d}[1],[x27],x14                    \n\t" // Load c11  into quad and increment by rs_c.
" ld1 {v4.d}[0],[x27],x14                    \n\t" // Load c12  into quad and increment by rs_c.
" ld1 {v4.d}[1],[x27],x14                    \n\t" // Load c13  into quad and increment by rs_c.
" ld1 {v5.d}[0],[x27],x14                    \n\t" // Load c14  into quad and increment by rs_c.
" ld1 {v5.d}[1],[x27],x14                    \n\t" // Load c15  into quad and increment by rs_c.
"                                            \n\t"
" fmul v0.2d,v0.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v1.2d,v1.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v2.2d,v2.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v3.2d,v3.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v4.2d,v4.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v5.2d,v5.2d,v7.d[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROGENSTOREDS1:                     \n\t"
"                                            \n\t"
" fmla v0.2d,v8.2d,v6.d[0]                   \n\t" // Scale by alpha
" fmla v1.2d,v9.2d,v6.d[0]                   \n\t" // Scale by alpha
" fmla v2.2d,v10.2d,v6.d[0]                  \n\t" // Scale by alpha
" fmla v3.2d,v11.2d,v6.d[0]                  \n\t" // Scale by alpha
" fmla v4.2d,v12.2d,v6.d[0]                  \n\t" // Scale by alpha
" fmla v5.2d,v13.2d,v6.d[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x2                                \n\t" // Load address of C.
"                                            \n\t"
" st1 {v0.d}[0],[x27],x14                    \n\t" // Store c00  into quad and increment by rs_c.
" st1 {v0.d}[1],[x27],x14                    \n\t" // Store c01  into quad and increment by rs_c.
" st1 {v1.d}[0],[x27],x14                    \n\t" // Store c02  into quad and increment by rs_c.
" st1 {v1.d}[1],[x27],x14                    \n\t" // Store c03  into quad and increment by rs_c.
" st1 {v2.d}[0],[x27],x14                    \n\t" // Store c04  into quad and increment by rs_c.
" st1 {v2.d}[1],[x27],x14                    \n\t" // Store c05  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x20                               \n\t" // Load address of C.
"                                            \n\t"
" st1 {v3.d}[0],[x27],x14                    \n\t" // Store c10  into quad and increment by rs_c.
" st1 {v3.d}[1],[x27],x14                    \n\t" // Store c11  into quad and increment by rs_c.
" st1 {v4.d}[0],[x27],x14                    \n\t" // Store c12  into quad and increment by rs_c.
" st1 {v4.d}[1],[x27],x14                    \n\t" // Store c13  into quad and increment by rs_c.
" st1 {v5.d}[0],[x27],x14                    \n\t" // Store c14  into quad and increment by rs_c.
" st1 {v5.d}[1],[x27],x14                    \n\t" // Store c15  into quad and increment by rs_c.
"                                            \n\t"
" dup  v8.2d, xzr                            \n\t"
" dup  v9.2d, xzr                            \n\t"
" dup  v10.2d, xzr                           \n\t"
" dup  v11.2d, xzr                           \n\t"
" dup  v12.2d, xzr                           \n\t"
" dup  v13.2d, xzr                           \n\t"
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x21                               \n\t" // Load address of C.
"                                            \n\t"
" ld1 {v8.d}[0], [x27],x14                   \n\t" // Load c20  into quad and increment by rs_c.
" ld1 {v8.d}[1], [x27],x14                   \n\t" // Load c21  into quad and increment by rs_c.
" ld1 {v9.d}[0], [x27],x14                   \n\t" // Load c22  into quad and increment by rs_c.
" ld1 {v9.d}[1], [x27],x14                   \n\t" // Load c23  into quad and increment by rs_c.
" ld1 {v10.d}[0],[x27],x14                   \n\t" // Load c24  into quad and increment by rs_c.
" ld1 {v10.d}[1],[x27],x14                   \n\t" // Load c25  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x22                               \n\t" // Load address of C.
"                                            \n\t"
" ld1 {v11.d}[0],[x27],x14                   \n\t" // Load c30  into quad and increment by rs_c.
" ld1 {v11.d}[1],[x27],x14                   \n\t" // Load c31  into quad and increment by rs_c.
" ld1 {v12.d}[0],[x27],x14                   \n\t" // Load c32  into quad and increment by rs_c.
" ld1 {v12.d}[1],[x27],x14                   \n\t" // Load c33  into quad and increment by rs_c.
" ld1 {v13.d}[0],[x27],x14                   \n\t" // Load c34  into quad and increment by rs_c.
" ld1 {v13.d}[1],[x27],x14                   \n\t" // Load c35  into quad and increment by rs_c.
"                                            \n\t"
" fmul v8.2d, v8.2d, v7.d[0]                 \n\t" // Scale by beta
" fmul v9.2d, v9.2d, v7.d[0]                 \n\t" // Scale by beta
" fmul v10.2d,v10.2d,v7.d[0]                 \n\t" // Scale by beta
" fmul v11.2d,v11.2d,v7.d[0]                 \n\t" // Scale by beta
" fmul v12.2d,v12.2d,v7.d[0]                 \n\t" // Scale by beta
" fmul v13.2d,v13.2d,v7.d[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROGENSTOREDS2:                     \n\t"
"                                            \n\t"
" fmla v8.2d, v14.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v9.2d, v15.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v10.2d,v16.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v11.2d,v17.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v12.2d,v18.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v13.2d,v19.2d,v6.d[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x21                               \n\t" // Load address of C.
"                                            \n\t"
" st1 {v8.d}[0], [x27],x14                   \n\t" // Store c20  into quad and increment by rs_c.
" st1 {v8.d}[1], [x27],x14                   \n\t" // Store c21  into quad and increment by rs_c.
" st1 {v9.d}[0], [x27],x14                   \n\t" // Store c22  into quad and increment by rs_c.
" st1 {v9.d}[1], [x27],x14                   \n\t" // Store c23  into quad and increment by rs_c.
" st1 {v10.d}[0],[x27],x14                   \n\t" // Store c24  into quad and increment by rs_c.
" st1 {v10.d}[1],[x27],x14                   \n\t" // Store c25  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x22                               \n\t" // Load address of C.
"                                            \n\t"
" st1 {v11.d}[0],[x27],x14                   \n\t" // Store c30  into quad and increment by rs_c.
" st1 {v11.d}[1],[x27],x14                   \n\t" // Store c31  into quad and increment by rs_c.
" st1 {v12.d}[0],[x27],x14                   \n\t" // Store c32  into quad and increment by rs_c.
" st1 {v12.d}[1],[x27],x14                   \n\t" // Store c33  into quad and increment by rs_c.
" st1 {v13.d}[0],[x27],x14                   \n\t" // Store c34  into quad and increment by rs_c.
" st1 {v13.d}[1],[x27],x14                   \n\t" // Store c35  into quad and increment by rs_c.
"                                            \n\t"
" dup  v0.2d, xzr                            \n\t"
" dup  v1.2d, xzr                            \n\t"
" dup  v2.2d, xzr                            \n\t"
" dup  v3.2d, xzr                            \n\t"
" dup  v4.2d, xzr                            \n\t"
" dup  v5.2d, xzr                            \n\t"
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x23                               \n\t" // Load address of C.
"                                            \n\t"
" ld1 {v0.d}[0],[x27],x14                    \n\t" // Load c40  into quad and increment by rs_c.
" ld1 {v0.d}[1],[x27],x14                    \n\t" // Load c41  into quad and increment by rs_c.
" ld1 {v1.d}[0],[x27],x14                    \n\t" // Load c42  into quad and increment by rs_c.
" ld1 {v1.d}[1],[x27],x14                    \n\t" // Load c43  into quad and increment by rs_c.
" ld1 {v2.d}[0],[x27],x14                    \n\t" // Load c44  into quad and increment by rs_c.
" ld1 {v2.d}[1],[x27],x14                    \n\t" // Load c45  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x24                               \n\t" // Load address of C.
"                                            \n\t"
" ld1 {v3.d}[0],[x27],x14                    \n\t" // Load c50  into quad and increment by rs_c.
" ld1 {v3.d}[1],[x27],x14                    \n\t" // Load c51  into quad and increment by rs_c.
" ld1 {v4.d}[0],[x27],x14                    \n\t" // Load c52  into quad and increment by rs_c.
" ld1 {v4.d}[1],[x27],x14                    \n\t" // Load c53  into quad and increment by rs_c.
" ld1 {v5.d}[0],[x27],x14                    \n\t" // Load c54  into quad and increment by rs_c.
" ld1 {v5.d}[1],[x27],x14                    \n\t" // Load c55  into quad and increment by rs_c.
"                                            \n\t"
" fmul v0.2d,v0.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v1.2d,v1.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v2.2d,v2.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v3.2d,v3.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v4.2d,v4.2d,v7.d[0]                   \n\t" // Scale by beta
" fmul v5.2d,v5.2d,v7.d[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROGENSTOREDS3:                     \n\t"
"                                            \n\t"
" fmla v0.2d,v20.2d,v6.d[0]                  \n\t" // Scale by alpha
" fmla v1.2d,v21.2d,v6.d[0]                  \n\t" // Scale by alpha
" fmla v2.2d,v22.2d,v6.d[0]                  \n\t" // Scale by alpha
" fmla v3.2d,v23.2d,v6.d[0]                  \n\t" // Scale by alpha
" fmla v4.2d,v24.2d,v6.d[0]                  \n\t" // Scale by alpha
" fmla v5.2d,v25.2d,v6.d[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x23                               \n\t" // Load address of C.
"                                            \n\t"
" st1 {v0.d}[0],[x27],x14                    \n\t" // Store c40  into quad and increment by rs_c.
" st1 {v0.d}[1],[x27],x14                    \n\t" // Store c41  into quad and increment by rs_c.
" st1 {v1.d}[0],[x27],x14                    \n\t" // Store c42  into quad and increment by rs_c.
" st1 {v1.d}[1],[x27],x14                    \n\t" // Store c43  into quad and increment by rs_c.
" st1 {v2.d}[0],[x27],x14                    \n\t" // Store c44  into quad and increment by rs_c.
" st1 {v2.d}[1],[x27],x14                    \n\t" // Store c45  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x24                               \n\t" // Load address of C.
"                                            \n\t"
" st1 {v3.d}[0],[x27],x14                    \n\t" // Store c50  into quad and increment by rs_c.
" st1 {v3.d}[1],[x27],x14                    \n\t" // Store c51  into quad and increment by rs_c.
" st1 {v4.d}[0],[x27],x14                    \n\t" // Store c52  into quad and increment by rs_c.
" st1 {v4.d}[1],[x27],x14                    \n\t" // Store c53  into quad and increment by rs_c.
" st1 {v5.d}[0],[x27],x14                    \n\t" // Store c54  into quad and increment by rs_c.
" st1 {v5.d}[1],[x27],x14                    \n\t" // Store c55  into quad and increment by rs_c.
"                                            \n\t"
" dup  v8.2d, xzr                            \n\t"
" dup  v9.2d, xzr                            \n\t"
" dup  v10.2d, xzr                           \n\t"
" dup  v11.2d, xzr                           \n\t"
" dup  v12.2d, xzr                           \n\t"
" dup  v13.2d, xzr                           \n\t"
"                                            \n\t"
" fcmp d7,#0.0                               \n\t"
" beq .DBETAZEROGENSTOREDS4                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x25                               \n\t"
"                                            \n\t"
" ld1 {v8.d}[0], [x27],x14                   \n\t" // Load c60  into quad and increment by rs_c.
" ld1 {v8.d}[1], [x27],x14                   \n\t" // Load c61  into quad and increment by rs_c.
" ld1 {v9.d}[0], [x27],x14                   \n\t" // Load c62  into quad and increment by rs_c.
" ld1 {v9.d}[1], [x27],x14                   \n\t" // Load c63  into quad and increment by rs_c.
" ld1 {v10.d}[0],[x27],x14                   \n\t" // Load c64  into quad and increment by rs_c.
" ld1 {v10.d}[1],[x27],x14                   \n\t" // Load c65  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x26                               \n\t" // Load address of C.
"                                            \n\t"
" ld1 {v11.d}[0],[x27],x14                   \n\t" // Load c70  into quad and increment by rs_c.
" ld1 {v11.d}[1],[x27],x14                   \n\t" // Load c71  into quad and increment by rs_c.
" ld1 {v12.d}[0],[x27],x14                   \n\t" // Load c72  into quad and increment by rs_c.
" ld1 {v12.d}[1],[x27],x14                   \n\t" // Load c73  into quad and increment by rs_c.
" ld1 {v13.d}[0],[x27],x14                   \n\t" // Load c74  into quad and increment by rs_c.
" ld1 {v13.d}[1],[x27],x14                   \n\t" // Load c75  into quad and increment by rs_c.
"                                            \n\t"
" fmul v8.2d, v8.2d, v7.d[0]                 \n\t" // Scale by beta
" fmul v9.2d, v9.2d, v7.d[0]                 \n\t" // Scale by beta
" fmul v10.2d,v10.2d,v7.d[0]                 \n\t" // Scale by beta
" fmul v11.2d,v11.2d,v7.d[0]                 \n\t" // Scale by beta
" fmul v12.2d,v12.2d,v7.d[0]                 \n\t" // Scale by beta
" fmul v13.2d,v13.2d,v7.d[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .DBETAZEROGENSTOREDS4:                     \n\t"
"                                            \n\t"
" prfm pldl2keep,[x3]                        \n\t"
" prfm pldl2keep,[x4]                        \n\t"
"                                            \n\t"
" fmla v8.2d, v26.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v9.2d, v27.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v10.2d,v28.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v11.2d,v29.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v12.2d,v30.2d,v6.d[0]                 \n\t" // Scale by alpha
" fmla v13.2d,v31.2d,v6.d[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x25                               \n\t" // Load address of C.
"                                            \n\t"
" st1 {v8.d}[0], [x27],x14                   \n\t" // Store c60  into quad and increment by rs_c.
" st1 {v8.d}[1], [x27],x14                   \n\t" // Store c61  into quad and increment by rs_c.
" st1 {v9.d}[0], [x27],x14                   \n\t" // Store c62  into quad and increment by rs_c.
" st1 {v9.d}[1], [x27],x14                   \n\t" // Store c63  into quad and increment by rs_c.
" st1 {v10.d}[0],[x27],x14                   \n\t" // Store c64  into quad and increment by rs_c.
" st1 {v10.d}[1],[x27],x14                   \n\t" // Store c65  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x26                               \n\t" // Load address of C.
"                                            \n\t"
" st1 {v11.d}[0],[x27],x14                   \n\t" // Store c70  into quad and increment by rs_c.
" st1 {v11.d}[1],[x27],x14                   \n\t" // Store c71  into quad and increment by rs_c.
" st1 {v12.d}[0],[x27],x14                   \n\t" // Store c72  into quad and increment by rs_c.
" st1 {v12.d}[1],[x27],x14                   \n\t" // Store c73  into quad and increment by rs_c.
" st1 {v13.d}[0],[x27],x14                   \n\t" // Store c74  into quad and increment by rs_c.
" st1 {v13.d}[1],[x27],x14                   \n\t" // Store c75  into quad and increment by rs_c.
"                                            \n\t"
" .DEND:                                     \n\t" // Done!
"                                            \n\t"
:// output operands (none)
:// input operands
 [aaddr]  "m" (a),      // 0
 [baddr]  "m" (b),      // 1
 [caddr]  "m" (c),      // 2
 [k_iter] "m" (k_iter), // 3
 [k_left] "m" (k_left), // 4
 [alpha]  "m" (alpha),  // 5
 [beta]   "m" (beta),   // 6
 [rs_c]   "m" (rs_c),   // 6
 [cs_c]   "m" (cs_c),   // 7
 [a_next] "m" (a_next), // 8
 [b_next] "m" (b_next)  // 9
:// Register clobber list
 "x0","x1","x2","x3",
 "x4","x5","x6",
 "x7","x8","x9",
 "x10","x11","x12","x13","x14","x16","x17",
 "x20","x21","x22","x23","x24","x25","x26",
 "x27",       
 "v0","v1","v2",
 "v3","v4","v5",
 "v6","v7","v8",
 "v9","v10","v11",
 "v12","v13","v14",
 "v15","v16","v17","v18","v19",
 "v20","v21","v22","v23",
 "v24","v25","v26","v27",
 "v28","v29","v30","v31"
);



}

//void bli_cgemm_opt_4x4(
//                        dim_t              k,
//                        scomplex* restrict alpha,
//                        scomplex* restrict a,
//                        scomplex* restrict b,
//                        scomplex* restrict beta,
//                        scomplex* restrict c, inc_t rs_c, inc_t cs_c,
//                        auxinfo_t*         data
//                      )
//{
//	/* Just call the reference implementation. */
//	BLIS_CGEMM_UKERNEL_REF( k,
//	                   alpha,
//	                   a,
//	                   b,
//	                   beta,
//	                   c, rs_c, cs_c,
//	                   data );
//}
//
//void bli_zgemm_opt_4x4(
//                        dim_t              k,
//                        dcomplex* restrict alpha,
//                        dcomplex* restrict a,
//                        dcomplex* restrict b,
//                        dcomplex* restrict beta,
//                        dcomplex* restrict c, inc_t rs_c, inc_t cs_c,
//                        auxinfo_t*         data
//                      )
//{
//	/* Just call the reference implementation. */
//	BLIS_ZGEMM_UKERNEL_REF( k,
//	                   alpha,
//	                   a,
//	                   b,
//	                   beta,
//	                   c, rs_c, cs_c,
//	                   data );
//}
//
