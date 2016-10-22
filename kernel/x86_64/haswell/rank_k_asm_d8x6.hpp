/*
 * This file is modified and redistribued from 
 * 
 * BLIS
 * An object-based framework for developing high-performance BLAS-like
 * libraries.
 *
 * Copyright (C) 2014, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *  - Neither the name of The University of Texas at Austin nor the names
 *    of its contributors may be used to endorse or promote products
 *    derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *
 * rank_k_asm_d8x4.hpp
 * 
 * Mofidifier:
 * Chenhan D. Yu - Department of Computer Science, 
 *                 The University of Texas at Austin
 *
 *
 * Purpose: 
 *
 *
 * Todo:
 *
 *
 * Modification:
 * 
 * Chenhan
 * Feb 01, 2015: This file is extracted from bli_gemm_asm_d8x4.c in 
 *               the Sandy-Bridge AVX micro-kernel directory of BLIS. 
 *               The double precision rank-k update with a typical mc leading 
 *               is kept in this file to work as a stand a long micro-kernel
 *               in GSKS. This variant compute C = A*B if pc = 0, otherwise
 *               C += A*B is computed.
 *
 *
 *
 * */

#include <stdio.h>
#include <immintrin.h> // AVX

#include <hmlp_internal.hxx>
#include <avx_type.h> // self-defined vector type


struct rank_k_asm_d8x6 
{
  inline void operator()( 
      int k, 
      double *a, 
      double *b, 
      double *c, int ldc, 
      aux_s<double, double, double, double> *aux ) const 
  {

    unsigned long long k_iter = k / 4;
    unsigned long long k_left = k % 4;
    unsigned long long pc     = aux->pc;

    __asm__ volatile
    (
    "                                            \n\t"
    "                                            \n\t"
    "movq                %2, %%rax               \n\t" // load address of a.              ( v )
    "movq                %3, %%rbx               \n\t" // load address of b.              ( v )
    "movq                %5, %%r15               \n\t" // load address of b_next.         ( v )
    "addq           $32 * 4, %%rax               \n\t"
    "                                            \n\t" // initialize loop by pre-loading
    "vmovaps           -4 * 32(%%rax), %%ymm0    \n\t" // a03
    "vmovaps           -3 * 32(%%rax), %%ymm1    \n\t" // a47
      "prefetcht2    0 * 8(%%r15)                  \n\t"
    "                                            \n\t"
    "movq                %4, %%rcx               \n\t" // load address of c
    "                                            \n\t"
    "prefetcht0    7 * 8(%%rcx)                  \n\t" // prefetch c                     
    "prefetcht0   15 * 8(%%rcx)                  \n\t" 
    "prefetcht0   23 * 8(%%rcx)                  \n\t" 
    "prefetcht0   31 * 8(%%rcx)                  \n\t" 
      "prefetcht0   39 * 8(%%rcx)                  \n\t"
      "prefetcht0   47 * 8(%%rcx)                  \n\t"
    "                                            \n\t"
    "vxorpd    %%ymm4,  %%ymm4,  %%ymm4          \n\t" // Zero out c03_: c47_:
    "vxorpd    %%ymm5,  %%ymm5,  %%ymm5          \n\t"
    "vxorpd    %%ymm6,  %%ymm6,  %%ymm6          \n\t"
    "vxorpd    %%ymm7,  %%ymm7,  %%ymm7          \n\t"
    "vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \n\t" 
    "vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
    "vxorpd    %%ymm10, %%ymm10, %%ymm10         \n\t"
    "vxorpd    %%ymm11, %%ymm11, %%ymm11         \n\t"
    "vxorpd    %%ymm12, %%ymm12, %%ymm12         \n\t"
    "vxorpd    %%ymm13, %%ymm13, %%ymm13         \n\t"
    "vxorpd    %%ymm14, %%ymm14, %%ymm14         \n\t"
    "vxorpd    %%ymm15, %%ymm15, %%ymm15         \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq      %0, %%rsi                         \n\t" // i = k_iter;                     ( v )
    "testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.        ( v )
    "je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that    ( v )
    "                                            \n\t" // contains the k_left loop.
    "                                            \n\t"
    "                                            \n\t"
    ".DLOOPKITER:                                \n\t" // MAIN LOOP
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 0
    "prefetcht0  16 * 32(%%rax)                  \n\t"
    "                                            \n\t"
    "vbroadcastsd       0 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd       1 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd       2 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd       3 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd       4 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd       5 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "vmovaps           -2 * 32(%%rax), %%ymm0    \n\t"
    "vmovaps           -1 * 32(%%rax), %%ymm1    \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 1
    "vbroadcastsd       6 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd       7 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd       8 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd       9 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      10 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd      11 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "vmovaps            0 * 32(%%rax), %%ymm0    \n\t"
    "vmovaps            1 * 32(%%rax), %%ymm1    \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 2
    "prefetcht0  20 * 32(%%rax)                  \n\t"
    "                                            \n\t"
    "vbroadcastsd      12 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd      13 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd      14 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd      15 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      16 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd      17 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "vmovaps            2 * 32(%%rax), %%ymm0    \n\t"
    "vmovaps            3 * 32(%%rax), %%ymm1    \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t" // iteration 3
    "vbroadcastsd      18 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd      19 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd      20 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd      21 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      22 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd      23 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "addq           $4 * 8 * 8, %%rax            \n\t" // a += 4*8 (unroll x mr)
    "addq           $4 * 6 * 8, %%rbx            \n\t" // b += 4*6 (unroll x nr)
    "                                            \n\t"
    "vmovaps           -4 * 32(%%rax), %%ymm0    \n\t"
    "vmovaps           -3 * 32(%%rax), %%ymm1    \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "decq   %%rsi                                \n\t" // i -= 1;
    "jne    .DLOOPKITER                          \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DCONSIDKLEFT:                              \n\t"
    "                                            \n\t"
    "movq      %1, %%rsi                         \n\t" // i = k_left;
    "testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
    "je     .DPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
    "                                            \n\t" // else, we prepare to enter k_left loop.
    "                                            \n\t"
    "                                            \n\t"
    ".DLOOPKLEFT:                                \n\t" // EDGE LOOP
    "                                            \n\t"
    "prefetcht0  16 * 32(%%rax)                  \n\t"
    "                                            \n\t"
    "vbroadcastsd       0 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd       1 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd       2 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd       3 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd       4 *  8(%%rbx), %%ymm2    \n\t"
    "vbroadcastsd       5 *  8(%%rbx), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "addq           $1 * 8 * 8, %%rax            \n\t" // a += 1*8 (unroll x mr)
    "addq           $1 * 6 * 8, %%rbx            \n\t" // b += 1*6 (unroll x nr)
    "                                            \n\t"
    "vmovaps           -4 * 32(%%rax), %%ymm0    \n\t"
    "vmovaps           -3 * 32(%%rax), %%ymm1    \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "decq   %%rsi                                \n\t" // i -= 1;
    "jne    .DLOOPKLEFT                          \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "                                            \n\t"
    ".DPOSTACCUM:                                \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "movq      %6, %%rdi                         \n\t" // load pc
    "testq  %%rdi, %%rdi                         \n\t" // check pc via logical AND.        ( v )
    "je     .DNOLOADC                            \n\t" // if pc == 0, jump to code
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    0 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = C_c( 0:3, 0 )
    "vaddpd            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // ymm0 += ymm4
    "vmovapd           %%ymm0,  0 * 32(%%rcx)    \n\t" // C_c( 0:3, 0 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    1 * 32(%%rcx),  %%ymm1           \n\t" // ymm0 = C_c( 4:7, 0 )
    "vaddpd            %%ymm5,  %%ymm1,  %%ymm1  \n\t" // ymm0 += ymm5
    "vmovapd           %%ymm1,  1 * 32(%%rcx)    \n\t" // C_c( 4:7, 0 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    2 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = C_c( 0:3, 1 )
    "vaddpd            %%ymm6,  %%ymm0,  %%ymm0  \n\t" // ymm0 += ymm6
    "vmovapd           %%ymm0,  2 * 32(%%rcx)    \n\t" // C_c( 0:3, 1 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    3 * 32(%%rcx),  %%ymm1           \n\t" // ymm0 = C_c( 4:7, 1 )
    "vaddpd            %%ymm7,  %%ymm1,  %%ymm1  \n\t" // ymm0 += ymm7
    "vmovapd           %%ymm1,  3 * 32 (%%rcx)   \n\t" // C_c( 4:7, 1 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    4 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = C_c( 0:3, 2 )
    "vaddpd            %%ymm8,  %%ymm0,  %%ymm0  \n\t" // ymm0 += ymm8
    "vmovapd           %%ymm0,  4 * 32(%%rcx)    \n\t" // C_c( 0:3, 2 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    5 * 32(%%rcx),  %%ymm1           \n\t" // ymm0 = C_c( 4:7, 2 )
    "vaddpd            %%ymm9,  %%ymm1,  %%ymm1  \n\t" // ymm0 += ymm9
    "vmovapd           %%ymm1,  5 * 32(%%rcx)    \n\t" // C_c( 4:7, 2 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    6 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = C_c( 0:3, 3 )
    "vaddpd            %%ymm10, %%ymm0,  %%ymm0  \n\t" // ymm0 += ymm10
    "vmovapd           %%ymm0,  6 * 32(%%rcx)    \n\t" // C_c( 0:3, 3 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    7 * 32(%%rcx),  %%ymm1           \n\t" // ymm0 = C_c( 4:7, 3 )
    "vaddpd            %%ymm11, %%ymm1,  %%ymm1  \n\t" // ymm0 += ymm11
    "vmovapd           %%ymm1,  7 * 32(%%rcx)    \n\t" // C_c( 4:7, 3 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    8 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = C_c( 0:3, 3 )
    "vaddpd            %%ymm12, %%ymm0,  %%ymm0  \n\t" // ymm0 += ymm12
    "vmovapd           %%ymm0,  8 * 32(%%rcx)    \n\t" // C_c( 0:3, 5 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd    9 * 32(%%rcx),  %%ymm1           \n\t" // ymm0 = C_c( 4:7, 3 )
    "vaddpd            %%ymm13, %%ymm1,  %%ymm1  \n\t" // ymm0 += ymm13
    "vmovapd           %%ymm1,  9 * 32(%%rcx)    \n\t" // C_c( 4:7, 5 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd   10 * 32(%%rcx),  %%ymm0           \n\t" // ymm0 = C_c( 0:3, 3 )
    "vaddpd            %%ymm14, %%ymm0,  %%ymm0  \n\t" // ymm0 += ymm14
    "vmovapd           %%ymm0, 10 * 32(%%rcx)    \n\t" // C_c( 0:3, 6 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"
    "vmovapd   11 * 32(%%rcx),  %%ymm1           \n\t" // ymm0 = C_c( 4:7, 3 )
    "vaddpd            %%ymm15, %%ymm1,  %%ymm1  \n\t" // ymm0 += ymm11
    "vmovapd           %%ymm1, 11 * 32(%%rcx)    \n\t" // C_c( 4:7, 6 ) = ymm0
    "                                            \n\t"
    "                                            \n\t"		
    "jmp    .DDONE                               \n\t"
    "                                            \n\t"
    "                                            \n\t"
    "                                            \n\t"
    ".DNOLOADC:                                  \n\t"
    "vmovapd           %%ymm4,   0 * 32(%%rcx)   \n\t" // C_c( 0:3, 0 ) = ymm4
    "vmovapd           %%ymm5,   1 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm6,   2 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm7,   3 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm8,   4 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm9,   5 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm10,  6 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm11,  7 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm12,  8 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm13,  9 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm14, 10 * 32(%%rcx)   \n\t" 
    "vmovapd           %%ymm15, 11 * 32(%%rcx)   \n\t" 				
    "                                            \n\t"
    "                                            \n\t"
    ".DDONE:                                     \n\t"
    "                                            \n\t"
    : // output operands (none)
    : // input operands
      "m" (k_iter),      // 0
      "m" (k_left),      // 1
      "m" (a),           // 2
      "m" (b),           // 3
      "m" (c),           // 4
      "m" (aux->b_next), // 5
        "m" (pc)           // 6
    : // register clobber list
      "rax", "rbx", "rcx", "rsi", "rdi",
        "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
      );
  }

  //printf( "ldc = %d\n", ldc );
  //printf( "%lf, %lf, %lf, %lf\n", c[0], c[ ldc + 0], c[ ldc * 2 + 0], c[ ldc * 3 + 0] );
  //printf( "%lf, %lf, %lf, %lf\n", c[1], c[ ldc + 1], c[ ldc * 2 + 1], c[ ldc * 3 + 1] );
  //printf( "%lf, %lf, %lf, %lf\n", c[2], c[ ldc + 2], c[ ldc * 2 + 2], c[ ldc * 3 + 2] );
  //printf( "%lf, %lf, %lf, %lf\n", c[3], c[ ldc + 3], c[ ldc * 2 + 3], c[ ldc * 3 + 3] );
  //printf( "%lf, %lf, %lf, %lf\n", c[4], c[ ldc + 4], c[ ldc * 2 + 4], c[ ldc * 3 + 4] );
  //printf( "%lf, %lf, %lf, %lf\n", c[5], c[ ldc + 5], c[ ldc * 2 + 5], c[ ldc * 3 + 5] );
  //printf( "%lf, %lf, %lf, %lf\n", c[6], c[ ldc + 6], c[ ldc * 2 + 6], c[ ldc * 3 + 6] );
  //printf( "%lf, %lf, %lf, %lf\n", c[7], c[ ldc + 7], c[ ldc * 2 + 7], c[ ldc * 3 + 7] );
};
