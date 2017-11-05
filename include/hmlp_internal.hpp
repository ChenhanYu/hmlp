/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  



#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <stdint.h>

#ifndef HMLP_INTERNAL_HPP
#define HMLP_INTERNAL_HPP

#define restrict __restrict__


typedef unsigned long long dim_t;
typedef unsigned long long inc_t;

template<typename TA, typename TB, typename TC, typename TV>
struct aux_s
{
  TA *a_next;

  TB *b_next;

  TC *c_buff;
   
  TV *V;

  int ldv;

  // For gsks

  TV *hi;

  TV *hj;

//  // For gsknn
//  TV *D;
//
//  int *I;
//
//  int ldr;

  // index for gkmx to access data in the closure of opkernel and opreduce.
  int i;

  int j;

  int b;

  // edge case problem size
  int ib;

  int jb;

  int m;

  int n;

  // whether this is the first rank-k update.
  int pc;

  int do_packC;

  int ldc;
};




#define BLIS_GEMM_KERNEL(name,type)    \
  void name                            \
  (                                    \
    dim_t             k,               \
    type*    restrict alpha,           \
    type*    restrict a,               \
    type*    restrict b,               \
    type*    restrict beta,            \
    type*    restrict c,               \
    inc_t rs_c, inc_t cs_c,            \
    aux_s<type, type, type, type> *aux \
  )                                    \

#define GEMM_OPERATOR(type)            \
  void operator()                      \
  (                                    \
    dim_t k,                           \
    type *a,                           \
    type *b,                           \
    type *c, inc_t rs_c, inc_t cs_c,   \
    aux_s<type, type, type, type> *aux \
  )                                    \

#define STRA_OPERATOR(type)            \
  void operator()                      \
  (                                    \
    int k,                             \
    type *a,                           \
    type *b,                           \
    int len,                           \
    type **c_list, int ldc,            \
    type *alpha_list,                  \
    aux_s<type, type, type, type> *aux \
  )                                    \

#define GSKS_OPERATOR(type)            \
  void operator()                      \
  (                                    \
    kernel_s<type> *ker,               \
    int k,                             \
    int rhs,                           \
    type *u,                           \
    type *a, type *aa,                 \
    type *b, type *bb,                 \
    type *w,                           \
    type *c, int ldc,                  \
    aux_s<type, type, type, type> *aux \
  )                                    \

#define GSKNN_OPERATOR(type)           \
  void operator()                      \
  (                                    \
    kernel_s<type> *ker,               \
    int k,                             \
    int r,                             \
    type *a, type *aa,                 \
    type *b, type *bb,                 \
    type *c,                           \
    aux_s<type, type, type, type> *aux \
    int *bmap                          \
  )                                    \




#endif /** define HMLP_INTERNAL_HPP */
