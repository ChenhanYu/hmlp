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

//#define dim_t std::size_t
//#define int_t std::size_t
typedef std::size_t dim_t;
typedef std::size_t inc_t;


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

  // For gsknn
  TV *D;

  int *I;

  int ldr;

  // index for gkmx to access data in the closure of opkernel and opreduce.
  int i;

  int j;

  int b;

  // edge case problem size
  int ib;

  int jb;

  // whether this is the first rank-k update.
  int pc;

  int do_packC;

  int ldc;
};


#endif // define HMLP_INTERNAL_HPP
