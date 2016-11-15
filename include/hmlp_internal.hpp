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
