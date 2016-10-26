#ifndef HMLP_INTERNAL_HPP
#define HMLP_INTERNAL_HPP

template<typename TA, typename TB, typename TC, typename TV>
struct aux_s 
{
  TA *a_next;
  TB *b_next;
  TC *c_buff;
  int pc;
  int do_packC;
};

#endif // define HMLP_INTERNAL_HPP
