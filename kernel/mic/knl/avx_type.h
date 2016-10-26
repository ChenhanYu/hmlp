#ifndef AVX_TYPE_H
#define AVX_TYPE_H

typedef union 
{
  __m512d v;
  double d[ 8 ];
  __m512i i;
  unsigned long long u[ 8 ];
  unsigned long u32[ 16 ];
} v8df_t;

typedef union 
{
  __m256d v;
  double d[ 4 ];
  __m256i i;
  unsigned long long u[ 4 ];
} v4df_t;

typedef union 
{
  __m128i v;
  int d[ 4 ];
} v4li_t;

#endif // define AVX_TYPE_H
