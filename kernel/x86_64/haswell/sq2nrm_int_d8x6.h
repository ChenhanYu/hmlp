
  // Prefetch aa and bb
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );

  double neg2 = -2.0;
  double dzero = 0.0;

  // Scale -2
  a03.v   = _mm256_broadcast_sd( &neg2 );
  c03_0.v = _mm256_mul_pd( a03.v, c03_0.v );
  c03_1.v = _mm256_mul_pd( a03.v, c03_1.v );
  c03_2.v = _mm256_mul_pd( a03.v, c03_2.v );
  c03_3.v = _mm256_mul_pd( a03.v, c03_3.v );
  c03_4.v = _mm256_mul_pd( a03.v, c03_4.v );
  c03_5.v = _mm256_mul_pd( a03.v, c03_5.v );

  c47_0.v = _mm256_mul_pd( a03.v, c47_0.v );
  c47_1.v = _mm256_mul_pd( a03.v, c47_1.v );
  c47_2.v = _mm256_mul_pd( a03.v, c47_2.v );
  c47_3.v = _mm256_mul_pd( a03.v, c47_3.v );
  c47_4.v = _mm256_mul_pd( a03.v, c47_4.v );
  c47_5.v = _mm256_mul_pd( a03.v, c47_5.v );

  a03.v   = _mm256_load_pd( (double*)aa );
  c03_0.v = _mm256_add_pd( a03.v, c03_0.v );
  c03_1.v = _mm256_add_pd( a03.v, c03_1.v );
  c03_2.v = _mm256_add_pd( a03.v, c03_2.v );
  c03_3.v = _mm256_add_pd( a03.v, c03_3.v );
  c03_4.v = _mm256_add_pd( a03.v, c03_4.v );
  c03_5.v = _mm256_add_pd( a03.v, c03_5.v );

  a47.v   = _mm256_load_pd( (double*)( aa + 4 ) );
  c47_0.v = _mm256_add_pd( a47.v, c47_0.v );
  c47_1.v = _mm256_add_pd( a47.v, c47_1.v );
  c47_2.v = _mm256_add_pd( a47.v, c47_2.v );
  c47_3.v = _mm256_add_pd( a47.v, c47_3.v );
  c47_4.v = _mm256_add_pd( a47.v, c47_4.v );
  c47_5.v = _mm256_add_pd( a47.v, c47_5.v );
  
  b0.v    = _mm256_broadcast_sd( (double*)( bb     ) );
  c03_0.v = _mm256_add_pd( b0.v, c03_0.v );
  c47_0.v = _mm256_add_pd( b0.v, c47_0.v );

  b1.v    = _mm256_broadcast_sd( (double*)( bb + 1 ) );
  c03_1.v = _mm256_add_pd( b1.v, c03_1.v );
  c47_1.v = _mm256_add_pd( b1.v, c47_1.v );

  b0.v    = _mm256_broadcast_sd( (double*)( bb + 2 ) );
  c03_2.v = _mm256_add_pd( b0.v, c03_2.v );
  c47_2.v = _mm256_add_pd( b0.v, c47_2.v );

  b1.v    = _mm256_broadcast_sd( (double*)( bb + 3 ) );
  c03_3.v = _mm256_add_pd( b1.v, c03_3.v );
  c47_3.v = _mm256_add_pd( b1.v, c47_3.v );

  b0.v    = _mm256_broadcast_sd( (double*)( bb + 4 ) );
  c03_4.v = _mm256_add_pd( b0.v, c03_4.v );
  c47_4.v = _mm256_add_pd( b0.v, c47_4.v );

  b1.v    = _mm256_broadcast_sd( (double*)( bb + 5 ) );
  c03_5.v = _mm256_add_pd( b1.v, c03_5.v );
  c47_5.v = _mm256_add_pd( b1.v, c47_5.v );

  // Check if there is any illegle value 
  a03.v   = _mm256_broadcast_sd( &dzero );
  c03_0.v = _mm256_max_pd( a03.v, c03_0.v );
  c03_1.v = _mm256_max_pd( a03.v, c03_1.v );
  c03_2.v = _mm256_max_pd( a03.v, c03_2.v );
  c03_3.v = _mm256_max_pd( a03.v, c03_3.v );
  c03_4.v = _mm256_max_pd( a03.v, c03_4.v );
  c03_5.v = _mm256_max_pd( a03.v, c03_5.v );

  c47_0.v = _mm256_max_pd( a03.v, c47_0.v );
  c47_1.v = _mm256_max_pd( a03.v, c47_1.v );
  c47_2.v = _mm256_max_pd( a03.v, c47_2.v );
  c47_3.v = _mm256_max_pd( a03.v, c47_3.v );
  c47_4.v = _mm256_max_pd( a03.v, c47_4.v );
  c47_5.v = _mm256_max_pd( a03.v, c47_5.v );

