// begin ks_kernel_summation_int_d8x4

  rhs_left = rhs % 2;
  rhs      = rhs / 2;

  //printf( "rhs: %d, rhs_left: %d\n", rhs, rhs_left );

  for ( i = 0; i < rhs; i ++ ) {
    A03.v    = _mm256_load_pd( u +  8 );
    A47.v    = _mm256_load_pd( u + 12 );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 16 ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w + 4 ) );

    w_tmp.v  = _mm256_broadcast_sd( (double*)w );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_0.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_0.v );
    u03.v    = _mm256_add_pd( u03.v, a03.v );
    u47.v    = _mm256_add_pd( u47.v, a47.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 1 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_1.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_1.v );
    u03.v    = _mm256_add_pd( u03.v, a03.v );
    u47.v    = _mm256_add_pd( u47.v, a47.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 2 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_2.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_2.v );
    u03.v    = _mm256_add_pd( u03.v, a03.v );
    u47.v    = _mm256_add_pd( u47.v, a47.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 3 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_3.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_3.v );
    u03.v    = _mm256_add_pd( u03.v, a03.v );
    u47.v    = _mm256_add_pd( u47.v, a47.v );

    _mm256_store_pd( u     , u03.v );
    _mm256_store_pd( u + 4 , u47.v );
    u03.v    = _mm256_load_pd( u + 16 );
    u47.v    = _mm256_load_pd( u + 20 );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 24 ) );
    __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w + 8 ) );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 4 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_0.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_0.v );
    A03.v    = _mm256_add_pd( A03.v, a03.v );
    A47.v    = _mm256_add_pd( A47.v, a47.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 5 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_1.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_1.v );
    A03.v    = _mm256_add_pd( A03.v, a03.v );
    A47.v    = _mm256_add_pd( A47.v, a47.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 6 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_2.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_2.v );
    A03.v    = _mm256_add_pd( A03.v, a03.v );
    A47.v    = _mm256_add_pd( A47.v, a47.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 7 ) );
    a03.v    = _mm256_mul_pd( w_tmp.v, c03_3.v );
    a47.v    = _mm256_mul_pd( w_tmp.v, c47_3.v );
    A03.v    = _mm256_add_pd( A03.v, a03.v );
    A47.v    = _mm256_add_pd( A47.v, a47.v );

    _mm256_store_pd( u +  8, A03.v );
    _mm256_store_pd( u + 12, A47.v );

    u += 16;
    w += 8;
  }

  
  if ( rhs_left ) {
    w_tmp.v  = _mm256_broadcast_sd( (double*)w );
    c03_0.v  = _mm256_mul_pd( w_tmp.v, c03_0.v );
    c47_0.v  = _mm256_mul_pd( w_tmp.v, c47_0.v );
    u03.v    = _mm256_add_pd( u03.v, c03_0.v );
    u47.v    = _mm256_add_pd( u47.v, c47_0.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 1 ) );
    c03_1.v  = _mm256_mul_pd( w_tmp.v, c03_1.v );
    c47_1.v  = _mm256_mul_pd( w_tmp.v, c47_1.v );
    u03.v    = _mm256_add_pd( u03.v, c03_1.v );
    u47.v    = _mm256_add_pd( u47.v, c47_1.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 2 ) );
    c03_2.v    = _mm256_mul_pd( w_tmp.v, c03_2.v );
    c47_2.v    = _mm256_mul_pd( w_tmp.v, c47_2.v );
    u03.v    = _mm256_add_pd( u03.v, c03_2.v );
    u47.v    = _mm256_add_pd( u47.v, c47_2.v );

    w_tmp.v  = _mm256_broadcast_sd( (double*)( w + 3 ) );
    c03_3.v    = _mm256_mul_pd( w_tmp.v, c03_3.v );
    c47_3.v    = _mm256_mul_pd( w_tmp.v, c47_3.v );
    u03.v    = _mm256_add_pd( u03.v, c03_3.v );
    u47.v    = _mm256_add_pd( u47.v, c47_3.v );

    _mm256_store_pd( u     , u03.v );
    _mm256_store_pd( u + 4 , u47.v );
  }

// end ks_kernel_summation_int_d8x4
