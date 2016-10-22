  //for ( i = 0; i < rhs; i ++ ) {
  //  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( u + 8 ) );
  //  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( w + 6 ) );

    b0.v    = _mm256_broadcast_sd( w      );
    b1.v    = _mm256_broadcast_sd( w +  1 );
    a03.v = _mm256_fmadd_pd( c03_0.v, b0.v, a03.v );
    a47.v = _mm256_fmadd_pd( c47_0.v, b0.v, a47.v );
    a03.v = _mm256_fmadd_pd( c03_1.v, b1.v, a03.v );
    a47.v = _mm256_fmadd_pd( c47_1.v, b1.v, a47.v );

    b0.v    = _mm256_broadcast_sd( w +  2  );
    b1.v    = _mm256_broadcast_sd( w +  3 );
    a03.v = _mm256_fmadd_pd( c03_2.v, b0.v, a03.v );
    a47.v = _mm256_fmadd_pd( c47_2.v, b0.v, a47.v );
    a03.v = _mm256_fmadd_pd( c03_3.v, b1.v, a03.v );
    a47.v = _mm256_fmadd_pd( c47_3.v, b1.v, a47.v );

    b0.v    = _mm256_broadcast_sd( w +  4  );
    b1.v    = _mm256_broadcast_sd( w +  5 );
    a03.v = _mm256_fmadd_pd( c03_4.v, b0.v, a03.v );
    a47.v = _mm256_fmadd_pd( c47_4.v, b0.v, a47.v );
    a03.v = _mm256_fmadd_pd( c03_5.v, b1.v, a03.v );
    a47.v = _mm256_fmadd_pd( c47_5.v, b1.v, a47.v );

    _mm256_store_pd( u     , a03.v );
    _mm256_store_pd( u + 4 , a47.v );
  //  a03.v    = _mm256_load_pd( (double*)( u +  8 ) );
  //  a47.v    = _mm256_load_pd( (double*)( u + 12 ) );

  //  u += 8;
  //  w += 6;
  //}

// end weighted_sum_int_d8x6
