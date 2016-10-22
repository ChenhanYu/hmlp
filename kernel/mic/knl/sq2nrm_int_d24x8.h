
  // Prefetch aa and bb
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );

  double neg2 = -2.0;
  double dzero = 0.0;

  // Scale -2
  a07.v   = _mm512_set1_pd( neg2 );

  c07_0.v = _mm512_mul_pd( a07.v, c07_0.v );
  c07_1.v = _mm512_mul_pd( a07.v, c07_1.v );
  c07_2.v = _mm512_mul_pd( a07.v, c07_2.v );
  c07_3.v = _mm512_mul_pd( a07.v, c07_3.v );
  c07_4.v = _mm512_mul_pd( a07.v, c07_4.v );
  c07_5.v = _mm512_mul_pd( a07.v, c07_5.v );
  c07_6.v = _mm512_mul_pd( a07.v, c07_6.v );
  c07_7.v = _mm512_mul_pd( a07.v, c07_7.v );

  c15_0.v = _mm512_mul_pd( a07.v, c15_0.v );
  c15_1.v = _mm512_mul_pd( a07.v, c15_1.v );
  c15_2.v = _mm512_mul_pd( a07.v, c15_2.v );
  c15_3.v = _mm512_mul_pd( a07.v, c15_3.v );
  c15_4.v = _mm512_mul_pd( a07.v, c15_4.v );
  c15_5.v = _mm512_mul_pd( a07.v, c15_5.v );
  c15_6.v = _mm512_mul_pd( a07.v, c15_6.v );
  c15_7.v = _mm512_mul_pd( a07.v, c15_7.v );

  c23_0.v = _mm512_mul_pd( a07.v, c23_0.v );
  c23_1.v = _mm512_mul_pd( a07.v, c23_1.v );
  c23_2.v = _mm512_mul_pd( a07.v, c23_2.v );
  c23_3.v = _mm512_mul_pd( a07.v, c23_3.v );
  c23_4.v = _mm512_mul_pd( a07.v, c23_4.v );
  c23_5.v = _mm512_mul_pd( a07.v, c23_5.v );
  c23_6.v = _mm512_mul_pd( a07.v, c23_6.v );
  c23_7.v = _mm512_mul_pd( a07.v, c23_7.v );

  a07.v   = _mm512_load_pd( aa );

  c07_0.v = _mm512_add_pd( a07.v, c07_0.v );
  c07_1.v = _mm512_add_pd( a07.v, c07_1.v );
  c07_2.v = _mm512_add_pd( a07.v, c07_2.v );
  c07_3.v = _mm512_add_pd( a07.v, c07_3.v );
  c07_4.v = _mm512_add_pd( a07.v, c07_4.v );
  c07_5.v = _mm512_add_pd( a07.v, c07_5.v );
  c07_6.v = _mm512_add_pd( a07.v, c07_6.v );
  c07_7.v = _mm512_add_pd( a07.v, c07_7.v );

  a15.v   = _mm512_load_pd( aa + 8 );

  c15_0.v = _mm512_add_pd( a15.v, c15_0.v );
  c15_1.v = _mm512_add_pd( a15.v, c15_1.v );
  c15_2.v = _mm512_add_pd( a15.v, c15_2.v );
  c15_3.v = _mm512_add_pd( a15.v, c15_3.v );
  c15_4.v = _mm512_add_pd( a15.v, c15_4.v );
  c15_5.v = _mm512_add_pd( a15.v, c15_5.v );
  c15_6.v = _mm512_add_pd( a15.v, c15_6.v );
  c15_7.v = _mm512_add_pd( a15.v, c15_7.v );

  a23.v   = _mm512_load_pd( aa + 16 );

  c23_0.v = _mm512_add_pd( a23.v, c23_0.v );
  c23_1.v = _mm512_add_pd( a23.v, c23_1.v );
  c23_2.v = _mm512_add_pd( a23.v, c23_2.v );
  c23_3.v = _mm512_add_pd( a23.v, c23_3.v );
  c23_4.v = _mm512_add_pd( a23.v, c23_4.v );
  c23_5.v = _mm512_add_pd( a23.v, c23_5.v );
  c23_6.v = _mm512_add_pd( a23.v, c23_6.v );
  c23_7.v = _mm512_add_pd( a23.v, c23_7.v );


  b0.v    = _mm512_set1_pd( bb[ 0 ] );
  c07_0.v = _mm512_add_pd( b0.v, c07_0.v );
  c15_0.v = _mm512_add_pd( b0.v, c15_0.v );
  c23_0.v = _mm512_add_pd( b0.v, c23_0.v );

  b1.v    = _mm512_set1_pd( bb[ 1 ] );
  c07_1.v = _mm512_add_pd( b1.v, c07_1.v );
  c15_1.v = _mm512_add_pd( b1.v, c15_1.v );
  c23_1.v = _mm512_add_pd( b1.v, c23_1.v );

  b0.v    = _mm512_set1_pd( bb[ 2 ] );
  c07_2.v = _mm512_add_pd( b0.v, c07_2.v );
  c15_2.v = _mm512_add_pd( b0.v, c15_2.v );
  c23_2.v = _mm512_add_pd( b0.v, c23_2.v );

  b1.v    = _mm512_set1_pd( bb[ 3 ] );
  c07_3.v = _mm512_add_pd( b1.v, c07_3.v );
  c15_3.v = _mm512_add_pd( b1.v, c15_3.v );
  c23_3.v = _mm512_add_pd( b1.v, c23_3.v );

  b0.v    = _mm512_set1_pd( bb[ 4 ] );
  c07_4.v = _mm512_add_pd( b0.v, c07_4.v );
  c15_4.v = _mm512_add_pd( b0.v, c15_4.v );
  c23_4.v = _mm512_add_pd( b0.v, c23_4.v );

  b1.v    = _mm512_set1_pd( bb[ 5 ] );
  c07_5.v = _mm512_add_pd( b1.v, c07_5.v );
  c15_5.v = _mm512_add_pd( b1.v, c15_5.v );
  c23_5.v = _mm512_add_pd( b1.v, c23_5.v );

  b0.v    = _mm512_set1_pd( bb[ 6 ] );
  c07_6.v = _mm512_add_pd( b0.v, c07_6.v );
  c15_6.v = _mm512_add_pd( b0.v, c15_6.v );
  c23_6.v = _mm512_add_pd( b0.v, c23_6.v );

  b1.v    = _mm512_set1_pd( bb[ 7 ] );
  c07_7.v = _mm512_add_pd( b1.v, c07_7.v );
  c15_7.v = _mm512_add_pd( b1.v, c15_7.v );
  c23_7.v = _mm512_add_pd( b1.v, c23_7.v );


  // Check if there is any illegle value 
  a07.v   = _mm512_set1_pd( dzero );

  c07_0.v = _mm512_max_pd( a07.v, c07_0.v );
  c07_1.v = _mm512_max_pd( a07.v, c07_1.v );
  c07_2.v = _mm512_max_pd( a07.v, c07_2.v );
  c07_3.v = _mm512_max_pd( a07.v, c07_3.v );
  c07_4.v = _mm512_max_pd( a07.v, c07_4.v );
  c07_5.v = _mm512_max_pd( a07.v, c07_5.v );
  c07_6.v = _mm512_max_pd( a07.v, c07_6.v );
  c07_7.v = _mm512_max_pd( a07.v, c07_7.v );

  c15_0.v = _mm512_max_pd( a07.v, c15_0.v );
  c15_1.v = _mm512_max_pd( a07.v, c15_1.v );
  c15_2.v = _mm512_max_pd( a07.v, c15_2.v );
  c15_3.v = _mm512_max_pd( a07.v, c15_3.v );
  c15_4.v = _mm512_max_pd( a07.v, c15_4.v );
  c15_5.v = _mm512_max_pd( a07.v, c15_5.v );
  c15_6.v = _mm512_max_pd( a07.v, c15_6.v );
  c15_7.v = _mm512_max_pd( a07.v, c15_7.v );

  c23_0.v = _mm512_max_pd( a07.v, c23_0.v );
  c23_1.v = _mm512_max_pd( a07.v, c23_1.v );
  c23_2.v = _mm512_max_pd( a07.v, c23_2.v );
  c23_3.v = _mm512_max_pd( a07.v, c23_3.v );
  c23_4.v = _mm512_max_pd( a07.v, c23_4.v );
  c23_5.v = _mm512_max_pd( a07.v, c23_5.v );
  c23_6.v = _mm512_max_pd( a07.v, c23_6.v );
  c23_7.v = _mm512_max_pd( a07.v, c23_7.v );
