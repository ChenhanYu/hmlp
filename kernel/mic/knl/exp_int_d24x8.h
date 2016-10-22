
  // y = exp( x )
  // log( y ) = x * log2e
  // y = 2 ^^ ( x * log2e )
  const double log2e  =  1.4426950408889634073599;

  // x * log2e
  a07.v   = _mm512_set1_pd( log2e );
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
  
  // 2 ^^ ( x * log2e )
  c07_0.v = _mm512_exp2a23_pd( c07_0.v );
  c07_1.v = _mm512_exp2a23_pd( c07_1.v );
  c07_2.v = _mm512_exp2a23_pd( c07_2.v );
  c07_3.v = _mm512_exp2a23_pd( c07_3.v );
  c07_4.v = _mm512_exp2a23_pd( c07_4.v );
  c07_5.v = _mm512_exp2a23_pd( c07_5.v );
  c07_6.v = _mm512_exp2a23_pd( c07_6.v );
  c07_7.v = _mm512_exp2a23_pd( c07_7.v );

  c15_0.v = _mm512_exp2a23_pd( c15_0.v );
  c15_1.v = _mm512_exp2a23_pd( c15_1.v );
  c15_2.v = _mm512_exp2a23_pd( c15_2.v );
  c15_3.v = _mm512_exp2a23_pd( c15_3.v );
  c15_4.v = _mm512_exp2a23_pd( c15_4.v );
  c15_5.v = _mm512_exp2a23_pd( c15_5.v );
  c15_6.v = _mm512_exp2a23_pd( c15_6.v );
  c15_7.v = _mm512_exp2a23_pd( c15_7.v );

  c23_0.v = _mm512_exp2a23_pd( c23_0.v );
  c23_1.v = _mm512_exp2a23_pd( c23_1.v );
  c23_2.v = _mm512_exp2a23_pd( c23_2.v );
  c23_3.v = _mm512_exp2a23_pd( c23_3.v );
  c23_4.v = _mm512_exp2a23_pd( c23_4.v );
  c23_5.v = _mm512_exp2a23_pd( c23_5.v );
  c23_6.v = _mm512_exp2a23_pd( c23_6.v );
  c23_7.v = _mm512_exp2a23_pd( c23_7.v );
