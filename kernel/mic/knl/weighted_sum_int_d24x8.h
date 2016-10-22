    b0.v  = _mm512_set1_pd( w[ 0 ] );
    b1.v  = _mm512_set1_pd( w[ 1 ] );
    a07.v = _mm512_fmadd_pd( c07_0.v, b0.v, a07.v );
    a15.v = _mm512_fmadd_pd( c15_0.v, b0.v, a15.v );
    a23.v = _mm512_fmadd_pd( c23_0.v, b0.v, a23.v );
    a07.v = _mm512_fmadd_pd( c07_1.v, b1.v, a07.v );
    a15.v = _mm512_fmadd_pd( c15_1.v, b1.v, a15.v );
    a23.v = _mm512_fmadd_pd( c23_1.v, b1.v, a23.v );

    b0.v  = _mm512_set1_pd( w[ 2 ] );
    b1.v  = _mm512_set1_pd( w[ 3 ] );
    a07.v = _mm512_fmadd_pd( c07_2.v, b0.v, a07.v );
    a15.v = _mm512_fmadd_pd( c15_2.v, b0.v, a15.v );
    a23.v = _mm512_fmadd_pd( c23_2.v, b0.v, a23.v );
    a07.v = _mm512_fmadd_pd( c07_3.v, b1.v, a07.v );
    a15.v = _mm512_fmadd_pd( c15_3.v, b1.v, a15.v );
    a23.v = _mm512_fmadd_pd( c23_3.v, b1.v, a23.v );

    b0.v  = _mm512_set1_pd( w[ 4 ] );
    b1.v  = _mm512_set1_pd( w[ 5 ] );
    a07.v = _mm512_fmadd_pd( c07_4.v, b0.v, a07.v );
    a15.v = _mm512_fmadd_pd( c15_4.v, b0.v, a15.v );
    a23.v = _mm512_fmadd_pd( c23_4.v, b0.v, a23.v );
    a07.v = _mm512_fmadd_pd( c07_5.v, b1.v, a07.v );
    a15.v = _mm512_fmadd_pd( c15_5.v, b1.v, a15.v );
    a23.v = _mm512_fmadd_pd( c23_5.v, b1.v, a23.v );

    b0.v  = _mm512_set1_pd( w[ 6 ] );
    b1.v  = _mm512_set1_pd( w[ 7 ] );
    a07.v = _mm512_fmadd_pd( c07_6.v, b0.v, a07.v );
    a15.v = _mm512_fmadd_pd( c15_6.v, b0.v, a15.v );
    a23.v = _mm512_fmadd_pd( c23_6.v, b0.v, a23.v );
    a07.v = _mm512_fmadd_pd( c07_7.v, b1.v, a07.v );
    a15.v = _mm512_fmadd_pd( c15_7.v, b1.v, a15.v );
    a23.v = _mm512_fmadd_pd( c23_7.v, b1.v, a23.v );

    _mm512_store_pd( u     , a07.v );
    _mm512_store_pd( u +  8, a15.v );
    _mm512_store_pd( u + 16, a23.v );
