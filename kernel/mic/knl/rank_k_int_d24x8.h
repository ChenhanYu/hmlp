  int k_iter = k / 2;
  int k_left = k % 2;

  v8df_t pmask = { 0, 1, 2, 3, 4, 5, 6, 7 };

  __asm__ volatile( "prefetcht2 0(%0)    \n\t" : :"r"( aux->b_next ) );

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( a ) );

  c07_0.v = _mm512_setzero_pd();
  c07_1.v = _mm512_setzero_pd();
  c07_2.v = _mm512_setzero_pd();
  c07_3.v = _mm512_setzero_pd();

  __asm__ volatile( "prefetcht0 64(%0)    \n\t" : :"r"( a ) );

  c07_4.v = _mm512_setzero_pd();
  c07_5.v = _mm512_setzero_pd();
  c07_6.v = _mm512_setzero_pd();
  c07_7.v = _mm512_setzero_pd();

  __asm__ volatile( "prefetcht0 128(%0)    \n\t" : :"r"( a ) );

  c15_0.v = _mm512_setzero_pd();
  c15_1.v = _mm512_setzero_pd();
  c15_2.v = _mm512_setzero_pd();
  c15_3.v = _mm512_setzero_pd();

  __asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( b ) );

  c15_4.v = _mm512_setzero_pd();
  c15_5.v = _mm512_setzero_pd();
  c15_6.v = _mm512_setzero_pd();
  c15_7.v = _mm512_setzero_pd();

  c23_0.v = _mm512_setzero_pd();
  c23_1.v = _mm512_setzero_pd();
  c23_2.v = _mm512_setzero_pd();
  c23_3.v = _mm512_setzero_pd();
  c23_4.v = _mm512_setzero_pd();
  c23_5.v = _mm512_setzero_pd();
  c23_6.v = _mm512_setzero_pd();
  c23_7.v = _mm512_setzero_pd();





  // Load a07 
  a07.v = _mm512_load_pd( a      );
  // Load a15
  a15.v = _mm512_load_pd( a +  8 );
  // Load a23
  a23.v = _mm512_load_pd( a + 16 );

  for ( i = 0; i < k_iter; ++ i ) {
    // Iteration #0

	// Broadcast b0
	b0.v = _mm512_set1_pd( b[ 0 ] );
	// Broadcast b1
    b1.v = _mm512_set1_pd( b[ 1 ] );
    
    c07_0.v = _mm512_fmadd_pd( a07.v, b0.v, c07_0.v );
    c15_0.v = _mm512_fmadd_pd( a15.v, b0.v, c15_0.v );
    c23_0.v = _mm512_fmadd_pd( a23.v, b0.v, c23_0.v );

    __asm__ volatile( "prefetcht0 3456(%0)    \n\t" : :"r"(a) ); // #18
    //__asm__ volatile( "prefetcht0 768(%0)    \n\t" : :"r"(a) ); // #4

    c07_1.v = _mm512_fmadd_pd( a07.v, b1.v, c07_1.v );
    c15_1.v = _mm512_fmadd_pd( a15.v, b1.v, c15_1.v );
    c23_1.v = _mm512_fmadd_pd( a23.v, b1.v, c23_1.v );


    // Preload A07
    A07.v = _mm512_load_pd( a + 24 );

	// Broadcast b2
	b0.v = _mm512_set1_pd( b[ 2 ] );
	// Broadcast b3
	b1.v = _mm512_set1_pd( b[ 3 ] );

    c07_2.v = _mm512_fmadd_pd( a07.v, b0.v, c07_2.v );
    c15_2.v = _mm512_fmadd_pd( a15.v, b0.v, c15_2.v );
    c23_2.v = _mm512_fmadd_pd( a23.v, b0.v, c23_2.v );

    __asm__ volatile( "prefetcht0 3520(%0)    \n\t" : :"r"(a) ); // #18
    //__asm__ volatile( "prefetcht0 832(%0)    \n\t" : :"r"(a) ); // #4

    c07_3.v = _mm512_fmadd_pd( a07.v, b1.v, c07_3.v );
    c15_3.v = _mm512_fmadd_pd( a15.v, b1.v, c15_3.v );
    c23_3.v = _mm512_fmadd_pd( a23.v, b1.v, c23_3.v );


    // Preload A15
    A15.v = _mm512_load_pd( a + 32 );

	// Broadcast b4
	b0.v = _mm512_set1_pd( b[ 4 ] );
	// Broadcast b5
	b1.v = _mm512_set1_pd( b[ 5 ] );

    c07_4.v = _mm512_fmadd_pd( a07.v, b0.v, c07_4.v );
    c15_4.v = _mm512_fmadd_pd( a15.v, b0.v, c15_4.v );
    c23_4.v = _mm512_fmadd_pd( a23.v, b0.v, c23_4.v );

    __asm__ volatile( "prefetcht0 3584(%0)    \n\t" : :"r"(a) ); // #18
    //__asm__ volatile( "prefetcht0 896(%0)    \n\t" : :"r"(a) ); // #4

    c07_5.v = _mm512_fmadd_pd( a07.v, b1.v, c07_5.v );
    c15_5.v = _mm512_fmadd_pd( a15.v, b1.v, c15_5.v );
    c23_5.v = _mm512_fmadd_pd( a23.v, b1.v, c23_5.v );


    // Preload A23
    A23.v = _mm512_load_pd( a + 40 );

	// Broadcast b6
	b0.v = _mm512_set1_pd( b[ 6 ] );
	// Broadcast b7
	b1.v = _mm512_set1_pd( b[ 7 ] );

    c07_6.v = _mm512_fmadd_pd( a07.v, b0.v, c07_6.v );
    c15_6.v = _mm512_fmadd_pd( a15.v, b0.v, c15_6.v );
    c23_6.v = _mm512_fmadd_pd( a23.v, b0.v, c23_6.v );

    __asm__ volatile( "prefetcht0 1152(%0)    \n\t" : :"r"(b) ); // #18
    //__asm__ volatile( "prefetcht0 256(%0)    \n\t" : :"r"(b) ); // #4


    c07_7.v = _mm512_fmadd_pd( a07.v, b1.v, c07_7.v );
    c15_7.v = _mm512_fmadd_pd( a15.v, b1.v, c15_7.v );
    c23_7.v = _mm512_fmadd_pd( a23.v, b1.v, c23_7.v );


    // Iteration #1
    //_mm512_prefetch_i64gather_pd( pmask.i, a + 72, 1, _MM_HINT_T0 );

	// Broadcast b8
	b0.v = _mm512_set1_pd( b[ 8 ] );
	// Broadcast b9
	b1.v = _mm512_set1_pd( b[ 9 ] );

    c07_0.v = _mm512_fmadd_pd( A07.v, b0.v, c07_0.v );
    c15_0.v = _mm512_fmadd_pd( A15.v, b0.v, c15_0.v );
    c23_0.v = _mm512_fmadd_pd( A23.v, b0.v, c23_0.v );

    __asm__ volatile( "prefetcht0 3648(%0)    \n\t" : :"r"(a) ); // #18
    //__asm__ volatile( "prefetcht0 960(%0)    \n\t" : :"r"(a) ); // #4

    c07_1.v = _mm512_fmadd_pd( A07.v, b1.v, c07_1.v );
    c15_1.v = _mm512_fmadd_pd( A15.v, b1.v, c15_1.v );
    c23_1.v = _mm512_fmadd_pd( A23.v, b1.v, c23_1.v );

    // Preload a07
    a07.v = _mm512_load_pd( a + 48 );

	// Broadcast b10
	b0.v = _mm512_set1_pd( b[ 10 ] );
	// Broadcast b11
	b1.v = _mm512_set1_pd( b[ 11 ] );

    c07_2.v = _mm512_fmadd_pd( A07.v, b0.v, c07_2.v );
    c15_2.v = _mm512_fmadd_pd( A15.v, b0.v, c15_2.v );
    c23_2.v = _mm512_fmadd_pd( A23.v, b0.v, c23_2.v );

    //__asm__ volatile( "prefetcht0 3712(%0)    \n\t" : :"r"(a) );
    __asm__ volatile( "prefetcht0 1024 (%0)    \n\t" : :"r"(a) );

    c07_3.v = _mm512_fmadd_pd( A07.v, b1.v, c07_3.v );
    c15_3.v = _mm512_fmadd_pd( A15.v, b1.v, c15_3.v );
    c23_3.v = _mm512_fmadd_pd( A23.v, b1.v, c23_3.v );


    // Preload a15
    a15.v = _mm512_load_pd( a + 56 );

	// Broadcast b12
	b0.v = _mm512_set1_pd( b[ 12 ] );
	// Broadcast b13
	b1.v = _mm512_set1_pd( b[ 13 ] );

    c07_4.v = _mm512_fmadd_pd( A07.v, b0.v, c07_4.v );
    c15_4.v = _mm512_fmadd_pd( A15.v, b0.v, c15_4.v );
    c23_4.v = _mm512_fmadd_pd( A23.v, b0.v, c23_4.v );

    __asm__ volatile( "prefetcht0 3776(%0)    \n\t" : :"r"(a) ); // #18
    //__asm__ volatile( "prefetcht0 1088(%0)    \n\t" : :"r"(a) ); // #4

    c07_5.v = _mm512_fmadd_pd( A07.v, b1.v, c07_5.v );
    c15_5.v = _mm512_fmadd_pd( A15.v, b1.v, c15_5.v );
    c23_5.v = _mm512_fmadd_pd( A23.v, b1.v, c23_5.v );


    // Preload a23
    a23.v = _mm512_load_pd( a + 64 );

	// Broadcast b14
	b0.v = _mm512_set1_pd( b[ 14 ] );
	// Broadcast b15
	b1.v = _mm512_set1_pd( b[ 15 ] );

    c07_6.v = _mm512_fmadd_pd( A07.v, b0.v, c07_6.v );
    c15_6.v = _mm512_fmadd_pd( A15.v, b0.v, c15_6.v );
    c23_6.v = _mm512_fmadd_pd( A23.v, b0.v, c23_6.v );

    __asm__ volatile( "prefetcht0 1216(%0)    \n\t" : :"r"(b) ); // #18
    //__asm__ volatile( "prefetcht0 320(%0)    \n\t" : :"r"(b) ); // #4

    c07_7.v = _mm512_fmadd_pd( A07.v, b1.v, c07_7.v );
    c15_7.v = _mm512_fmadd_pd( A15.v, b1.v, c15_7.v );
    c23_7.v = _mm512_fmadd_pd( A23.v, b1.v, c23_7.v );

	// Increment
    a += 48;
    b += 16;
  }

  for ( i = 0; i < k_left; ++ i ) {

	// Broadcast b0
	b0.v = _mm512_set1_pd( b[ 0 ] );
	// Broadcast b1
    b1.v = _mm512_set1_pd( b[ 1 ] );
    
    c07_0.v = _mm512_fmadd_pd( a07.v, b0.v, c07_0.v );
    c15_0.v = _mm512_fmadd_pd( a15.v, b0.v, c15_0.v );
    c23_0.v = _mm512_fmadd_pd( a23.v, b0.v, c23_0.v );

    c07_1.v = _mm512_fmadd_pd( a07.v, b1.v, c07_1.v );
    c15_1.v = _mm512_fmadd_pd( a15.v, b1.v, c15_1.v );
    c23_1.v = _mm512_fmadd_pd( a23.v, b1.v, c23_1.v );

	// Broadcast b2
	b0.v = _mm512_set1_pd( b[ 2 ] );
	// Broadcast b3
	b1.v = _mm512_set1_pd( b[ 3 ] );

    c07_2.v = _mm512_fmadd_pd( a07.v, b0.v, c07_2.v );
    c15_2.v = _mm512_fmadd_pd( a15.v, b0.v, c15_2.v );
    c23_2.v = _mm512_fmadd_pd( a23.v, b0.v, c23_2.v );

    c07_3.v = _mm512_fmadd_pd( a07.v, b1.v, c07_3.v );
    c15_3.v = _mm512_fmadd_pd( a15.v, b1.v, c15_3.v );
    c23_3.v = _mm512_fmadd_pd( a23.v, b1.v, c23_3.v );

	// Broadcast b4
	b0.v = _mm512_set1_pd( b[ 4 ] );
	// Broadcast b5
	b1.v = _mm512_set1_pd( b[ 5 ] );

    c07_4.v = _mm512_fmadd_pd( a07.v, b0.v, c07_4.v );
    c15_4.v = _mm512_fmadd_pd( a15.v, b0.v, c15_4.v );
    c23_4.v = _mm512_fmadd_pd( a23.v, b0.v, c23_4.v );

    c07_5.v = _mm512_fmadd_pd( a07.v, b1.v, c07_5.v );
    c15_5.v = _mm512_fmadd_pd( a15.v, b1.v, c15_5.v );
    c23_5.v = _mm512_fmadd_pd( a23.v, b1.v, c23_5.v );

	// Broadcast b6
	b0.v = _mm512_set1_pd( b[ 6 ] );
	// Broadcast b7
	b1.v = _mm512_set1_pd( b[ 7 ] );

    c07_6.v = _mm512_fmadd_pd( a07.v, b0.v, c07_6.v );
    c15_6.v = _mm512_fmadd_pd( a15.v, b0.v, c15_6.v );
    c23_6.v = _mm512_fmadd_pd( a23.v, b0.v, c23_6.v );

    c07_7.v = _mm512_fmadd_pd( a07.v, b1.v, c07_7.v );
    c15_7.v = _mm512_fmadd_pd( a15.v, b1.v, c15_7.v );
    c23_7.v = _mm512_fmadd_pd( a23.v, b1.v, c23_7.v );

    a += 24;
    b += 8;
  }
 

  //printf( "c (tid %d)\n", omp_get_thread_num() );
  //printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[0], c07_1.d[0], c07_2.d[0], c07_3.d[0], c07_4.d[0], c07_5.d[0], c07_6.d[0], c07_7.d[0] );

  // Prefetch aa and bb
  //__asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( aa ) );
  //__asm__ volatile( "prefetcht0 0(%0)    \n\t" : :"r"( bb ) );
  //_mm512_prefetch_i64gather_pd( pmask.i, aa, 8, _MM_HINT_T0 );
  //_mm512_prefetch_i64gather_pd( pmask.i, bb, 8, _MM_HINT_T0 );
  
  if ( aux->pc != 0 ) {
    // nonpacked
    /*
    tmpc03_0.v = _mm256_load_pd( (double*)( c               ) );
    tmpc47_0.v = _mm256_load_pd( (double*)( c + 4           ) );

    tmpc03_1.v = _mm256_load_pd( (double*)( c + ldc * 1     ) );
    tmpc47_1.v = _mm256_load_pd( (double*)( c + ldc * 1 + 4 ) );

    tmpc03_2.v = _mm256_load_pd( (double*)( c + ldc * 2     ) );
    tmpc47_2.v = _mm256_load_pd( (double*)( c + ldc * 2 + 4 ) );

    tmpc03_3.v = _mm256_load_pd( (double*)( c + ldc * 3     ) );
    tmpc47_3.v = _mm256_load_pd( (double*)( c + ldc * 3 + 4 ) );
    */

	double *cptr;
	//int j;
	//double ctmp[ 24 * 8 ], *cptr;

	//for ( i = 0; i < 24; i ++ ) {
	//  for ( j = 0; j < 8; j ++ ) {
	//	ctmp[ j * 24 + i ] = c[ i * 8 + j ];
	//  }
	//}

	//cptr = ctmp;
	cptr = c;

    //#pragma omp critical 
    //{
    //  printf( "c (tid %d)\n", omp_get_thread_num() );
    //  printf( "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", c07_0.d[0], c07_1.d[0], c07_2.d[0], c07_3.d[0], c07_4.d[0], c07_5.d[0], c07_6.d[0], c07_7.d[0] );
    //  printf( "%lf, %lf, %lf, %lf\n", c[0], c[24], c[48], c[72] );
    //}


    // packed
    a07.v   = _mm512_load_pd( cptr      );
    c07_0.v = _mm512_add_pd( a07.v, c07_0.v );

    a15.v   = _mm512_load_pd( cptr +  8 );
    c15_0.v = _mm512_add_pd( a15.v, c15_0.v );

    a23.v   = _mm512_load_pd( cptr + 16 );
    c23_0.v = _mm512_add_pd( a23.v, c23_0.v );

    a07.v   = _mm512_load_pd( cptr + 24 );
    c07_1.v = _mm512_add_pd( a07.v, c07_1.v );

    a15.v   = _mm512_load_pd( cptr + 32 );
    c15_1.v = _mm512_add_pd( a15.v, c15_1.v );

    a23.v   = _mm512_load_pd( cptr + 40 );
    c23_1.v = _mm512_add_pd( a23.v, c23_1.v );

    a07.v   = _mm512_load_pd( cptr + 48 );
    c07_2.v = _mm512_add_pd( a07.v, c07_2.v );

    a15.v   = _mm512_load_pd( cptr + 56 );
    c15_2.v = _mm512_add_pd( a15.v, c15_2.v );

    a23.v   = _mm512_load_pd( cptr + 64 );
    c23_2.v = _mm512_add_pd( a23.v, c23_2.v );

    a07.v   = _mm512_load_pd( cptr + 72 );
    c07_3.v = _mm512_add_pd( a07.v, c07_3.v );

    a15.v   = _mm512_load_pd( cptr + 80 );
    c15_3.v = _mm512_add_pd( a15.v, c15_3.v );

    a23.v   = _mm512_load_pd( cptr + 88 );
    c23_3.v = _mm512_add_pd( a23.v, c23_3.v );

    a07.v   = _mm512_load_pd( cptr + 96 );
    c07_4.v = _mm512_add_pd( a07.v, c07_4.v );

    a15.v   = _mm512_load_pd( cptr + 104 );
    c15_4.v = _mm512_add_pd( a15.v, c15_4.v );

    a23.v   = _mm512_load_pd( cptr + 112 );
    c23_4.v = _mm512_add_pd( a23.v, c23_4.v );

    a07.v   = _mm512_load_pd( cptr + 120 );
    c07_5.v = _mm512_add_pd( a07.v, c07_5.v );

    a15.v   = _mm512_load_pd( cptr + 128 );
    c15_5.v = _mm512_add_pd( a15.v, c15_5.v );

    a23.v   = _mm512_load_pd( cptr + 136 );
    c23_5.v = _mm512_add_pd( a23.v, c23_5.v );

    a07.v   = _mm512_load_pd( cptr + 144 );
    c07_6.v = _mm512_add_pd( a07.v, c07_6.v );

    a15.v   = _mm512_load_pd( cptr + 152 );
    c15_6.v = _mm512_add_pd( a15.v, c15_6.v );

    a23.v   = _mm512_load_pd( cptr + 160 );
    c23_6.v = _mm512_add_pd( a23.v, c23_6.v );

    a07.v   = _mm512_load_pd( cptr + 168 );
    c07_7.v = _mm512_add_pd( a07.v, c07_7.v );

    a15.v   = _mm512_load_pd( cptr + 176 );
    c15_7.v = _mm512_add_pd( a15.v, c15_7.v );

    a23.v   = _mm512_load_pd( cptr + 184 );
    c23_7.v = _mm512_add_pd( a23.v, c23_7.v );
  }
