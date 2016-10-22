  if ( aux->pc ) {
	for ( j = 0; j < 6; j ++ ) {
	  for ( i = 0; i < 8; i ++ ) {
		K[ j * 8 + i ] += c[ j * 8 + i ];
	  }
	}
  }
  for ( p = 0; p < k; p ++ ) {
	for ( j = 0; j < 6; j ++ ) {
	  for ( i = 0; i < 8; i ++ ) {
		K[ j * 8 + i ] += a[ i ] * b [ j ];
	  }
	}
	a += 8;
	b += 8;
  }
