/*
 * test_dgsknn.c
 *
 * Leslie Rice
 *
 * Department of Computer Science, University of Texas at Austin
 *
 * Purpose:
 * this is the main function to exam the correctness between dgsknn()
 * and dgsknn_ref().
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define NUM_POINTS 10240
#define GFLOPS 1073741824
#define TOLERANCE 1E-13

void compute_error(
    int    r,
    int    n,
    double *D,
    int    *I,
    double *D_gold,
    int    *I_gold
    )
{

  int    i, j, p;
  double *D1, *D2;
  int    *I1, *I2, *Set1, *Set2;

  #ifdef HMLP_MIC_AVX512
    Set1 = (int*)hbw_malloc( sizeof(int) * NUM_POINTS );
    Set2 = (int*)hbw_malloc( sizeof(int) * NUM_POINTS );
  #else
    Set1 = (int*)malloc( sizeof(int) * NUM_POINTS );
    Set2 = (int*)malloc( sizeof(int) * NUM_POINTS );
  #endif

  // Check set equvilent.
  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < NUM_POINTS; i ++ ) {
      Set1[ i ] = 0;
      Set2[ i ] = 0;
    }
    for ( i = 0; i < r; i ++ ) {
      p = I[ j * r + i ];
      Set1[ p ] = i;
      Set2[ p ] = 1;
    }
    for ( i = 0; i < r; i ++ ) {
      p = I_gold[ j * r + i ];
      if ( Set2[ p ] == 0 ) {
        Set1[ p ] = i;
        Set2[ p ] = 2;
      }
      else {
        Set2[ p ] = 0;
      }
    }
    for ( i = 0; i < NUM_POINTS; i ++ ) {
      if ( Set2[ i ] == 1 && D[ j * r ] != D[ j * r + Set1[ i ] ] ) {
        printf( "(%E,%E,%d,%d,%E,1,%d)\n", D[ j * r ], D_gold[ j * r ],
            j, i, D[ j * r + Set1[ i ] ], I[ j * r ] );
      }
      if ( Set2[ i ] == 2 && D_gold[ j * r ] != D_gold[ j * r + Set1[ i ] ] ) {
        printf( "(%E,%E,%d,%d,%E,2,%d)\n", D[ j * r ], D_gold[ j * r ],
            j, i, D_gold[ j * r + Set1[ i ] ], I_gold[ j * r ] );
        if ( D_gold[ j * r ] < D_gold[ j * r + Set1[ i ] ] ) {
          printf( "bug\n" );
        }
      }
    }
  }

    free( Set1 );
    free( Set2 );
}


/*
 * --------------------------------------------------------------------------
 * @brief  This is the test routine to exam the correctness of GSKNN. XA and
 *         XB are d leading coordinate tables, and u, w have to be rhs
 *         leading. In this case, dgsknn() doesn't need to know the size of
 *         nxa and nxb as long as those index map--amap, bmap, umap and wmap
 *         --are within the legal range.
 *
 * @param  *kernel gsknn data structure
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * --------------------------------------------------------------------------
 */
template<typename T>
void test_gsknn( int m, int n, int k, int r )
{
  int    i, j, p, nx, iter, n_iter;
  int    *amap, *bmap, *I, *I_mkl;
  double *XA, *XB, *XA2, *XB2, *D, *D_mkl;
  double tmp, error, flops;
  double ref_beg, ref_time, dgsknn_beg, dgsknn_time;

  nx     = NUM_POINTS;
  n_iter = 1;


  // ------------------------------------------------------------------------
  // Memory allocation for all common buffers
  // ------------------------------------------------------------------------
#ifdef HMLP_MIC_AVX512
  amap  = (int*)hbw_malloc( sizeof(int) * m );
  bmap  = (int*)hbw_malloc( sizeof(int) * n );
  I     = (int*)hbw_malloc( sizeof(int) * r * n );
  I_mkl = (int*)hbw_malloc( sizeof(int) * r * n );
  XA    = (double*)hbw_malloc( sizeof(double) * k * nx );   // k   leading
  XA2   = (double*)hbw_malloc( sizeof(double) * nx );
  D     = (double*)hbw_malloc( sizeof(double) * r * n );
  D_mkl = (double*)hbw_malloc( sizeof(double) * r * n );
#else
  amap = (int*)malloc( sizeof(int) * m );
  bmap = (int*)malloc( sizeof(int) * n );
  I     = (int*)malloc( sizeof(int) * r * n );
  I_mkl = (int*)malloc( sizeof(int) * r * n );
  XA   = (double*)malloc( sizeof(double) * k * nx );   // k   leading
  XA2  = (double*)malloc( sizeof(double) * nx );
  D     = (double*)malloc( sizeof(double) * r * n );
  D_mkl = (double*)malloc( sizeof(double) * r * n );
#endif
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------

  for ( i = 0; i < m; i ++ )
  {
    amap[ i ] = i;
  }

  for ( j = 0; j < n; j ++ )
  {
    bmap[ j ] = j;
  }

  // random[ 0, 0.1 ]
  for ( i = 0; i < nx; i ++ )
  {
    for ( p = 0; p < k; p ++ )
    {
      XA[ i * k + p ] = (double)( rand() % 100 ) / 1000.0;
    }
  }
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Compute XA2
  // ------------------------------------------------------------------------
  for ( i = 0; i < nx; i ++ ) {
    tmp = 0.0;
    for ( p = 0; p < k; p ++ ) {
      tmp += XA[ i * k + p ] * XA[ i * k + p ];
    }
    XA2[ i ] = tmp;
  }
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Use the same coordinate table
  // ------------------------------------------------------------------------
  XB  = XA;
  XB2 = XA2;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call my implementation
  // ------------------------------------------------------------------------
  for ( iter = -1; iter < n_iter; iter ++ )
  {
    if ( iter == 0 ) dgsknn_beg = omp_get_wtime();
    dgsknn(
        m, n, k, r,
        XA, XA2, amap,
        XB, XB2, bmap,
        D,       I
    );
  }
  dgsknn_time = omp_get_wtime() - dgsknn_beg;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Initialize D ( distance ) to the maximum double and I ( index ) to -1.
  // ------------------------------------------------------------------------

  for ( j = 0; j < n; j ++ ) {
    for ( i = 0; i < r; i ++ ) {
      D[ j * r + i ]     = 1.79E+308;
      I[ j * r + i ]     = -1;
      D_mkl[ j * r + i ] = 1.79E+308;
      I_mkl[ j * r + i ] = -1;
    }
  }

  // ------------------------------------------------------------------------
  // Call the reference function
  // ------------------------------------------------------------------------
  for ( iter = -1; iter < n_iter; iter ++ )
  {
    if ( iter == 0 ) ref_beg = omp_get_wtime();

    dgsknn_ref(
        m, n, k, r,
        XA, XA2, amap,
        XB, XB2, bmap,
        D_mkl,   I_mkl
        );

  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time   /= n_iter;
  dgsknn_time /= n_iter;

  compute_error(
      r,
      n,
      D,
      I,
      D_mkl,
      I_mkl
      );

  flops = ( (double)( m * n ) / GFLOPS ) * ( 2.0 * k + 37.0 );
  printf( "%d, %d, %d, %d, %5.2lf, %5.2lf;\n",
      m, n, k, r, flops / dgsknn_time, flops / ref_time );

  free( amap );
  free( bmap );
  free( XA );
  free( XA2 );
  free( D );
  free( I );
  free( D_mkl );
  free( I_mkl );
}



int main( int argc, char *argv[] )
{
  int    m, n, k, r;

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );
  sscanf( argv[ 4 ], "%d", &r );

  test_gsknn<double>( m, n, k, r );

  return 0;
}
