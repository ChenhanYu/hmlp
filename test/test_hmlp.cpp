/*
 * test_dgsks.c
 *
 * Chenhan D. Yu
 *
 * Department of Computer Science, University of Texas at Austin
 *
 * Purpose: 
 * this is the main function to exam the correctness between dgsks()
 * and dgsks_ref().
 *
 * Todo:
 *
 * Chenhan
 * Apr 27, 2015: readjust the flops count of the polynomail, 
 * laplace, tanh kernel.
 *
 * Modification:
 * Chenhan
 * Apr 27, 2015: New tanh kernel configuration. 
 *
 * Chenhan
 * Dec  7, 2015: Simplify 
 *
 * */


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define NUM_POINTS 24000
#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

void compute_error(
    int    m,
    int    rhs,
    double *u_test,
    double *u_gold
    )
{
  int    i, p, max_idx;
  double max_err, abs_err, rel_err;
  double tmp, nrm2;

  max_idx = -1;
  max_err = 0.0;
  nrm2    = 0.0;
  rel_err = 0.0;

  for ( i = 0; i < m; i ++ ) {
    for ( p = 0; p < rhs; p ++ ) {
      tmp = fabs( u_test[ i * rhs + p ] - u_gold[ i * rhs + p ] );
      if ( tmp > max_err ) {
        max_err = tmp;
        max_idx = i;
      }
      rel_err += tmp * tmp;
      nrm2    += u_gold[ i * rhs + p ] * u_gold[ i * rhs + p ];
    }
  }

  abs_err = sqrt( rel_err );
  rel_err /= nrm2;
  rel_err = sqrt( rel_err );

  if ( rel_err > TOLERANCE ) {
	  printf( "rel error = %E, abs error = %E, max error = %E, idx = %d\n", 
		  rel_err, abs_err, max_err, max_idx );
  }
}


/* 
 * --------------------------------------------------------------------------
 * @brief  This is the test routine to exam the correctness of GSKS. XA and
 *         XB are d leading coordinate tables, and u, w have to be rhs
 *         leading. In this case, dgsks() doesn't need to know the size of
 *         nxa and nxb as long as those index map--amap, bmap, umap and wmap
 *         --are within the legal range.
 *
 * @param  *kernel gsks data structure
 * @param  m       Number of target points
 * @param  n       Number of source points
 * @param  k       Data point dimension
 * --------------------------------------------------------------------------
 */
template<typename T>
void test_hmlp( int m, int n, int k ) 
{
  int    i, j, p, nx, iter, n_iter, rhs;
  int    *amap, *bmap, *wmap, *umap;
  double *XA, *XB, *XA2, *XB2, *u, *w, *h, *umkl, *C, *C_ref;
  double tmp, error, flops;
  double ref_beg, ref_time, dgsks_beg, dgsks_time;

  nx     = NUM_POINTS;
  rhs    = 1;
  n_iter = 1;


  // ------------------------------------------------------------------------
  // Memory allocation for all common buffers
  // ------------------------------------------------------------------------
#ifdef HMLP_MIC_AVX512
  amap = (int*)hwb_malloc( sizeof(int) * m );
  umap = (int*)hwb_malloc( sizeof(int) * m );
  bmap = (int*)hwb_malloc( sizeof(int) * n );
  wmap = (int*)hwb_malloc( sizeof(int) * n );
  XA   = (double*)hwb_malloc( sizeof(double) * k * nx );   // k   leading
  XA2  = (double*)hwb_malloc( sizeof(double) * nx );
  u    = (double*)hwb_malloc( sizeof(double) * nx * rhs ); // rhs leading
  w    = (double*)hwb_malloc( sizeof(double) * nx * rhs ); // rhs leading
  umkl = (double*)hwb_malloc( sizeof(double) * nx * rhs ); // rhs leading
  C    = (double*)hwb_malloc( sizeof(double) * m * n );
  C_ref= (double*)hwb_malloc( sizeof(double) * m * n );
#else
  amap = (int*)malloc( sizeof(int) * m );
  umap = (int*)malloc( sizeof(int) * m );
  bmap = (int*)malloc( sizeof(int) * n );
  wmap = (int*)malloc( sizeof(int) * n );
  XA   = (double*)malloc( sizeof(double) * k * nx );   // k   leading
  XA2  = (double*)malloc( sizeof(double) * nx );
  u    = (double*)malloc( sizeof(double) * nx * rhs ); // rhs leading
  w    = (double*)malloc( sizeof(double) * nx * rhs ); // rhs leading
  umkl = (double*)malloc( sizeof(double) * nx * rhs ); // rhs leading
  C    = (double*)malloc( sizeof(double) * m * n );
  C_ref= (double*)malloc( sizeof(double) * m * n );
#endif
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < rhs; p ++ ) {
      u[ i * rhs + p ]    = 0.0;
      umkl[ i * rhs + p ] = 0.0;
      w[ i * rhs + p ]    = (double)( rand() % 1000 ) / 1000.0;
    }
  }

  for ( i = 0; i < m; i ++ ) {
    amap[ i ] = i;
    umap[ i ] = i;
  }

  for ( j = 0; j < n; j ++ ) {
    bmap[ j ] = j;
    wmap[ j ] = j;
  }

  // random[ 0, 0.1 ]
  for ( i = 0; i < nx; i ++ ) {
    for ( p = 0; p < k; p ++ ) {
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
  // for ( iter = -1; iter < n_iter; iter ++ ) 
  // {
  //   if ( iter == 0 ) dgsks_beg = omp_get_wtime();
  //   dgemm_tn(
  //       m, n, k,
  //       XA, k, amap,
  //       XB, k, bmap,
  //       C, m 
  //   );
  // }
  // dgsks_time = omp_get_wtime() - dgsks_beg;
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // Call my implementation
  // ------------------------------------------------------------------------
  ks_t kernel;
  kernel.type = KS_GAUSSIAN;
  kernel.scal = -0.5;
  for ( iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) dgsks_beg = omp_get_wtime();
    dgsks(
        &kernel,
        m, n, k,
        u,       umap,
        XA, XA2, amap,
        XB, XB2, bmap,
        w,       wmap 
    );
  }
  dgsks_time = omp_get_wtime() - dgsks_beg;
  // ------------------------------------------------------------------------



  // ------------------------------------------------------------------------
  // Call the reference function 
  // ------------------------------------------------------------------------
  for ( iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) ref_beg = omp_get_wtime();

    dgsks_ref(
        &kernel,
        m, n, k,
        umkl,    umap,
        XA, XA2, amap,
        XB, XB2, bmap,
        w,       wmap
        );


    //for ( j = 0; j < n; j ++ ) 
    //{
    //  for ( i = 0; i < m; i ++ ) 
    //  {
    //    C_ref[ j * m + i ] = 0.0;    
    //    for ( p = 0; p < k; p ++ ) 
    //    {
    //      C_ref[ j * m + i ] += XA[ i * k + p ] * XB[ j * k + p ];    
    //    }
    //  }
    //}
  }
  ref_time = omp_get_wtime() - ref_beg;
  // ------------------------------------------------------------------------

  ref_time   /= n_iter;
  dgsks_time /= n_iter;

  //for ( j = 0; j < n; j ++ ) 
  //{
  //  for ( i = 0; i < m; i ++ ) 
  //  {
  //    double myc, goldc;
  //    myc = C[ j * m + i ];
  //    goldc = C_ref[ j * m + i ];
  //    if ( fabs( myc - goldc ) > 1E-9 ) 
  //    {
  //      printf( "C( %d, %d ) diff %E, %E\n", i, j, myc, goldc );
  //    }
  //  }
  //}

  for ( i = 0; i < nx; i ++ )
  {
    double u_test, u_goal;

    u_test = u[ i ];
    u_goal = umkl[ i ];
    if ( fabs( u_test - u_goal ) > 1E-9 ) 
    {
      printf( "u( %d ) diff %E, %E\n", i, u_test, u_goal );
      break;
    }
  }




  flops = ( (double)( m * n ) / GFLOPS ) * ( 2.0 * k + 37.0 );
  printf( "%d, %d, %d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / dgsks_time, flops / ref_time );

}



int main( int argc, char *argv[] )
{
  int    m, n, k;

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );

  test_hmlp<double>( m, n, k );

  return 0;
}
