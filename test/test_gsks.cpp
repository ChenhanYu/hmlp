/*
 * test_gsks.c
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
#include <string.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <iostream>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define NUM_POINTS 24000
#define GFLOPS 1073741824 
#define TOLERANCE 1E-13

void compute_error
(
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

  for ( i = 0; i < m; i ++ ) 
  {
    for ( p = 0; p < rhs; p ++ ) 
    {
      tmp = fabs( u_test[ i * rhs + p ] - u_gold[ i * rhs + p ] );
      if ( tmp > max_err ) 
      {
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

  if ( rel_err > TOLERANCE ) 
  {
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
bool test_gsks( kernel_s<T> *kernel, int m, int n, int k ) 
{
  bool failure = false;
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
  amap = (int*)hbw_malloc( sizeof(int) * m );
  umap = (int*)hbw_malloc( sizeof(int) * m );
  bmap = (int*)hbw_malloc( sizeof(int) * n );
  wmap = (int*)hbw_malloc( sizeof(int) * n );
  XA   = (double*)hbw_malloc( sizeof(double) * k * nx );   // k   leading
  XA2  = (double*)hbw_malloc( sizeof(double) * nx );
  u    = (double*)hbw_malloc( sizeof(double) * nx * rhs ); // rhs leading
  w    = (double*)hbw_malloc( sizeof(double) * nx * rhs ); // rhs leading
  umkl = (double*)hbw_malloc( sizeof(double) * nx * rhs ); // rhs leading
  C    = (double*)hbw_malloc( sizeof(double) * m * n );
  C_ref= (double*)hbw_malloc( sizeof(double) * m * n );
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
  //C    = (double*)malloc( sizeof(double) * m * n );
  //C_ref= (double*)malloc( sizeof(double) * m * n );
  posix_memalign( (void**)&C,     32, sizeof(T) * m * n );
  posix_memalign( (void**)&C_ref, 32, sizeof(T) * m * n );
#endif
  // ------------------------------------------------------------------------


  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------
  for ( i = 0; i < nx; i ++ ) 
  {
    for ( p = 0; p < rhs; p ++ ) 
    {
      u[ i * rhs + p ]    = 0.0;
      umkl[ i * rhs + p ] = 0.0;
      w[ i * rhs + p ]    = (double)( rand() % 1000 ) / 1000.0;
    }
  }

  for ( i = 0; i < m; i ++ ) 
  {
    amap[ i ] = i;
    umap[ i ] = i;
  }

  for ( j = 0; j < n; j ++ ) 
  {
    bmap[ j ] = j;
    wmap[ j ] = j;
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
  // Test Variable Bandwidth Gaussian Kernel
  // ------------------------------------------------------------------------
  if ( kernel->type == KS_GAUSSIAN_VAR_BANDWIDTH ) 
  {
#ifdef HMLP_MIC_AVX512
    kernel->hi = (double*)hbw_malloc( sizeof(double) * nx );
    kernel->hj = (double*)hbw_malloc( sizeof(double) * nx );
#else
    kernel->hi = (double*)malloc( sizeof(double) * nx );
    kernel->hj = (double*)malloc( sizeof(double) * nx );
#endif
    for ( i = 0; i < nx; i ++ ) 
    {
      kernel->hi[ i ] = ( 1.0 + 0.5 / ( 1 + exp( -1.0 * XA2[ i ] ) ) );
      kernel->hi[ i ] = -1.0 / ( 2.0 * kernel->hi[ i ] * kernel->hi[ i ] );
      kernel->hj[ i ] = kernel->hi[ i ];
    }
  }
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
  for ( iter = -1; iter < n_iter; iter ++ ) 
  {
    if ( iter == 0 ) dgsks_beg = omp_get_wtime();
    dgsks
    (
      kernel,
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

    dgsks_ref
    (
      kernel,
      m, n, k,
      umkl,    umap,
      XA, XA2, amap,
      XB, XB2, bmap,
      w,       wmap
    );
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
      failure = true;
      break;
    }
  }


  switch ( kernel->type ) 
  {
    case KS_GAUSSIAN:
      {
        flops = ( (double)( m * n ) / GFLOPS ) * ( 2 * k + 35 + 2 );
        break;
      }
    case KS_GAUSSIAN_VAR_BANDWIDTH:
      {
        flops = ( (double)( m * n ) / GFLOPS ) * ( 2 * k + 35 );
#ifdef HMLP_MIC_AVX512
        hbw_free( kernel->hi );
        hbw_free( kernel->hj );
#else
        free( kernel->hi );
        free( kernel->hj );
#endif
        break;
      }
    case KS_POLYNOMIAL:
      {
        flops = ( (double)( m * n ) / GFLOPS ) * ( 2 * k + 6 );
        break;
      }
    case KS_LAPLACE:
      {
        flops = ( (double)( m * n ) / GFLOPS ) * ( 2 * k + 60 );
        break;
      }
    case KS_TANH:
      {
        flops = ( (double)( m * n ) / GFLOPS ) * ( 2 * k + 89 );
        break;
      }
    case KS_QUARTIC:
      {
        flops = ( (double)( m * n ) / GFLOPS ) * ( 2 * k + 8 );
        break;
      }
    case KS_MULTIQUADRATIC:
      {
        flops = ( (double)( m * n ) / GFLOPS ) * ( 2 * k + 6 );
        break;
      }
    case KS_EPANECHNIKOV:
      {
        flops = ( (double)( m * n ) / GFLOPS ) * ( 2 * k + 7 );
        break;
      }
    default:
      exit( 1 );
  }

#ifdef MATLAB_OUTPUT
  printf( "%d, %d, %d, %5.2lf, %5.2lf;\n", 
      m, n, k, flops / dgsks_time, flops / ref_time );
#else
  printf( "%d, %5.2lf, %5.2lf\n", 
      k, flops / dgsks_time, flops / ref_time );
#endif

  free( amap );
  free( umap );
  free( bmap );
  free( wmap );
  free( XA );
  free( XA2 );
  free( u );
  free( w );
  free( umkl );
  free( C );
  free( C_ref );

  return failure;
}; // end test_gsks()



int main( int argc, char *argv[] )
{
  int m, n, k;
  char type[ 30 ];

  kernel_s<double> kernel;

  sscanf( argv[ 1 ], "%s", type );
  sscanf( argv[ 2 ], "%d", &m );
  sscanf( argv[ 3 ], "%d", &n );
  sscanf( argv[ 4 ], "%d", &k );

  if ( !strcmp( type, "Gaussian" ) ) 
  {
    kernel.type = KS_GAUSSIAN;
    kernel.scal = -0.5;
  }
  else if ( !strcmp( type, "Polynomial" ) ) 
  {
    kernel.type = KS_POLYNOMIAL;
    kernel.powe = 4.0;
    kernel.scal = 0.1;
    kernel.cons = 0.1;
  }
  else if ( !strcmp( type, "Laplace" ) ) 
  {
    kernel.type = KS_LAPLACE;
  }
  else if ( !strcmp( type, "Var_bandwidth" ) ) 
  {
    kernel.type = KS_GAUSSIAN_VAR_BANDWIDTH;
  }
  else if ( !strcmp( type, "Tanh" ) ) 
  {
    kernel.type = KS_TANH;
    kernel.scal = 0.1;
    kernel.cons = 0.1;
  }
  else if ( !strcmp( type, "Quartic" ) ) 
  {
    kernel.type = KS_QUARTIC;
  }
  else if ( !strcmp( type, "Multiquadratic" ) ) 
  {
    kernel.type = KS_MULTIQUADRATIC;
    kernel.cons = 1.0 ;
  }
  else if ( !strcmp( type, "Epanechnikov" ) ) 
  {
    kernel.type = KS_EPANECHNIKOV;
  }
  else 
  {
    printf( "gsksMain(): kernel type mismatch %s\n", type );
    exit( 1 );
  }

  auto failure = test_gsks<double>( &kernel, m, n, k );

  return failure;
}; // end main()
