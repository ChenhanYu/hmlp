/**
 *  -- GKMX (version 1.1.0) --
 *
 *  NVIDIA Corp, Santa Clara
 *
 *  @date June 2016
 *  @author Chenhan D. Yu
 *
 */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <hmlp.h>
#include <hmlp_blas_lapack.h>

#define GFLOPS 1073741824
#define TOLERANCE 1E-13

using namespace hmlp;


template<typename T>
void compute_error
(
  int m, int n,
  T *test, int ldtest,
  T *goal, int ldgoal
)
{
  thrust::tuple<int, int, T> max_err( -1, -1, (T)0.0 );
  T abs_err = 0.0, rel_err = 0.0, nrm2 = 0.0;

  for ( int j = 0; j < n; j ++ ) 
  {
    for ( int i = 0; i < m; i ++ ) 
    {
      T tmp_goal = goal[ j * ldgoal + i ];
      T tmp_test = test[ j * ldtest + i ];
      T err = fabs( tmp_test - tmp_goal );
      if ( err > thrust::get<2>( max_err ) ) 
      {
        max_err = thrust::make_tuple( i, j, err );
      }
      rel_err += err * err;
      nrm2    += tmp_goal * tmp_goal;
    }
  }

  abs_err = sqrt( rel_err );
  rel_err /= nrm2;
  rel_err = sqrt( rel_err );

  if ( rel_err > TOLERANCE ) 
  {
	  printf( "rel error %E, abs error %E, max error %E at (%d %d)\n", 
		  rel_err, abs_err, thrust::get<2>( max_err ), 
          thrust::get<0>( max_err ), thrust::get<1>( max_err ) );
  }
};

template<typename T>
void compute_error
(
  int m, int n,
  T *test, int ldtest,
  T *goal, int ldgoal,
  int batchSize, int stride
)
{
  for ( int b = 0; b < batchSize; b ++ )
  {
    compute_error
    ( 
      m, n, 
      test + b * stride, ldtest, 
      goal + b * stride, ldgoal
    );
  }
};




/** 
 *  @brief This is an example to invoke the gkmm instance with a Gaussian
 *         kernel function. A2 and B2 are precomputed to accelerate the
 *         computation of the square 2-norm using expansion ||a-b||^2 = ||a||^2
 *         + 2a^{T}b + ||b||^2.
 *
 */ 
template <typename T>
void test_gkmm( int m, int n, int k, int batchSize )
{
  float gkmm_time = 0.0, gkmm_strided_time = 0.0, ref_time = 0.0;
  double flops = 0.0;

  cublasHandle_t handle;
  cublasCreate( &handle );

  cudaEvent_t gkmx_beg, gkmx_end;
  // cublasOperation_t transa = CUBLAS_OP_N; 
  // cublasOperation_t transb = CUBLAS_OP_N;
  cudaEventCreate( &gkmx_beg );
  cudaEventCreate( &gkmx_end );

  // Thrust style
  thrust::host_vector<T> h_A( m * k * batchSize );
  thrust::host_vector<T> h_B( k * n * batchSize );
  thrust::host_vector<T> h_C( m * n * batchSize, (T)0.0 );
  thrust::host_vector<T> h_Cref( m * n * batchSize, (T)0.0 );

  for ( int i = 0; i < batchSize; i ++ ) 
  {
    for ( int j = 0; j < m * k; j ++ ) h_A[ i * m * k + j ] = (T) rand() / (T) RAND_MAX;
    for ( int j = 0; j < k * n; j ++ ) h_B[ i * k * n + j ] = (T) rand() / (T) RAND_MAX;
    //for ( int j = 0; j < m * n; j ++ ) h_C[ i * m * n + j ] = (T) 0.0;
  }

  // memcpyHostToDevice
  thrust::device_vector<T> d_A = h_A;
  thrust::device_vector<T> d_B = h_B;
  thrust::device_vector<T> d_C = h_C;
  thrust::device_vector<T> d_Cref = h_Cref;

  // Thrust style
  thrust::device_vector<T*> d_Ap( batchSize );
  thrust::device_vector<T*> d_Bp( batchSize );
  thrust::device_vector<T*> d_Cp( batchSize );
  thrust::device_vector<T*> d_Crefp( batchSize );

  // Caculate device pointers
  for ( int i = 0; i < batchSize; ++ i ) 
  {
    d_Ap[ i ] = d_A.data().get() + i * m * k;
    d_Bp[ i ] = d_B.data().get() + i * k * n;
    d_Cp[ i ] = d_C.data().get() + i * m * n;
    d_Crefp[ i ] = d_Cref.data().get() + i * m * n;
  }
  
  // Non-strided GKMM
  cudaEventRecord( gkmx_beg, 0 );
  gkmm_dfma
  (
    0, 
    HMLP_OP_N, HMLP_OP_N,
    m, n, k, 
    (const T**)d_Ap.data().get(), m,
    (const T**)d_Bp.data().get(), k,
               d_Cp.data().get(), m,
    batchSize 
  );
  cudaThreadSynchronize();
  cudaEventRecord( gkmx_end, 0 );
  cudaEventSynchronize( gkmx_end );
  cudaEventElapsedTime( &gkmm_time, gkmx_beg, gkmx_end );
  gkmm_time /= 1000.0;

  // Strided GKMM
  cudaEventRecord( gkmx_beg, 0 );
  gkmm_dfma
  (
    0, 
    HMLP_OP_N, HMLP_OP_N,
    m, n, k, 
    (const T*)d_A.data().get(), m, m * k,
    (const T*)d_B.data().get(), k, k * n,
              d_C.data().get(), m, m * n,
    batchSize
  );
  cudaThreadSynchronize();
  cudaEventRecord( gkmx_end, 0 );
  cudaEventSynchronize( gkmx_end );
  cudaEventElapsedTime( &gkmm_strided_time, gkmx_beg, gkmx_end );
  gkmm_strided_time /= 1000.0;


  // Reference 
  cudaEventRecord( gkmx_beg, 0 );
  xgemm_batched 
  (
    handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    1.0,
    d_Ap.data().get(), m,
    d_Bp.data().get(), k, 0.0,
    d_Crefp.data().get(), m,
    batchSize 
  );
  cudaThreadSynchronize();
  cudaEventRecord( gkmx_end, 0 );
  cudaEventSynchronize( gkmx_end );
  cudaEventElapsedTime( &ref_time, gkmx_beg, gkmx_end );
  ref_time /= 1000.0;

   
  thrust::copy( d_C.begin(), d_C.end(), h_C.begin() );
  thrust::copy( d_Cref.begin(), d_Cref.end(), h_Cref.begin() );

  compute_error
  (
    m, n, 
    h_C.data(),    m, 
    h_Cref.data(), m,
    batchSize,
    m * n
  );

  //flops = ( (double)( m * n ) / GFLOPS ) * ( 2.0 * k ) * batchSize;
  flops = ( 2.0 * m * n * k ) * (double)batchSize / ( 1000.0 * 1000.0 * 1000.0 );
  printf( "%4d, %4d, %4d, %4d, %f, %f, %f;\n", m, n, k, batchSize, 
      flops / gkmm_time, flops / gkmm_strided_time, flops / ref_time
      //gkmm_time, gkmm_strided_time 
      );
}


/*
 *  @brief This is an example to invoke the gkrm instance that implements the
 *         nearst neighbor search (one neighbor). 2-norm option is used in this
 *         case; so A2 and B2 are precomputed. The kernel function take the
 *         square distance and its column index, returning a pair. The reduce
 *         function computes argmin between two pairs and keep the one with the
 *         smaller distance.
 *        
 */ 
template <typename T, typename TC>
void test_gkrm( int m, int n, int k, int batchSize )
{
  float  gkrm_time = 0.0, gkrm_strided_time = 0.0;
  double flops = 0.0;

  cudaEvent_t gkmx_beg, gkmx_end;
  // cublasOperation_t transa = CUBLAS_OP_N; 
  // cublasOperation_t transb = CUBLAS_OP_N; 
  cudaEventCreate( &gkmx_beg );
  cudaEventCreate( &gkmx_end );

  // Thrust style
  thrust::host_vector<T>  h_A( m * k * batchSize );
  thrust::host_vector<T>  h_B( k * n * batchSize );
  thrust::host_vector<TC> h_C( m * n * batchSize );
  thrust::host_vector<TC> h_O( m * n * batchSize ); // reference

  for ( int i = 0; i < batchSize; i ++ ) 
  {
    for ( int j = 0; j < m * k; j ++ ) h_A[ i * m * k + j ] = (T) rand() / (T) RAND_MAX;
    for ( int j = 0; j < k * n; j ++ ) h_B[ i * k * n + j ] = (T) rand() / (T) RAND_MAX;
    for ( int j = 0; j < m * n; j ++ ) 
    {
      h_C[ i * m * n + j ].first  = (T) 0.0;
      h_C[ i * m * n + j ].second = (int) -1;
      h_O[ i * m * n + j ].first  = (T) 0.0;
      h_O[ i * m * n + j ].second = (int) -1;
    } 
  }

  // memcpyHostToDevice
  thrust::device_vector<T>  d_A = h_A;
  thrust::device_vector<T>  d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;
  thrust::device_vector<TC> d_O = h_O; // reference
  thrust::device_vector<double>  d_D( m * n * batchSize ); // reference pairwise distnaces

  // Thrust style
  thrust::device_vector<T*>  d_Ap( batchSize );
  thrust::device_vector<T*>  d_Bp( batchSize );
  thrust::device_vector<TC*> d_Cp( batchSize );
  thrust::device_vector<TC*> d_Op( batchSize );

  thrust::host_vector<TC*>   h_Cp( batchSize );
  thrust::host_vector<TC*>   h_Op( batchSize );


  for ( int i = 0; i < batchSize; ++ i ) 
  {
    d_Ap[ i ] = d_A.data().get() + i * m * k;
    d_Bp[ i ] = d_B.data().get() + i * k * n;
    d_Cp[ i ] = d_C.data().get() + i * m * n;
    d_Op[ i ] = d_O.data().get() + i * m * n;

    h_Cp[ i ] = h_C.data() + i * m * n;
    h_Op[ i ] = h_O.data() + i * m * n;
  }

  // Compute squared 2 norm for RBF kernel.
  thrust::device_vector<T>  d_A2( m * batchSize );
  thrust::device_vector<T>  d_B2( n * batchSize );
  thrust::device_vector<T*> d_A2p( batchSize );
  thrust::device_vector<T*> d_B2p( batchSize );
  for ( int i = 0; i < batchSize; i ++ ) 
  {
    d_A2p[ i ] = d_A2.data().get() + i * m;
    d_B2p[ i ] = d_B2.data().get() + i * n;
  }

  dsq2nrm
  (
    HMLP_OP_T,
    k, m, 
    d_A2p.data().get(), 
    (const T**)d_Ap.data().get(), NULL, m, batchSize 
  );
  dsq2nrm
  (
    HMLP_OP_N,
    k, n, 
    d_B2p.data().get(), 
    (const T**)d_Bp.data().get(), NULL, k, batchSize 
  );

  // Setup the initial value for k-mean
  TC initC( 999999.99, -1 );

  // batch_cuda_gkrm
  cudaEventRecord( gkmx_beg, 0 );
  gkrm_dkmeans
  (
    0, 
    HMLP_OP_T, HMLP_OP_N,
    m, n, k, 
    d_Ap.data().get(), d_A2p.data().get(), m,
    d_Bp.data().get(), d_B2p.data().get(), k,
    d_Cp.data().get(), m,
    batchSize
  );
  cudaThreadSynchronize();
  cudaEventRecord( gkmx_end, 0 );
  cudaEventSynchronize( gkmx_end );
  cudaEventElapsedTime( &gkrm_time, gkmx_beg, gkmx_end );
  gkrm_time /= 1000.0;

  // Strided GKRM
  cudaEventRecord( gkmx_beg, 0 );
  //gkrm_dkmean
  //(
  //  0, 
  //  transa, transb,
  //  m, n, k, 
  //  d_A.data().get(), d_A2.data().get(), m, m * k,
  //  d_B.data().get(), d_B2.data().get(), k, k * n,
  //  d_C.data().get(), m, m * n,
  //  batchSize
  //);
  cudaThreadSynchronize();
  cudaEventRecord( gkmx_end, 0 );
  cudaEventSynchronize( gkmx_end );
  cudaEventElapsedTime( &gkrm_strided_time, gkmx_beg, gkmx_end );
  gkrm_strided_time /= 1000.0;

  // Reference
  cudaEventRecord( gkmx_beg, 0 );
  dkmeans
  (
    0, 
    //transa, transb,
    m, n, k, 
    d_Ap.data().get(), d_A2p.data().get(), m,
    d_Bp.data().get(), d_B2p.data().get(), k,
    d_Op.data().get(), m,
    batchSize
  );
  cudaThreadSynchronize();
  cudaEventRecord( gkmx_end, 0 );
  cudaEventSynchronize( gkmx_end );
  cudaEventElapsedTime( &gkrm_strided_time, gkmx_beg, gkmx_end );
  gkrm_strided_time /= 1000.0;

  thrust::copy( d_C.begin(), d_C.end(), h_C.begin() );
  thrust::copy( d_O.begin(), d_O.end(), h_O.begin() );

  // Compute error
  for ( int b = 0; b < batchSize; b++ )
  {
    for ( int i = 0; i < m; i ++ )
    {
      if ( h_C[ i ].second != h_O[ i ].second )
      {
        printf( "C[%4d] = (%5.2lf,%4d) != (%5.2lf,%4d) = O[%4d]\n",
            i, h_C[ i + b * m * n ].first, h_C[ i + b * m * n ].second, 
            h_O[ i + b * m * n ].first, h_C[ i + b * m * n ].second, i );
        break;
      }
    }
  }


  flops = ( 2.0 * m * n * k + 3.0 ) * (double)batchSize / ( 1000.0 * 1000.0 * 1000.0 );
  printf( "%4d, %4d, %4d, %4d, %f, %f;\n", m, n, k, batchSize, 
      flops / gkrm_time, flops / gkrm_strided_time );
}


///*
// *  @brief This is an example to invoke the gkmmv instance that implements the
// *         kernel summation. 2-norm option is used in this
// *         case; so A2 and B2 are precomputed. The kernel function take the
// *         square distance and its weight, returning a weighted scalar. The reduce
// *         function computes summation between two saclars and return the potential.
// *        
// */ 
//template <typename T>
//void test_gkmmv( int m, int n, int k, int batchSize )
//{
//  float  gkmmv_time = 0.0, gkmmv_strided_time = 0.0;
//  double flops = 0.0;
//  struct gkmx_s<T> gkmx;
//
//  cudaEvent_t gkmx_beg, gkmx_end;
//  cublasOperation_t transa = CUBLAS_OP_N; 
//  cublasOperation_t transb = CUBLAS_OP_N; 
//  cudaEventCreate( &gkmx_beg );
//  cudaEventCreate( &gkmx_end );
//
//
//  // Thrust style
//  thrust::host_vector<T> h_A( m * k * batchSize );
//  thrust::host_vector<T> h_B( k * n * batchSize );
//  thrust::host_vector<T> h_C( m * n * batchSize );
//
//  for ( int i = 0; i < batchSize; i ++ ) {
//    for ( int j = 0; j < m * k; j ++ ) h_A[ i * m * k + j ] = (T) rand() / (T) RAND_MAX;
//    for ( int j = 0; j < k * n; j ++ ) h_B[ i * k * n + j ] = (T) rand() / (T) RAND_MAX;
//    for ( int j = 0; j < m * n; j ++ ) h_C[ i * m * n + j ] = (T) 0.0;
//  }
//
//  // memcpyHostToDevice
//  thrust::device_vector<T> d_A = h_A;
//  thrust::device_vector<T> d_B = h_B;
//  thrust::device_vector<T> d_C = h_C;
//
//  // Thrust style
//  thrust::device_vector<T*> d_Ap( batchSize );
//  thrust::device_vector<T*> d_Bp( batchSize );
//  thrust::device_vector<T*> d_Cp( batchSize );
//
//  
//  for ( int i = 0; i < batchSize; ++ i ) {
//    d_Ap[ i ] = d_A.data().get() + i * m * k;
//    d_Bp[ i ] = d_B.data().get() + i * k * n;
//    d_Cp[ i ] = d_C.data().get() + i * m * n;
//  }
//  
//  // Setup struct gkmx 
//  gkmx.type = GKMX_GAUSSIAN;
//  gkmx.h    = 1.0;
//
//  // Compute squared 2 norm for RBF kernel.
//  thrust::device_vector<T>  d_A2( m * batchSize );
//  thrust::device_vector<T>  d_B2( n * batchSize );
//  thrust::device_vector<T*> d_A2p( batchSize );
//  thrust::device_vector<T*> d_B2p( batchSize );
//  for ( int i = 0; i < batchSize; i ++ ) {
//    d_A2p[ i ] = d_A2.data().get() + i * m;
//    d_B2p[ i ] = d_B2.data().get() + i * n;
//  }
//  gkmx.A2 = d_A2p.data().get();
//  gkmx.B2 = d_B2p.data().get();
//
//  dsq2nrm_batched_t( k, m, gkmx.A2, (const T**)d_Ap.data().get(), NULL, m, batchSize );
//  dsq2nrm_batched_n( k, n, gkmx.B2, (const T**)d_Bp.data().get(), NULL, k, batchSize );
//
//  // Setup u and w
//  thrust::host_vector<T>  h_u( m * batchSize, (T)0.0 );
//  thrust::device_vector<T>  d_u = h_u;
//  thrust::device_vector<T*>  d_up( batchSize );
//  for ( int i = 0; i < batchSize; ++ i ) d_up[ i ] = d_u.data().get() + i * m;
//  gkmx.u = d_up.data().get();
//
//  thrust::host_vector<T>  h_w( n * batchSize );
//  for ( int i = 0; i < n; i ++ ) h_w[ i ] = 0.1 * (double)i;
//  thrust::device_vector<T>  d_w = h_w;
//  thrust::device_vector<T*>  d_wp( batchSize );
//  for ( int i = 0; i < batchSize; ++ i ) d_wp[ i ] = d_w.data().get() + i * n;
//  gkmx.w = d_wp.data().get();
//  
//
//  // batch_cuda_gkmx
//  cudaEventRecord( gkmx_beg, 0 );
//  dgkmmv_batched_instance(
//      0, transa, transb,
//      m, n, k, 
//      (const T**)d_Ap.data().get(), m,
//      (const T**)d_Bp.data().get(), k,
//      d_Cp.data().get(), m,
//      batchSize,
//      &gkmx );
//  cudaThreadSynchronize();
//  cudaEventRecord( gkmx_end, 0 );
//  cudaEventSynchronize( gkmx_end );
//  cudaEventElapsedTime( &gkmmv_time, gkmx_beg, gkmx_end );
//  gkmmv_time /= 1000.0;
//
//
//  // Strided GKMMV
//  cudaEventRecord( gkmx_beg, 0 );
//  dgkmmv_batched_strided_instance(
//      0, transa, transb,
//      m, n, k, 
//      d_A.data().get(), m, m * k,
//      d_B.data().get(), k, k * n,
//      d_C.data().get(), m, m * n,
//      batchSize,
//      &gkmx );
//  cudaThreadSynchronize();
//  cudaEventRecord( gkmx_end, 0 );
//  cudaEventSynchronize( gkmx_end );
//  cudaEventElapsedTime( &gkmmv_strided_time, gkmx_beg, gkmx_end );
//  gkmmv_strided_time /= 1000.0;
//
//  flops = ( 2.0 * m * n * k + 2.0 + 35.0 ) * (double)batchSize / ( 1000.0 * 1000.0 * 1000.0 );
//  printf( "%4d, %4d, %4d, %4d, %f, %f;\n", m, n, k, batchSize, 
//      flops / gkmmv_time, flops / gkmmv_strided_time );
//}


/**
 *  @brief The main program that calls three instances.
 *
 */
int main (int argc, char *argv[])
{
  int m, n, k, batchSize;

  if ( argc < 5 ) 
  {
    printf( "test_gkmm_gpu: requires 4 arguments\n" );
    exit( 1 );
  }

  sscanf( argv[ 1 ], "%d", &m );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &k );
  sscanf( argv[ 4 ], "%d", &batchSize );

  test_gkmm<double>( m, n, k, batchSize );
  //test_gkrm<double, thrust::pair<double, int> >( m, n, k, batchSize );
  //test_gkmmv<double>( m, n, k, batchSize );

  return 0;
}
