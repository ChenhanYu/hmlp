/**
 *  -- GKMX (version 1.1.1) --
 *
 *  NVIDIA Corp, Santa Clara
 *
 *  @date June 2016
 *  @author Chenhan D. Yu
 *
 *  Modification
 *
 *
 *
 */

#ifndef GKMX_CUH
#define GKMX_CUH

#include <hmlp.h>
#include <hmlp_util.hpp>

#ifndef mymax
#define mymax(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef mymin
#define mymin(a, b) ((a) < (b) ? (a) : (b))
#endif
#define fetch(A, m, n, bound) offs_d##A[mymin(n*LD##A+m, bound)]

#define strfetch(A, i, m, n, bound) offs_d##A##i[mymin(n*LD##A, bound)]

namespace hmlp
{
namespace strassen
{

//#define version(s,v) s ## _V_ ## v
//
//// GKMM macros (see gkmx_template_kernel_batched.hxx for the definition.)
//#define gkmm_macro(ta,tb,s,v) gkmm_internal \
//  < ta, tb, s ## _V_ ## v, SQ2NRM, OPKERNEL, OP1, OP2, TA, TB, TC, TV> \
//  ( \
//  stream, \
//  m, n, k, \
//  Aarray, lda, \
//  Barray, ldb, \
//  Carray, ldc, \
//  batchSize, \
//  opkernel, op1, op2, initV ) 
//
//#define gkmm_strided_macro(ta,tb,s,v) gkmm_internal \
//  < ta, tb, s ## _V_ ## v, SQ2NRM, OPKERNEL, OP1, OP2, TA, TB, TC, TV> \
//  ( \
//  stream, \
//  m, n, k, \
//  Aarray, lda, loa, \
//  Barray, ldb, lob, \
//  Carray, ldc, loc, \
//  batchSize, \
//  opkernel, op1, o2, initV ) 


#if __CUDA_ARCH__ < 600
  __device__ double atomicAdd(double* address, double val)
  {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do 
    {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
          __double_as_longlong(val + __longlong_as_double(assumed)));

      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return
      __longlong_as_double(old);
  }
#endif




template<
bool TRANSA, bool TRANSB,
int DIM_X, int DIM_Y,
int BLK_M, int BLK_N, int BLK_K, 
int DIM_XA, int DIM_YA, int DIM_XB, int DIM_YB, 
int THR_M, int THR_N, 
int GAMMA, int DELTA, int ALPHA0, int ALPHA1,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
static __device__ void strassen_device
(
  int M, int N, int K,
  const TA* __restrict__ A0, const TA* __restrict__ A1, int LDA,
  const TB* __restrict__ B0, const TB* __restrict__ B1, int LDB,
        TC* __restrict__ C0,       TC* __restrict__ C1, int LDC,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV 
)
{
#if (__CUDA_ARCH__ >= 200)

  // Semi-ring rank-k update template
  #include <strassen_stencil.hpp>

  // Store C regs->dev
  #pragma unroll
  for ( n = 0; n < THR_N; n ++ ) 
  {
    int coord_dCn = bly * BLK_N + n * DIM_Y + idy;
    #pragma unroll
    for ( m = 0; m < THR_M; m ++ ) 
    {
      int coord_dCm = blx * BLK_M + m * DIM_X + idx;
      if ( coord_dCm < M && coord_dCn < N ) 
      {
        int offsC = coord_dCn * LDC + coord_dCm;
        TV &regC = rC[ n ][ m ];

        if ( ALPHA0 )
        {
          TC &memC0 = C0[ offsC ];
          memC0 += ALPHA0 * regC;
          //atomicAdd( C0 + offsC, ALPHA0 * regC );
        }

        if ( ALPHA1 )
        {
          TC &memC1 = C1[ offsC ];
          memC1 += ALPHA1 * regC;
          //atomicAdd( C1 + offsC, ALPHA1 * regC );
        }

        // Not sure what's the operation for C0 and C1
        //memC0 = opkernel( regC, coord_dCm, coord_dCn, blockIdx.z );
        //memC1 = opkernel( regC, coord_dCm, coord_dCn, blockIdx.z );
      }
    }
  }

#endif /* (__CUDA_ARCH__ >= 200) */
};



template<
bool TRANSA, bool TRANSB,
int DIM_X, int DIM_Y, 
int BLK_M, int BLK_N, int BLK_K,
int DIM_XA, int DIM_YA, 
int DIM_XB, int DIM_YB,
int CASE,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
static __global__ void strassen_kernel
(
  int M, int N, int K,
  const TA *Aarray[], int LDA, 
  const TB *Barray[], int LDB, 
        TC *Carray[], int LDC, 
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV
)
{
  // acquire all partitions
  const TA *A00, *A01, *A10, *A11;
  const TB *B00, *B01, *B10, *B11;
        TC *C00, *C01, *C10, *C11;

  if ( TRANSA )
  {
    A00 = Aarray[ blockIdx.z ];
    A01 = A00 + K / 2;
    A10 = A00 + ( M / 2 ) * LDA;
    A11 = A10 + K / 2;
  }
  else
  {
    A00 = Aarray[ blockIdx.z ];
    A01 = A00 + ( K / 2 ) * LDA;
    A10 = A00 + M / 2;
    A11 = A01 + M / 2;
  }

  if ( TRANSB )
  {
    B00 = Barray[ blockIdx.z ];
    B01 = B00 + N / 2;
    B10 = B00 + ( K / 2 ) * LDB;
    B11 = B10 + N / 2;
  }
  else
  {
    B00 = Barray[ blockIdx.z ];
    B01 = B00 + ( N / 2 ) * LDA;
    B10 = B00 + K / 2;
    B11 = B01 + K / 2;
  }

  C00 = Carray[ blockIdx.z ];
  C01 = C00 + ( N / 2 ) * LDC;
  C10 = C00 + M / 2;
  C11 = C01 + M / 2;


  switch ( CASE )
  {
    case 1:
      // M1
      strassen_device<
        TRANSA, TRANSB,
        DIM_X, DIM_Y, 
        BLK_M, BLK_N, BLK_K,
        DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
        (BLK_M/DIM_X), (BLK_N/DIM_Y),
        1, 1, 1, 1,
        SQ2NRM, OPKERNEL, OP1, OP2,
        TA, TB, TC, TV>
      (
        M / 2, N / 2, K / 2, 
        A00, A11, LDA, 
        B00, B11, LDB, 
        C00, C11, LDC,
        opkernel, op1, op2, initV 
      );
      break;

    case 2:
      // M2
      strassen_device<
        TRANSA, TRANSB,
        DIM_X, DIM_Y, 
        BLK_M, BLK_N, BLK_K,
        DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
        (BLK_M/DIM_X), (BLK_N/DIM_Y),
        1, 0, 1, -1,
        SQ2NRM, OPKERNEL, OP1, OP2,
        TA, TB, TC, TV>
      (
        M / 2, N / 2, K / 2, 
        A10,  A11, LDA, 
        B00, NULL, LDB, 
        C10,  C11, LDC,
        opkernel, op1, op2, initV 
      );
      break;

    case 3:
      // M3
      strassen_device<
        TRANSA, TRANSB,
        DIM_X, DIM_Y, 
        BLK_M, BLK_N, BLK_K,
        DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
        (BLK_M/DIM_X), (BLK_N/DIM_Y),
        0, -1, 1, 1,
        SQ2NRM, OPKERNEL, OP1, OP2,
        TA, TB, TC, TV>
      (
        M / 2, N / 2, K / 2, 
        A00, NULL, LDA, 
        B01,  B11, LDB, 
        C01,  C11, LDC,
        opkernel, op1, op2, initV 
      );
      break;

    case 4:
      // M4
      strassen_device<
        TRANSA, TRANSB,
        DIM_X, DIM_Y, 
        BLK_M, BLK_N, BLK_K,
        DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
        (BLK_M/DIM_X), (BLK_N/DIM_Y),
        0, -1, 1, 1,
        SQ2NRM, OPKERNEL, OP1, OP2,
        TA, TB, TC, TV>
      (
        M / 2, N / 2, K / 2, 
        A11, NULL, LDA, 
        B10,  B00, LDB, 
        C00,  C10, LDC,
        opkernel, op1, op2, initV 
      );
      break;

    case 5:
      // M5
      strassen_device<
        TRANSA, TRANSB,
        DIM_X, DIM_Y, 
        BLK_M, BLK_N, BLK_K,
        DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
        (BLK_M/DIM_X), (BLK_N/DIM_Y),
        1, 0, -1, 1,
        SQ2NRM, OPKERNEL, OP1, OP2,
        TA, TB, TC, TV>
      (
        M / 2, N / 2, K / 2, 
        A00,  A01, LDA,
        B11, NULL, LDB,
        C00,  C01, LDC,
        opkernel, op1, op2, initV 
      );
      break;

    case 6:
      // M6
      strassen_device<
        TRANSA, TRANSB,
        DIM_X, DIM_Y, 
        BLK_M, BLK_N, BLK_K,
        DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
        (BLK_M/DIM_X), (BLK_N/DIM_Y),
        -1, 1, 1, 0,
        SQ2NRM, OPKERNEL, OP1, OP2,
        TA, TB, TC, TV>
      (
        M / 2, N / 2, K / 2, 
        A10,  A00, LDA,
        B00,  B01, LDB,
        C11, NULL, LDC,
        opkernel, op1, op2, initV 
      );
      break;

    case 7:
      // M7
      strassen_device<
        TRANSA, TRANSB,
        DIM_X, DIM_Y, 
        BLK_M, BLK_N, BLK_K,
        DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
        (BLK_M/DIM_X), (BLK_N/DIM_Y),
        -1, 1, 1, 0,
        SQ2NRM, OPKERNEL, OP1, OP2,
        TA, TB, TC, TV>
      (
        M / 2, N / 2, K / 2, 
        A01,  A11, LDA,
        B10,  B11, LDB,
        C00, NULL, LDC,
        opkernel, op1, op2, initV 
      );
      break;
    default:
      break;
  }

};


template<bool TRANSA, bool TRANSB,
int DIM_X, int DIM_Y, 
int BLK_M, int BLK_N, int BLK_K,
int DIM_XA, int DIM_YA, 
int DIM_XB, int DIM_YB,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
static __global__ void strassen_kernel
(
  int M, int N, int K,
  const TA *A, int LDA, int LOA,
  const TB *B, int LDB, int LOB,
        TC *C, int LDC, int LOC,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV 
)
{
  // To be implemented
};


template<
bool TRANSA, bool TRANSB,
int DIM_X, int DIM_Y, 
int BLK_M, int BLK_N, int BLK_K,
int dim_vec,
int DIM_XA, int DIM_YA, int DIM_XB, int DIM_YB,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
void strassen_internal
(
  cudaStream_t stream, 
  int m, int n, int k,
  const TA *Aarray[], int lda,
  const TB *Barray[], int ldb,
        TC *Carray[], int ldc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV
)
{
  cudaStream_t str_stream[ 2 ];
  cudaStreamCreate( &str_stream[ 0 ] ) ;
  cudaStreamCreate( &str_stream[ 1 ] ) ;

  dim3 dimBlock( DIM_X, DIM_Y );
  dim3 dimGrid( hmlp_ceildiv( m / 2, BLK_M ), hmlp_ceildiv( n / 2, BLK_N ), batchSize );

  strassen_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    7,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, str_stream[ 0 ] >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    opkernel, op1, op2, initV
  );
  strassen_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    4,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, str_stream[ 0 ] >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    opkernel, op1, op2, initV
  );
  strassen_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    3,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, str_stream[ 1 ] >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    opkernel, op1, op2, initV
  );
  strassen_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    6,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, str_stream[ 1 ] >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    opkernel, op1, op2, initV
  );

  cudaStreamSynchronize( str_stream[ 0 ] );
  cudaStreamSynchronize( str_stream[ 1 ] );

  strassen_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    5,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, str_stream[ 0 ] >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    opkernel, op1, op2, initV
  );
  strassen_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    2,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, str_stream[ 1 ] >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    opkernel, op1, op2, initV
  );

  cudaStreamSynchronize( str_stream[ 0 ] );
  cudaStreamSynchronize( str_stream[ 1 ] );

  strassen_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    1,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, str_stream[ 0 ] >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    opkernel, op1, op2, initV
  );

  cudaStreamSynchronize( str_stream[ 0 ] );
  cudaStreamSynchronize( str_stream[ 1 ] );
};


/**
 *  batched version
 */ 
template<
bool TRANSA, bool TRANSB,
int DIM_X, int DIM_Y, 
int BLK_M, int BLK_N, int BLK_K,
int dim_vec,
int DIM_XA, int DIM_YA, int DIM_XB, int DIM_YB,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
void strassen_internal
(
  cudaStream_t stream, 
  int m, int n, int k,
  const TA *Aarray, int lda, int loa,
  const TB *Barray, int ldb, int lob,
        TC *Carray, int ldc, int loc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV 
)
{
  dim3 dimBlock( DIM_X, DIM_Y );
  dim3 dimGrid( hmlp_ceildiv( m / 2, BLK_M ), hmlp_ceildiv( n / 2, BLK_N ), batchSize );

  strassen_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, stream >>>
  ( 
    m, n, k, 
    Aarray, lda, loa,
    Barray, ldb, lob,
    Carray, ldc, loc,
    opkernel, op1, op2, initV
  );
};


/**
 *  @brief This is the GKMM (General Kernel Matrix Matrix) template wrapper.
 *         This interface accepts double pointers. Here the type rules are
 *         op2: <TA,TB> to <TV>,
 *         op1: <TV,TV> to <TV>, and
 *         opkernel: <TV> to <TC>.
 *
 *  @param TA Type of *Aarray[]
 *  @param TB Type of *Barray[]
 *  @param TC Type of *Carray[]
 *  @param TV Type of of the output of op1 and op2
 *  @param SQ@NRM Whether opkernel uses a^2-2ab+b^2 expansion or not
 *  @param OPKERNEL Type of opkernel
 *  @param OP1 Type of op1
 *  @param OP2 Type of op2
 *
 *  @param stream CUDA stream
 *  @param transA Can either be CUBLAS_OP_N or CUBLAS_OP_T
 *  @param transB Can either be CUBLAS_OP_N or CUBLAS_OP_T
 *  @param m Input matrix dimension
 *  @param n Input matrix dimension
 *  @param k Input matrix dimension
 *  @param Aarray Input matrices in double pointers
 *  @param lda Leading dimension of matrix A
 *  @param Barray Input matrices in double pointers
 *  @param ldb Leading dimension of matrix B
 *  @param Carray Ounput matrices in double pointers
 *  @param ldc Leading dimension of matrix C
 *  @param batchSize number of indepedent gkmm operations
 *  @opkernel Closure of the kernel operators
 *  @op1 Closure of the semi-ring operator
 *  @op2 Closure of the semi-ring operator
 *  @init1 Initial value of the semi-ring operators
 *
 */ 
template<
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV> 
void strassen
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  const TA *Aarray[], int lda, 
  const TB *Barray[], int ldb, 
        TC *Carray[], int ldc, 
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV 
)
{
  // Early return.
  if ( m <= 0 || n <= 0 || k <= 0 ) return;

  // Specify input formats
  int shape = 0;
  if      ( transA == HMLP_OP_N && transB == HMLP_OP_N ) { shape = 0; } // nn
  else if ( transA == HMLP_OP_N && transB == HMLP_OP_T ) { shape = 1; } // nt
  else if ( transA == HMLP_OP_T && transB == HMLP_OP_N ) { shape = 3; } // tn
  else if ( transA == HMLP_OP_T && transB == HMLP_OP_T ) { shape = 4; } // tt

  // Autotuned decision tree
  //#include <gkmm_autotune.hpp>
 
  // NN case for testing
  strassen_internal
  <false,false, 16, 16, 96, 80, 16, 1, 16, 16, 16, 16,
  SQ2NRM, OPKERNEL, OP1, OP2>
  (
    stream,
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    batchSize,
    opkernel, op1, op2, initV
  );
    

}

/**
 *  @brief This is the GKMM (General Kernel Matrix Matrix) template wrapper.
 *         This interface accepts pointers with strided access. 
 *         Here the type rules are
 *         op2: <TA,TB> to <TV>,
 *         op1: <TV,TV> to <TV>, and
 *         opkernel: <TV> to <TC>.
 *
 *  @param TA Type of *Aarray
 *  @param TB Type of *Barray
 *  @param TC Type of *Carray
 *  @param TV Type of of the output of op1 and op2
 *  @param SQ@NRM Whether opkernel uses a^2-2ab+b^2 expansion or not
 *  @param OPKERNEL Type of opkernel
 *  @param OP1 Type of op1
 *  @param OP2 Type of op2
 *
 *  @param stream CUDA stream
 *  @param transA Can either be CUBLAS_OP_N or CUBLAS_OP_T
 *  @param transB Can either be CUBLAS_OP_N or CUBLAS_OP_T
 *  @param m Input matrix dimension
 *  @param n Input matrix dimension
 *  @param k Input matrix dimension
 *  @param Aarray Input matrices (at least m-by-k-by-batchSize)
 *  @param lda Leading dimension of matrix A
 *  @param loa Stride between each matrix A
 *  @param Barray Input matrices (at least k-by-n-by-batchSize)
 *  @param ldb Leading dimension of matrix B
 *  @param lob Stride between each matrix B
 *  @param Carray Ounput matrices (at least m-by-n-by-batchSize)
 *  @param ldc Leading dimension of matrix C
 *  @param loc Stride between each matrix C
 *  @param batchSize number of indepedent gkmm operations
 *  @opkernel Closure of the kernel operators
 *  @op1 Closure of the semi-ring operator
 *  @op2 Closure of the semi-ring operator
 *  @init1 Initial value of the semi-ring operators
 *
 */ 
template<
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV> 
void strassen
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  const TA *Aarray, int lda, int loa,
  const TB *Barray, int ldb, int lob,
        TC *Carray, int ldc, int loc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV
)
{
  // Early return.
  if ( m <= 0 || n <= 0 || k <= 0 ) return;

  // Specify input formats
  int shape = 0;
  if      ( transA == HMLP_OP_N && transB == HMLP_OP_N ) { shape = 0; } // nn
  else if ( transA == HMLP_OP_N && transB == HMLP_OP_T ) { shape = 1; } // nt
  else if ( transA == HMLP_OP_T && transB == HMLP_OP_N ) { shape = 3; } // tn
  else if ( transA == HMLP_OP_T && transB == HMLP_OP_T ) { shape = 4; } // tt

  // Autotuned decision tree
  //#include <gkmm_strided_autotune.hpp>

  // NN case for testing
  strassen_internal
  <false,false, 16, 16, 64, 80, 16, 1, 16, 16, 16, 16,
  SQ2NRM, OPKERNEL, OP1, OP2>
  (
    stream,
    m, n, k, 
    Aarray, lda, loa, 
    Barray, ldb, lob,
    Carray, ldc, loc,
    batchSize,
    opkernel, op1, op2, initV
  );

};


}; // end namespace gkmm
}; // end namespace hmlp


#endif // define GKMX_CUH
