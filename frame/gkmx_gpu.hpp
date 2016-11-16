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

#include <thrust/functional.h>

#include <hmlp.h>
#include <hmlp_util.hpp>

#ifndef mymax
#define mymax(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef mymin
#define mymin(a, b) ((a) < (b) ? (a) : (b))
#endif
#define fetch(A, m, n, bound) offs_d##A[mymin(n*LD##A+m, bound)]
//#define fetch(A, m, n, bound) offs_d##A[thrust::minimum<int>(n*LD##A+m, bound)]

#define GKMX_GPU_CONFIG \
bool TRANSA, bool TRANSB,\
int DIM_X, int DIM_Y,\
int BLK_M, int BLK_N, int BLK_K,\
int DIM_XA, int DIM_YA, int DIM_XB, int DIM_YB

#define version(s,v) s ## _V_ ## v


// GKMM macros (see gkmx_template_kernel_batched.hxx for the definition.)
#define gkmm_macro(ta,tb,s,v) gkmm_internal \
  < ta, tb, s ## _V_ ## v, SQ2NRM, OPKERNEL, OP1, OP2, TA, TB, TC, TV> \
  ( \
  stream, \
  m, n, k, \
  Aarray, lda, \
  Barray, ldb, \
  Carray, ldc, \
  batchSize, \
  opkernel, op1, op2, initV ) 

#define gkmm_strided_macro(ta,tb,s,v) gkmm_internal \
  < ta, tb, s ## _V_ ## v, SQ2NRM, OPKERNEL, OP1, OP2, TA, TB, TC, TV> \
  ( \
  stream, \
  m, n, k, \
  Aarray, lda, loa, \
  Barray, ldb, lob, \
  Carray, ldc, loc, \
  batchSize, \
  opkernel, op1, op2, initV ) 

// GKRM macros (see gkmx_template_kernel_batched.hxx for the definition.)
#define gkrm_macro(ta,tb,s,v) gkrm_internal \
  < ta, tb, s ## _V_ ## v, SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE, TA, TB, TC, TV> \
  ( \
  stream, \
  m, n, k, \
  Aarray, lda, \
  Barray, ldb, \
  Carray, ldc, \
  batchSize, \
  opkernel, op1, op2, initV, opreduce, initC ) 

#define gkrm_strided_macro(ta,tb,s,v) gkrm_internal \
  < ta, tb, s ## _V_ ## v, SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE, TA, TB, TC, TV> \
  ( \
  stream, \
  m, n, k, \
  Aarray, lda, loa, \
  Barray, ldb, lob, \
  Carray, ldc, loc, \
  batchSize, \
  opkernel, op1, op2, initV, opreduce, initC ) 


namespace hmlp
{
namespace gkmx
{


template<
GKMX_GPU_CONFIG, int THR_M, int THR_N,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
static __device__ void gkmm_device
(
  int M, int N, int K,
  const TA* __restrict__ A, int LDA,
  const TB* __restrict__ B, int LDB,
        TC* __restrict__ C, int LDC,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV 
)
{
#if (__CUDA_ARCH__ >= 200)

  // Semi-ring rank-k update template
  #include <gkmm_stencil.hpp>

  // SQ2NRM option
  if ( SQ2NRM ) 
  {
    __syncthreads();
    if ( idt < BLK_M && blx * BLK_M + idt < M ) 
    {
      sA[ 0 ][ idt ] = opkernel.A2[ blockIdx.z ][ blx * BLK_M + idt ];
    }
    if ( idt < BLK_N && bly * BLK_N + idt < N ) 
    {
      sB[ idt ][ 0 ] = opkernel.B2[ blockIdx.z ][ bly * BLK_N + idt ];
    }
    __syncthreads();
  }

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
        TC &memC = C[ offsC ];
        if ( SQ2NRM ) 
        {
          regC *= -2.0;
          regC += sA[ 0 ][ m * DIM_X + idx ] + sB[ n * DIM_Y + idy ][ 0 ];
        }
        memC = opkernel( regC, coord_dCm, coord_dCn, blockIdx.z );
      }
    }
  }

#endif /* (__CUDA_ARCH__ >= 200) */
};


template<
GKMX_GPU_CONFIG,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
static __global__ void gkmm_kernel
(
  int M, int N, int K,
  const TA *Aarray[], int LDA, 
  const TB *Barray[], int LDB, 
        TC *Carray[], int LDC, 
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV
)
{
  gkmm_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  (
    M, N, K, 
    Aarray[ blockIdx.z ], LDA,
    Barray[ blockIdx.z ], LDB,
    Carray[ blockIdx.z ], LDC,
    opkernel, op1, op2, initV 
  );
};


template<
GKMX_GPU_CONFIG,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
static __global__ void gkmm_kernel
(
  int M, int N, int K,
  const TA *A, int LDA, int LOA,
  const TB *B, int LDB, int LOB,
        TC *C, int LDC, int LOC,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV 
)
{
  gkmm_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  (
    M, N, K, 
    A + LOA * blockIdx.z, LDA,
    B + LOB * blockIdx.z, LDB,
    C + LOC * blockIdx.z, LDC,
    opkernel, op1, op2, initV 
  );
};


template<
bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
void gkmm_internal
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
  dim3 dimBlock( DIM_X, DIM_Y );
  dim3 dimGrid( hmlp_ceildiv( m, BLK_M ), hmlp_ceildiv( n, BLK_N ), batchSize );

  gkmm_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    SQ2NRM, OPKERNEL, OP1, OP2,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, stream >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc,
    opkernel, op1, op2, initV
  );
};


/**
 *  batched version
 */ 
template<
bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2,
typename TA, typename TB, typename TC, typename TV>
void gkmm_internal
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
  dim3 dimGrid( hmlp_ceildiv( m, BLK_M ), hmlp_ceildiv( n, BLK_N ), batchSize );

  gkmm_kernel<
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
void gkmm
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
  #include <gkmm_autotune.hpp>
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
void gkmm
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
  #include <gkmm_strided_autotune.hpp>
};



/**
 *  @brief gkrm 
 *
 *
 */

template<
GKMX_GPU_CONFIG, int THR_M, int THR_N, 
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV>
static __device__ void gkrm_device
(
  int M, int N, int K,
  const TA* __restrict__ A, int LDA,
  const TB* __restrict__ B, int LDB,
        TC* __restrict__ C, int LDC,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, OPREDUCE opreduce, TC initC 
)
{
  TC rc[THR_M];

  // Semi-ring rank-k update template
  #include <gkmm_stencil.hpp>

  // SQ2NRM option
  if ( SQ2NRM ) 
  {
    __syncthreads();
    if ( idt < BLK_M && blx * BLK_M + idt < M ) 
    {
      sA[ 0 ][ idt ] = opkernel.A2[ blockIdx.z ][ blx * BLK_M + idt ];
    }
    if ( idt < BLK_N && bly * BLK_N + idt < N ) 
    {
      sB[ idt ][ 0 ] = opkernel.B2[ blockIdx.z ][ bly * BLK_N + idt ];
    }
    __syncthreads();
  }

  #pragma unroll
  for ( m = 0; m < THR_M; m ++ ) 
  {
    int coord_dCm = blx*BLK_M + m*DIM_X + idx;
    int offsC = ( bly * DIM_Y + idy ) * LDC + coord_dCm;
    #pragma unroll
    for ( n = 0; n < THR_N; n ++ ) 
    {
      int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
      if ( coord_dCm < M && coord_dCn < N ) 
      {
        TV  &regC = rC[ n ][ m ];
        TC regK;
        if ( SQ2NRM ) 
        {
          regC *= -2.0;
          regC += sA[ 0 ][ m * DIM_X + idx ] + sB[ n * DIM_Y + idy ][ 0 ];
        }
        regK = opkernel( regC, coord_dCm, coord_dCn, blockIdx.z );
        if ( !n ) 
        {
          rc[ m ] = regK;
        }
        else 
        {
          rc[ m ] = opreduce( rc[ m ], regK, coord_dCm, coord_dCn, blockIdx.z );
        }
      }
    }
    // For special case where DIM_Y < N, we need condition idy < N.
    if ( coord_dCm < M && bly * BLK_N < N && idy < N ) 
    {
      C[ offsC ] = rc[ m ];
    }
  }
};


/** 
 *  @brief A simple row-wise reduction, which does not exploit the parallelism
 *         of the binary tree.
 */ 
template<typename TC, typename OPREDUCE>
static __device__
void reduce_device
( 
  int M, int N, 
  TC* __restrict__ C, int LDC,
  OPREDUCE opreduce
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx < M ) 
  {
    TC ru = C[ idx ];
    for ( int j = 1; j < N; j ++ ) 
    {
      ru = opreduce( ru, C[ j * LDC + idx ], idx, j, blockIdx.z );    
    }
    C[ idx ] = ru;
  }
}


template<typename TC, typename OPREDUCE>
static __global__
void reduce_kernel
( 
  int M, int N, 
  TC** Carray, int LDC,
  OPREDUCE opreduce 
)
{
  reduce_device<TC, OPREDUCE>
  ( 
    M, N, 
    Carray[ blockIdx.z ], LDC, 
    opreduce 
  );
};


/*
 *
 */ 
template<typename TC, typename OPREDUCE>
static __global__
void reduce_kernel
( 
  int M, int N, 
  TC* Carray, int LDC, int LOC,
  OPREDUCE opreduce 
)
{
  reduce_device<TC, OPREDUCE>
  ( 
    M, N, 
    Carray + blockIdx.z * LOC, LDC, 
    opreduce 
  );
};


template <typename TC, bool STRIDED, typename OPREDUCE>
void reduce
(
  cudaStream_t stream,
  int m, int n, 
  TC* Carray[], TC* C, int ldc, int loc,
  int batchSize, 
  OPREDUCE opreduce 
)
{
  dim3 dimBlock( 256, 1 );
  dim3 dimGrid( ( m - 1 ) / 256 + 1, 1, batchSize );
  reduce_kernel<TC, OPREDUCE>
  <<<dimGrid, dimBlock, 0, stream>>>
  ( 
    m, n, 
    Carray, ldc,
    opreduce 
  );
};








template<
GKMX_GPU_CONFIG,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV>
static __global__ void gkrm_kernel
(
  int M, int N, int K,
  const TA *Aarray[], int LDA,
  const TB *Barray[], int LDB,
        TC *Carray[], int LDC,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, OPREDUCE opreduce, TC initC 
)
{
  gkrm_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE,
    TA, TB, TC, TV>
  (
    M, N, K, 
    Aarray[ blockIdx.z ], LDA,
    Barray[ blockIdx.z ], LDB,
    Carray[ blockIdx.z ], LDC,
    opkernel, op1, op2, initV, opreduce, initC 
  );
};

template<
GKMX_GPU_CONFIG,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV>
static __global__ void gkrm_kernel
(
  int M, int N, int K,
  const TA *A, int LDA, int LOA,
  const TB *B, int LDB, int LOB,
        TC *C, int LDC, int LOC,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, OPREDUCE opreduce, TC initC 
)
{
  gkrm_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE,
    TA, TB, TC, TV>
  (
    M, N, K, 
    A + LOA * blockIdx.z, LDA,
    B + LOB * blockIdx.z, LDB,
    C + LOC * blockIdx.z, LDC,
    opkernel, op1, op2, initV, opreduce, initC 
  );
};


template<
bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV>
void gkrm_internal
(
  cudaStream_t stream, 
  int m, int n, int k,
  TA *Aarray[], int lda,
  TB *Barray[], int ldb,
  TC *Carray[], int ldc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, OPREDUCE opreduce, TC initC
)
{
  dim3 dimBlock( DIM_X, DIM_Y );
  dim3 dimGrid( hmlp_ceildiv( m, BLK_M ), hmlp_ceildiv( n, BLK_N ), batchSize );

  gkrm_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, stream >>>
  ( 
    m, n, k, 
    (const TA**)Aarray, lda,
    (const TB**)Barray, ldb,
    Carray, ldc, 
    opkernel, op1, op2, initV, opreduce, initC
  );

  dim3 dimBlockReduce( 256, 1 );
  dim3 dimGridReduce( ( m - 1 ) / 256 + 1, 1, batchSize );

  reduce_kernel<TC, OPREDUCE>
  <<< dimGridReduce, dimBlockReduce, 0, stream >>>
  ( 
    m, ( ( n - 1 ) / BLK_N + 1 ) * DIM_Y, 
    Carray, ldc, 
    opreduce 
  );
};

template <
bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV>
void gkrm_internal
(
  cudaStream_t stream, 
  int m, int n, int k,
  TA *Aarray, int lda, int loa,
  TB *Barray, int ldb, int lob,
  TC *Carray, int ldc, int loc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, OPREDUCE opreduce, TC initC 
)
{
  dim3 dimBlock( DIM_X, DIM_Y );
  dim3 dimGrid( gkmx_ceildiv( m, BLK_M ), gkmx_ceildiv( n, BLK_N ), batchSize );

  gkrm_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE,
    TA, TB, TC, TV>
  <<< dimGrid, dimBlock, 0, stream >>>
  ( 
    m, n, k, 
    (const TA*)Aarray, lda, loa,
    (const TB*)Barray, ldb, lob,
    Carray, ldc, loc, 
    opkernel, op1, op2, initV, opreduce, initC 
  );

  dim3 dimBlockReduce( 256, 1 );
  dim3 dimGridReduce( ( m - 1 ) / 256 + 1, 1, batchSize );

  reduce_kernel
  <TC, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
  <<< dimGridReduce, dimBlockReduce, 0, stream >>>
  ( 
    m, thrust::minimum<int>( n, ( ( n - 1 ) / BLK_N + 1 ) * DIM_Y ), 
    Carray, ldc, loc, 
    opreduce 
  );
};


template<
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV> 
void gkrm
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  TA *Aarray[], int lda, 
  TB *Barray[], int ldb, 
  TC *Carray[], int ldc, 
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV,
  OPREDUCE opreduce, TC initC
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
  #include <gkrm_autotune.hpp>
};


template<
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV> 
void gkrm
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  TA *Aarray, int lda, int loa,
  TB *Barray, int ldb, int lob,
  TC *Carray, int ldc, int loc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, 
  OPREDUCE opreduce, TC initC
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
  #include <gkrm_strided_autotune.hpp>
};



/** 
 *  @brief  Compute square 2-norms for an input matrix X. 
 *  @param 
 *
 */
template <typename T, bool TRANSX>
static __device__ void sq2nrm_device
(
  int d, int n, 
  T* __restrict__ X2, const T* __restrict__ X, int ldx 
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0.0, tmp;
  if ( idx < n ) 
  {
    if ( TRANSX ) 
    {
      for ( int p = 0; p < d; p ++ ) 
      {
        tmp = X[ p * ldx + idx ];
        sum += tmp * tmp;
      }
    }
    else 
    {
      for ( int p = 0; p < d; p ++ ) 
      {
        tmp = X[ idx * ldx + p ];
        sum += tmp * tmp;
      }
    }
    X2[ idx ] = sum;
  }
};

template <typename T, bool STRIDED, bool TRANSX>
static void __global__ sq2nrm_kernel
(
  int d, int n, 
  T* X2array[], const T* Xarray[], const T* X, int ldx, long long int lox 
)
{
  int batchid = blockIdx.z;
  if ( STRIDED ) 
  {
    sq2nrm_device<T, TRANSX>
    ( 
      d, n, 
      X2array[ batchid ], X + batchid * lox, ldx 
    );
  }
  else 
  {
    sq2nrm_device<T, TRANSX>
    ( 
      d, n, 
      X2array[ batchid ], Xarray[ batchid ], ldx 
    );
  }
};

template <typename T, bool STRIDED, bool TRANSX>
void sq2nrm
(
  int d, int n, 
  T* X2array[], const T* Xarray[], const T* X, int ldx, long long int lox, 
  int batchSize 
)
{
  dim3 dimBlock( 256, 1 );
  dim3 dimGrid( ( n - 1 ) / 256 + 1, 1, batchSize );

  sq2nrm_kernel<T, STRIDED, TRANSX>
  <<<dimGrid, dimBlock, 0, 0>>>
  ( 
    d, n, 
    X2array, Xarray, X, ldx, lox 
  );
};



/**
 *  @brief  Compute kernel value element-wise.
 *
 */ 
template <typename TV, typename TC, bool SQ2NRM, typename OPKERNEL>
static __device__ void transform_device
(
  int m, int n, 
  TV* __restrict__ V, 
  TC* __restrict__ C, int ldc, 
  OPKERNEL opkernel 
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ TV sA2[ 16 ];
  __shared__ TV sB2[ 16 ];

  if ( idx < m && idy < n ) 
  {
    if ( SQ2NRM ) 
    {
      sA2[ threadIdx.x ] = opkernel.A2[ blockIdx.z ][ idx ];
      sB2[ threadIdx.y ] = opkernel.B2[ blockIdx.z ][ idy ];
      __syncthreads();
      V[ idy * ldc + idx ] *= -2.0;
      V[ idy * ldc + idx ] += sA2[ threadIdx.x ] + sB2[ threadIdx.y ];
    }
    C[ idy * ldc + idx ] = opkernel( V[ idy * ldc + idx ], idx, idy, blockIdx.z );
  }
};

template <typename TV, typename TC, bool STRIDED, bool SQ2NRM, typename OPKERNEL>
static __global__ void transform_kernel
(
  int m, int n, 
  TV* Varray[], TV* V, 
  TC* Carray[], TC* C, int ldc, int loc,
  OPKERNEL opkernel 
)
{
  int batchid = blockIdx.z;
  transform_device<TV, TC, SQ2NRM, OPKERNEL>
  ( 
    m, n, 
    Varray[ batchid ], 
    Carray[ batchid ], ldc, 
    opkernel 
  );
};

template <typename TV, typename TC, bool STRIDED, bool SQ2NRM, typename OPKERNEL>
void transform
(
  cudaStream_t stream,
  int m, int n, 
  TV* Varray[], TV* V, 
  TC* Carray[], TC* C, int ldc, int loc,
  int batchSize, 
  OPKERNEL opkernel 
)
{
  dim3 dimBlock( 16, 16 );
  dim3 dimGrid( ( m - 1 ) / 16 + 1, ( n - 1 ) / 16 + 1, batchSize );
  transform_kernel<TV, TC, STRIDED, SQ2NRM, OPKERNEL>
  <<<dimGrid, dimBlock, 0, stream>>>
  ( 
    m, n, 
    Varray, V, 
    Carray, C, ldc, loc, 
    opkernel 
  );
};












}; // end namespace gkmx
}; // end namespace hmlp


#endif // define GKMX_CUH
