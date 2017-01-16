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

#ifndef GKRM_GPU_HPP
#define GKRM_GPU_HPP

#include <hmlp.h>
#include <hmlp_util.h>

namespace hmlp
{
namespace gkrm
{

#define GKMX_GPU_CONFIG 

#define version(s,v) s ## _V_ ## v

// GKRM macros (see gkmx_template_kernel_batched.hxx for the definition.)
#define gkrm_macro(ta,tb,s,v) gkrm_template_batched_internal \
  < ta, tb, s ## _V_ ## v, TA, TB, TC, TV, SQ2NRM, OPKERNEL, OP1, OP2> \
  ( \
  stream, \
  m, n, k, \
  Aarray, lda, \
  Barray, ldb, \
  Carray, ldc, \
  batchSize, \
  opkernel, op1, op2, initV, opreduce, initC ) 

#define gkrm_strided_macro(ta,tb,s,v) gkrm_template_batched_strided_internal \
  < ta, tb, s ## _V_ ## v, TA, TB, TC, TV, SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE> \
  ( \
  stream, \
  m, n, k, \
  Aarray, lda, loa, \
  Barray, ldb, lob, \
  Carray, ldc, loc, \
  batchSize, \
  opkernel, op1, op2, initV, opreduce, initC ) 


/**
 *  @brief
 *
 *  TODO: It is possible to use atomic to reduce C. The problem is
 *        we are not sure whether size( TC ) is within 64 bits.
 */ 
template<
bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y,
const int BLK_M, const int BLK_N, const int BLK_K, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
const int THR_M, const int THR_N, 
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
  for ( m = 0; m < THR_M; m ++ ) rc[ m ] = init2;

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
}; // end gkrm_device()




/**
 *  @brief
 */ 
template<
bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
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
}; // end gkrm_kernel()

/**
 *  @breif
 */ 
template<
bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
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
  const TA *Aarray[], int lda,
  const TB *Barray[], int ldb,
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
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>
  <<< dimGrid, dimBlock, 0, stream >>>
  ( 
    m, n, k, 
    Aarray, lda,
    Barray, ldb,
    Carray, ldc, 
    opkernel, op1, op2, init1, opreduce, init2 
  );

  dim3 dimBlockReduce( 256, 1 );
  dim3 dimGridReduce( ( m - 1 ) / 256 + 1, 1, batchSize );


  printf( "%d\n", min( n, ( ( n - 1 ) / BLK_N + 1 ) * DIM_Y ) ); 

  reduce_template_batched_kernel
  <TC, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
  <<< dimGridReduce, dimBlockReduce, 0, stream >>>
  ( 
    m, min( n, ( ( n - 1 ) / BLK_N + 1 ) * DIM_Y ), 
    Carray, ldc, 
    opreduce 
  );

};


/**
 *  @brief GKRM cuda kernel wrapper with strided access. A global reduction
 *         kernel is followed after the local reduction in gkrm.
 */ 
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
  const TA *Aarray, int lda, int loa,
  const TB *Barray, int ldb, int lob,
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
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>
  <<< dimGrid, dimBlock, 0, stream >>>
  ( 
    m, n, k, 
    Aarray, lda, loa,
    Barray, ldb, lob,
    Carray, ldc, loc, 
    opkernel, op1, op2, initV, opreduce, initC 
  );

  dim3 dimBlockReduce( 256, 1 );
  dim3 dimGridReduce( ( m - 1 ) / 256 + 1, 1, batchSize );

  printf( "%d\n", min( n, ( ( n - 1 ) / BLK_N + 1 ) * DIM_Y ) ); 

  reduce_template_batched_strided_kernel
  <TC, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
  <<< dimGridReduce, dimBlockReduce, 0, stream >>>
  ( 
    m, min( n, ( ( n - 1 ) / BLK_N + 1 ) * DIM_Y ), 
    Carray, ldc, loc, 
    opreduce 
  );
};









/**
 *  @brief This is the GKRM (General Kernel Reduced Matrix) template wrapper.
 *         This interface accepts double pointers. Here the type rules are
 *         op2: <TA,TB> to <TV>,
 *         op1: <TV,TV> to <TV>,
 *         opkernel: <TV> to <TC>,
 *         opreduce: <TC,TC> to <TC>
 *
 *  @param TA Type of *Aarray[]
 *  @param TB Type of *Barray[]
 *  @param TC Type of *Carray[]
 *  @param TV Type of of the output of op1 and op2
 *  @param SQ@NRM Whether opkernel uses a^2-2ab+b^2 expansion or not
 *  @param OPKERNEL Type of opkernel
 *  @param OP1 Type of op1
 *  @param OP2 Type of op2
 *  @param OPREDUCE Type of opreduce
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
 *  @opreduce Closure of the reduce operator
 *  @init2 Initial value of the reduce operator
 *
 */ 
template<
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV> 
void gkrm
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  const TA *Aarray[], int lda,
  const TB *Barray[], int ldb,
        TC *Carray[], int ldc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, OPREDUCE opreduce, TC initC 
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

/**
 *  @brief This is the GKRM (General Kernel Reduced Matrix) template wrapper.
 *         This interface accepts pointers with strided access. 
 *         Here the type rules are
 *         op2: <TA,TB> to <TV>,
 *         op1: <TV,TV> to <TV>,
 *         opkernel: <TV> to <TC>,
 *         opreduce: <TC,TC> to <TC>.
 *
 *  @param TA Type of *Aarray
 *  @param TB Type of *Barray
 *  @param TC Type of *Carray
 *  @param TV Type of of the output of op1 and op2
 *  @param SQ@NRM Whether opkernel uses a^2-2ab+b^2 expansion or not
 *  @param OPKERNEL Type of opkernel
 *  @param OP1 Type of op1
 *  @param OP2 Type of op2
 *  @param OPREDUCE Type of opreduce
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
 *  @opreduce Closure of the reduce operator
 *  @init2 Initial value of the reduce operator
 *
 */ 
template<
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE,
typename TA, typename TB, typename TC, typename TV> 
void gkrm
(
  cudaStream_t stream, 
  hmlpOperation_t transA, hmlpOperation_t transB, 
  int m, int n, int k,
  const TA *Aarray, int lda, int loa,
  const TB *Barray, int ldb, int lob,
        TC *Carray, int ldc, int loc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV initV, OPREDUCE opreduce, TC initC 
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
  #include <gkrm_strided_autotune.hxx>
};




#endif // define GKRM_GPU_HPP
