/**
 *  -- GKMX (version 1.1.0) --
 *
 *  NVIDIA Corp, Santa Clara
 *
 *  @date June 2016
 *  @author Chenhan D. Yu
 *
 */

#ifndef GKMX_TEMPLATE_KERNEL_BATCHED_HXX
#define GKMX_TEMPLATE_KERNEL_BATCHED_HXX

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#define fetch(A, m, n, bound) offs_d##A[min(n*LD##A+m, bound)]

#include "gkmx_template_device.hxx"

/**
 *  @brief The kernel wrapper of GKMM. Memroy access between each z-dim block
 *         is through the double pointer arrays.
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int DIM_XA, const int DIM_YA, 
const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2>
static __global__ void gkmm_template_batched_kernel(
    int M, int N, int K,
    const TA *Aarray[], int LDA, 
    const TB *Barray[], int LDB, 
          TC *Carray[], int LDC, 
    OPKERNEL opkernel,
    OP1 op1, OP2 op2, TV init1 )
{
  int batchid = blockIdx.z;
  gkmm_template_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2>(
        M, N, K, 
        Aarray[ batchid ], LDA,
        Barray[ batchid ], LDB,
        Carray[ batchid ], LDC,
        opkernel, op1, op2, init1 );
}

/**
 *  @brief The kernel wrapper of GKMM. Memroy access between each z-dim block
 *         is through computing the address offset using LOA, LOB and LOC.
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int DIM_XA, const int DIM_YA, 
const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2>
static __global__ void gkmm_template_batched_strided_kernel(
    int M, int N, int K,
    const TA *A, int LDA, int LOA,
    const TB *B, int LDB, int LOB,
          TC *C, int LDC, int LOC,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1 )
{
  int batchid = blockIdx.z;
  gkmm_template_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2>(
        M, N, K, 
        A + LOA * batchid, LDA,
        B + LOB * batchid, LDB,
        C + LOC * batchid, LDC,
        opkernel, op1, op2, init1 );
}

/**
 *  @brief The kernel wrapper of GKRM. Memroy access between each z-dim block
 *         is through the double pointer arrays.
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
static __global__ void gkrm_template_batched_kernel(
    int M, int N, int K,
    const TA *Aarray[], int LDA,
    const TB *Barray[], int LDB,
          TC *Carray[], int LDC,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  int batchid = blockIdx.z;
  gkrm_template_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>(
        M, N, K, 
        Aarray[ batchid ], LDA,
        Barray[ batchid ], LDB,
        Carray[ batchid ], LDC,
        opkernel, op1, op2, init1, opreduce, init2 );
}

/**
 *  @brief The kernel wrapper of GKRM. Memroy access between each z-dim block
 *         is through computing the address offset using LOA, LOB and LOC.
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
static __global__ void gkrm_template_batched_strided_kernel(
    int M, int N, int K,
    const TA *A, int LDA, int LOA,
    const TB *B, int LDB, int LOB,
          TC *C, int LDC, int LOC,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  int batchid = blockIdx.z;
  gkrm_template_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>(
        M, N, K, 
        A + LOA * batchid, LDA,
        B + LOB * batchid, LDB,
        C + LOC * batchid, LDC,
        opkernel, op1, op2, init1, opreduce, init2 );
}


/**
 *  @brief The kernel wrapper of GKMMV. Memroy access between each z-dim block
 *         is through the double pointer arrays.
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
static __global__ void gkmmv_template_batched_kernel(
    int M, int N, int K,
    const TA *Aarray[], int LDA,
    const TB *Barray[], int LDB,
          TC *Carray[], int LDC,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  int batchid = blockIdx.z;
  gkmmv_template_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>(
        M, N, K, 
        Aarray[ batchid ], LDA,
        Barray[ batchid ], LDB,
        Carray[ batchid ], LDC,
        opkernel, op1, op2, init1, opreduce, init2 );
}


/**
 *  @brief The kernel wrapper of GKMMV. Memroy access between each z-dim block
 *         is through computing the address offset using LOA, LOB and LOC.
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
static __global__ void gkmmv_template_batched_strided_kernel(
    int M, int N, int K,
    const TA *A, int LDA, int LOA,
    const TB *B, int LDB, int LOB,
          TC *C, int LDC, int LOC,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  int batchid = blockIdx.z;
  gkmmv_template_device<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K,
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    (BLK_M/DIM_X), (BLK_N/DIM_Y), 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>(
        M, N, K, 
        A + LOA * batchid, LDA,
        B + LOB * batchid, LDB,
        C + LOC * batchid, LDC,
        opkernel, op1, op2, init1, opreduce, init2 );
}


/*
 *
 */ 
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const
int BLK_N, typename OPREDUCE>
static __global__
void reduce_template_batched_kernel( int M, int N, T** Carray, int LDC,
    OPREDUCE opreduce )
{
  int batchid = blockIdx.z;
  reduce_template_device <T, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
    ( M, N, Carray[batchid], LDC, opreduce );
}


/*
 *
 */ 
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const
int BLK_N, typename OPREDUCE>
static __global__
void reduce_template_batched_strided_kernel( int M, int N, T* Carray, int LDC, int LOC,
    OPREDUCE opreduce )
{
  int batchid = blockIdx.z;
  reduce_template_device <T, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
    ( M, N, Carray + batchid * LOC, LDC, opreduce );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 *  @brief GKMM cuda kernel wrapper with double pointers. 
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2>
void gkmm_template_batched_internal(
    cudaStream_t stream, 
    int m, int n, int k,
    const TA *Aarray[], int lda,
    const TB *Barray[], int ldb,
          TC *Carray[], int ldc,
    int batchSize,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1 )
{
  dim3 dimBlock(DIM_X, DIM_Y);
  dim3 dimGrid( gkmx_ceildiv( m, BLK_M ), gkmx_ceildiv( n, BLK_N ), batchSize );
  gkmm_template_batched_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2>
      <<< dimGrid, dimBlock, 0, stream >>>
      ( m, n, k, 
        Aarray, lda,
        Barray, ldb,
        Carray, ldc,
        opkernel, 
        op1, op2, init1 );
}


/**
 *  @brief GKMM cuda kernel wrapper with strided access. 
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec,
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2>
void gkmm_template_batched_strided_internal(
    cudaStream_t stream, 
    int m, int n, int k,
    const TA *Aarray, int lda, int loa,
    const TB *Barray, int ldb, int lob,
          TC *Carray, int ldc, int loc,
    int batchSize,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1 )
{
  dim3 dimBlock(DIM_X, DIM_Y);
  dim3 dimGrid( gkmx_ceildiv( m, BLK_M ), gkmx_ceildiv( n, BLK_N ), batchSize );

  gkmm_template_batched_strided_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB,
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2>
      <<< dimGrid, dimBlock, 0, stream >>>
      ( m, n, k, 
        Aarray, lda, loa,
        Barray, ldb, lob,
        Carray, ldc, loc,
        opkernel, 
        op1, op2, init1 );
}


/**
 *  @brief GKRM cuda kernel wrapper with double pointers. A global reduction
 *         kernel is followed after the local reduction in gkrm.
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
void gkrm_template_batched_internal(
    cudaStream_t stream, 
    int m, int n, int k,
    const TA *Aarray[], int lda,
    const TB *Barray[], int ldb,
          TC *Carray[], int ldc,
    int batchSize,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  dim3 dimBlock(DIM_X, DIM_Y);
  dim3 dimGrid( gkmx_ceildiv( m, BLK_M ), gkmx_ceildiv( n, BLK_N ), batchSize );
  gkrm_template_batched_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>
      <<< dimGrid, dimBlock, 0, stream >>>
      ( m, n, k, 
        Aarray, lda,
        Barray, ldb,
        Carray, ldc, 
        opkernel, op1, op2, init1, opreduce, init2 );
  dim3 dimBlockReduce( 256, 1 );
  dim3 dimGridReduce( ( m - 1 ) / 256 + 1, 1, batchSize );
  reduce_template_batched_kernel
    <TC, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
    <<< dimGridReduce, dimBlockReduce, 0, stream >>>
    ( m, ( ( n - 1 ) / BLK_N + 1 ) * DIM_Y, Carray, ldc, opreduce );
}


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
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
void gkrm_template_batched_strided_internal(
    cudaStream_t stream, 
    int m, int n, int k,
    const TA *Aarray, int lda, int loa,
    const TB *Barray, int ldb, int lob,
          TC *Carray, int ldc, int loc,
    int batchSize,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  dim3 dimBlock(DIM_X, DIM_Y);
  dim3 dimGrid( gkmx_ceildiv( m, BLK_M ), gkmx_ceildiv( n, BLK_N ), batchSize );
  gkrm_template_batched_strided_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>
      <<< dimGrid, dimBlock, 0, stream >>>
      ( m, n, k, 
        Aarray, lda, loa,
        Barray, ldb, lob,
        Carray, ldc, loc, 
        opkernel, op1, op2, init1, opreduce, init2 );
  dim3 dimBlockReduce( 256, 1 );
  dim3 dimGridReduce( ( m - 1 ) / 256 + 1, 1, batchSize );
  reduce_template_batched_strided_kernel
    <TC, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
    <<< dimGridReduce, dimBlockReduce, 0, stream >>>
    ( m, min( n, ( ( n - 1 ) / BLK_N + 1 ) * DIM_Y ), Carray, ldc, loc, opreduce );
}


/**
 *  @brief GKMMV cuda kernel wrapper with double pointers. A global reduction
 *         kernel is followed after the local reduction in gkrm.
 */ 
template< bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
void gkmmv_template_batched_internal(
    cudaStream_t stream, 
    int m, int n, int k,
    const TA *Aarray[], int lda,
    const TB *Barray[], int ldb,
          TC *Carray[], int ldc,
    int batchSize,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  dim3 dimBlock(DIM_X, DIM_Y);
  dim3 dimGrid( gkmx_ceildiv( m, BLK_M ), gkmx_ceildiv( n, BLK_N ), batchSize );
  gkmmv_template_batched_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>
      <<< dimGrid, dimBlock, 0, stream >>>
      ( m, n, k, 
        Aarray, lda,
        Barray, ldb,
        Carray, ldc,
        opkernel, op1, op2, init1, opreduce, init2 );
  dim3 dimBlockReduce( 256, 1 );
  dim3 dimGridReduce( ( m - 1 ) / 256 + 1, 1, batchSize );
  reduce_template_batched_kernel
    <TC, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
    <<< dimGridReduce, dimBlockReduce, 0, stream >>>
    ( m, ( n - 1 ) / BLK_N + 1, Carray, ldc, opreduce );
}

/**
 *  @brief GKMMV cuda kernel wrapper with strided access. A global reduction
 *         kernel is followed after the local reduction in gkrm.
 */ 
template< bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y, 
const int BLK_M, const int BLK_N, const int BLK_K,
const int dim_vec, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
void gkmmv_template_batched_strided_internal(
    cudaStream_t stream, 
    int m, int n, int k,
    const TA *Aarray, int lda, int loa,
    const TB *Barray, int ldb, int lob,
          TC *Carray, int ldc, int loc,
    int batchSize,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  dim3 dimBlock(DIM_X, DIM_Y);
  dim3 dimGrid( gkmx_ceildiv( m, BLK_M ), gkmx_ceildiv( n, BLK_N ), batchSize );
  gkmmv_template_batched_strided_kernel<
    TRANSA, TRANSB,
    DIM_X, DIM_Y, 
    BLK_M, BLK_N, BLK_K, 
    DIM_XA, DIM_YA, DIM_XB, DIM_YB, 
    TA, TB, TC, TV, 
    SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE>
      <<< dimGrid, dimBlock, 0, stream >>>
      ( m, n, k, 
        Aarray, lda, loa,
        Barray, ldb, lob,
        Carray, ldc, loc, 
        opkernel, op1, op2, init1, opreduce, init2 );
  dim3 dimBlockReduce( 256, 1 );
  dim3 dimGridReduce( ( m - 1 ) / 256 + 1, 1, batchSize );
  reduce_template_batched_strided_kernel
    <TC, DIM_X, DIM_Y, BLK_M, BLK_N, OPREDUCE>
    <<< dimGridReduce, dimBlockReduce, 0, stream >>>
    ( m, ( n - 1 ) / BLK_N + 1, Carray, ldc, loc, opreduce );
}

#endif // define GKMX_TEMPLATE_KERNEL_BATCHED_HXX
