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

#include "gkmx.h"


namespace hmlp
{
namespace gkmmv
{





#define PRECISION_d

// GKMX kernel implementations
#include "gkmx_template_kernel_batched.hxx"

// Tunned parameters 
#include "gemm_config/dgemm_param_nn.h"
#include "gemm_config/dgemm_param_nt.h"
#include "gemm_config/dgemm_param_tn.h"
#include "gemm_config/dgemm_param_tt.h"
#include "gemm_config/sgemm_param_nn.h"
#include "gemm_config/sgemm_param_nt.h"
#include "gemm_config/sgemm_param_tn.h"
#include "gemm_config/sgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

// GKMMV macros (see gkmx_template_kernel_batched.hxx for the definition.)
#define gkmmv(ta,tb,s,v) gkmmv_template_batched_internal \
  < ta, tb, s ## _V_ ## v, TA, TB, TC, TV, SQ2NRM, OPKERNEL, OP1, OP2> \
  ( \
  stream, \
  m, n, k, \
  Aarray, lda, \
  Barray, ldb, \
  Carray, ldc, \
  batchSize, \
  opkernel, op1, op2, init1, opreduce, init2 ) 

#define gkmmv_strided(ta,tb,s,v) gkmmv_template_batched_strided_internal \
  < ta, tb, s ## _V_ ## v, TA, TB, TC, TV, SQ2NRM, OPKERNEL, OP1, OP2, OPREDUCE> \
  ( \
  stream, \
  m, n, k, \
  Aarray, lda, loa, \
  Barray, ldb, lob, \
  Carray, ldc, loc, \
  batchSize, \
  opkernel, op1, op2, init1, opreduce, init2 ) 





















/**
 *  @brief This is the GKMMV (General Kernel Matrix Matrix Vector) template 
 *         wrapper. This interface accepts double pointers. 
 *         Here the type rules are
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
void gkmmv
(
  cudaStream_t stream, 
  cublasOperation_t transA, cublasOperation_t transB, 
  int m, int n, int k,
  const TA *Aarray[], int lda,
  const TB *Barray[], int ldb,
        TC *Carray[], int ldc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 
)
{
  // Early return.
  if ( m <= 0 || n <= 0 || k <= 0 ) return;

  // Specify input formats
  int shape = 0;
  if      ( transA == CUBLAS_OP_N && transB == CUBLAS_OP_N ) { shape = 0; } // nn
  else if ( transA == CUBLAS_OP_N && transB == CUBLAS_OP_T ) { shape = 1; } // nt
  else if ( transA == CUBLAS_OP_T && transB == CUBLAS_OP_N ) { shape = 3; } // tn
  else if ( transA == CUBLAS_OP_T && transB == CUBLAS_OP_T ) { shape = 4; } // tt

  // Autotuned decision tree
  #include "gkmx_autotune/gkmmv_autotune.hxx"
}

/**
 *  @brief This is the GKMMV (General Kernel Matrix Matrix Vector) template wrapper.
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
void gkmmv
(
  cudaStream_t stream, 
  cublasOperation_t transA, cublasOperation_t transB, 
  int m, int n, int k,
  const TA *Aarray, int lda, int loa,
  const TB *Barray, int ldb, int lob,
        TC *Carray, int ldc, int loc,
  int batchSize,
  OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
  // Early return.
  if ( m <= 0 || n <= 0 || k <= 0 ) return;

  // Specify input formats
  int shape = 0;
  if      ( transA == CUBLAS_OP_N && transB == CUBLAS_OP_N ) { shape = 0; } // nn
  else if ( transA == CUBLAS_OP_N && transB == CUBLAS_OP_T ) { shape = 1; } // nt
  else if ( transA == CUBLAS_OP_T && transB == CUBLAS_OP_N ) { shape = 3; } // tn
  else if ( transA == CUBLAS_OP_T && transB == CUBLAS_OP_T ) { shape = 4; } // tt

  // Autotuned decision tree
  #include "gkmx_autotune/gkmmv_strided_autotune.hxx"
}

}; // end namespace gkmmv
}; // end namespace hmlp


#endif // define GKMX_CUH
