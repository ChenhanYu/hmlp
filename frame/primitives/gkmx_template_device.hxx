/**
 *  -- GKMX (version 1.1.0) --
 *
 *  NVIDIA Corp, Santa Clara
 *
 *  @date June 2016
 *  @author Chenhan D. Yu
 *
 */
#ifndef GKMX_TEMPLATE_DEVICE_HXX
#define GKMX_TEMPLATE_DEVICE_HXX


/**
 *  @breif The main kernel (per z-dim block) of GKMM. Four types NN, NT, TN
 *         TT are unified in this case. Stride access and double pointers 
 *         are also unified here. SQ2NRM is used to determine whether to 
 *         compute a^2 -2ab + b^2. See gkmm_template_stencil.hxx for the
 *         special rank-k update we perform in all GKMM, GKRM and GKMMV.
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y,
const int BLK_M, const int BLK_N, const int BLK_K, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
const int THR_M, const int THR_N, 
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2>
static __device__ void gkmm_template_device(
    int M, int N, int K,
    const TA* __restrict__ A, int LDA,
    const TB* __restrict__ B, int LDB,
          TC* __restrict__ C, int LDC,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1 )
{
#if (__CUDA_ARCH__ >= 200)

  // Semi-ring rank-k update template
  #include "gkmm_template_stencil.hxx"

  // SQ2NRM option
  if ( SQ2NRM ) {
    __syncthreads();
    if ( idt < BLK_M && blx * BLK_M + idt < M ) {
      sA[ 0 ][ idt ] = opkernel.A2[ blockIdx.z ][ blx * BLK_M + idt ];
    }
    if ( idt < BLK_N && bly * BLK_N + idt < N ) {
      sB[ idt ][ 0 ] = opkernel.B2[ blockIdx.z ][ bly * BLK_N + idt ];
    }
    __syncthreads();
  }

  // Store C regs->dev
  #pragma unroll
  for ( n = 0; n < THR_N; n ++ ) {
    int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
    #pragma unroll
    for ( m = 0; m < THR_M; m ++ ) {
      int coord_dCm = blx*BLK_M + m*DIM_X + idx;
      if ( coord_dCm < M && coord_dCn < N ) {
        int offsC = coord_dCn*LDC + coord_dCm;
        TV &regC = rC[ n ][ m ];
        TC &memC = C[ offsC ];
        if ( SQ2NRM ) {
          regC *= -2.0;
          regC += sA[ 0 ][ m * DIM_X + idx ] + sB[ n * DIM_Y + idy ][ 0 ];
        }
        memC = opkernel( regC, coord_dCm, coord_dCn, blockIdx.z );
      }
    }
  }

#endif /* (__CUDA_ARCH__ >= 200) */
}



/**
 *  @breif The main kernel (per z-dim block) of GKRM. Row-wise in thread
 *  reduction is performed at the end by opreduce. Number of columns of the
 *  output matrix C are reduced 1 / THR_N. A global reduction is required 
 *  after this kernel.
 *
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y,
const int BLK_M, const int BLK_N, const int BLK_K, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
const int THR_M, const int THR_N, 
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
static __device__ void gkrm_template_device(
    int M, int N, int K,
    const TA* __restrict__ A, int LDA,
    const TB* __restrict__ B, int LDB,
          TC* __restrict__ C, int LDC,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
#if (__CUDA_ARCH__ >= 200)
  TC rc[THR_M];

  // Semi-ring rank-k update template
  #include "gkmm_template_stencil.hxx"

  // SQ2NRM option
  if ( SQ2NRM ) {
    __syncthreads();
    if ( idt < BLK_M && blx * BLK_M + idt < M ) {
      sA[ 0 ][ idt ] = opkernel.A2[ blockIdx.z ][ blx * BLK_M + idt ];
    }
    if ( idt < BLK_N && bly * BLK_N + idt < N ) {
      sB[ idt ][ 0 ] = opkernel.B2[ blockIdx.z ][ bly * BLK_N + idt ];
    }
    __syncthreads();
  }

  #pragma unroll
  for ( m = 0; m < THR_M; m ++ ) rc[ m ] = init2;

  #pragma unroll
  for ( m = 0; m < THR_M; m ++ ) {
    int coord_dCm = blx*BLK_M + m*DIM_X + idx;
    int offsC = ( bly * DIM_Y + idy ) * LDC + coord_dCm;
    #pragma unroll
    for ( n = 0; n < THR_N; n ++ ) {
      int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
      if ( coord_dCm < M && coord_dCn < N ) {
        TV  &regC = rC[ n ][ m ];
        TC regK;
        if ( SQ2NRM ) {
          regC *= -2.0;
          regC += sA[ 0 ][ m * DIM_X + idx ] + sB[ n * DIM_Y + idy ][ 0 ];
        }
        regK = opkernel( regC, coord_dCm, coord_dCn, blockIdx.z );
        if ( !n ) {
          rc[ m ] = regK;
        }
        else {
          rc[ m ] = opreduce( rc[ m ], regK, coord_dCm, coord_dCn, blockIdx.z );
        }
      }
    }
    // For special case where DIM_Y < N, we need condition idy < N.
    if ( coord_dCm < M && bly * BLK_N < N && idy < N ) {
      C[ offsC ] = rc[ m ];
    }
  }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/**
 *  @breif The main kernel (per z-dim block) of GKMMV. Row-wise in block
 *  reduction is performed at the end by opreduce. Number of columns of the
 *  output matrix C are first reduced 1 / THR_N. Then lafer C is reduced to a
 *  vector per block. Still a global reduction is required after this kernel.
 *
 */ 
template<bool TRANSA, bool TRANSB,
const int DIM_X, const int DIM_Y,
const int BLK_M, const int BLK_N, const int BLK_K, 
const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
const int THR_M, const int THR_N, 
typename TA, typename TB, typename TC, typename TV,
bool SQ2NRM, typename OPKERNEL, typename OP1, typename OP2, typename OPREDUCE>
static __device__ void gkmmv_template_device(
    int M, int N, int K,
    const TA* __restrict__ A, int LDA,
    const TB* __restrict__ B, int LDB,
          TC* __restrict__ C, int LDC,
    OPKERNEL opkernel, OP1 op1, OP2 op2, TV init1, OPREDUCE opreduce, TC init2 )
{
#if (__CUDA_ARCH__ >= 200)
  // Semi-ring rank-k update template
  #include "gkmm_template_stencil.hxx"

  __syncthreads();
  if ( idt < BLK_N && bly * BLK_N + idt < N ) {
    sB[ idt ][ BLK_K ] = opkernel.w[ blockIdx.z ][ bly * BLK_N + idt ];
  }

  // SQ2NRM option
  if ( SQ2NRM ) {
    if ( idt < BLK_M && blx * BLK_M + idt < M ) {
      sA[ 0 ][ idt ] = opkernel.A2[ blockIdx.z ][ blx * BLK_M + idt ];
    }
    if ( idt < BLK_N && bly * BLK_N + idt < N ) {
      sB[ idt ][ 0 ] = opkernel.B2[ blockIdx.z ][ bly * BLK_N + idt ];
    }
  }
  __syncthreads();


  #pragma unroll
  for ( m = 0; m < THR_M; m ++ ) {
    int coord_dCm = blx*BLK_M + m*DIM_X + idx;
    rA[ m ] = 0.0;
    #pragma unroll
    for ( n = 0; n < THR_N; n ++ ) {
      int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
      if ( coord_dCm < M && coord_dCn < N ) {
        TV &regC = rC[ n ][ m ];
        if ( SQ2NRM ) {
          regC *= -2.0;
          regC += sA[ 0 ][ m * DIM_X + idx ] + sB[ n * DIM_Y + idy ][ 0 ];
        }
        regC = opkernel( regC, coord_dCm, coord_dCn, blockIdx.z ) * sB[ n *
          DIM_Y + idy ][ BLK_K ];
        rA[ m ] = opreduce( rA[ m ], regC, coord_dCm, coord_dCn, blockIdx.z );
      }
    }
  }

  // Reduce 2D row potentials for each block.
  __syncthreads();
  #pragma unroll
  for ( m = 0; m < THR_M; m ++ ) {
    sA[ idy ][ m * DIM_X + idx ] = rA[ m ];
  }
  __syncthreads();

  int coord_dCm = blx*BLK_M + idt;
  if ( idt < BLK_M && coord_dCm < M ) {
    rA[ 0 ] = 0.0;
    int offsC = bly*LDC + coord_dCm;
    #pragma unroll
    for ( n = 0; n < DIM_Y; n ++ ) {
      rA[ 0 ] = opreduce( rA[ 0 ], sA[ n ][ idt ], coord_dCm, n, blockIdx.z );
    }
    C[ offsC ] = rA[ 0 ];
  }

#endif /* (__CUDA_ARCH__ >= 200) */
}

/** 
 *  @brief A simple row-wise reduction, which does not exploit the parallelism
 *         of the binary tree.
 */ 
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const
int BLK_N, typename OPREDUCE>
static __device__
void reduce_template_device( int M, int N, T* __restrict__ C, int LDC,
    OPREDUCE opreduce)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx < M ) {
    T ru = C[ idx ];
    for ( int j = 1; j < N; j ++ ) {
      ru = opreduce( ru, C[ j * LDC + idx ], idx, j, blockIdx.z );    
    }
    C[ idx ] = ru;
  }
}

#endif // define GKMX_TEMPLATE_DEVICE_HXX
