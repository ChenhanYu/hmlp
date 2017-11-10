/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  


namespace hmlp
{
namespace gofmm
{
namespace gpu
{

template<typename T>
static __global__
void assemble_kernel
( 
  int m, int n, 
  const      T * __restrict__ a, 
  const size_t * __restrict__ amap, 
             T * __restrict__ A, int lda )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if ( idx < m && idy < n )
  {
    //A[ amap[ idy ] * m + idx ] += a[ idy * m + idx ];
    A[ idy * lda + amap[ idx ] ] += a[ idy * m + idx ];
  }
};

void assemble( 
    cudaStream_t stream, 
    int m, int n, 
    const  float *  __restrict__ a, 
    const size_t *  __restrict__ amap, 
           float *  __restrict__ A, int lda )
{
  dim3 dimBlock( 16, 16 );
  dim3 dimGrid( ( m - 1 ) / 16 + 1, ( n - 1 ) / 16 + 1 );
  assemble_kernel<<< dimGrid, dimBlock, 0, stream >>>( m, n, a, amap, A, lda );
};

void assemble( 
    cudaStream_t stream, 
    int m, int n, 
    const double *  __restrict__ a, 
    const size_t *  __restrict__ amap, 
          double *  __restrict__ A, int lda )
{
  dim3 dimBlock( 16, 16 );
  dim3 dimGrid( ( m - 1 ) / 16 + 1, ( n - 1 ) / 16 + 1 );
  assemble_kernel<<< dimGrid, dimBlock, 0, stream >>>( m, n, a, amap, A, lda );
};


}; /** end namespace gpu */ 
}; /** end namespace gofmm */
}; /** end namespace hmlp */
