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
  const T * __restrict__ a, const size_t *  __restrict__ amap, 
  T * __restrict__ A )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if ( idx < m && idy < n )
  {
    A[ amap[ idy ] * m + idx ] += a[ idy * m + idx ];
  }
};

void assemble
( cudaStream_t stream, int m, int n, const float *  __restrict__ a, const size_t *  __restrict__ amap, 
float *  __restrict__ A )
{
  dim3 dimBlock( 16, 16 );
  dim3 dimGrid( ( m - 1 ) / 16 + 1, ( n - 1 ) / 16 + 1 );
  assemble_kernel<<< dimGrid, dimBlock, 0, stream >>>( m, n, a, amap, A );
};

void assemble
( cudaStream_t stream, int m, int n, const double *  __restrict__ a, const size_t *  __restrict__ amap, double *  __restrict__ A )
{
  dim3 dimBlock( 16, 16 );
  dim3 dimGrid( ( m - 1 ) / 16 + 1, ( n - 1 ) / 16 + 1 );
  assemble_kernel<<< dimGrid, dimBlock, 0, stream >>>( m, n, a, amap, A );
};


}; /** namespace gpu */ 
}; /** namespace gofmm */
}; /** namespace hmlp */
