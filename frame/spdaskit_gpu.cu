namespace hmlp
{
namespace spdaskit
{
namespace gpu
{

template<typename T>
static __global__
void assemble_kernel
( int m, int n, T *a, size_t *amap, T *A )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if ( idx < m && idy < n )
  {
    A[ amap[ idy ] * m + idx ] += a[ idy * m + idx ];
  }
};

void assemble
( cudaStream_t stream, int m, int n, float *a, size_t *amap, float *A )
{
  dim3 dimBlock( 16, 16 );
  dim3 dimGrid( ( m - 1 ) / 16 + 1, ( n - 1 ) / 16 + 1 );
  assemble_kernel<<< dimGrid, dimBlock, 0, stream >>>( m, n, a, amap, A );
};

void assemble
( cudaStream_t stream, int m, int n, double *a, size_t *amap, double *A )
{
  dim3 dimBlock( 16, 16 );
  dim3 dimGrid( ( m - 1 ) / 16 + 1, ( n - 1 ) / 16 + 1 );
  assemble_kernel<<< dimGrid, dimBlock, 0, stream >>>( m, n, a, amap, A );
};

};
};
};
