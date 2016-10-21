#ifndef HMLP_BLAS_LAPACK
#define HMLP_BLAS_LAPACK

namespace hmlp
{

void xgemm(
    char *transA, char *transB,
    int m, int n, int k, 
    float alpha,
    float *A, int lda,
    float *B, int ldb, float beta,
    float *C, int ldc
    );

void xgemm(
    char *transA, char *transB,
    int m, int n, int k, 
    double alpha,
    double *A, int lda,
    double *B, int ldb, double beta,
    double *C, int ldc
    );

} // end namespace hmlp

#endif // define HMLP_BLAS_LAPACK
