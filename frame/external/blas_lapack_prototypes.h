#ifndef BLAS_LAPACK_PROTOTYPES_H
#define BLAS_LAPACK_PROTOTYPES_H

/** BLAS level-1 */
double ddot_( int* n, const double* dx, int* incx, const double* dy, int* incy );
float  sdot_( int* n, const  float* dx, int* incx, const  float* dy, int* incy );
double dnrm2_( int *n, double *x, int *incx );
float  snrm2_( int *n,  float *x, int *incx );
void saxpy_( const int* n, const float* alpha, const float* x, const int* incx, float* y, const int *incy );
void daxpy_( const int* n, const double* alpha, const double* x, const int* incx, double* y, const int *incy );


/** BLAS level-2 */


/** BLAS level-3 */
void dgemm_( 
    const char *transA, const char *transB, 
    int *m, int *n, int *k, 
    double *alpha,
    const double *A, int *lda, 
    const double *B, int *ldb, double *beta, 
    double *C, int *ldc );
void sgemm_( 
    const char *transA, const char *transB, 
    int *m, int *n, int *k, 
    float *alpha,
    const float *A, int *lda, 
    const float *B, int *ldb, float *beta, 
    float *C, int *ldc );
void dsyrk_( 
    const char *uplo, const char *trans, 
    int *n, int *k, 
    double *alpha,
    double *A, int *lda, 
    double *beta, 
    double *C, int *ldc );
void ssyrk_( 
    const char *uplo, const char *trans, 
    int *n, int *k, 
    float *alpha,
    float *A, int *lda, 
    float *beta, 
    float *C, int *ldc );
void dtrsm_( 
    const char *side, const char *uplo,
    const char *transA, const char *diag,
    int *m, int *n,
    double *alpha,
    double *A, int *lda,
    double *B, int *ldb );
void strsm_( 
    const char *side, const char *uplo,
    const char *transA, const char *diag,
    int *m, int *n,
    float *alpha,
    float *A, int *lda,
    float *B, int *ldb );
void dtrmm_( 
    const char *side, const char *uplo,
    const char *transA, const char *diag,
    int *m, int *n,
    double *alpha,
    double *A, int *lda,
    double *B, int *ldb );
void strmm_( 
    const char *side, const char *uplo,
    const char *transA, const char *diag,
    int *m, int *n,
    float *alpha,
    float *A, int *lda,
    float *B, int *ldb );

/** LAPACK */
void xerbla_( const char *srname, int *info );
void dswap_( 
    int *n, 
    double *dx, int *incx, 
    double *dy, int *incy );
void sswap_( 
    int *n, 
    float  *sx, int *incx, 
    float  *sy, int *incy );
void dlacpy_( 
    const char *uplo, int *m, int *n, 
    double *a, int *lda, 
    double *b, int *ldb );
void slacpy_( 
    const char *uplo, int *m, int *n, 
    float *a, int *lda, 
    float *b, int *ldb );
void dlaswp_( 
  int *n, double *A, int *lda, 
  int *k1, int *k2, int *ipiv, int *incx );
void slaswp_( 
  int *n, float *A, int *lda, 
  int *k1, int *k2, int *ipiv, int *incx );
void dpotrf_( 
    const char *uplo, 
    int *n, double *A, int *lda, int *info );
void spotrf_( 
    const char *uplo, 
    int *n, float *A, int *lda, int *info );
void dpotrs_( 
    const char *uplo, 
    int *n, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info );
void spotrs_( 
    const char *uplo, 
    int *n, int *nrhs, float *A, int *lda, float *B, int *ldb, int *info );
void dposv_(
    const char *uplo, 
    int *n, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info );
void sposv_(
    const char *uplo, 
    int *n, int *nrhs, float *A, int *lda, float *B, int *ldb, int *info );
void dgetrf_(
    int *m, int *n, 
    double *A, int *lda, int *ipiv, int *info );
void sgetrf_(
    int *m, int *n, 
    float *A, int *lda, int *ipiv, int *info );
void dgetrs_(
    const char *trans,
    int *m, int *nrhs, 
    double *A, int *lda, int *ipiv,
    double *B, int *ldb, int *info );
void sgetrs_(
    const char *trans,
    int *m, int *nrhs, 
    float *A, int *lda, int *ipiv,
    float *B, int *ldb, int *info );
void dgecon_(
    const char *norm,
    int *n,
    double *A, int *lda, 
    double *anorm, 
    double *rcond, 
    double *work, int *iwork, int *info );
void sgecon_(
    const char *norm,
    int *n,
    float *A, int *lda, 
    float *anorm, 
    float *rcond, 
    float *work, int *iwork, int *info );
void dgeqrf_(
    int *m, int *n, 
    double *A, int *lda, 
    double *tau, 
    double *work, int *lwork, int *info );
void sgeqrf_(
    int *m, int *n, 
    float *A, int *lda, 
    float *tau, 
    float *work, int *lwork, int *info );
void dorgqr_(
    int *m, int *n, int *k,
    double *A, int *lda, 
    double *tau,
    double *work, int *lwork, int *info );
void sorgqr_(
    int *m, int *n, int *k,
    float *A, int *lda, 
    float *tau,
    float *work, int *lwork, int *info );
void dormqr_( 
    const char *side, const char *trans,
    int *m, int *n, int *k, 
    double *A, int *lda, 
    double *tau,
    double *C, int *ldc, 
    double *work, int *lwork, int *info );
void sormqr_( 
    const char *side, const char *trans,
    int *m, int *n, int *k, 
    float *A, int *lda, 
    float *tau,
    float *C, int *ldc, 
    float *work, int *lwork, int *info );
void dgeqp3_(
    int *m, int *n,
    double *A, int *lda, int *jpvt,
    double *tau,
    double *work, int *lwork, int *info ); 
void sgeqp3_(
    int *m, int *n,
    float *A, int *lda, int *jpvt,
    float *tau,
    float *work, int *lwork, int *info );
void dgeqp4( 
    int *m, int *n, 
    double *A, int *lda, int *jpvt, 
    double *tau,
    double *work, int *lwork, int *info );
void sgeqp4( 
    int *m, int *n, 
    float *A, int *lda, int *jpvt, 
    float *tau,
    float *work, int *lwork, int *info );
void dgels_(
    const char *trans,
    int *m, int *n, int *nrhs,
    double *A, int *lda,
    double *B, int *ldb,
    double *work, int *lwork, int *info );
void sgels_(
    const char *trans,
    int *m, int *n, int *nrhs,
    float *A, int *lda,
    float *B, int *ldb,
    float *work, int *lwork, int *info );
void dgesdd_( 
    const char *jobz, 
    int *m, int *n, 
    double *A, int *lda, 
    double *S, 
    double *U, int *ldu, 
    double *VT, int *ldvt, 
    double *work, int *lwork, int *iwork, int *info );
void sgesdd_( 
    const char *jobz, 
    int *m, int *n, 
    float *A, int *lda, 
    float *S, 
    float *U, int *ldu, 
    float *VT, int *ldvt, 
    float *work, int *lwork, int *iwork, int *info );
void dstev_(
  const char *jobz,
  int *n, 
  double *D, 
  double *E, 
  double *Z, int *ldz, 
  double *work, int *info );
void sstev_(
  const char *jobz,
  int *n, 
  float *D, 
  float *E, 
  float *Z, int *ldz, 
  float *work, int *info );


void dlarf_( 
    const char *side, 
    int *m, int *n, 
    double *v, int *incv, 
    double *tau, 
    double *c, int *ldc, 
    double *work );
void slarf_( 
    const char *side, 
    int *m, int *n, 
    float *v, int *incv, 
    float *tau, 
    float *c, int *ldc, 
    float *work );
void dlarft_( 
    const char *direct, const char *storev, 
    int *n, int *k, 
    double *v, int *ldv, 
    double *tau, 
    double *t, int *ldt );
void slarft_( 
    const char *direct, const char *storev, 
    int *n, int *k, 
    float *v, int *ldv, 
    float *tau, 
    float *t, int *ldt );
void dlarfb_( 
    const char *side, const char *trans, const char *direct, const char *storev, 
    int *m, int *n, int *k, 
    double *v, int *ldv,
    double *t, int *ldt, 
    double *c, int *ldc, 
    double *work, int *ldwork );
void slarfb_( 
    const char *side, const char *trans, const char *direct, const char *storev, 
    int *m, int *n, int *k, 
    float *v, int *ldv,
    float *t, int *ldt, 
    float *c, int *ldc, 
    float *work, int *ldwork );
void dlarfg_( int *n, double *alpha, double *x, int *incx, double *tau );
void slarfg_( int *n, float  *alpha, float  *x, int *incx, float  *tau );

#endif /** define BLAS_LAPACK_PROTOTYPES_H */
