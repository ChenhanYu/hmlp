#ifndef HMLP_H
#define HMLP_H


void dgemm_tn(
	int m, int n, int k,
	double *A, int lda, int *amap,
	double *B, int ldb, int *bmap,
	double *C, int ldc
	);









typedef enum {
  KS_GAUSSIAN,
  KS_POLYNOMIAL,
  KS_LAPLACE,
  KS_GAUSSIAN_VAR_BANDWIDTH,
  KS_TANH,
  KS_QUARTIC,
  KS_MULTIQUADRATIC,
  KS_EPANECHNIKOV
} ks_type;

struct kernel_s {
  ks_type type;
  double powe;
  double scal;
  double cons;
  double *hi;
  double *hj;
  double *h;
};

typedef struct kernel_s ks_t;

void dgsks(
    ks_t *kernel,
    int m, int n, int k,
    double *u,             int *umap,
    double *A, double *A2, int *amap,
    double *B, double *B2, int *bmap,
    double *w,             int *wmap
    );

void dgsks_ref(
    ks_t *kernel,
    int m, int n, int k,
    double *u,             int *umap,
    double *A, double *A2, int *amap,
    double *B, double *B2, int *bmap,
    double *w,             int *wmap
    );


#endif // define HMLP_H
