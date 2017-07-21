
void sgeqp4( int * m, int * n, float * A, int * lda, int * jpvt, float * tau,
         float * work, int * lwork, int * info );

int sgeqp4_HQRRP_WY_blk_var4( int m_A, int n_A, float * buff_A, int ldim_A,
        int * buff_jpvt, float * buff_tau,
        int nb_alg, int pp, int panel_pivoting );

