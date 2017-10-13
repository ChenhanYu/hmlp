#ifndef _PVFMM_KERNEL_HPP_
#define _PVFMM_KERNEL_HPP_

#include <pvfmm/intrin_wrapper.hpp>
#include <pvfmm/mem_mgr.hpp>
#include <pvfmm/matrix.hpp>
#include <pvfmm/vector.hpp>
#include <pvfmm/common.hpp>

#include <functional>

namespace pvfmm {

template <class ValueType, Integer DIM> class KernelFunction {

 public:
  typedef void (KerFn)(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx);

  KernelFunction(KerFn& ker_, Integer d_src, Integer d_trg, std::string ker_name_, void* ctx_ = NULL, bool verbose = false) : ker(ker_), ctx(ctx_) {
    ker_name = ker_name_ + std::to_string(sizeof(ValueType) * 8);
    ker_dim[0] = d_src;
    ker_dim[1] = d_trg;
    scale_invar = true;
    Init(verbose);
  }

  virtual ~KernelFunction() {}

  virtual void operator()(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg) const { ker(r_src, n_src, v_src, r_trg, v_trg, ctx); }

  std::string Name() const { return ker_name; }

  Integer Dim(Integer i) const { return ker_dim[i]; }

  bool ScaleInvariant() const { return scale_invar; }

  void BuildMatrix(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& r_trg, Matrix<ValueType>& M) const {
    Long Nsrc = r_src.Dim() / DIM;
    Long Ntrg = r_trg.Dim() / DIM;
    PVFMM_ASSERT(r_src.Dim() == Nsrc * DIM);
    PVFMM_ASSERT(r_trg.Dim() == Ntrg * DIM);
    if (M.Dim(0) != Nsrc * ker_dim[0] || M.Dim(1) != Ntrg * ker_dim[1]) {
      M.ReInit(Nsrc * ker_dim[0], Ntrg * ker_dim[1]);
    }

    Integer omp_p = omp_get_max_threads();
    #pragma omp parallel for schedule(static)
    for (Integer tid = 0; tid < omp_p; tid++) {
      Vector<ValueType> r_src_, n_src_, v_src_(ker_dim[0]), v_trg;
      Long a = Nsrc * (tid + 0) / omp_p;
      Long b = Nsrc * (tid + 1) / omp_p;
      for (Long i = a; i < b; i++) {
        {  // Set r_src_
          r_src_.ReInit(DIM, (Iterator<ValueType>)r_src.Begin() + i * DIM, false);
        }
        if (n_src.Dim()) {  // Set n_src_
          n_src_.ReInit(DIM, (Iterator<ValueType>)n_src.Begin() + i * DIM, false);
        }
        for (Integer j = 0; j < ker_dim[0]; j++) {
          {  // Set v_src_
            v_src_.SetZero();
            v_src_[j] = 1;
          }
          {  // Set v_trg
            v_trg.ReInit(Ntrg * ker_dim[1], M[i * ker_dim[0] + j], false);
            v_trg.SetZero();
          }
          (*this)(r_src_, n_src_, v_src_, r_trg, v_trg);
        }
      }
    }
  }

  const Permutation<ValueType>& SrcPerm(Integer idx) const {
    PVFMM_ASSERT(idx < SymmTransCount);
    return src_perm_vec[idx];
  }

  const Permutation<ValueType>& TrgPerm(Integer idx) const {
    PVFMM_ASSERT(idx < SymmTransCount);
    return trg_perm_vec[idx];
  }

 protected:
  void Init(bool verbose) {
    ValueType eps = 1.0;
    while (eps + (ValueType)1.0 > 1.0) eps *= 0.5;

    Long N = 10240;
    ValueType eps_ = N * N * eps;
    Vector<ValueType> trg_coord(DIM);
    Vector<ValueType> src_coord1(N * DIM);
    Vector<ValueType> src_norml1(N * DIM);
    Matrix<ValueType> M1(N, ker_dim[0] * ker_dim[1]);
    {  // Set trg_coord, src_coord1, src_norml1, M1
      ValueType scal = 1.0;
      trg_coord.SetZero();
      while (true) {
        ValueType abs_sum = 0;
        for (Long i = 0; i < N; i++) {
          ValueType x[DIM], r;
          do {
            r = 0;
            for (Integer j = 0; j < DIM; j++) {
              x[j] = (drand48() - 0.5);
              r += x[j] * x[j];
            }
            r = pvfmm::sqrt<ValueType>(r);
          } while (r < 0.25);
          for (Integer j = 0; j < DIM; j++) {
            src_coord1[i * DIM + j] = x[j] * scal;
          }

          do {
            r = 0;
            for (Integer j = 0; j < DIM; j++) {
              x[j] = (drand48() - 0.5);
              r += x[j] * x[j];
            }
            r = pvfmm::sqrt<ValueType>(r);
          } while (r < 0.25);
          for (Integer j = 0; j < DIM; j++) {
            src_norml1[i * DIM + j] = x[j] / r;
          }
        }
        {  // Set M1
          M1.SetZero();
          Matrix<ValueType> M(N * ker_dim[0], ker_dim[1], M1.Begin(), false);
          BuildMatrix(src_coord1, src_norml1, trg_coord, M);
        }
        for (Long i = 0; i < M1.Dim(0) * M1.Dim(1); i++) {
          abs_sum += pvfmm::fabs<ValueType>(M1[0][i]);
        }
        if (abs_sum > pvfmm::sqrt<ValueType>(eps) || scal < eps) break;
        scal = scal * 0.5;
      }
    }

    if (ker_dim[0] * ker_dim[1] > 0) {  // Determine scaling
      Vector<ValueType> src_coord2(N * DIM);
      Vector<ValueType> src_norml2(N * DIM);
      Matrix<ValueType> M2(N, ker_dim[0] * ker_dim[1]);
      for (Long i = 0; i < N * DIM; i++) {  // Set src_coord1, src_norml1
        src_coord2[i] = src_coord1[i] * 0.5;
        src_norml2[i] = src_norml1[i];
      }
      {  // Set M2
        M2.SetZero();
        Matrix<ValueType> M(N * ker_dim[0], ker_dim[1], M2.Begin(), false);
        BuildMatrix(src_coord2, src_norml2, trg_coord, M);
      }

      ValueType max_val = 0;
      for (Long i = 0; i < ker_dim[0] * ker_dim[1]; i++) {
        ValueType dot11 = 0, dot22 = 0;
        for (Long j = 0; j < N; j++) {
          dot11 += M1[j][i] * M1[j][i];
          dot22 += M2[j][i] * M2[j][i];
        }
        max_val = std::max<ValueType>(max_val, dot11);
        max_val = std::max<ValueType>(max_val, dot22);
      }

      Matrix<ValueType> M_scal(ker_dim[0], ker_dim[1]);
      if (scale_invar) {  // Set M_scal[i][j] = log_2(s[i][j])
        for (Long i = 0; i < ker_dim[0] * ker_dim[1]; i++) {
          ValueType dot11 = 0, dot12 = 0, dot22 = 0;
          for (Long j = 0; j < N; j++) {
            dot11 += M1[j][i] * M1[j][i];
            dot12 += M1[j][i] * M2[j][i];
            dot22 += M2[j][i] * M2[j][i];
          }
          if (dot11 > max_val * eps && dot22 > max_val * eps) {
            PVFMM_ASSERT(dot12 > 0);
            ValueType s = dot12 / dot11;
            M_scal[0][i] = pvfmm::log<ValueType>(s) / pvfmm::log<ValueType>(2.0);
            ValueType err = pvfmm::sqrt<ValueType>((dot22 / dot11) / (s * s) - 1.0);
            if (err > eps_) {
              scale_invar = false;
              M_scal[0][i] = 0.0;
            }
            // assert(M_scal[0][i]>=0.0); // Kernel function must decay
          } else if (dot11 > max_val * eps || dot22 > max_val * eps) {
            scale_invar = false;
            M_scal[0][i] = 0.0;
          } else {
            M_scal[0][i] = -1;
          }
        }
      }

      if (scale_invar) {  // Set src_perm_vec[0], trg_perm_vec[0]
        Matrix<ValueType> b(ker_dim[0] * ker_dim[1] + 1, 1);
        {  // Set b
          memcopy(b.Begin(), M_scal.Begin(), ker_dim[0] * ker_dim[1]);
          b[ker_dim[0] * ker_dim[1]][0] = 0;
        }

        Matrix<ValueType> M(ker_dim[0] * ker_dim[1] + 1, ker_dim[0] + ker_dim[1]);
        {  // Set M
          M.SetZero();
          for (Long i0 = 0; i0 < ker_dim[0]; i0++) {
            for (Long i1 = 0; i1 < ker_dim[1]; i1++) {
              Long j = i0 * ker_dim[1] + i1;
              if (b[j][0] > 0) {
                M[j][i0 + 0] = 1;
                M[j][i1 + ker_dim[0]] = 1;
              }
            }
          }
          M[ker_dim[0] * ker_dim[1]][0] = 1;
        }

        Matrix<ValueType> x = M.pinv() * b;
        for (Long i0 = 0; i0 < ker_dim[0]; i0++) {  // Check solution
          for (Long i1 = 0; i1 < ker_dim[1]; i1++) {
            if (M_scal[i0][i1] >= 0) {
              if (pvfmm::fabs<ValueType>(x[i0][0] + x[ker_dim[0] + i1][0] - M_scal[i0][i1]) > eps_) {
                scale_invar = false;
              }
            }
          }
        }

        Permutation<ValueType> P_src(ker_dim[0]);
        Permutation<ValueType> P_trg(ker_dim[1]);
        if (scale_invar) {  // Set P_src, P_trg (coarsen)
          for (Long i = 0; i < ker_dim[0]; i++) {
            P_src.scal[i] = pvfmm::pow<ValueType>(0.5, x[i][0]);
          }
          for (Long i = 0; i < ker_dim[1]; i++) {
            P_trg.scal[i] = pvfmm::pow<ValueType>(0.5, x[ker_dim[0] + i][0]);
          }
        }
        src_perm_vec[0] = P_src;
        trg_perm_vec[0] = P_trg;
        if (scale_invar) {  // Set P_src, P_trg (refine)
          for (Long i = 0; i < ker_dim[0]; i++) {
            P_src.scal[i] = pvfmm::pow<ValueType>(0.5, -x[i][0]);
          }
          for (Long i = 0; i < ker_dim[1]; i++) {
            P_trg.scal[i] = pvfmm::pow<ValueType>(0.5, -x[ker_dim[0] + i][0]);
          }
        }
        src_perm_vec[1] = P_src;
        trg_perm_vec[1] = P_trg;
      }
    }
    if (ker_dim[0] * ker_dim[1] > 0) {                               // Determine symmetry
      for (Integer p_type = 2; p_type < SymmTransCount; p_type++) {  // For each symmetry transform
        Vector<ValueType> src_coord2(N * DIM);
        Vector<ValueType> src_norml2(N * DIM);
        Matrix<ValueType> M2(N, ker_dim[0] * ker_dim[1]);
        if (p_type < 2 + DIM) {  // Reflect
          for (Long i = 0; i < N; i++) {
            for (Integer j = 0; j < DIM; j++) {
              if (p_type == 2 + j) {
                src_coord2[i * DIM + j] = -src_coord1[i * DIM + j];
                src_norml2[i * DIM + j] = -src_norml1[i * DIM + j];
              } else {
                src_coord2[i * DIM + j] = src_coord1[i * DIM + j];
                src_norml2[i * DIM + j] = src_norml1[i * DIM + j];
              }
            }
          }
        } else {
          Integer swap0, swap1;
          {  // Set swap0, swap1
            Integer iter = 2 + DIM;
            for (Integer i = 0; i < DIM; i++) {
              for (Integer j = i + 1; j < DIM; j++) {
                if (iter == p_type) {
                  swap0 = i;
                  swap1 = j;
                }
                iter++;
              }
            }
          }
          for (Long i = 0; i < N; i++) {
            for (Integer j = 0; j < DIM; j++) {
              if (j == swap0) {
                src_coord2[i * DIM + j] = src_coord1[i * DIM + swap1];
                src_norml2[i * DIM + j] = src_norml1[i * DIM + swap1];
              } else if (j == swap1) {
                src_coord2[i * DIM + j] = src_coord1[i * DIM + swap0];
                src_norml2[i * DIM + j] = src_norml1[i * DIM + swap0];
              } else {
                src_coord2[i * DIM + j] = src_coord1[i * DIM + j];
                src_norml2[i * DIM + j] = src_norml1[i * DIM + j];
              }
            }
          }
        }
        {  // Set M2
          M2.SetZero();
          Matrix<ValueType> M(N * ker_dim[0], ker_dim[1], M2.Begin(), false);
          BuildMatrix(src_coord2, src_norml2, trg_coord, M);
        }

        Matrix<Long> M11, M22;
        {  // Set M11, M22
          Matrix<ValueType> dot11(ker_dim[0] * ker_dim[1], ker_dim[0] * ker_dim[1]);
          Matrix<ValueType> dot12(ker_dim[0] * ker_dim[1], ker_dim[0] * ker_dim[1]);
          Matrix<ValueType> dot22(ker_dim[0] * ker_dim[1], ker_dim[0] * ker_dim[1]);
          Vector<ValueType> norm1(ker_dim[0] * ker_dim[1]);
          Vector<ValueType> norm2(ker_dim[0] * ker_dim[1]);
          {  // Set dot11, dot12, dot22, norm1, norm2
            dot11.SetZero();
            dot12.SetZero();
            dot22.SetZero();
            for (Long k = 0; k < N; k++) {
              for (Long i = 0; i < ker_dim[0] * ker_dim[1]; i++) {
                for (Long j = 0; j < ker_dim[0] * ker_dim[1]; j++) {
                  dot11[i][j] += M1[k][i] * M1[k][j];
                  dot12[i][j] += M1[k][i] * M2[k][j];
                  dot22[i][j] += M2[k][i] * M2[k][j];
                }
              }
            }
            for (Long i = 0; i < ker_dim[0] * ker_dim[1]; i++) {
              norm1[i] = pvfmm::sqrt<ValueType>(dot11[i][i]);
              norm2[i] = pvfmm::sqrt<ValueType>(dot22[i][i]);
            }
            for (Long i = 0; i < ker_dim[0] * ker_dim[1]; i++) {
              for (Long j = 0; j < ker_dim[0] * ker_dim[1]; j++) {
                dot11[i][j] /= (norm1[i] * norm1[j]);
                dot12[i][j] /= (norm1[i] * norm2[j]);
                dot22[i][j] /= (norm2[i] * norm2[j]);
              }
            }
          }

          Long flag = 1;
          M11.ReInit(ker_dim[0], ker_dim[1]);
          M22.ReInit(ker_dim[0], ker_dim[1]);
          M11.SetZero();
          M22.SetZero();
          for (Long i = 0; i < ker_dim[0] * ker_dim[1]; i++) {
            if (norm1[i] > eps_ && M11[0][i] == 0) {
              for (Long j = 0; j < ker_dim[0] * ker_dim[1]; j++) {
                if (pvfmm::fabs<ValueType>(norm1[i] - norm1[j]) < eps_ && pvfmm::fabs<ValueType>(pvfmm::fabs<ValueType>(dot11[i][j]) - 1.0) < eps_) {
                  M11[0][j] = (dot11[i][j] > 0 ? flag : -flag);
                }
                if (pvfmm::fabs<ValueType>(norm1[i] - norm2[j]) < eps_ && pvfmm::fabs<ValueType>(pvfmm::fabs<ValueType>(dot12[i][j]) - 1.0) < eps_) {
                  M22[0][j] = (dot12[i][j] > 0 ? flag : -flag);
                }
              }
              flag++;
            }
          }
        }

        Matrix<Long> P1, P2;
        {  // P1
          Matrix<Long>& P = P1;
          Matrix<Long> M1 = M11;
          Matrix<Long> M2 = M22;
          for (Long i = 0; i < M1.Dim(0); i++) {
            for (Long j = 0; j < M1.Dim(1); j++) {
              if (M1[i][j] < 0) M1[i][j] = -M1[i][j];
              if (M2[i][j] < 0) M2[i][j] = -M2[i][j];
            }
            std::sort(M1[i], M1[i] + M1.Dim(1));
            std::sort(M2[i], M2[i] + M2.Dim(1));
          }
          P.ReInit(M1.Dim(0), M1.Dim(0));
          for (Long i = 0; i < M1.Dim(0); i++) {
            for (Long j = 0; j < M1.Dim(0); j++) {
              P[i][j] = 1;
              for (Long k = 0; k < M1.Dim(1); k++) {
                if (M1[i][k] != M2[j][k]) {
                  P[i][j] = 0;
                  break;
                }
              }
            }
          }
        }
        {  // P2
          Matrix<Long>& P = P2;
          Matrix<Long> M1 = M11.Transpose();
          Matrix<Long> M2 = M22.Transpose();
          for (Long i = 0; i < M1.Dim(0); i++) {
            for (Long j = 0; j < M1.Dim(1); j++) {
              if (M1[i][j] < 0) M1[i][j] = -M1[i][j];
              if (M2[i][j] < 0) M2[i][j] = -M2[i][j];
            }
            std::sort(M1[i], M1[i] + M1.Dim(1));
            std::sort(M2[i], M2[i] + M2.Dim(1));
          }
          P.ReInit(M1.Dim(0), M1.Dim(0));
          for (Long i = 0; i < M1.Dim(0); i++) {
            for (Long j = 0; j < M1.Dim(0); j++) {
              P[i][j] = 1;
              for (Long k = 0; k < M1.Dim(1); k++) {
                if (M1[i][k] != M2[j][k]) {
                  P[i][j] = 0;
                  break;
                }
              }
            }
          }
        }

        Vector<Permutation<Long>> P1vec, P2vec;
        {  // P1vec
          Matrix<Long>& Pmat = P1;
          Vector<Permutation<Long>>& Pvec = P1vec;

          Permutation<Long> P(Pmat.Dim(0));
          Vector<Long>& perm = P.perm;
          perm.SetZero();

          // First permutation
          for (Long i = 0; i < P.Dim(); i++) {
            for (Long j = 0; j < P.Dim(); j++) {
              if (Pmat[i][j]) {
                perm[i] = j;
                break;
              }
            }
          }

          Vector<Long> perm_tmp;
          while (true) {  // Next permutation
            perm_tmp = perm;
            std::sort(&perm_tmp[0], &perm_tmp[0] + perm_tmp.Dim());
            for (Long i = 0; i < perm_tmp.Dim(); i++) {
              if (perm_tmp[i] != i) break;
              if (i == perm_tmp.Dim() - 1) {
                Pvec.PushBack(P);
              }
            }

            bool last = false;
            for (Long i = 0; i < P.Dim(); i++) {
              Long tmp = perm[i];
              for (Long j = perm[i] + 1; j < P.Dim(); j++) {
                if (Pmat[i][j]) {
                  perm[i] = j;
                  break;
                }
              }
              if (perm[i] > tmp) break;
              for (Long j = 0; j < P.Dim(); j++) {
                if (Pmat[i][j]) {
                  perm[i] = j;
                  break;
                }
              }
              if (i == P.Dim() - 1) last = true;
            }
            if (last) break;
          }
        }
        {  // P2vec
          Matrix<Long>& Pmat = P2;
          Vector<Permutation<Long>>& Pvec = P2vec;

          Permutation<Long> P(Pmat.Dim(0));
          Vector<Long>& perm = P.perm;
          perm.SetZero();

          // First permutation
          for (Long i = 0; i < P.Dim(); i++) {
            for (Long j = 0; j < P.Dim(); j++) {
              if (Pmat[i][j]) {
                perm[i] = j;
                break;
              }
            }
          }

          Vector<Long> perm_tmp;
          while (true) {  // Next permutation
            perm_tmp = perm;
            std::sort(&perm_tmp[0], &perm_tmp[0] + perm_tmp.Dim());
            for (Long i = 0; i < perm_tmp.Dim(); i++) {
              if (perm_tmp[i] != i) break;
              if (i == perm_tmp.Dim() - 1) {
                Pvec.PushBack(P);
              }
            }

            bool last = false;
            for (Long i = 0; i < P.Dim(); i++) {
              Long tmp = perm[i];
              for (Long j = perm[i] + 1; j < P.Dim(); j++) {
                if (Pmat[i][j]) {
                  perm[i] = j;
                  break;
                }
              }
              if (perm[i] > tmp) break;
              for (Long j = 0; j < P.Dim(); j++) {
                if (Pmat[i][j]) {
                  perm[i] = j;
                  break;
                }
              }
              if (i == P.Dim() - 1) last = true;
            }
            if (last) break;
          }
        }

        {  // Find pairs which acutally work (neglect scaling)
          Vector<Permutation<Long>> P1vec_, P2vec_;
          Matrix<Long> M1 = M11;
          Matrix<Long> M2 = M22;
          for (Long i = 0; i < M1.Dim(0); i++) {
            for (Long j = 0; j < M1.Dim(1); j++) {
              if (M1[i][j] < 0) M1[i][j] = -M1[i][j];
              if (M2[i][j] < 0) M2[i][j] = -M2[i][j];
            }
          }

          Matrix<Long> M;
          for (Long i = 0; i < P1vec.Dim(); i++) {
            for (Long j = 0; j < P2vec.Dim(); j++) {
              M = P1vec[i] * M2 * P2vec[j];
              for (Long k = 0; k < M.Dim(0) * M.Dim(1); k++) {
                if (M[0][k] != M1[0][k]) break;
                if (k == M.Dim(0) * M.Dim(1) - 1) {
                  P1vec_.PushBack(P1vec[i]);
                  P2vec_.PushBack(P2vec[j]);
                }
              }
            }
          }

          P1vec = P1vec_;
          P2vec = P2vec_;
        }

        Permutation<ValueType> P1_, P2_;
        {  // Find pairs which acutally work
          for (Long k = 0; k < P1vec.Dim(); k++) {
            Permutation<Long> P1 = P1vec[k];
            Permutation<Long> P2 = P2vec[k];
            Matrix<Long> M1 = M11;
            Matrix<Long> M2 = P1 * M22 * P2;

            Matrix<ValueType> M(M1.Dim(0) * M1.Dim(1) + 1, M1.Dim(0) + M1.Dim(1));
            M.SetZero();
            M[M1.Dim(0) * M1.Dim(1)][0] = 1.0;
            for (Long i = 0; i < M1.Dim(0); i++) {
              for (Long j = 0; j < M1.Dim(1); j++) {
                Long k = i * M1.Dim(1) + j;
                M[k][i] = M1[i][j];
                M[k][M1.Dim(0) + j] = -M2[i][j];
              }
            }
            M = M.pinv();
            {  // Construct new permutation
              Permutation<Long> P1_(M1.Dim(0));
              Permutation<Long> P2_(M1.Dim(1));
              for (Long i = 0; i < M1.Dim(0); i++) {
                P1_.scal[i] = (M[i][M1.Dim(0) * M1.Dim(1)] > 0 ? 1 : -1);
              }
              for (Long i = 0; i < M1.Dim(1); i++) {
                P2_.scal[i] = (M[M1.Dim(0) + i][M1.Dim(0) * M1.Dim(1)] > 0 ? 1 : -1);
              }
              P1 = P1_ * P1;
              P2 = P2 * P2_;
            }

            bool done = true;
            Matrix<Long> Merr = P1 * M22 * P2 - M11;
            for (Long i = 0; i < Merr.Dim(0) * Merr.Dim(1); i++) {
              if (Merr[0][i]) {
                done = false;
                break;
              }
            }
            {  // Check if permutation is symmetric
              Permutation<Long> P1_ = P1.Transpose();
              Permutation<Long> P2_ = P2.Transpose();
              for (Long i = 0; i < P1.Dim(); i++) {
                if (P1_.perm[i] != P1.perm[i] || P1_.scal[i] != P1.scal[i]) {
                  done = false;
                  break;
                }
              }
              for (Long i = 0; i < P2.Dim(); i++) {
                if (P2_.perm[i] != P2.perm[i] || P2_.scal[i] != P2.scal[i]) {
                  done = false;
                  break;
                }
              }
            }
            if (done) {
              P1_ = Permutation<ValueType>(P1.Dim());
              P2_ = Permutation<ValueType>(P2.Dim());
              for (Long i = 0; i < P1.Dim(); i++) {
                P1_.perm[i] = P1.perm[i];
                P1_.scal[i] = P1.scal[i];
              }
              for (Long i = 0; i < P2.Dim(); i++) {
                P2_.perm[i] = P2.perm[i];
                P2_.scal[i] = P2.scal[i];
              }
              break;
            }
          }
          assert(P1_.Dim() && P2_.Dim());
        }
        src_perm_vec[p_type] = P1_;
        trg_perm_vec[p_type] = P2_;
      }
      for (Integer i = 2; i < SymmTransCount; i++) {
        PVFMM_ASSERT_MSG(src_perm_vec[i].Dim() && trg_perm_vec[i].Dim(), "no-symmetry for: " << ker_name);
      }
    }
    if (verbose) {  // Display kernel information
      std::cout << "\n";
      std::cout << "Kernel Name    : " << ker_name << '\n';
      std::cout << "Precision      : " << (double)eps << '\n';
      std::cout << "Scale Invariant: " << (scale_invar ? "yes" : "no") << '\n';
      if (scale_invar && ker_dim[0] * ker_dim[1] > 0) {
        std::cout << "Scaling Matrix :\n";
        Matrix<ValueType> Src(ker_dim[0], 1);
        Matrix<ValueType> Trg(1, ker_dim[1]);
        for (Long i = 0; i < ker_dim[0]; i++) Src[i][0] = 1.0 / src_perm_vec[0].scal[i];
        for (Long i = 0; i < ker_dim[1]; i++) Trg[0][i] = 1.0 / trg_perm_vec[0].scal[i];
        std::cout << Src* Trg;
      }
      std::cout << "\n";
    }
    PVFMM_ASSERT(scale_invar);
  }

  void* ctx;
  KerFn& ker;
  std::string ker_name;
  StaticArray<Integer, 2> ker_dim;

  bool scale_invar;
  static const Integer SymmTransCount = 2 + DIM + DIM * (DIM - 1) / 2;  // Scaling + Reflection + Coordinate-Swap
  StaticArray<Permutation<ValueType>, SymmTransCount> src_perm_vec, trg_perm_vec;
};

class KernelFnWrapper {
 public:
  /**
   * \brief Generic kernel which rearranges data for vectorization, calls the
   * actual uKernel and copies data to the output array in the original order.
   */
  template <class ValueType, class Real, class Vec, Integer DIM, Integer SRC_DIM, Integer TRG_DIM, void (*uKernel)(const Matrix<Real>&, const Matrix<Real>&, const Matrix<Real>&, const Matrix<Real>&, Matrix<Real>&)> static void kernel_wrapper(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg) {
    Integer VecLen = sizeof(Vec) / sizeof(Real);
    Long src_cnt = r_src.Dim() / DIM;
    Long trg_cnt = r_trg.Dim() / DIM;

    #define STACK_BUFF_SIZE 4096
    StaticArray<Real, STACK_BUFF_SIZE + PVFMM_MEM_ALIGN> stack_buff;
    Iterator<Real> buff = NULL;

    Matrix<Real> src_coord, src_norml, src_value, trg_coord, trg_value;
    {  // Rearrange data in src_coord, src_norml, src_value, trg_coord, trg_value
      Long src_cnt_ = ((src_cnt + VecLen - 1) / VecLen) * VecLen;
      Long trg_cnt_ = ((trg_cnt + VecLen - 1) / VecLen) * VecLen;

      Iterator<Real> buff_ptr = NULL;
      {  // Set buff_ptr
        Long buff_size = 0;
        if (r_src.Dim()) buff_size += src_cnt_ * DIM;
        if (n_src.Dim()) buff_size += src_cnt_ * DIM;
        if (v_src.Dim()) buff_size += src_cnt_ * SRC_DIM;
        if (r_trg.Dim()) buff_size += trg_cnt_ * DIM;
        if (v_trg.Dim()) buff_size += trg_cnt_ * TRG_DIM;
        if (buff_size > STACK_BUFF_SIZE) {  // Allocate buff
          buff = aligned_new<Real>(buff_size);
          buff_ptr = buff;
        } else {  // use stack_buff
          const uintptr_t ptr = (uintptr_t) & stack_buff[0];
          const uintptr_t ALIGN_MINUS_ONE = PVFMM_MEM_ALIGN - 1;
          const uintptr_t NOT_ALIGN_MINUS_ONE = ~ALIGN_MINUS_ONE;
          const uintptr_t offset = ((ptr + ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE) - ptr;
          buff_ptr = (Iterator<Real>)((Iterator<char>)stack_buff + offset);
        }
      }
      if (r_src.Dim()) {  // Set src_coord
        src_coord.ReInit(DIM, src_cnt_, buff_ptr, false);
        buff_ptr += DIM * src_cnt_;
        Long i = 0;
        for (; i < src_cnt; i++) {
          for (Long j = 0; j < DIM; j++) {
            src_coord[j][i] = r_src[i * DIM + j];
          }
        }
        for (; i < src_cnt_; i++) {
          for (Long j = 0; j < DIM; j++) {
            src_coord[j][i] = 0;
          }
        }
      }
      if (n_src.Dim()) {  // Set src_norml
        src_norml.ReInit(DIM, src_cnt_, buff_ptr, false);
        buff_ptr += DIM * src_cnt_;
        Long i = 0;
        for (; i < src_cnt; i++) {
          for (Long j = 0; j < DIM; j++) {
            src_norml[j][i] = n_src[i * DIM + j];
          }
        }
        for (; i < src_cnt_; i++) {
          for (Long j = 0; j < DIM; j++) {
            src_norml[j][i] = 0;
          }
        }
      }
      if (v_src.Dim()) {  // Set src_value
        src_value.ReInit(SRC_DIM, src_cnt_, buff_ptr, false);
        buff_ptr += SRC_DIM * src_cnt_;
        Long i = 0;
        for (; i < src_cnt; i++) {
          for (Long j = 0; j < SRC_DIM; j++) {
            src_value[j][i] = v_src[i * SRC_DIM + j];
          }
        }
        for (; i < src_cnt_; i++) {
          for (Long j = 0; j < SRC_DIM; j++) {
            src_value[j][i] = 0;
          }
        }
      }
      if (r_trg.Dim()) {  // Set trg_coord
        trg_coord.ReInit(DIM, trg_cnt_, buff_ptr, false);
        buff_ptr += DIM * trg_cnt_;
        Long i = 0;
        for (; i < trg_cnt; i++) {
          for (Long j = 0; j < DIM; j++) {
            trg_coord[j][i] = r_trg[i * DIM + j];
          }
        }
        for (; i < trg_cnt_; i++) {
          for (Long j = 0; j < DIM; j++) {
            trg_coord[j][i] = 0;
          }
        }
      }
      if (v_trg.Dim()) {  // Set trg_value
        trg_value.ReInit(TRG_DIM, trg_cnt_, buff_ptr, false);
        buff_ptr += TRG_DIM * trg_cnt_;
        Long i = 0;
        for (; i < trg_cnt_; i++) {
          for (Long j = 0; j < TRG_DIM; j++) {
            trg_value[j][i] = 0;
          }
        }
      }
    }
    uKernel(src_coord, src_norml, src_value, trg_coord, trg_value);
    {  // Set v_trg
      for (Long i = 0; i < trg_cnt; i++) {
        for (Long j = 0; j < TRG_DIM; j++) {
          v_trg[i * TRG_DIM + j] += trg_value[j][i];
        }
      }
    }
    if (buff != NULL) {  // Free memory: buff
      aligned_delete<Real>(buff);
    }
  }
};

template <class ValueType> struct Laplace3D {
 public:
  static const Integer DIM = 3;

  inline static const KernelFunction<ValueType, DIM>& single_layer() {
    static KernelFunction<ValueType, DIM> ker(sl_poten<2>, 1, 1, "laplace-sl");
    return ker;
  }

  inline static const KernelFunction<ValueType, DIM>& double_layer() {
    static KernelFunction<ValueType, DIM> ker(dl_poten, 1, 1, "laplace-dl");
    return ker;
  }

  inline static const KernelFunction<ValueType, DIM>& single_layer_gradient() {
    static KernelFunction<ValueType, DIM> ker(sl_grad, 1, DIM, "laplace-sl-grad");
    return ker;
  }

  inline static const KernelFunction<ValueType, DIM>& double_layer_gradient() {
    static KernelFunction<ValueType, DIM> ker(dl_grad, 1, DIM, "laplace-dl-grad");
    return ker;
  }

 protected:
  template <class Vec = ValueType, Vec (*RSQRT_INTRIN)(Vec) = rsqrt_intrin0<Vec>> static void sl_poten_uKernel(const Matrix<ValueType>& src_coord, const Matrix<ValueType>& src_norml, const Matrix<ValueType>& src_value, const Matrix<ValueType>& trg_coord, Matrix<ValueType>& trg_value) {
    #define SRC_BLK 1000
    Integer VecLen = sizeof(Vec) / sizeof(ValueType);

    //// Number of newton iterations
    Integer NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin0<Vec, ValueType>) NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin1<Vec, ValueType>) NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin2<Vec, ValueType>) NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin3<Vec, ValueType>) NWTN_ITER = 3;

    ValueType nwtn_scal = 1;  // scaling factor for newton iterations
    for (Integer i = 0; i < NWTN_ITER; i++) {
      nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const ValueType OOFP = 1.0 / (4 * nwtn_scal * const_pi<ValueType>());

    Long src_cnt_ = src_coord.Dim(1);
    Long trg_cnt_ = trg_coord.Dim(1);
    for (Long sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
      Long src_cnt = src_cnt_ - sblk;
      if (src_cnt > SRC_BLK) src_cnt = SRC_BLK;
      for (Long t = 0; t < trg_cnt_; t += VecLen) {
        Vec tx = load_intrin<Vec>(&trg_coord[0][t]);
        Vec ty = load_intrin<Vec>(&trg_coord[1][t]);
        Vec tz = load_intrin<Vec>(&trg_coord[2][t]);
        Vec tv = zero_intrin<Vec>();
        for (Long s = sblk; s < sblk + src_cnt; s++) {
          Vec dx = sub_intrin(tx, bcast_intrin<Vec>(&src_coord[0][s]));
          Vec dy = sub_intrin(ty, bcast_intrin<Vec>(&src_coord[1][s]));
          Vec dz = sub_intrin(tz, bcast_intrin<Vec>(&src_coord[2][s]));
          Vec sv = bcast_intrin<Vec>(&src_value[0][s]);

          Vec r2 = mul_intrin(dx, dx);
          r2 = add_intrin(r2, mul_intrin(dy, dy));
          r2 = add_intrin(r2, mul_intrin(dz, dz));

          Vec rinv = RSQRT_INTRIN(r2);
          tv = add_intrin(tv, mul_intrin(rinv, sv));
        }
        Vec oofp = set_intrin<Vec, ValueType>(OOFP);
        tv = add_intrin(mul_intrin(tv, oofp), load_intrin<Vec>(&trg_value[0][t]));
        store_intrin(&trg_value[0][t], tv);
      }
    }

    {  // Add FLOPS
      #ifndef __MIC__
      //Profile::Add_FLOP((Long)trg_cnt_ * (Long)src_cnt_ * (12 + 4 * (NWTN_ITER)));
      #endif
    }
    #undef SRC_BLK
  }

 private:
  template <Integer newton_iter = 0> static void sl_poten(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    #define PVFMM_KER_NWTN(nwtn) \
  if (newton_iter == nwtn) KernelFnWrapper::kernel_wrapper<ValueType, Real, Vec, DIM, 1, 1, Laplace3D<Real>::template sl_poten_uKernel<Vec, rsqrt_intrin##nwtn<Vec, Real>>>(r_src, n_src, v_src, r_trg, v_trg)
    #define PVFMM_KERNEL_MACRO \
  PVFMM_KER_NWTN(0);       \
  PVFMM_KER_NWTN(1);       \
  PVFMM_KER_NWTN(2);       \
  PVFMM_KER_NWTN(3);
    if (TypeTraits<ValueType>::ID() == TypeTraits<float>::ID()) {
      typedef float Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256
      #elif defined __SSE3__
      #define Vec __m128
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else if (TypeTraits<ValueType>::ID() == TypeTraits<double>::ID()) {
      typedef double Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256d
      #elif defined __SSE3__
      #define Vec __m128d
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else {
      typedef ValueType Real;
      #define Vec Real
      PVFMM_KERNEL_MACRO;
      #undef Vec
    }
    #undef PVFMM_KER_NWTN
  }

  static void dl_poten(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    const ValueType OOFP = 1.0 / (4 * const_pi<ValueType>());
    Long Nsrc = r_src.Dim() / DIM;
    Long Ntrg = r_trg.Dim() / DIM;
    PVFMM_ASSERT(r_src.Dim() == Nsrc * DIM);
    PVFMM_ASSERT(n_src.Dim() == Nsrc * DIM);
    PVFMM_ASSERT(v_src.Dim() == Nsrc * 1);
    PVFMM_ASSERT(r_trg.Dim() == Ntrg * DIM);
    PVFMM_ASSERT(v_trg.Dim() == Ntrg * 1);
    for (Long i = 0; i < Ntrg; i++) {
      ValueType trg_val = 0.0;
      for (Long j = 0; j < Nsrc; j++) {
        ValueType dx[DIM];
        for (Integer k = 0; k < DIM; k++) {
          dx[k] = r_trg[i * DIM + k] - r_src[j * DIM + k];
        }

        ValueType n_dot_r = 0;
        for (Integer k = 0; k < DIM; k++) {
          n_dot_r += n_src[j * DIM + k] * dx[k];
        }

        ValueType r2 = 0;
        for (Integer k = 0; k < DIM; k++) {
          r2 += dx[k] * dx[k];
        }
        ValueType r2inv = (r2 ? 1.0 / r2 : 0.0);
        ValueType rinv = sqrt(r2inv);

        trg_val += v_src[j] * n_dot_r * r2inv * rinv;
      }
      v_trg[i] += trg_val * OOFP;
    }
    //Profile::Add_FLOP((Long)Ntrg * (Long)Nsrc * 19);
  }

  static void sl_grad(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    Long Nsrc = r_src.Dim() / DIM;
    Long Ntrg = r_trg.Dim() / DIM;
    PVFMM_ASSERT(r_src.Dim() == Nsrc * DIM);
    PVFMM_ASSERT(v_src.Dim() == Nsrc * 1);
    PVFMM_ASSERT(r_trg.Dim() == Ntrg * DIM);
    PVFMM_ASSERT(v_trg.Dim() == Ntrg * DIM);
    for (Long i = 0; i < Ntrg; i++) {
      ValueType trg_val[DIM];
      for (Integer k = 0; k < DIM; k++) {
        trg_val[k] = 0;
      }
      for (Long j = 0; j < Nsrc; j++) {
        ValueType dx[DIM];
        for (Integer k = 0; k < DIM; k++) {
          dx[k] = r_trg[i * DIM + k] - r_src[j * DIM + k];
        }

        ValueType r2 = 0;
        for (Integer k = 0; k < DIM; k++) {
          r2 += dx[k] * dx[k];
        }
        ValueType r2inv = (r2 ? 1.0 / r2 : 0.0);
        ValueType rinv = sqrt(r2inv);

        ValueType v_r3inv = v_src[j] * r2inv * rinv;
        for (Integer k = 0; k < DIM; k++) {
          trg_val[k] += dx[k] * v_r3inv;
        }
      }
      for (Integer k = 0; k < DIM; k++) {
        v_trg[i * DIM + k] += trg_val[k];
      }
    }
    //Profile::Add_FLOP((Long)Ntrg * (Long)Nsrc * 18);
  }

  static void dl_grad(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    Long Nsrc = r_src.Dim() / DIM;
    Long Ntrg = r_trg.Dim() / DIM;
    PVFMM_ASSERT(r_src.Dim() == Nsrc * DIM);
    PVFMM_ASSERT(n_src.Dim() == Nsrc * DIM);
    PVFMM_ASSERT(v_src.Dim() == Nsrc * 1);
    PVFMM_ASSERT(r_trg.Dim() == Ntrg * DIM);
    PVFMM_ASSERT(v_trg.Dim() == Ntrg * DIM);
    for (Long i = 0; i < Ntrg; i++) {
      ValueType trg_val[DIM];
      for (Integer k = 0; k < DIM; k++) {
        trg_val[k] = 0;
      }
      for (Long j = 0; j < Nsrc; j++) {
        ValueType dx[DIM];
        for (Integer k = 0; k < DIM; k++) {
          dx[k] = r_trg[i * DIM + k] - r_src[j * DIM + k];
        }

        ValueType n_dot_r = 0;
        for (Integer k = 0; k < DIM; k++) {
          n_dot_r += n_src[j * DIM + k] * dx[k];
        }

        ValueType r2 = 0;
        for (Integer k = 0; k < DIM; k++) {
          r2 += dx[k] * dx[k];
        }
        ValueType r2inv = (r2 ? 1.0 / r2 : 0.0);
        ValueType rinv = sqrt(r2inv);

        ValueType v_nr_r5inv = v_src[j] * r2inv * r2inv * rinv * n_dot_r;
        for (Integer k = 0; k < DIM; k++) {
          trg_val[k] += dx[k] * v_nr_r5inv;
        }
      }
      for (Integer k = 0; k < DIM; k++) {
        v_trg[i * DIM + k] += trg_val[k];
      }
    }
    //Profile::Add_FLOP((Long)Ntrg * (Long)Nsrc * 25);
  }

  friend class KernelFnWrapper;
};

template <class ValueType> struct Stokes3D {
 public:
  static const Integer DIM = 3;

  inline static const KernelFunction<ValueType, DIM>& FxP() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = 1;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (2*invr3*fdotr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FxdP() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (2*invr3*f[0]-6*x[0]*invr5*fdotr) * scal;
      v[1] += (2*invr3*f[1]-6*x[1]*invr5*fdotr) * scal;
      v[2] += (2*invr3*f[2]-6*x[2]*invr5*fdotr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FxU() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (x[0]*invr3*fdotr+f[0]*invr) * scal;
      v[1] += (x[1]*invr3*fdotr+f[1]*invr) * scal;
      v[2] += (x[2]*invr3*fdotr+f[2]*invr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FxdU() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM * DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (invr3*fdotr-3*x[0]*x[0]*invr5*fdotr                       ) * scal;
      v[1] += ((-3*x[0]*x[1]*invr5*fdotr)+x[0]*invr3*f[1]-x[1]*invr3*f[0]) * scal;
      v[2] += ((-3*x[0]*x[2]*invr5*fdotr)+x[0]*invr3*f[2]-x[2]*invr3*f[0]) * scal;
      v[3] += ((-3*x[0]*x[1]*invr5*fdotr)-x[0]*invr3*f[1]+x[1]*invr3*f[0]) * scal;
      v[4] += (invr3*fdotr-3*x[1]*x[1]*invr5*fdotr                       ) * scal;
      v[5] += ((-3*x[1]*x[2]*invr5*fdotr)+x[1]*invr3*f[2]-x[2]*invr3*f[1]) * scal;
      v[6] += ((-3*x[0]*x[2]*invr5*fdotr)-x[0]*invr3*f[2]+x[2]*invr3*f[0]) * scal;
      v[7] += ((-3*x[1]*x[2]*invr5*fdotr)-x[1]*invr3*f[2]+x[2]*invr3*f[1]) * scal;
      v[8] += (invr3*fdotr-3*x[2]*x[2]*invr5*fdotr                       ) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& Fxd2U() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM * DIM * DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[ 0] += ((-9*x[0]*invr5*fdotr)+15*x[0]*x[0]*x[0]*invr7*fdotr+invr3*f[0]-3*x[0]*x[0]*invr5*f[0]                       ) * scal;
      v[ 1] += ((-3*x[1]*invr5*fdotr)+15*x[0]*x[0]*x[1]*invr7*fdotr+invr3*f[1]-3*x[0]*x[0]*invr5*f[1]                       ) * scal;
      v[ 2] += ((-3*x[2]*invr5*fdotr)+15*x[0]*x[0]*x[2]*invr7*fdotr+invr3*f[2]-3*x[0]*x[0]*invr5*f[2]                       ) * scal;
      v[ 3] += ((-3*x[1]*invr5*fdotr)+15*x[0]*x[0]*x[1]*invr7*fdotr+invr3*f[1]-3*x[0]*x[0]*invr5*f[1]                       ) * scal;
      v[ 4] += ((-3*x[0]*invr5*fdotr)+15*x[0]*x[1]*x[1]*invr7*fdotr-6*x[0]*x[1]*invr5*f[1]-invr3*f[0]+3*x[1]*x[1]*invr5*f[0]) * scal;
      v[ 5] += (15*x[0]*x[1]*x[2]*invr7*fdotr-3*x[0]*x[1]*invr5*f[2]-3*x[0]*x[2]*invr5*f[1]+3*x[1]*x[2]*invr5*f[0]          ) * scal;
      v[ 6] += ((-3*x[2]*invr5*fdotr)+15*x[0]*x[0]*x[2]*invr7*fdotr+invr3*f[2]-3*x[0]*x[0]*invr5*f[2]                       ) * scal;
      v[ 7] += (15*x[0]*x[1]*x[2]*invr7*fdotr-3*x[0]*x[1]*invr5*f[2]-3*x[0]*x[2]*invr5*f[1]+3*x[1]*x[2]*invr5*f[0]          ) * scal;
      v[ 8] += ((-3*x[0]*invr5*fdotr)+15*x[0]*x[2]*x[2]*invr7*fdotr-6*x[0]*x[2]*invr5*f[2]-invr3*f[0]+3*x[2]*x[2]*invr5*f[0]) * scal;
      v[ 9] += ((-3*x[1]*invr5*fdotr)+15*x[0]*x[0]*x[1]*invr7*fdotr-invr3*f[1]+3*x[0]*x[0]*invr5*f[1]-6*x[0]*x[1]*invr5*f[0]) * scal;
      v[10] += ((-3*x[0]*invr5*fdotr)+15*x[0]*x[1]*x[1]*invr7*fdotr+invr3*f[0]-3*x[1]*x[1]*invr5*f[0]                       ) * scal;
      v[11] += (15*x[0]*x[1]*x[2]*invr7*fdotr-3*x[0]*x[1]*invr5*f[2]+3*x[0]*x[2]*invr5*f[1]-3*x[1]*x[2]*invr5*f[0]          ) * scal;
      v[12] += ((-3*x[0]*invr5*fdotr)+15*x[0]*x[1]*x[1]*invr7*fdotr+invr3*f[0]-3*x[1]*x[1]*invr5*f[0]                       ) * scal;
      v[13] += ((-9*x[1]*invr5*fdotr)+15*x[1]*x[1]*x[1]*invr7*fdotr+invr3*f[1]-3*x[1]*x[1]*invr5*f[1]                       ) * scal;
      v[14] += ((-3*x[2]*invr5*fdotr)+15*x[1]*x[1]*x[2]*invr7*fdotr+invr3*f[2]-3*x[1]*x[1]*invr5*f[2]                       ) * scal;
      v[15] += (15*x[0]*x[1]*x[2]*invr7*fdotr-3*x[0]*x[1]*invr5*f[2]+3*x[0]*x[2]*invr5*f[1]-3*x[1]*x[2]*invr5*f[0]          ) * scal;
      v[16] += ((-3*x[2]*invr5*fdotr)+15*x[1]*x[1]*x[2]*invr7*fdotr+invr3*f[2]-3*x[1]*x[1]*invr5*f[2]                       ) * scal;
      v[17] += ((-3*x[1]*invr5*fdotr)+15*x[1]*x[2]*x[2]*invr7*fdotr-6*x[1]*x[2]*invr5*f[2]-invr3*f[1]+3*x[2]*x[2]*invr5*f[1]) * scal;
      v[18] += ((-3*x[2]*invr5*fdotr)+15*x[0]*x[0]*x[2]*invr7*fdotr-invr3*f[2]+3*x[0]*x[0]*invr5*f[2]-6*x[0]*x[2]*invr5*f[0]) * scal;
      v[19] += (15*x[0]*x[1]*x[2]*invr7*fdotr+3*x[0]*x[1]*invr5*f[2]-3*x[0]*x[2]*invr5*f[1]-3*x[1]*x[2]*invr5*f[0]          ) * scal;
      v[20] += ((-3*x[0]*invr5*fdotr)+15*x[0]*x[2]*x[2]*invr7*fdotr+invr3*f[0]-3*x[2]*x[2]*invr5*f[0]                       ) * scal;
      v[21] += (15*x[0]*x[1]*x[2]*invr7*fdotr+3*x[0]*x[1]*invr5*f[2]-3*x[0]*x[2]*invr5*f[1]-3*x[1]*x[2]*invr5*f[0]          ) * scal;
      v[22] += ((-3*x[2]*invr5*fdotr)+15*x[1]*x[1]*x[2]*invr7*fdotr-invr3*f[2]+3*x[1]*x[1]*invr5*f[2]-6*x[1]*x[2]*invr5*f[1]) * scal;
      v[23] += ((-3*x[1]*invr5*fdotr)+15*x[1]*x[2]*x[2]*invr7*fdotr+invr3*f[1]-3*x[2]*x[2]*invr5*f[1]                       ) * scal;
      v[24] += ((-3*x[0]*invr5*fdotr)+15*x[0]*x[2]*x[2]*invr7*fdotr+invr3*f[0]-3*x[2]*x[2]*invr5*f[0]                       ) * scal;
      v[25] += ((-3*x[1]*invr5*fdotr)+15*x[1]*x[2]*x[2]*invr7*fdotr+invr3*f[1]-3*x[2]*x[2]*invr5*f[1]                       ) * scal;
      v[26] += ((-9*x[2]*invr5*fdotr)+15*x[2]*x[2]*x[2]*invr7*fdotr+invr3*f[2]-3*x[2]*x[2]*invr5*f[2]                       ) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FxPU() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM+1;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (2*invr3*fdotr             ) * scal;
      v[1] += (x[0]*invr3*fdotr+f[0]*invr) * scal;
      v[2] += (x[1]*invr3*fdotr+f[1]*invr) * scal;
      v[3] += (x[2]*invr3*fdotr+f[2]*invr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  /////////////////////////////////////////////////////////////////////////////

  inline static const KernelFunction<ValueType, DIM>& FSxP() {
    constexpr Integer k_dim0 = DIM+1;
    constexpr Integer k_dim1 = 1;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (2*invr3*fdotr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FSxdP() {
    constexpr Integer k_dim0 = DIM+1;
    constexpr Integer k_dim1 = DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (2*invr3*f[0]-6*x[0]*invr5*fdotr) * scal;
      v[1] += (2*invr3*f[1]-6*x[1]*invr5*fdotr) * scal;
      v[2] += (2*invr3*f[2]-6*x[2]*invr5*fdotr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FSxU() {
    constexpr Integer k_dim0 = DIM+1;
    constexpr Integer k_dim1 = DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (x[0]*invr3*f[3]+x[0]*invr3*fdotr+f[0]*invr) * scal;
      v[1] += (x[1]*invr3*f[3]+x[1]*invr3*fdotr+f[1]*invr) * scal;
      v[2] += (x[2]*invr3*f[3]+x[2]*invr3*fdotr+f[2]*invr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FSxdU() {
    constexpr Integer k_dim0 = DIM+1;
    constexpr Integer k_dim1 = DIM * DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (invr3*f[3]-3*x[0]*x[0]*invr5*f[3]+invr3*fdotr-3*x[0]*x[0]*invr5*fdotr            ) * scal;
      v[1] += ((-3*x[0]*x[1]*invr5*f[3])-3*x[0]*x[1]*invr5*fdotr+x[0]*invr3*f[1]-x[1]*invr3*f[0]) * scal;
      v[2] += ((-3*x[0]*x[2]*invr5*f[3])-3*x[0]*x[2]*invr5*fdotr+x[0]*invr3*f[2]-x[2]*invr3*f[0]) * scal;
      v[3] += ((-3*x[0]*x[1]*invr5*f[3])-3*x[0]*x[1]*invr5*fdotr-x[0]*invr3*f[1]+x[1]*invr3*f[0]) * scal;
      v[4] += (invr3*f[3]-3*x[1]*x[1]*invr5*f[3]+invr3*fdotr-3*x[1]*x[1]*invr5*fdotr            ) * scal;
      v[5] += ((-3*x[1]*x[2]*invr5*f[3])-3*x[1]*x[2]*invr5*fdotr+x[1]*invr3*f[2]-x[2]*invr3*f[1]) * scal;
      v[6] += ((-3*x[0]*x[2]*invr5*f[3])-3*x[0]*x[2]*invr5*fdotr-x[0]*invr3*f[2]+x[2]*invr3*f[0]) * scal;
      v[7] += ((-3*x[1]*x[2]*invr5*f[3])-3*x[1]*x[2]*invr5*fdotr-x[1]*invr3*f[2]+x[2]*invr3*f[1]) * scal;
      v[8] += (invr3*f[3]-3*x[2]*x[2]*invr5*f[3]+invr3*fdotr-3*x[2]*x[2]*invr5*fdotr            ) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FSxd2U() {
    constexpr Integer k_dim0 = DIM+1;
    constexpr Integer k_dim1 = DIM * DIM * DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[ 0] += ((-9*x[0]*invr5*f[3])+15*x[0]*x[0]*x[0]*invr7*f[3]-9*x[0]*invr5*fdotr+15*x[0]*x[0]*x[0]*invr7*fdotr+invr3*f[0]-3*x[0]*x[0]*invr5*f[0]                       ) * scal;
      v[ 1] += ((-3*x[1]*invr5*f[3])+15*x[0]*x[0]*x[1]*invr7*f[3]-3*x[1]*invr5*fdotr+15*x[0]*x[0]*x[1]*invr7*fdotr+invr3*f[1]-3*x[0]*x[0]*invr5*f[1]                       ) * scal;
      v[ 2] += ((-3*x[2]*invr5*f[3])+15*x[0]*x[0]*x[2]*invr7*f[3]-3*x[2]*invr5*fdotr+15*x[0]*x[0]*x[2]*invr7*fdotr+invr3*f[2]-3*x[0]*x[0]*invr5*f[2]                       ) * scal;
      v[ 3] += ((-3*x[1]*invr5*f[3])+15*x[0]*x[0]*x[1]*invr7*f[3]-3*x[1]*invr5*fdotr+15*x[0]*x[0]*x[1]*invr7*fdotr+invr3*f[1]-3*x[0]*x[0]*invr5*f[1]                       ) * scal;
      v[ 4] += ((-3*x[0]*invr5*f[3])+15*x[0]*x[1]*x[1]*invr7*f[3]-3*x[0]*invr5*fdotr+15*x[0]*x[1]*x[1]*invr7*fdotr-6*x[0]*x[1]*invr5*f[1]-invr3*f[0]+3*x[1]*x[1]*invr5*f[0]) * scal;
      v[ 5] += (15*x[0]*x[1]*x[2]*invr7*f[3]+15*x[0]*x[1]*x[2]*invr7*fdotr-3*x[0]*x[1]*invr5*f[2]-3*x[0]*x[2]*invr5*f[1]+3*x[1]*x[2]*invr5*f[0]                            ) * scal;
      v[ 6] += ((-3*x[2]*invr5*f[3])+15*x[0]*x[0]*x[2]*invr7*f[3]-3*x[2]*invr5*fdotr+15*x[0]*x[0]*x[2]*invr7*fdotr+invr3*f[2]-3*x[0]*x[0]*invr5*f[2]                       ) * scal;
      v[ 7] += (15*x[0]*x[1]*x[2]*invr7*f[3]+15*x[0]*x[1]*x[2]*invr7*fdotr-3*x[0]*x[1]*invr5*f[2]-3*x[0]*x[2]*invr5*f[1]+3*x[1]*x[2]*invr5*f[0]                            ) * scal;
      v[ 8] += ((-3*x[0]*invr5*f[3])+15*x[0]*x[2]*x[2]*invr7*f[3]-3*x[0]*invr5*fdotr+15*x[0]*x[2]*x[2]*invr7*fdotr-6*x[0]*x[2]*invr5*f[2]-invr3*f[0]+3*x[2]*x[2]*invr5*f[0]) * scal;
      v[ 9] += ((-3*x[1]*invr5*f[3])+15*x[0]*x[0]*x[1]*invr7*f[3]-3*x[1]*invr5*fdotr+15*x[0]*x[0]*x[1]*invr7*fdotr-invr3*f[1]+3*x[0]*x[0]*invr5*f[1]-6*x[0]*x[1]*invr5*f[0]) * scal;
      v[10] += ((-3*x[0]*invr5*f[3])+15*x[0]*x[1]*x[1]*invr7*f[3]-3*x[0]*invr5*fdotr+15*x[0]*x[1]*x[1]*invr7*fdotr+invr3*f[0]-3*x[1]*x[1]*invr5*f[0]                       ) * scal;
      v[11] += (15*x[0]*x[1]*x[2]*invr7*f[3]+15*x[0]*x[1]*x[2]*invr7*fdotr-3*x[0]*x[1]*invr5*f[2]+3*x[0]*x[2]*invr5*f[1]-3*x[1]*x[2]*invr5*f[0]                            ) * scal;
      v[12] += ((-3*x[0]*invr5*f[3])+15*x[0]*x[1]*x[1]*invr7*f[3]-3*x[0]*invr5*fdotr+15*x[0]*x[1]*x[1]*invr7*fdotr+invr3*f[0]-3*x[1]*x[1]*invr5*f[0]                       ) * scal;
      v[13] += ((-9*x[1]*invr5*f[3])+15*x[1]*x[1]*x[1]*invr7*f[3]-9*x[1]*invr5*fdotr+15*x[1]*x[1]*x[1]*invr7*fdotr+invr3*f[1]-3*x[1]*x[1]*invr5*f[1]                       ) * scal;
      v[14] += ((-3*x[2]*invr5*f[3])+15*x[1]*x[1]*x[2]*invr7*f[3]-3*x[2]*invr5*fdotr+15*x[1]*x[1]*x[2]*invr7*fdotr+invr3*f[2]-3*x[1]*x[1]*invr5*f[2]                       ) * scal;
      v[15] += (15*x[0]*x[1]*x[2]*invr7*f[3]+15*x[0]*x[1]*x[2]*invr7*fdotr-3*x[0]*x[1]*invr5*f[2]+3*x[0]*x[2]*invr5*f[1]-3*x[1]*x[2]*invr5*f[0]                            ) * scal;
      v[16] += ((-3*x[2]*invr5*f[3])+15*x[1]*x[1]*x[2]*invr7*f[3]-3*x[2]*invr5*fdotr+15*x[1]*x[1]*x[2]*invr7*fdotr+invr3*f[2]-3*x[1]*x[1]*invr5*f[2]                       ) * scal;
      v[17] += ((-3*x[1]*invr5*f[3])+15*x[1]*x[2]*x[2]*invr7*f[3]-3*x[1]*invr5*fdotr+15*x[1]*x[2]*x[2]*invr7*fdotr-6*x[1]*x[2]*invr5*f[2]-invr3*f[1]+3*x[2]*x[2]*invr5*f[1]) * scal;
      v[18] += ((-3*x[2]*invr5*f[3])+15*x[0]*x[0]*x[2]*invr7*f[3]-3*x[2]*invr5*fdotr+15*x[0]*x[0]*x[2]*invr7*fdotr-invr3*f[2]+3*x[0]*x[0]*invr5*f[2]-6*x[0]*x[2]*invr5*f[0]) * scal;
      v[19] += (15*x[0]*x[1]*x[2]*invr7*f[3]+15*x[0]*x[1]*x[2]*invr7*fdotr+3*x[0]*x[1]*invr5*f[2]-3*x[0]*x[2]*invr5*f[1]-3*x[1]*x[2]*invr5*f[0]                            ) * scal;
      v[20] += ((-3*x[0]*invr5*f[3])+15*x[0]*x[2]*x[2]*invr7*f[3]-3*x[0]*invr5*fdotr+15*x[0]*x[2]*x[2]*invr7*fdotr+invr3*f[0]-3*x[2]*x[2]*invr5*f[0]                       ) * scal;
      v[21] += (15*x[0]*x[1]*x[2]*invr7*f[3]+15*x[0]*x[1]*x[2]*invr7*fdotr+3*x[0]*x[1]*invr5*f[2]-3*x[0]*x[2]*invr5*f[1]-3*x[1]*x[2]*invr5*f[0]                            ) * scal;
      v[22] += ((-3*x[2]*invr5*f[3])+15*x[1]*x[1]*x[2]*invr7*f[3]-3*x[2]*invr5*fdotr+15*x[1]*x[1]*x[2]*invr7*fdotr-invr3*f[2]+3*x[1]*x[1]*invr5*f[2]-6*x[1]*x[2]*invr5*f[1]) * scal;
      v[23] += ((-3*x[1]*invr5*f[3])+15*x[1]*x[2]*x[2]*invr7*f[3]-3*x[1]*invr5*fdotr+15*x[1]*x[2]*x[2]*invr7*fdotr+invr3*f[1]-3*x[2]*x[2]*invr5*f[1]                       ) * scal;
      v[24] += ((-3*x[0]*invr5*f[3])+15*x[0]*x[2]*x[2]*invr7*f[3]-3*x[0]*invr5*fdotr+15*x[0]*x[2]*x[2]*invr7*fdotr+invr3*f[0]-3*x[2]*x[2]*invr5*f[0]                       ) * scal;
      v[25] += ((-3*x[1]*invr5*f[3])+15*x[1]*x[2]*x[2]*invr7*f[3]-3*x[1]*invr5*fdotr+15*x[1]*x[2]*x[2]*invr7*fdotr+invr3*f[1]-3*x[2]*x[2]*invr5*f[1]                       ) * scal;
      v[26] += ((-9*x[2]*invr5*f[3])+15*x[2]*x[2]*x[2]*invr7*f[3]-9*x[2]*invr5*fdotr+15*x[2]*x[2]*x[2]*invr7*fdotr+invr3*f[2]-3*x[2]*x[2]*invr5*f[2]                       ) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& FSxPU() {
    constexpr Integer k_dim0 = DIM+1;
    constexpr Integer k_dim1 = DIM+1;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += x[k] * f[k];

      v[0] += (2*invr3*fdotr                             ) * scal;
      v[1] += (x[0]*invr3*f[3]+x[0]*invr3*fdotr+f[0]*invr) * scal;
      v[2] += (x[1]*invr3*f[3]+x[1]*invr3*fdotr+f[1]*invr) * scal;
      v[3] += (x[2]*invr3*f[3]+x[2]*invr3*fdotr+f[2]*invr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  /////////////////////////////////////////////////////////////////////////////

  inline static const KernelFunction<ValueType, DIM>& DxP() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = 1;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType ndotr = 0;
      ValueType ndotf = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += f[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotr += n[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotf += n[k] * f[k];

      v[0] += 4*(invr3*(-ndotf)+3*invr5*fdotr*ndotr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& DxdP() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType ndotr = 0;
      ValueType ndotf = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += f[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotr += n[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotf += n[k] * f[k];

      v[0] += 4*((-3*x[0]*invr5*(-ndotf))-15*x[0]*invr7*fdotr*ndotr+3*invr5*f[0]*ndotr+3*invr5*fdotr*n[0]) * scal;
      v[1] += 4*((-3*x[1]*invr5*(-ndotf))-15*x[1]*invr7*fdotr*ndotr+3*invr5*f[1]*ndotr+3*invr5*fdotr*n[1]) * scal;
      v[2] += 4*((-3*x[2]*invr5*(-ndotf))-15*x[2]*invr7*fdotr*ndotr+3*invr5*f[2]*ndotr+3*invr5*fdotr*n[2]) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& DxU() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType ndotr = 0;
      ValueType ndotf = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += f[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotr += n[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotf += n[k] * f[k];

      v[0] += (6*x[0]*invr5*fdotr*ndotr) * scal;
      v[1] += (6*x[1]*invr5*fdotr*ndotr) * scal;
      v[2] += (6*x[2]*invr5*fdotr*ndotr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& DxdU() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM * DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType ndotr = 0;
      ValueType ndotf = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += f[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotr += n[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotf += n[k] * f[k];

      v[0] += (6*invr5*fdotr*ndotr-30*x[0]*x[0]*invr7*fdotr*ndotr+6*x[0]*invr5*f[0]*ndotr+6*x[0]*invr5*fdotr*n[0]) * scal;
      v[1] += ((-30*x[0]*x[1]*invr7*fdotr*ndotr)+6*x[0]*invr5*f[1]*ndotr+6*x[0]*invr5*fdotr*n[1]                 ) * scal;
      v[2] += ((-30*x[0]*x[2]*invr7*fdotr*ndotr)+6*x[0]*invr5*f[2]*ndotr+6*x[0]*invr5*fdotr*n[2]                 ) * scal;
      v[3] += ((-30*x[0]*x[1]*invr7*fdotr*ndotr)+6*x[1]*invr5*f[0]*ndotr+6*x[1]*invr5*fdotr*n[0]                 ) * scal;
      v[4] += (6*invr5*fdotr*ndotr-30*x[1]*x[1]*invr7*fdotr*ndotr+6*x[1]*invr5*f[1]*ndotr+6*x[1]*invr5*fdotr*n[1]) * scal;
      v[5] += ((-30*x[1]*x[2]*invr7*fdotr*ndotr)+6*x[1]*invr5*f[2]*ndotr+6*x[1]*invr5*fdotr*n[2]                 ) * scal;
      v[6] += ((-30*x[0]*x[2]*invr7*fdotr*ndotr)+6*x[2]*invr5*f[0]*ndotr+6*x[2]*invr5*fdotr*n[0]                 ) * scal;
      v[7] += ((-30*x[1]*x[2]*invr7*fdotr*ndotr)+6*x[2]*invr5*f[1]*ndotr+6*x[2]*invr5*fdotr*n[1]                 ) * scal;
      v[8] += (6*invr5*fdotr*ndotr-30*x[2]*x[2]*invr7*fdotr*ndotr+6*x[2]*invr5*f[2]*ndotr+6*x[2]*invr5*fdotr*n[2]) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& Dxd2U() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM * DIM * DIM;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType ndotr = 0;
      ValueType ndotf = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      ValueType invr9 = invr2*invr7;
      for(Integer k=0;k<DIM;k++) fdotr += f[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotr += n[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotf += n[k] * f[k];

      v[ 0] += ((-90*x[0]*invr7*fdotr*ndotr)+210*x[0]*x[0]*x[0]*invr9*fdotr*ndotr+12*invr5*f[0]*ndotr-60*x[0]*x[0]*invr7*f[0]*ndotr+12*invr5*fdotr*n[0]-60*x[0]*x[0]*invr7*fdotr*n[0]+12*x[0]*invr5*f[0]*n[0]                                                                                ) * scal;
      v[ 1] += ((-30*x[1]*invr7*fdotr*ndotr)+210*x[0]*x[0]*x[1]*invr9*fdotr*ndotr+6*invr5*f[1]*ndotr-30*x[0]*x[0]*invr7*f[1]*ndotr-30*x[0]*x[1]*invr7*f[0]*ndotr+6*invr5*fdotr*n[1]-30*x[0]*x[0]*invr7*fdotr*n[1]+6*x[0]*invr5*f[0]*n[1]-30*x[0]*x[1]*invr7*fdotr*n[0]+6*x[0]*invr5*f[1]*n[0]) * scal;
      v[ 2] += ((-30*x[2]*invr7*fdotr*ndotr)+210*x[0]*x[0]*x[2]*invr9*fdotr*ndotr+6*invr5*f[2]*ndotr-30*x[0]*x[0]*invr7*f[2]*ndotr-30*x[0]*x[2]*invr7*f[0]*ndotr+6*invr5*fdotr*n[2]-30*x[0]*x[0]*invr7*fdotr*n[2]+6*x[0]*invr5*f[0]*n[2]-30*x[0]*x[2]*invr7*fdotr*n[0]+6*x[0]*invr5*f[2]*n[0]) * scal;
      v[ 3] += ((-30*x[1]*invr7*fdotr*ndotr)+210*x[0]*x[0]*x[1]*invr9*fdotr*ndotr+6*invr5*f[1]*ndotr-30*x[0]*x[0]*invr7*f[1]*ndotr-30*x[0]*x[1]*invr7*f[0]*ndotr+6*invr5*fdotr*n[1]-30*x[0]*x[0]*invr7*fdotr*n[1]+6*x[0]*invr5*f[0]*n[1]-30*x[0]*x[1]*invr7*fdotr*n[0]+6*x[0]*invr5*f[1]*n[0]) * scal;
      v[ 4] += ((-30*x[0]*invr7*fdotr*ndotr)+210*x[0]*x[1]*x[1]*invr9*fdotr*ndotr-60*x[0]*x[1]*invr7*f[1]*ndotr-60*x[0]*x[1]*invr7*fdotr*n[1]+12*x[0]*invr5*f[1]*n[1]                                                                                                                        ) * scal;
      v[ 5] += (210*x[0]*x[1]*x[2]*invr9*fdotr*ndotr-30*x[0]*x[1]*invr7*f[2]*ndotr-30*x[0]*x[2]*invr7*f[1]*ndotr-30*x[0]*x[1]*invr7*fdotr*n[2]+6*x[0]*invr5*f[1]*n[2]-30*x[0]*x[2]*invr7*fdotr*n[1]+6*x[0]*invr5*f[2]*n[1]                                                                   ) * scal;
      v[ 6] += ((-30*x[2]*invr7*fdotr*ndotr)+210*x[0]*x[0]*x[2]*invr9*fdotr*ndotr+6*invr5*f[2]*ndotr-30*x[0]*x[0]*invr7*f[2]*ndotr-30*x[0]*x[2]*invr7*f[0]*ndotr+6*invr5*fdotr*n[2]-30*x[0]*x[0]*invr7*fdotr*n[2]+6*x[0]*invr5*f[0]*n[2]-30*x[0]*x[2]*invr7*fdotr*n[0]+6*x[0]*invr5*f[2]*n[0]) * scal;
      v[ 7] += (210*x[0]*x[1]*x[2]*invr9*fdotr*ndotr-30*x[0]*x[1]*invr7*f[2]*ndotr-30*x[0]*x[2]*invr7*f[1]*ndotr-30*x[0]*x[1]*invr7*fdotr*n[2]+6*x[0]*invr5*f[1]*n[2]-30*x[0]*x[2]*invr7*fdotr*n[1]+6*x[0]*invr5*f[2]*n[1]                                                                   ) * scal;
      v[ 8] += ((-30*x[0]*invr7*fdotr*ndotr)+210*x[0]*x[2]*x[2]*invr9*fdotr*ndotr-60*x[0]*x[2]*invr7*f[2]*ndotr-60*x[0]*x[2]*invr7*fdotr*n[2]+12*x[0]*invr5*f[2]*n[2]                                                                                                                        ) * scal;
      v[ 9] += ((-30*x[1]*invr7*fdotr*ndotr)+210*x[0]*x[0]*x[1]*invr9*fdotr*ndotr-60*x[0]*x[1]*invr7*f[0]*ndotr-60*x[0]*x[1]*invr7*fdotr*n[0]+12*x[1]*invr5*f[0]*n[0]                                                                                                                        ) * scal;
      v[10] += ((-30*x[0]*invr7*fdotr*ndotr)+210*x[0]*x[1]*x[1]*invr9*fdotr*ndotr-30*x[0]*x[1]*invr7*f[1]*ndotr+6*invr5*f[0]*ndotr-30*x[1]*x[1]*invr7*f[0]*ndotr-30*x[0]*x[1]*invr7*fdotr*n[1]+6*x[1]*invr5*f[0]*n[1]+6*invr5*fdotr*n[0]-30*x[1]*x[1]*invr7*fdotr*n[0]+6*x[1]*invr5*f[1]*n[0]) * scal;
      v[11] += (210*x[0]*x[1]*x[2]*invr9*fdotr*ndotr-30*x[0]*x[1]*invr7*f[2]*ndotr-30*x[1]*x[2]*invr7*f[0]*ndotr-30*x[0]*x[1]*invr7*fdotr*n[2]+6*x[1]*invr5*f[0]*n[2]-30*x[1]*x[2]*invr7*fdotr*n[0]+6*x[1]*invr5*f[2]*n[0]                                                                   ) * scal;
      v[12] += ((-30*x[0]*invr7*fdotr*ndotr)+210*x[0]*x[1]*x[1]*invr9*fdotr*ndotr-30*x[0]*x[1]*invr7*f[1]*ndotr+6*invr5*f[0]*ndotr-30*x[1]*x[1]*invr7*f[0]*ndotr-30*x[0]*x[1]*invr7*fdotr*n[1]+6*x[1]*invr5*f[0]*n[1]+6*invr5*fdotr*n[0]-30*x[1]*x[1]*invr7*fdotr*n[0]+6*x[1]*invr5*f[1]*n[0]) * scal;
      v[13] += ((-90*x[1]*invr7*fdotr*ndotr)+210*x[1]*x[1]*x[1]*invr9*fdotr*ndotr+12*invr5*f[1]*ndotr-60*x[1]*x[1]*invr7*f[1]*ndotr+12*invr5*fdotr*n[1]-60*x[1]*x[1]*invr7*fdotr*n[1]+12*x[1]*invr5*f[1]*n[1]                                                                                ) * scal;
      v[14] += ((-30*x[2]*invr7*fdotr*ndotr)+210*x[1]*x[1]*x[2]*invr9*fdotr*ndotr+6*invr5*f[2]*ndotr-30*x[1]*x[1]*invr7*f[2]*ndotr-30*x[1]*x[2]*invr7*f[1]*ndotr+6*invr5*fdotr*n[2]-30*x[1]*x[1]*invr7*fdotr*n[2]+6*x[1]*invr5*f[1]*n[2]-30*x[1]*x[2]*invr7*fdotr*n[1]+6*x[1]*invr5*f[2]*n[1]) * scal;
      v[15] += (210*x[0]*x[1]*x[2]*invr9*fdotr*ndotr-30*x[0]*x[1]*invr7*f[2]*ndotr-30*x[1]*x[2]*invr7*f[0]*ndotr-30*x[0]*x[1]*invr7*fdotr*n[2]+6*x[1]*invr5*f[0]*n[2]-30*x[1]*x[2]*invr7*fdotr*n[0]+6*x[1]*invr5*f[2]*n[0]                                                                   ) * scal;
      v[16] += ((-30*x[2]*invr7*fdotr*ndotr)+210*x[1]*x[1]*x[2]*invr9*fdotr*ndotr+6*invr5*f[2]*ndotr-30*x[1]*x[1]*invr7*f[2]*ndotr-30*x[1]*x[2]*invr7*f[1]*ndotr+6*invr5*fdotr*n[2]-30*x[1]*x[1]*invr7*fdotr*n[2]+6*x[1]*invr5*f[1]*n[2]-30*x[1]*x[2]*invr7*fdotr*n[1]+6*x[1]*invr5*f[2]*n[1]) * scal;
      v[17] += ((-30*x[1]*invr7*fdotr*ndotr)+210*x[1]*x[2]*x[2]*invr9*fdotr*ndotr-60*x[1]*x[2]*invr7*f[2]*ndotr-60*x[1]*x[2]*invr7*fdotr*n[2]+12*x[1]*invr5*f[2]*n[2]                                                                                                                        ) * scal;
      v[18] += ((-30*x[2]*invr7*fdotr*ndotr)+210*x[0]*x[0]*x[2]*invr9*fdotr*ndotr-60*x[0]*x[2]*invr7*f[0]*ndotr-60*x[0]*x[2]*invr7*fdotr*n[0]+12*x[2]*invr5*f[0]*n[0]                                                                                                                        ) * scal;
      v[19] += (210*x[0]*x[1]*x[2]*invr9*fdotr*ndotr-30*x[0]*x[2]*invr7*f[1]*ndotr-30*x[1]*x[2]*invr7*f[0]*ndotr-30*x[0]*x[2]*invr7*fdotr*n[1]+6*x[2]*invr5*f[0]*n[1]-30*x[1]*x[2]*invr7*fdotr*n[0]+6*x[2]*invr5*f[1]*n[0]                                                                   ) * scal;
      v[20] += ((-30*x[0]*invr7*fdotr*ndotr)+210*x[0]*x[2]*x[2]*invr9*fdotr*ndotr-30*x[0]*x[2]*invr7*f[2]*ndotr+6*invr5*f[0]*ndotr-30*x[2]*x[2]*invr7*f[0]*ndotr-30*x[0]*x[2]*invr7*fdotr*n[2]+6*x[2]*invr5*f[0]*n[2]+6*invr5*fdotr*n[0]-30*x[2]*x[2]*invr7*fdotr*n[0]+6*x[2]*invr5*f[2]*n[0]) * scal;
      v[21] += (210*x[0]*x[1]*x[2]*invr9*fdotr*ndotr-30*x[0]*x[2]*invr7*f[1]*ndotr-30*x[1]*x[2]*invr7*f[0]*ndotr-30*x[0]*x[2]*invr7*fdotr*n[1]+6*x[2]*invr5*f[0]*n[1]-30*x[1]*x[2]*invr7*fdotr*n[0]+6*x[2]*invr5*f[1]*n[0]                                                                   ) * scal;
      v[22] += ((-30*x[2]*invr7*fdotr*ndotr)+210*x[1]*x[1]*x[2]*invr9*fdotr*ndotr-60*x[1]*x[2]*invr7*f[1]*ndotr-60*x[1]*x[2]*invr7*fdotr*n[1]+12*x[2]*invr5*f[1]*n[1]                                                                                                                        ) * scal;
      v[23] += ((-30*x[1]*invr7*fdotr*ndotr)+210*x[1]*x[2]*x[2]*invr9*fdotr*ndotr-30*x[1]*x[2]*invr7*f[2]*ndotr+6*invr5*f[1]*ndotr-30*x[2]*x[2]*invr7*f[1]*ndotr-30*x[1]*x[2]*invr7*fdotr*n[2]+6*x[2]*invr5*f[1]*n[2]+6*invr5*fdotr*n[1]-30*x[2]*x[2]*invr7*fdotr*n[1]+6*x[2]*invr5*f[2]*n[1]) * scal;
      v[24] += ((-30*x[0]*invr7*fdotr*ndotr)+210*x[0]*x[2]*x[2]*invr9*fdotr*ndotr-30*x[0]*x[2]*invr7*f[2]*ndotr+6*invr5*f[0]*ndotr-30*x[2]*x[2]*invr7*f[0]*ndotr-30*x[0]*x[2]*invr7*fdotr*n[2]+6*x[2]*invr5*f[0]*n[2]+6*invr5*fdotr*n[0]-30*x[2]*x[2]*invr7*fdotr*n[0]+6*x[2]*invr5*f[2]*n[0]) * scal;
      v[25] += ((-30*x[1]*invr7*fdotr*ndotr)+210*x[1]*x[2]*x[2]*invr9*fdotr*ndotr-30*x[1]*x[2]*invr7*f[2]*ndotr+6*invr5*f[1]*ndotr-30*x[2]*x[2]*invr7*f[1]*ndotr-30*x[1]*x[2]*invr7*fdotr*n[2]+6*x[2]*invr5*f[1]*n[2]+6*invr5*fdotr*n[1]-30*x[2]*x[2]*invr7*fdotr*n[1]+6*x[2]*invr5*f[2]*n[1]) * scal;
      v[26] += ((-90*x[2]*invr7*fdotr*ndotr)+210*x[2]*x[2]*x[2]*invr9*fdotr*ndotr+12*invr5*f[2]*ndotr-60*x[2]*x[2]*invr7*f[2]*ndotr+12*invr5*fdotr*n[2]-60*x[2]*x[2]*invr7*fdotr*n[2]+12*x[2]*invr5*f[2]*n[2]                                                                                ) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

  inline static const KernelFunction<ValueType, DIM>& DxPU() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM+1;
    static const ValueType scal = 1.0/(8.0*const_pi<ValueType>());
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType fdotr = 0;
      ValueType ndotr = 0;
      ValueType ndotf = 0;
      ValueType invr2 = invr*invr;
      ValueType invr3 = invr2*invr;
      ValueType invr5 = invr2*invr3;
      ValueType invr7 = invr2*invr5;
      for(Integer k=0;k<DIM;k++) fdotr += f[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotr += n[k] * x[k];
      for(Integer k=0;k<DIM;k++) ndotf += n[k] * f[k];

      v[0] += 4*(invr3*(-ndotf)+3*invr5*fdotr*ndotr) * scal;
      v[1] += (6*x[0]*invr5*fdotr*ndotr) * scal;
      v[2] += (6*x[1]*invr5*fdotr*ndotr) * scal;
      v[3] += (6*x[2]*invr5*fdotr*ndotr) * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

 private:

  static void GenKer(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx){
    const auto& ker = *(std::function<typename KernelFunction<ValueType, DIM>::KerFn>*)(ctx);
    ker(r_src, n_src, v_src, r_trg, v_trg, NULL);
  }

  template <Integer k_dim0, Integer k_dim1, class LambdaType> inline static const KernelFunction<ValueType, DIM>& KernelFromLambda(std::string name, LambdaType&& micro_ker) {
    static std::function<typename KernelFunction<ValueType, DIM>::KerFn> ker_fn = [&](const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx){
      Long src_cnt = r_src.Dim() / DIM;
      Long trg_cnt = r_trg.Dim() / DIM;
      for (Long t=0;t<trg_cnt;t++){
        ValueType v[k_dim1];
        for (Integer i=0;i<k_dim1;i++) {
          v[i]=0;
        }
        for (Long s=0;s<src_cnt;s++){
          ValueType dx[DIM], f[k_dim0], r2 = 0, invr = 0;
          for(Integer k=0;k<DIM;k++) {
            dx[k] = r_trg[t*DIM+k] - r_src[s*DIM+k];
            r2 += dx[k] * dx[k];
          }
          for(Integer k=0;k<k_dim0;k++) {
            f[k] = v_src[s*k_dim0+k];
          }
          if(r2>0) invr=1.0/sqrt(r2);
          micro_ker(v, dx, invr, &n_src[s*DIM], f);
        }
        for (Integer i=0;i<k_dim1;i++) {
          v_trg[t*k_dim1+i] += v[i];
        }
      }
    };
    static KernelFunction<ValueType, DIM> ker(GenKer, k_dim0, k_dim1, name, &ker_fn);
    return ker;
  }

  /////////////////////////////////////////////////////////////////////////////

 public:
  inline static const KernelFunction<ValueType, DIM>& single_layer_velocity() {
    static KernelFunction<ValueType, DIM> ker(sl_vel<2>, DIM, DIM, "stokes-vel-sl");
    return ker;
  }

  inline static const KernelFunction<ValueType, DIM>& single_layer_velocity_m2x() {
    static KernelFunction<ValueType, DIM> ker(sl_vel_m2x<2>, DIM + 1, DIM, "stokes-vel-sl-m2x");
    return ker;
  }

  inline static const KernelFunction<ValueType, DIM>& double_layer_velocity() {
    static KernelFunction<ValueType, DIM> ker(dl_vel<2>, DIM, DIM, "stokes-vel-dl");
    return ker;
  }

  inline static const KernelFunction<ValueType, DIM>& single_layer_pressure() {
    static KernelFunction<ValueType, DIM> ker(sl_press<2>, DIM, 1, "stokes-press-sl");
    return ker;
  }

  inline static const KernelFunction<ValueType, DIM>& single_layer_pressure_m2x() {
    static KernelFunction<ValueType, DIM> ker(sl_press_m2x<2>, DIM + 1, 1, "stokes-press-sl-m2x");
    return ker;
  }

 protected:
  template <class Vec = ValueType, Vec (*RSQRT_INTRIN)(Vec) = rsqrt_intrin0<Vec>> static void sl_vel_uKernel(const Matrix<ValueType>& src_coord, const Matrix<ValueType>& src_norml, const Matrix<ValueType>& src_value, const Matrix<ValueType>& trg_coord, Matrix<ValueType>& trg_value) {
    #define SRC_BLK 500
    static ValueType eps = machine_eps() * 128;
    size_t VecLen = sizeof(Vec) / sizeof(ValueType);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin0<Vec, ValueType>) NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin1<Vec, ValueType>) NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin2<Vec, ValueType>) NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin3<Vec, ValueType>) NWTN_ITER = 3;

    ValueType nwtn_scal = 1;  // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
      nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const ValueType OOEP = 1.0 / (8 * nwtn_scal * const_pi<ValueType>());
    Vec inv_nwtn_scal2 = set_intrin<Vec, ValueType>(1.0 / (nwtn_scal * nwtn_scal));

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
      size_t src_cnt = src_cnt_ - sblk;
      if (src_cnt > SRC_BLK) src_cnt = SRC_BLK;
      for (size_t t = 0; t < trg_cnt_; t += VecLen) {
        Vec tx = load_intrin<Vec>(&trg_coord[0][t]);
        Vec ty = load_intrin<Vec>(&trg_coord[1][t]);
        Vec tz = load_intrin<Vec>(&trg_coord[2][t]);

        Vec tvx = zero_intrin<Vec>();
        Vec tvy = zero_intrin<Vec>();
        Vec tvz = zero_intrin<Vec>();
        for (size_t s = sblk; s < sblk + src_cnt; s++) {
          Vec dx = sub_intrin(tx, bcast_intrin<Vec>(&src_coord[0][s]));
          Vec dy = sub_intrin(ty, bcast_intrin<Vec>(&src_coord[1][s]));
          Vec dz = sub_intrin(tz, bcast_intrin<Vec>(&src_coord[2][s]));

          Vec svx = bcast_intrin<Vec>(&src_value[0][s]);
          Vec svy = bcast_intrin<Vec>(&src_value[1][s]);
          Vec svz = bcast_intrin<Vec>(&src_value[2][s]);

          Vec r2 = mul_intrin(dx, dx);
          r2 = add_intrin(r2, mul_intrin(dy, dy));
          r2 = add_intrin(r2, mul_intrin(dz, dz));
          r2 = and_intrin(cmplt_intrin(set_intrin<Vec, ValueType>(eps), r2), r2);

          Vec rinv = RSQRT_INTRIN(r2);
          Vec rinv2 = mul_intrin(mul_intrin(rinv, rinv), inv_nwtn_scal2);

          Vec inner_prod = mul_intrin(svx, dx);
          inner_prod = add_intrin(inner_prod, mul_intrin(svy, dy));
          inner_prod = add_intrin(inner_prod, mul_intrin(svz, dz));
          inner_prod = mul_intrin(inner_prod, rinv2);

          tvx = add_intrin(tvx, mul_intrin(rinv, add_intrin(svx, mul_intrin(dx, inner_prod))));
          tvy = add_intrin(tvy, mul_intrin(rinv, add_intrin(svy, mul_intrin(dy, inner_prod))));
          tvz = add_intrin(tvz, mul_intrin(rinv, add_intrin(svz, mul_intrin(dz, inner_prod))));
        }
        Vec ooep = set_intrin<Vec, ValueType>(OOEP);

        tvx = add_intrin(mul_intrin(tvx, ooep), load_intrin<Vec>(&trg_value[0][t]));
        tvy = add_intrin(mul_intrin(tvy, ooep), load_intrin<Vec>(&trg_value[1][t]));
        tvz = add_intrin(mul_intrin(tvz, ooep), load_intrin<Vec>(&trg_value[2][t]));

        store_intrin(&trg_value[0][t], tvx);
        store_intrin(&trg_value[1][t], tvy);
        store_intrin(&trg_value[2][t], tvz);
      }
    }

    {  // Add FLOPS
      #ifndef __MIC__
      //Profile::Add_FLOP((long long)trg_cnt_ * (long long)src_cnt_ * (29 + 4 * (NWTN_ITER)));
      #endif
    }
    #undef SRC_BLK
  }

  template <class Vec = ValueType, Vec (*RSQRT_INTRIN)(Vec) = rsqrt_intrin0<Vec>> static void sl_vel_m2x_uKernel(const Matrix<ValueType>& src_coord, const Matrix<ValueType>& src_norml, const Matrix<ValueType>& src_value, const Matrix<ValueType>& trg_coord, Matrix<ValueType>& trg_value) {
    #define SRC_BLK 500
    size_t VecLen = sizeof(Vec) / sizeof(ValueType);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin0<Vec, ValueType>) NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin1<Vec, ValueType>) NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin2<Vec, ValueType>) NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin3<Vec, ValueType>) NWTN_ITER = 3;

    ValueType nwtn_scal = 1;  // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
      nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const ValueType OOEP = 1.0 / (8 * nwtn_scal * const_pi<ValueType>());
    Vec inv_nwtn_scal2 = set_intrin<Vec, ValueType>(1.0 / (nwtn_scal * nwtn_scal));

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
      size_t src_cnt = src_cnt_ - sblk;
      if (src_cnt > SRC_BLK) src_cnt = SRC_BLK;
      for (size_t t = 0; t < trg_cnt_; t += VecLen) {
        Vec tx = load_intrin<Vec>(&trg_coord[0][t]);
        Vec ty = load_intrin<Vec>(&trg_coord[1][t]);
        Vec tz = load_intrin<Vec>(&trg_coord[2][t]);

        Vec tvx = zero_intrin<Vec>();
        Vec tvy = zero_intrin<Vec>();
        Vec tvz = zero_intrin<Vec>();
        for (size_t s = sblk; s < sblk + src_cnt; s++) {
          Vec dx = sub_intrin(tx, bcast_intrin<Vec>(&src_coord[0][s]));
          Vec dy = sub_intrin(ty, bcast_intrin<Vec>(&src_coord[1][s]));
          Vec dz = sub_intrin(tz, bcast_intrin<Vec>(&src_coord[2][s]));

          Vec svx = bcast_intrin<Vec>(&src_value[0][s]);
          Vec svy = bcast_intrin<Vec>(&src_value[1][s]);
          Vec svz = bcast_intrin<Vec>(&src_value[2][s]);
          Vec inner_prod = bcast_intrin<Vec>(&src_value[3][s]);

          Vec r2 = mul_intrin(dx, dx);
          r2 = add_intrin(r2, mul_intrin(dy, dy));
          r2 = add_intrin(r2, mul_intrin(dz, dz));

          Vec rinv = RSQRT_INTRIN(r2);
          Vec rinv2 = mul_intrin(mul_intrin(rinv, rinv), inv_nwtn_scal2);

          inner_prod = add_intrin(inner_prod, mul_intrin(svx, dx));
          inner_prod = add_intrin(inner_prod, mul_intrin(svy, dy));
          inner_prod = add_intrin(inner_prod, mul_intrin(svz, dz));
          inner_prod = mul_intrin(inner_prod, rinv2);

          tvx = add_intrin(tvx, mul_intrin(rinv, add_intrin(svx, mul_intrin(dx, inner_prod))));
          tvy = add_intrin(tvy, mul_intrin(rinv, add_intrin(svy, mul_intrin(dy, inner_prod))));
          tvz = add_intrin(tvz, mul_intrin(rinv, add_intrin(svz, mul_intrin(dz, inner_prod))));
        }
        Vec ooep = set_intrin<Vec, ValueType>(OOEP);

        tvx = add_intrin(mul_intrin(tvx, ooep), load_intrin<Vec>(&trg_value[0][t]));
        tvy = add_intrin(mul_intrin(tvy, ooep), load_intrin<Vec>(&trg_value[1][t]));
        tvz = add_intrin(mul_intrin(tvz, ooep), load_intrin<Vec>(&trg_value[2][t]));

        store_intrin(&trg_value[0][t], tvx);
        store_intrin(&trg_value[1][t], tvy);
        store_intrin(&trg_value[2][t], tvz);
      }
    }

    {  // Add FLOPS
      #ifndef __MIC__
      //Profile::Add_FLOP((long long)trg_cnt_ * (long long)src_cnt_ * (29 + 4 * (NWTN_ITER)));
      #endif
    }
    #undef SRC_BLK
  }

  template <class Vec = ValueType, Vec (*RSQRT_INTRIN)(Vec) = rsqrt_intrin0<Vec>> static void dl_vel_uKernel(const Matrix<ValueType>& src_coord, const Matrix<ValueType>& src_norml, const Matrix<ValueType>& src_value, const Matrix<ValueType>& trg_coord, Matrix<ValueType>& trg_value) {
    #define SRC_BLK 500
    static ValueType eps = machine_eps() * 128;
    size_t VecLen = sizeof(Vec) / sizeof(ValueType);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin0<Vec, ValueType>) NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin1<Vec, ValueType>) NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin2<Vec, ValueType>) NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin3<Vec, ValueType>) NWTN_ITER = 3;

    ValueType nwtn_scal = 1;  // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
      nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const ValueType SCAL_CONST = 3.0 / (4.0 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<ValueType>());

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
      size_t src_cnt = src_cnt_ - sblk;
      if (src_cnt > SRC_BLK) src_cnt = SRC_BLK;
      for (size_t t = 0; t < trg_cnt_; t += VecLen) {
        Vec tx = load_intrin<Vec>(&trg_coord[0][t]);
        Vec ty = load_intrin<Vec>(&trg_coord[1][t]);
        Vec tz = load_intrin<Vec>(&trg_coord[2][t]);

        Vec tvx = zero_intrin<Vec>();
        Vec tvy = zero_intrin<Vec>();
        Vec tvz = zero_intrin<Vec>();
        for (size_t s = sblk; s < sblk + src_cnt; s++) {
          Vec dx = sub_intrin(tx, bcast_intrin<Vec>(&src_coord[0][s]));
          Vec dy = sub_intrin(ty, bcast_intrin<Vec>(&src_coord[1][s]));
          Vec dz = sub_intrin(tz, bcast_intrin<Vec>(&src_coord[2][s]));

          Vec snx = bcast_intrin<Vec>(&src_value[0][s]);
          Vec sny = bcast_intrin<Vec>(&src_value[1][s]);
          Vec snz = bcast_intrin<Vec>(&src_value[2][s]);

          Vec svx = bcast_intrin<Vec>(&src_norml[0][s]);
          Vec svy = bcast_intrin<Vec>(&src_norml[1][s]);
          Vec svz = bcast_intrin<Vec>(&src_norml[2][s]);

          Vec r2 = mul_intrin(dx, dx);
          r2 = add_intrin(r2, mul_intrin(dy, dy));
          r2 = add_intrin(r2, mul_intrin(dz, dz));
          r2 = and_intrin(cmplt_intrin(set_intrin<Vec, ValueType>(eps), r2), r2);

          Vec rinv = RSQRT_INTRIN(r2);
          Vec rinv2 = mul_intrin(rinv, rinv);
          Vec rinv5 = mul_intrin(mul_intrin(rinv2, rinv2), rinv);

          Vec r_dot_n = mul_intrin(snx, dx);
          r_dot_n = add_intrin(r_dot_n, mul_intrin(sny, dy));
          r_dot_n = add_intrin(r_dot_n, mul_intrin(snz, dz));

          Vec r_dot_f = mul_intrin(svx, dx);
          r_dot_f = add_intrin(r_dot_f, mul_intrin(svy, dy));
          r_dot_f = add_intrin(r_dot_f, mul_intrin(svz, dz));

          Vec p = mul_intrin(mul_intrin(r_dot_n, r_dot_f), rinv5);
          tvx = add_intrin(tvx, mul_intrin(dx, p));
          tvy = add_intrin(tvy, mul_intrin(dy, p));
          tvz = add_intrin(tvz, mul_intrin(dz, p));
        }
        Vec scal_const = set_intrin<Vec, ValueType>(SCAL_CONST);

        tvx = add_intrin(mul_intrin(tvx, scal_const), load_intrin<Vec>(&trg_value[0][t]));
        tvy = add_intrin(mul_intrin(tvy, scal_const), load_intrin<Vec>(&trg_value[1][t]));
        tvz = add_intrin(mul_intrin(tvz, scal_const), load_intrin<Vec>(&trg_value[2][t]));

        store_intrin(&trg_value[0][t], tvx);
        store_intrin(&trg_value[1][t], tvy);
        store_intrin(&trg_value[2][t], tvz);
      }
    }

    {  // Add FLOPS
      #ifndef __MIC__
      //Profile::Add_FLOP((long long)trg_cnt_ * (long long)src_cnt_ * (31 + 4 * (NWTN_ITER)));
      #endif
    }
    #undef SRC_BLK
  }

  template <class Vec = ValueType, Vec (*RSQRT_INTRIN)(Vec) = rsqrt_intrin0<Vec>> static void sl_press_uKernel(const Matrix<ValueType>& src_coord, const Matrix<ValueType>& src_norml, const Matrix<ValueType>& src_value, const Matrix<ValueType>& trg_coord, Matrix<ValueType>& trg_value) {
    #define SRC_BLK 500
    static ValueType eps = machine_eps() * 128;
    size_t VecLen = sizeof(Vec) / sizeof(ValueType);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin0<Vec, ValueType>) NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin1<Vec, ValueType>) NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin2<Vec, ValueType>) NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin3<Vec, ValueType>) NWTN_ITER = 3;

    ValueType nwtn_scal = 1;  // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
      nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const ValueType OOEP = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<ValueType>());

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
      size_t src_cnt = src_cnt_ - sblk;
      if (src_cnt > SRC_BLK) src_cnt = SRC_BLK;
      for (size_t t = 0; t < trg_cnt_; t += VecLen) {
        Vec tx = load_intrin<Vec>(&trg_coord[0][t]);
        Vec ty = load_intrin<Vec>(&trg_coord[1][t]);
        Vec tz = load_intrin<Vec>(&trg_coord[2][t]);

        Vec tv = zero_intrin<Vec>();
        for (size_t s = sblk; s < sblk + src_cnt; s++) {
          Vec dx = sub_intrin(tx, bcast_intrin<Vec>(&src_coord[0][s]));
          Vec dy = sub_intrin(ty, bcast_intrin<Vec>(&src_coord[1][s]));
          Vec dz = sub_intrin(tz, bcast_intrin<Vec>(&src_coord[2][s]));

          Vec svx = bcast_intrin<Vec>(&src_value[0][s]);
          Vec svy = bcast_intrin<Vec>(&src_value[1][s]);
          Vec svz = bcast_intrin<Vec>(&src_value[2][s]);

          Vec r2 = mul_intrin(dx, dx);
          r2 = add_intrin(r2, mul_intrin(dy, dy));
          r2 = add_intrin(r2, mul_intrin(dz, dz));
          r2 = and_intrin(cmplt_intrin(set_intrin<Vec, ValueType>(eps), r2), r2);

          Vec rinv = RSQRT_INTRIN(r2);
          Vec rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);

          Vec inner_prod = mul_intrin(svx, dx);
          inner_prod = add_intrin(inner_prod, mul_intrin(svy, dy));
          inner_prod = add_intrin(inner_prod, mul_intrin(svz, dz));
          tv = add_intrin(tv, mul_intrin(inner_prod, rinv3));
        }
        Vec ooep = set_intrin<Vec, ValueType>(OOEP);

        tv = add_intrin(mul_intrin(tv, ooep), load_intrin<Vec>(&trg_value[0][t]));
        store_intrin(&trg_value[0][t], tv);
      }
    }

    {  // Add FLOPS
      #ifndef __MIC__
      //Profile::Add_FLOP((long long)trg_cnt_ * (long long)src_cnt_ * (29 + 4 * (NWTN_ITER)));
      #endif
    }
    #undef SRC_BLK
  }

  template <class Vec = ValueType, Vec (*RSQRT_INTRIN)(Vec) = rsqrt_intrin0<Vec>> static void sl_press_m2x_uKernel(const Matrix<ValueType>& src_coord, const Matrix<ValueType>& src_norml, const Matrix<ValueType>& src_value, const Matrix<ValueType>& trg_coord, Matrix<ValueType>& trg_value) {
    #define SRC_BLK 500
    size_t VecLen = sizeof(Vec) / sizeof(ValueType);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin0<Vec, ValueType>) NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin1<Vec, ValueType>) NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin2<Vec, ValueType>) NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec (*)(Vec))rsqrt_intrin3<Vec, ValueType>) NWTN_ITER = 3;

    ValueType nwtn_scal = 1;  // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
      nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const ValueType OOEP = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<ValueType>());

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
      size_t src_cnt = src_cnt_ - sblk;
      if (src_cnt > SRC_BLK) src_cnt = SRC_BLK;
      for (size_t t = 0; t < trg_cnt_; t += VecLen) {
        Vec tx = load_intrin<Vec>(&trg_coord[0][t]);
        Vec ty = load_intrin<Vec>(&trg_coord[1][t]);
        Vec tz = load_intrin<Vec>(&trg_coord[2][t]);

        Vec tv = zero_intrin<Vec>();
        for (size_t s = sblk; s < sblk + src_cnt; s++) {
          Vec dx = sub_intrin(tx, bcast_intrin<Vec>(&src_coord[0][s]));
          Vec dy = sub_intrin(ty, bcast_intrin<Vec>(&src_coord[1][s]));
          Vec dz = sub_intrin(tz, bcast_intrin<Vec>(&src_coord[2][s]));

          Vec svx = bcast_intrin<Vec>(&src_value[0][s]);
          Vec svy = bcast_intrin<Vec>(&src_value[1][s]);
          Vec svz = bcast_intrin<Vec>(&src_value[2][s]);

          Vec r2 = mul_intrin(dx, dx);
          r2 = add_intrin(r2, mul_intrin(dy, dy));
          r2 = add_intrin(r2, mul_intrin(dz, dz));

          Vec rinv = RSQRT_INTRIN(r2);
          Vec rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);

          Vec inner_prod = mul_intrin(svx, dx);
          inner_prod = add_intrin(inner_prod, mul_intrin(svy, dy));
          inner_prod = add_intrin(inner_prod, mul_intrin(svz, dz));
          tv = add_intrin(tv, mul_intrin(inner_prod, rinv3));
        }
        Vec ooep = set_intrin<Vec, ValueType>(OOEP);

        tv = add_intrin(mul_intrin(tv, ooep), load_intrin<Vec>(&trg_value[0][t]));
        store_intrin(&trg_value[0][t], tv);
      }
    }

    {  // Add FLOPS
      #ifndef __MIC__
      //Profile::Add_FLOP((long long)trg_cnt_ * (long long)src_cnt_ * (29 + 4 * (NWTN_ITER)));
      #endif
    }
    #undef SRC_BLK
  }

 private:
  static ValueType machine_eps() {
    ValueType eps = 1;
    while (eps * (ValueType)0.5 + (ValueType)1.0 > 1.0) eps *= 0.5;
    return eps;
  }

  template <Integer newton_iter = 0> static void sl_vel(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    #define PVFMM_KER_NWTN(nwtn) \
  if (newton_iter == nwtn) KernelFnWrapper::kernel_wrapper<ValueType, Real, Vec, DIM, 3, 3, Stokes3D<Real>::template sl_vel_uKernel<Vec, rsqrt_intrin##nwtn<Vec, Real>>>(r_src, n_src, v_src, r_trg, v_trg)
    #define PVFMM_KERNEL_MACRO \
  PVFMM_KER_NWTN(0);       \
  PVFMM_KER_NWTN(1);       \
  PVFMM_KER_NWTN(2);       \
  PVFMM_KER_NWTN(3);
    if (TypeTraits<ValueType>::ID() == TypeTraits<float>::ID()) {
      typedef float Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256
      #elif defined __SSE3__
      #define Vec __m128
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else if (TypeTraits<ValueType>::ID() == TypeTraits<double>::ID()) {
      typedef double Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256d
      #elif defined __SSE3__
      #define Vec __m128d
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else {
      typedef ValueType Real;
      #define Vec Real
      PVFMM_KERNEL_MACRO;
      #undef Vec
    }
    #undef PVFMM_KER_NWTN
  }

  template <Integer newton_iter = 0> static void sl_vel_m2x(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    #define PVFMM_KER_NWTN(nwtn) \
  if (newton_iter == nwtn) KernelFnWrapper::kernel_wrapper<ValueType, Real, Vec, DIM, 4, 3, Stokes3D<Real>::template sl_vel_m2x_uKernel<Vec, rsqrt_intrin##nwtn<Vec, Real>>>(r_src, n_src, v_src, r_trg, v_trg)
    #define PVFMM_KERNEL_MACRO \
  PVFMM_KER_NWTN(0);       \
  PVFMM_KER_NWTN(1);       \
  PVFMM_KER_NWTN(2);       \
  PVFMM_KER_NWTN(3);
    if (TypeTraits<ValueType>::ID() == TypeTraits<float>::ID()) {
      typedef float Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256
      #elif defined __SSE3__
      #define Vec __m128
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else if (TypeTraits<ValueType>::ID() == TypeTraits<double>::ID()) {
      typedef double Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256d
      #elif defined __SSE3__
      #define Vec __m128d
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else {
      typedef ValueType Real;
      #define Vec Real
      PVFMM_KERNEL_MACRO;
      #undef Vec
    }
    #undef PVFMM_KER_NWTN
    #undef PVFMM_KERNEL_MACRO
  }

  template <Integer newton_iter = 0> static void dl_vel(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    #define PVFMM_KER_NWTN(nwtn) \
  if (newton_iter == nwtn) KernelFnWrapper::kernel_wrapper<ValueType, Real, Vec, DIM, 3, 3, Stokes3D<Real>::template dl_vel_uKernel<Vec, rsqrt_intrin##nwtn<Vec, Real>>>(r_src, n_src, v_src, r_trg, v_trg)
    #define PVFMM_KERNEL_MACRO \
  PVFMM_KER_NWTN(0);       \
  PVFMM_KER_NWTN(1);       \
  PVFMM_KER_NWTN(2);       \
  PVFMM_KER_NWTN(3);
    if (TypeTraits<ValueType>::ID() == TypeTraits<float>::ID()) {
      typedef float Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256
      #elif defined __SSE3__
      #define Vec __m128
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else if (TypeTraits<ValueType>::ID() == TypeTraits<double>::ID()) {
      typedef double Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256d
      #elif defined __SSE3__
      #define Vec __m128d
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else {
      typedef ValueType Real;
      #define Vec Real
      PVFMM_KERNEL_MACRO;
      #undef Vec
    }
    #undef PVFMM_KER_NWTN
    #undef PVFMM_KERNEL_MACRO
  }

  template <Integer newton_iter = 0> static void sl_press(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    #define PVFMM_KER_NWTN(nwtn) \
  if (newton_iter == nwtn) KernelFnWrapper::kernel_wrapper<ValueType, Real, Vec, DIM, 3, 1, Stokes3D<Real>::template sl_press_uKernel<Vec, rsqrt_intrin##nwtn<Vec, Real>>>(r_src, n_src, v_src, r_trg, v_trg)
    #define PVFMM_KERNEL_MACRO \
  PVFMM_KER_NWTN(0);       \
  PVFMM_KER_NWTN(1);       \
  PVFMM_KER_NWTN(2);       \
  PVFMM_KER_NWTN(3);
    if (TypeTraits<ValueType>::ID() == TypeTraits<float>::ID()) {
      typedef float Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256
      #elif defined __SSE3__
      #define Vec __m128
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else if (TypeTraits<ValueType>::ID() == TypeTraits<double>::ID()) {
      typedef double Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256d
      #elif defined __SSE3__
      #define Vec __m128d
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else {
      typedef ValueType Real;
      #define Vec Real
      PVFMM_KERNEL_MACRO;
      #undef Vec
    }
    #undef PVFMM_KER_NWTN
  }

  template <Integer newton_iter = 0> static void sl_press_m2x(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx) {
    #define PVFMM_KER_NWTN(nwtn) \
  if (newton_iter == nwtn) KernelFnWrapper::kernel_wrapper<ValueType, Real, Vec, DIM, 4, 1, Stokes3D<Real>::template sl_press_m2x_uKernel<Vec, rsqrt_intrin##nwtn<Vec, Real>>>(r_src, n_src, v_src, r_trg, v_trg)
    #define PVFMM_KERNEL_MACRO \
  PVFMM_KER_NWTN(0);       \
  PVFMM_KER_NWTN(1);       \
  PVFMM_KER_NWTN(2);       \
  PVFMM_KER_NWTN(3);
    if (TypeTraits<ValueType>::ID() == TypeTraits<float>::ID()) {
      typedef float Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256
      #elif defined __SSE3__
      #define Vec __m128
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else if (TypeTraits<ValueType>::ID() == TypeTraits<double>::ID()) {
      typedef double Real;
      #if defined __MIC__
      #define Vec Real
      #elif defined __AVX__
      #define Vec __m256d
      #elif defined __SSE3__
      #define Vec __m128d
      #else
      #define Vec Real
      #endif
      PVFMM_KERNEL_MACRO;
      #undef Vec
    } else {
      typedef ValueType Real;
      #define Vec Real
      PVFMM_KERNEL_MACRO;
      #undef Vec
    }
    #undef PVFMM_KER_NWTN
    #undef PVFMM_KERNEL_MACRO
  }

  friend class KernelFnWrapper;
};

template <class ValueType> struct Smoother {
 public:
  static const Integer DIM = 3;

  inline static const KernelFunction<ValueType, DIM>& ker3x3() {
    constexpr Integer k_dim0 = DIM;
    constexpr Integer k_dim1 = DIM;
    static const ValueType sigma2 = 0.0001;
    static const ValueType scal = pvfmm::pow(2.0 * const_pi<ValueType>() * sigma2, -(DIM - 1) * 0.5);
    static auto micro_ker = [&](ValueType* v, const ValueType* x, ValueType invr, const ValueType* n, const ValueType* f) {
      ValueType r2 = 1.0/(invr*invr);
      v[0] += exp(-r2/sigma2*0.5) * f[0] * scal;
      v[1] += exp(-r2/sigma2*0.5) * f[1] * scal;
      v[2] += exp(-r2/sigma2*0.5) * f[2] * scal;
    };
    return KernelFromLambda<k_dim0, k_dim1>(__FUNCTION__, micro_ker);
  }

 private:
  static void GenKer(const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx){
    const auto& ker = *(std::function<typename KernelFunction<ValueType, DIM>::KerFn>*)(ctx);
    ker(r_src, n_src, v_src, r_trg, v_trg, NULL);
  }

  template <Integer k_dim0, Integer k_dim1, class LambdaType> inline static const KernelFunction<ValueType, DIM>& KernelFromLambda(std::string name, LambdaType&& micro_ker) {
    static std::function<typename KernelFunction<ValueType, DIM>::KerFn> ker_fn = [&](const Vector<ValueType>& r_src, const Vector<ValueType>& n_src, const Vector<ValueType>& v_src, const Vector<ValueType>& r_trg, Vector<ValueType>& v_trg, const void* ctx){
      Long src_cnt = r_src.Dim() / DIM;
      Long trg_cnt = r_trg.Dim() / DIM;
      for (Long t=0;t<trg_cnt;t++){
        ValueType v[k_dim1];
        for (Integer i=0;i<k_dim1;i++) {
          v[i]=0;
        }
        for (Long s=0;s<src_cnt;s++){
          ValueType dx[DIM], f[k_dim0], r2 = 0, invr = 0;
          for(Integer k=0;k<DIM;k++) {
            dx[k] = r_trg[t*DIM+k] - r_src[s*DIM+k];
            r2 += dx[k] * dx[k];
          }
          for(Integer k=0;k<k_dim0;k++) {
            f[k] = v_src[s*k_dim0+k];
          }
          if(r2>0) invr=1.0/sqrt(r2);
          micro_ker(v, dx, invr, &n_src[s*DIM], f);
        }
        for (Integer i=0;i<k_dim1;i++) {
          v_trg[t*k_dim1+i] += v[i];
        }
      }
    };
    static KernelFunction<ValueType, DIM> ker(GenKer, k_dim0, k_dim1, name, &ker_fn);
    return ker;
  }
};

}  // end namespace

#endif  //_PVFMM_KERNEL_HPP_
