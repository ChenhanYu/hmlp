#ifndef _PVFMM_KERNEL_MATRIX_HPP_
#define _PVFMM_KERNEL_MATRIX_HPP_

#include <omp.h>

#include <pvfmm/cheb_utils.hpp>

namespace pvfmm {

template <class Real> class KernelMatrix {

  static const Integer DIM = 3;
  static const Integer ELEMDIM = DIM - 1;
  static const Integer Nface = 2 * DIM;

  public:

   KernelMatrix(size_t m=0, size_t n=0) {
     size_t N = std::max(m, n);
     Integer order = 6, depth = 0;
     auto& ker = Stokes3D<Real>::FxU();
     Integer Ncoeff = order * (order + 1) / 2 * ker.Dim(0);
     depth = log(N / (2.0 * DIM * Ncoeff))/log(4.0);
     Initialize(order, depth, ker);
   }

   Real operator() (size_t i, size_t j) const {
     return (GetElem(i,j) + GetElem(j,i)) * 0.5; // Symmetric part
   }

   Long Rows() const {
     return row_;
   }

   Long Columns() const {
     return col_;
   }

  private:

   void Initialize(Integer order, Integer depth, const KernelFunction<Real, DIM>& ker) {
     order_ = order;
     depth_ = depth;
     ker_ = &ker;
     {  // Set Ncoeff_
       Ncoeff_ = 1;
       for (Integer i = 0; i < ELEMDIM; i++) Ncoeff_ = (Ncoeff_ * (order_ + i)) / (i + 1);
     }
     GridDim_ = pvfmm::pow<Long>(2, depth);
     row_ = Nface * pvfmm::pow<ELEMDIM, Long>(GridDim_) * ker_->Dim(0) * Ncoeff_;
     col_ = Nface * pvfmm::pow<ELEMDIM, Long>(GridDim_) * ker_->Dim(1) * Ncoeff_;
   }

   Real GetElem(Long t, Long s) const {
     if (t >= row_ || s >= col_) return (s==t ? 1.0 : 0.0);

     Integer src_coeff_idx = s % (ker_->Dim(0) * Ncoeff_);
     Integer trg_coeff_idx = t % (ker_->Dim(1) * Ncoeff_);

     Integer src_elem = s / (pvfmm::pow<ELEMDIM, Long>(GridDim_) * ker_->Dim(0) * Ncoeff_);
     Integer trg_elem = t / (pvfmm::pow<ELEMDIM, Long>(GridDim_) * ker_->Dim(1) * Ncoeff_);

     StaticArray<Real, DIM> scoord, tcoord;
     { // Set scoord
       Long idx = (s / (ker_->Dim(0) * Ncoeff_)) % pvfmm::pow<ELEMDIM, Long>(GridDim_);
       Real s = pvfmm::pow<Real>(0.5, depth_);
       for (Integer i = 0; i < DIM; i++) {
         if (i == src_elem / 2) {
           scoord[i] = (src_elem % 2 ? 1.0 - s : 0.0);
         } else {
           scoord[i] = (idx % GridDim_) * (1.0 / GridDim_);
           idx = idx / GridDim_;
         }
       }
     }
     { // Set tcoord
       Long idx = (t / (ker_->Dim(1) * Ncoeff_)) % pvfmm::pow<ELEMDIM, Long>(GridDim_);
       Real s = pvfmm::pow<Real>(0.5, depth_);
       for (Integer i = 0; i < DIM; i++) {
         if (i == trg_elem / 2) {
           tcoord[i] = (trg_elem % 2 ? 1.0 - s : 0.0);
         } else {
           tcoord[i] = (idx % GridDim_) * (1.0 / GridDim_);
           idx = idx / GridDim_;
         }
       }
     }

     return GetMatrixElem(order_, scoord, src_elem, tcoord, trg_elem, src_coeff_idx, trg_coeff_idx);
   }

   Real GetMatrixElem(Integer order, ConstIterator<Real> scoord, Integer src_elem, ConstIterator<Real> tcoord, Integer trg_elem, Long src_coeff_idx, Long trg_coeff_idx) const {
     Real elem = 0;
     StaticArray<Long, DIM + 2> idx_arr;
     for (Integer i = 0; i < DIM; i++) idx_arr[i] = (tcoord[i] - scoord[i]) * GridDim_;
     if ( 0 || std::abs(idx_arr[0]) < 1000
				    && std::abs(idx_arr[1]) < 1000
					  && std::abs(idx_arr[2]) < 1000 ) 
		 { // Store near-interaction matrices
       idx_arr[DIM + 0] = src_elem;
       idx_arr[DIM + 1] = trg_elem;
       Long idx = cantor_signed(idx_arr, DIM + 2);
       #pragma omp critical
       { // Set elem
         if (mat.find(idx) == mat.end()) { // Compute and store new matrix
           mat[idx] = Integ(order_, scoord, src_elem, depth_, tcoord, trg_elem, depth_, *ker_);
         }
         elem = mat[idx][src_coeff_idx][trg_coeff_idx];
       }
     } else { // TODO: implement optimized algorithm for well-separated octants
       elem = IntegFar(order_, scoord, src_elem, depth_, tcoord, trg_elem, depth_, *ker_, src_coeff_idx, trg_coeff_idx);
       //Matrix<Real> M = Integ(order_, scoord, src_elem, depth_, tcoord, trg_elem, depth_, *ker_);
       //auto elem_ = M[src_coeff_idx][trg_coeff_idx];
       //static Real max_err = 0;
       //max_err = std::max(fabs(elem_-elem), max_err);
       //std::cout<<max_err<<'\n';
     }
     return elem;
   }

   static void ReferenceElemNodes(Integer order, Vector<Vector<Real>> &ref_coord) {
     ref_coord.ReInit(Nface);
     Long Neval = pvfmm::pow<Long>(order, ELEMDIM);
     {  // Set ref_coord
       Vector<Real> ref_coord_;
       ChebBasis<Real>::template Nodes<ELEMDIM>(order, ref_coord_);
       assert(ref_coord_.Dim() == Neval * ELEMDIM);
       if (ELEMDIM == DIM) {
         assert(Nface == 1);
         ref_coord[0].Swap(ref_coord_);
       } else if (ELEMDIM == DIM - 1) {
         for (Integer elem = 0; elem < Nface; elem++) {
           Integer k0 = (elem >> 1);
           ref_coord[elem].ReInit(Neval * DIM);
           for (Long j = 0; j < Neval; j++) {
             for (Integer k = 0; k < ELEMDIM; k++) {
               ref_coord[elem][j * DIM + ((k + k0 + 1) % DIM)] = ref_coord_[j * ELEMDIM + k];
             }
           }
           if (elem & 1) {
             for (Long j = 0; j < Neval; j++) {
               ref_coord[elem][j * DIM + k0] = 1;
             }
           } else {
             for (Long j = 0; j < Neval; j++) {
               ref_coord[elem][j * DIM + k0] = 0;
             }
           }
         }
       }
     }
   }

   static Vector<Real> ChebNodes(ConstIterator<Real> coord, Integer elem, Integer depth, Integer order) {
     static Vector<Vector<Real>> ref_nodes;
     #pragma omp critical (REF_NODES_CRITICAL)
     if (!ref_nodes.Dim()) {
       ReferenceElemNodes(order, ref_nodes);
     }

     Vector<Real> nodes;
     nodes = ref_nodes[elem];

     Integer N = nodes.Dim() / DIM;
     Real s = pvfmm::pow<Real>(0.5, depth);
     for (Integer i = 0; i < N; i++) {
       for (Integer j = DIM - DIM; j < DIM; j++) {
         nodes[i * DIM + j] = nodes[i * DIM + j] * s + coord[j];
       }
     }
     return nodes;
   }

   static Matrix<Real> Integ(Integer order, ConstIterator<Real> scoord, Integer src_elem, Integer src_depth, ConstIterator<Real> tcoord, Integer trg_elem, Integer trg_depth, const KernelFunction<Real, DIM> &ker, Real tol = -1) {
     Matrix<Real> Mcoeff2nodes;
     { // Set Mcoeff2nodes
       Vector<Real> trg_nodes = ChebNodes(tcoord, trg_elem, trg_depth, order);
       Integer Ntrg = trg_nodes.Dim() / DIM;
       PVFMM_ASSERT(Ntrg);

       for (Integer i = 0; i < Ntrg; i++) {  // Shift trg_nodes by scoord
         for (Integer j = 0; j < DIM; j++) {
           trg_nodes[i * DIM + j] -= scoord[j];
         }
       }

       Vector<Matrix<Real>> Mcoeff(Ntrg);
       Real s = pvfmm::pow<Real>(0.5, src_depth);
       #pragma omp parallel for schedule(dynamic)
       for (Integer i = 0; i < Ntrg; i++) {
         ChebBasis<Real>::template Integ<DIM, ELEMDIM>(Mcoeff[i], order, trg_nodes.Begin() + i * DIM, s, src_elem, ker, tol);
       }

       Mcoeff2nodes.ReInit(Mcoeff[0].Dim(0), Mcoeff[0].Dim(1) * Ntrg);
       for (Integer j = 0; j < Mcoeff[0].Dim(0); j++) {
         for (Integer i0 = 0; i0 < Mcoeff[0].Dim(1); i0++) {
           for (Integer i1 = 0; i1 < Ntrg; i1++) {
             Mcoeff2nodes[j][i1 * Mcoeff[0].Dim(1) + i0] = Mcoeff[i1][j][i0];
           }
         }
       }
     }

     Matrix<Real> Mcoeff2coeff;
     {  // Set Mcoeff2coeff
       Integer Ntrg = Mcoeff2nodes.Dim(1) / ker.Dim(1);
       assert(Ntrg * ker.Dim(1) == Mcoeff2nodes.Dim(1));
       for (Integer i = 0; i < Mcoeff2nodes.Dim(0); i++) {
         Matrix<Real> M(Ntrg, ker.Dim(1), Mcoeff2nodes[i], false);
         M = M.Transpose();
       }

       Vector<Real> coeff;
       const Vector<Real> fn_v(Mcoeff2nodes.Dim(0) * Mcoeff2nodes.Dim(1), Mcoeff2nodes.Begin(), false);
       ChebBasis<Real>::template Approx<ELEMDIM>(order, fn_v, coeff);
       ChebBasis<Real>::template Truncate<ELEMDIM>(coeff, order, order);
       Mcoeff2coeff.ReInit(Mcoeff2nodes.Dim(0), coeff.Dim() / Mcoeff2nodes.Dim(0), coeff.Begin());
     }

     return Mcoeff2coeff;
   }

   static Long cantor_idx(ConstIterator<Long> x, Integer N) {
     Long idx = 0;
     for (Integer d = 0; d < N; d++) {
       Long sum = 0;
       for (Integer i = 0; i < N - d; i++) {
         sum += std::abs(x[i]);
       }
       Long offset = 1;
       for (Integer i = 0; i < N - d; i++) {
         offset = offset * (sum + i) / (i + 1);
       }
       idx += offset;
     }
     return idx;
   }

   static Long cantor_signed(ConstIterator<Long> x, Integer N) {
     Long idx = cantor_idx(x, N);
     for (Integer i = 0; i < N; i++) {
       idx = 2 * idx + (x[i] < 0 ? 0 : 1);
     }
     return idx;
   }




   template <Integer DIM, Integer SUBDIM> static Real IntegFar(Integer order, const Vector<Real>& r_trg, Real side, Integer src_face, const KernelFunction<Real, DIM>& ker, Integer src_coeff_idx, Integer trg_coeff_idx) {
     static const Real eps = ChebBasis<Real>::machine_eps() * 64;
     Real side_inv = 1.0 / side;
     Integer Nq = 6;

     static Vector<Real> qp, qw, p0;
     #pragma omp critical
     if (!qp.Dim()) {
       ChebBasis<Real>::quad_rule(Nq, qp, qw);
       ChebBasis<Real>::EvalBasis1D(order, qp, p0);
     }

     Integer Ncoeff;
     {  // Set Ncoeff
       Ncoeff = 1;
       for (Integer i = 0; i < SUBDIM; i++) Ncoeff = (Ncoeff * (order + i)) / (i + 1);
     }
     StaticArray<Integer, 2> kdim;
     kdim[0] = ker.Dim(0);
     kdim[1] = ker.Dim(1);

     Integer sidx0=0, sidx1=0, sidx2=0;
     Integer tidx0=0, tidx1=0, tidx2=0;
     { // Set sidx
       sidx2 = src_coeff_idx / Ncoeff;
       Integer idx=0, sidx = src_coeff_idx % Ncoeff;
       for (Integer i0=0;i0<order;i0++){
         for (Integer i1=0;i0+i1<order;i1++){
           if(idx==sidx) {
             sidx0 = i1;
             sidx1 = i0;
           }
           idx++;
         }
       }
     }
     { // Set tidx
       tidx2 = trg_coeff_idx / Ncoeff;
       Integer idx=0, tidx = trg_coeff_idx % Ncoeff;
       for (Integer i0=0;i0<order;i0++){
         for (Integer i1=0;i0+i1<order;i1++){
           if(idx==tidx) {
             tidx0 = i1;
             tidx1 = i0;
           }
           idx++;
         }
       }
     }

     PVFMM_ASSERT(SUBDIM==2 && DIM==3);
     Vector<Real> n_src(pvfmm::pow<SUBDIM, Integer>(Nq)*DIM);
     Vector<Real> r_src(pvfmm::pow<SUBDIM, Integer>(Nq)*DIM);
     Vector<Real> v_src(pvfmm::pow<SUBDIM, Integer>(Nq)*kdim[0]);
     n_src.SetZero();
     v_src.SetZero();
     for (Integer i0=0;i0<Nq;i0++){
       for (Integer i1=0;i1<Nq;i1++){
         n_src[(i0 * Nq + i1) * DIM + src_face >> 1] = (src_face & 1 ? -1.0 : 1.0);
         r_src[(i0 * Nq + i1) * DIM + ((src_face>>1)+0)%DIM] = (src_face & 1 ? 1.0 : 0.0) * side;
         r_src[(i0 * Nq + i1) * DIM + ((src_face>>1)+1)%DIM] = qp[i0] * side;
         r_src[(i0 * Nq + i1) * DIM + ((src_face>>1)+2)%DIM] = qp[i1] * side;
         v_src[(i0 * Nq + i1) * kdim[0] + sidx2] = p0[sidx0 * Nq + i0] * p0[sidx1 * Nq + i1] * qw[i0] * qw[i1]*side*side;
       }
     }

     Long Ntrg = r_trg.Dim() / DIM;
     Vector<Real> v_trg(Ntrg * kdim[1]);
     v_trg.SetZero();
     ker(r_src, n_src, v_src, r_trg, v_trg);

     Vector<Real> fn_v(Ntrg), coeff;
     for (Long i=0;i<Ntrg;i++) fn_v[i] = v_trg[i * kdim[1] + tidx2];
     ChebBasis<Real>::template Approx<ELEMDIM>(order, fn_v, coeff);
     return coeff[trg_coeff_idx % Ncoeff];
   }

   static Real IntegFar(Integer order, ConstIterator<Real> scoord, Integer src_elem, Integer src_depth, ConstIterator<Real> tcoord, Integer trg_elem, Integer trg_depth, const KernelFunction<Real, DIM> &ker, Integer src_coeff_idx, Integer trg_coeff_idx) {
     Vector<Real> trg_nodes = ChebNodes(tcoord, trg_elem, trg_depth, order);
     Integer Ntrg = trg_nodes.Dim() / DIM; PVFMM_ASSERT(Ntrg);
     for (Integer i = 0; i < Ntrg; i++) {
       for (Integer j = 0; j < DIM; j++) {
         trg_nodes[i * DIM + j] -= scoord[j];
       }
     }
     Real s = pvfmm::pow<Real>(0.5, src_depth);
     return IntegFar<DIM, ELEMDIM>(order, trg_nodes, s, src_elem, ker, src_coeff_idx, trg_coeff_idx);
   }

   mutable std::map<Long, Matrix<Real>> mat;
   const KernelFunction<Real, DIM>* ker_;
   Integer order_, depth_, Ncoeff_;
   Long GridDim_, row_, col_;
};

}  // end namespace

#endif  //_PVFMM_KERNEL_MATRIX_HPP_
