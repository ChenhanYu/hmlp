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




#ifndef KERNELMATRIX_HPP
#define KERNELMATRIX_HPP

/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif

/** BLAS/LAPACK support */
#include <hmlp_blas_lapack.h>
/** KernelMatrix uses VirtualMatrix<T> as base */
#include <containers/VirtualMatrix.hpp>
#include <DistData.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{

template<typename T, class Allocator = std::allocator<T>>
class KernelMatrix : public VirtualMatrix<T, Allocator>, 
                     public ReadWrite
{
  public:

    /** (Default) constructor for symmetric kernel matrices. */
    KernelMatrix( size_t m, size_t n, size_t d, kernel_s<T> &kernel, 
        Data<T> &sources )
      : sources( sources ), 
        targets( sources ), 
        VirtualMatrix<T>( m, n ),
        all_dimensions( d )
    {
      assert( m == n );
      this->is_symmetric = true;
      this->d = d;
      this->kernel = kernel;
      /** Compute square 2-norms for Euclidian family. */
      ComputeSquare2Norm();
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
    };

    KernelMatrix( Data<T> &sources )
      : sources( sources ), 
        targets( sources ), 
        VirtualMatrix<T>( sources.col(), sources.col() ),
        all_dimensions( sources.row() )
    {
      this->is_symmetric = true;
      this->d = sources.row();
      this->kernel.type = KS_GAUSSIAN;
      this->kernel.scal = -0.5;
      /** Compute square 2-norms for Euclidian family. */
      ComputeSquare2Norm();
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
    };

    /** (Default) constructor for non-symmetric kernel matrices. */
    KernelMatrix( size_t m, size_t n, size_t d, kernel_s<T> &kernel, 
        Data<T> &sources, Data<T> &targets )
      : sources( sources ), 
        targets( targets ), 
        VirtualMatrix<T>( m, n ),
        all_dimensions( d )
    {
      this->is_symmetric = false;
      this->d = d;
      this->kernel = kernel;
      /** Compute square 2-norms for Euclidian family. */
      ComputeSquare2Norm();
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
    };

    /** (Default) destructor. */
    ~KernelMatrix() {};





    void ComputeSquare2Norm()
    {
      source_sqnorms.resize( this->col(), 0.0 );

      /** compute 2-norm using xdot */
      #pragma omp parallel for
      for ( size_t j = 0; j < this->col(); j ++ )
      {
        source_sqnorms[ j ] = xdot(
          d, sources.data() + j * d, 1, 
					   sources.data() + j * d, 1 );
      }

      /** compute 2-norms for targets if unsymmetric */
      if ( is_symmetric ) target_sqnorms = source_sqnorms;
      else
      {
        target_sqnorms.resize( this->row(), 0.0 );

        #pragma omp parallel for
        for ( size_t i = 0; i < this->row(); i ++ )
        {
          target_sqnorms[ i ] = xdot(
              d, targets.data() + i * d, 1, 
							   targets.data() + i * d, 1 );
        }
      }
    };


		/** ESSENTIAL: override the virtual function */
    T operator()( size_t i, size_t j )
    {
      T Kij = 0.0;

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          for ( size_t k = 0; k < d; k++ )
          {
						T tar = targets[ i * d + k ];
						T src = sources[ j * d + k ];
            Kij += ( tar - src ) * ( tar - src );
          }
          Kij = std::exp( kernel.scal * Kij );
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
			}
      return Kij;
		};


//    /** ESSENTIAL: return K( imap, jmap ) */
//    Data<T> operator() ( const vector<size_t> &imap, 
//                         const vector<size_t> &jmap )
//    {
//      Data<T> submatrix( imap.size(), jmap.size() );
//
//      if ( !submatrix.size() ) return submatrix;
//
//      /** Get coordinates of sources and targets */
//      Data<T> itargets = targets( imap );
//      Data<T> jsources = sources( jmap );
//
//      assert( itargets.col() == submatrix.row() );
//      assert( itargets.row() == d );
//      assert( jsources.col() == submatrix.col() );
//      assert( jsources.row() == d );
//
//      /** compute inner products */
//      xgemm
//      (
//        "T", "N",
//        imap.size(), jmap.size(), d,
//        -2.0, itargets.data(),   itargets.row(),
//              jsources.data(),   jsources.row(),
//         0.0, submatrix.data(), submatrix.row()
//      );
//
//      /** compute square norms */
//      vector<T> itarget_sqnorms( imap.size() );
//      vector<T> jsource_sqnorms( jmap.size() );
//
//
//      #pragma omp parallel for
//      for ( size_t i = 0; i < imap.size(); i ++ )
//      {
//        if ( target_sqnorms.size() )
//        {
//          /** if precomputed then directly copy */
//          itarget_sqnorms[ i ] = target_sqnorms[ imap[ i ] ];
//        }
//        else
//        {
//          /** otherwise compute them now */
//          itarget_sqnorms[ i ] = xdot(
//              d, itargets.data() + i * d, 1, itargets.data() + i * d, 1 );
//        }
//      }
//
//      #pragma omp parallel for
//      for ( size_t j = 0; j < jmap.size(); j ++ )
//      {
//        if ( source_sqnorms.size() )
//        {
//          /** if precomputed then directly copy */
//          jsource_sqnorms[ j ] = source_sqnorms[ jmap[ j ] ];
//        }
//        else
//        {
//          /** otherwise compute them now */
//          jsource_sqnorms[ j ] = xdot(
//              d, jsources.data() + j * d, 1, jsources.data() + j * d, 1 );
//        }
//      }
//
//      /** add square norms to inner products to get square distances */
//      #pragma omp parallel for
//      for ( size_t j = 0; j < jmap.size(); j ++ )
//      {
//        for ( size_t i = 0; i < imap.size(); i ++ )
//        {
//          submatrix( i, j ) += itarget_sqnorms[ i ] + jsource_sqnorms[ j ];
//        }
//      }
//
//      switch ( kernel.type )
//      {
//        case KS_GAUSSIAN:
//        {
//          /** apply the scaling factor and exponentiate */
//          #pragma omp parallel for
//          for ( size_t i = 0; i < submatrix.size(); i ++ )
//          {
//            submatrix[ i ] = std::exp( kernel.scal * submatrix[ i ] );
//          }
//
//          // gemm: 2 * i * j * d
//          // compute sqnorms: 2 * ( i + j ) * d
//          // add sqnorms: 2 * i * j
//          // scale and exponentiate: 2 * i * j
//          //flopcount += 2 * ( imap.size() * jmap.size() + imap.size() + jmap.size() ) * d
//          //           + 4 * imap.size() * jmap.size();
//          break;
//        }
//        default:
//        {
//          printf( "invalid kernel type\n" );
//          exit( 1 );
//          break;
//        }
//      }
//
//      return submatrix;
//    }; 
//

    /** (Overwrittable) ESSENTIAL: return K( I, J ) */
    virtual Data<T> operator() ( vector<size_t> &I, 
                                 vector<size_t> &J )
    {
      Data<T> KIJ = GeometryDistances( I, J );
			/** Early return */
			if ( !I.size() || !J.size() ) return KIJ;

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          /** Apply the scaling factor and exponent */
          #pragma omp parallel for 
          for ( size_t i = 0; i < KIJ.size(); i ++ )
          {
            KIJ[ i ] = std::exp( kernel.scal * KIJ[ i ] );
          }
          break;
        }
        case KS_LAPLACE:
        {
          //T scale = 0.0;

          //if ( d == 1 ) scale = 1.0;
          //else if ( d == 2 ) scale = ( 0.5 / M_PI );
          //else
          //{
          //  scale = tgamma( 0.5 * d + 1.0 ) / ( d * ( d - 2 ) * pow( M_PI, 0.5 * d ) ); 
          //}

          #pragma omp parallel for
          for ( size_t i = 0; i < KIJ.size(); i ++ )
          {
            if ( KIJ[ i ] < 1E-6 ) KIJ[ i ] = 0.0;
            else
            {
              if      ( d == 1 ) KIJ[ i ] = sqrt( KIJ[ i ] );
              else if ( d == 2 ) KIJ[ i ] = std::log( KIJ[ i ] ) / 2;
              else               KIJ[ i ] = 1 / sqrt( KIJ[ i ] );
            }
          }
          break;
        }
        case KS_QUARTIC:
        {
          #pragma omp parallel for
          for ( size_t i = 0; i < KIJ.size(); i ++ )
          {
            if ( KIJ[ i ] < 0.5 ) KIJ[ i ] = 1.0;
            else KIJ[ i ] = 0.0;
          }
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
      }

      return KIJ;
    };


    Data<T> GeometryDistances( const vector<size_t> &I, 
                               const vector<size_t> &J )
    {
      Data<T> KIJ( I.size(), J.size() );
			/** Early return if possible. */
			if ( !I.size() || !J.size() ) return KIJ;
			/** Request for coordinates: A (targets), B (sources). */
      Data<T> A, B;
      /** For symmetry matrices, extract from sources. */
      if ( is_symmetric ) 
      {
        A = sources( all_dimensions, I );
        B = sources( all_dimensions, J );
      }
      /** For non-symmetry matrices, extract from targets and sources. */
      else
      {
        A = targets( all_dimensions, I );
        B = sources( all_dimensions, J );
      }
      /** Compute inner products. */
      xgemm( "Transpose", "No-transpose", I.size(), J.size(), d,
        -2.0, A.data(), A.row(),
              B.data(), B.row(),
         0.0, KIJ.data(), KIJ.row() );
      /** Compute square 2-norms. */
      vector<T> A2( I.size() ), B2( J.size() );

      #pragma omp parallel for
      for ( size_t i = 0; i < I.size(); i ++ )
      {
        A2[ i ] = xdot( d, A.columndata( i ), 1, A.columndata( i ), 1 );
      } /** end omp parallel for */

      #pragma omp parallel for
      for ( size_t j = 0; j < J.size(); j ++ )
      {
        B2[ j ] = xdot( d, B.columndata( j ), 1, B.columndata( j ), 1 );
      } /** end omp parallel for */

      /** Add square norms to inner products to get square distances. */
      #pragma omp parallel for
      for ( size_t j = 0; j < J.size(); j ++ )
        for ( size_t i = 0; i < I.size(); i ++ )
          KIJ( i, j ) += A2[ i ] + B2[ j ];
      /** Return all pair-wise distances. */
      return KIJ;
    }; /** end GeometryDistances() */




    /** get the diagonal of KII, i.e. diag( K( I, I ) ) */
    Data<T> Diagonal( vector<size_t> &I )
    {
      /** 
       *  return values
       */
      Data<T> DII( I.size(), 1, 0.0 );

      /** at this moment we already have the corrdinates on this process */
      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          for ( size_t i = 0; i < I.size(); i ++ ) DII[ i ] = 1.0;
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
      }

      return DII;

    };






    /** important sampling */
    pair<T, size_t> ImportantSample( size_t j )
    {
      size_t i = std::rand() % this->col();
      pair<T, size_t> sample( (*this)( i, j ), i );
      return sample; 
    };

    void ComputeDegree()
    {
      printf( "ComputeDegree(): K * ones\n" );
      assert( is_symmetric );
      Data<T> ones( this->row(), (size_t)1, 1.0 );
      Degree.resize( this->row(), (size_t)1, 0.0 );
      Multiply( 1, Degree, ones );
    };

    Data<T> &GetDegree()
    {
      assert( is_symmetric );
      if ( Degree.row() != this->row() ) ComputeDegree();
      return Degree;
    };

    void NormalizedMultiply( size_t nrhs, T *u, T *w )
    {
      assert( is_symmetric );
      if ( Degree.row() != this->row() ) 
      {
        ComputeDegree();
      }
      Data<T> invDw( this->row(), (size_t)1, 0.0 );

      /** D^{-1/2}w */
      for ( size_t i = 0; i < this->row(); i ++ )
      {
        u[ i ] = 0.0;
        invDw[ i ] = w[ i ] / std::sqrt( Degree[ i ] );
      }

      /** KD^{-1/2}w */
      Multiply( nrhs, u, invDw.data() );

      /** D^{-1/2}KD^{-1/2}w */
      for ( size_t i = 0; i < this->row(); i ++ )
        u[ i ] = u[ i ] / std::sqrt( Degree[ i ] );
    };


    /**
     *
     *
     */ 
    void Multiply( size_t nrhs, T* u, T* w )
    {
      vector<int> amap( this->row() );
      vector<int> bmap( this->col() );
      for ( size_t i = 0; i < amap.size(); i ++ ) amap[ i ] = i;
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;

      if ( nrhs == 1 )
      {
        gsks( &kernel, amap.size(), bmap.size(), d,
                         u,                        amap.data(),
            targets.data(), target_sqnorms.data(), amap.data(), 
            sources.data(), source_sqnorms.data(), bmap.data(), 
                         w,                        bmap.data() );
      }
      else
      {
        printf( "gsks with multiple rhs is not implemented yet\n" );
        exit( 1 );
      }
    };

    /** u( umap ) += K( amap, bmap ) * w( wmap ) */
    template<typename TINDEX>
    void Multiply( 
        size_t nrhs,
        vector<T, Allocator> &u, vector<TINDEX> &umap, 
                                      vector<TINDEX> &amap,
                                      vector<TINDEX> &bmap,
        vector<T, Allocator> &w, vector<TINDEX> &wmap )
    {
      if ( nrhs == 1 )
      {
        gsks( &kernel, amap.size(), bmap.size(), d,
                  u.data(),                        umap.data(),
            targets.data(), target_sqnorms.data(), amap.data(), 
            sources.data(), source_sqnorms.data(), bmap.data(), 
                  w.data(),                        wmap.data() );
      }
      else
      {
        printf( "gsks with multiple rhs is not implemented yet\n" );
        exit( 1 );
      }
    };

    /** u( amap ) += K( amap, bmap ) * w( bmap ) */
    template<typename TINDEX>
    void Multiply( 
        size_t nrhs,
        vector<T, Allocator> &u,
                           vector<TINDEX> &amap,
                           vector<TINDEX> &bmap,
        vector<T, Allocator> &w )
    {
      Multiply( nrhs, u, amap, amap, bmap, w, bmap );
    };


    /** u += K * w */
    void Multiply( size_t nrhs, vector<T, Allocator> &u, vector<T, Allocator> &w )
    {
      vector<int> amap( this->row() );
      vector<int> bmap( this->col() );
      for ( size_t i = 0; i < amap.size(); i ++ ) amap[ i ] = i;
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;
      Multiply( nrhs, u, amap, bmap, w );
    };

    void Multiply( Data<T, Allocator> &u, Data<T, Allocator> &w )
    {
      assert( u.row() == this->row() );
      assert( w.row() == this->col() );
      assert( u.col() == w.col() );
      size_t nrhs = u.col();
      Multiply( nrhs, u, w );
    };


    void Print()
    {
      for ( size_t j = 0; j < this->col(); j ++ )
      {
        printf( "%8lu ", j );
      }
      printf( "\n" );
      for ( size_t i = 0; i < this->row(); i ++ )
      {
        for ( size_t j = 0; j < this->col(); j ++ )
        {
          printf( "% 3.1E ", (*this)( i, j ) );
        }
        printf( "\n" );
      }
    }; /** end Print() */

    /** Return number of attributes. */
    size_t dim() { return d; };

    /** flops required for Kab */
    double flops( size_t na, size_t nb ) 
    {
      double flopcount = 0.0;

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          flopcount = na * nb * ( 2.0 * d + 35.0 );
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
      }
      return flopcount; 
    };

  private:

    bool is_symmetric = true;

    size_t d;

    Data<T> &sources;

    Data<T> &targets;

    /** Degree (diagonal matrix) */
    Data<T> Degree;

    vector<T> source_sqnorms;

    vector<T> target_sqnorms;

    /** legacy data structure */
    kernel_s<T> kernel;

    /** [ 0, 1, ..., d-1 ] */
    vector<size_t> all_dimensions;

}; /** end class KernelMatrix */



template<typename T, typename TP, class Allocator = std::allocator<TP>>
class DistKernelMatrix : public DistVirtualMatrix<T, Allocator>, 
                         public ReadWrite
{
  public:

    /** (Default) unsymmetric kernel matrix */
    DistKernelMatrix
    ( 
      size_t m, size_t n, size_t d, 
      /** by default we assume sources are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &sources, 
      /** by default we assume targets are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &targets, 
      mpi::Comm comm 
    )
    : all_dimensions( d ), sources_user( sources ), targets_user( target ),
      DistVirtualMatrix<T>( m, n, comm )
    {
      this->is_symmetric = false;
      this->d = d;
      this->sources = &sources;
      this->targets = &targets;
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
    };

    /** Unsymmetric kernel matrix with legacy kernel_s<T> */
    DistKernelMatrix
    ( 
      size_t m, size_t n, size_t d,
      kernel_s<T> &kernel, 
      /** by default we assume sources are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &sources, 
      /** by default we assume targets are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &targets, 
      mpi::Comm comm 
    )
    : DistKernelMatrix( m, n, d, sources, targets, comm ) 
    {
      this->kernel = kernel;
    };

    /** (Default) symmetric kernel matrix */
    DistKernelMatrix
    ( 
      size_t n, size_t d, 
      /** by default we assume sources are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &sources, 
      mpi::Comm comm 
    ) 
    : all_dimensions( d ), sources_user( sources ), targets_user( d, 0, comm ),
      DistVirtualMatrix<T>( n, n, comm )
    {
      this->is_symmetric = true;
      this->d = d;
      this->sources = &sources;
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
    };

    /** Symmetric kernel matrix with legacy kernel_s<T> */
    DistKernelMatrix
    ( 
      size_t n, size_t d, 
      kernel_s<T> &kernel,
      /** by default we assume sources are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &sources, 
      mpi::Comm comm 
    )
    : DistKernelMatrix( n, d, sources, comm )
    {
      this->kernel = kernel;
    };


    DistKernelMatrix( DistData<STAR, CBLK, TP> &sources, mpi::Comm comm )
    : DistKernelMatrix( sources.col(), sources.row(), sources, comm ) 
    {
      this->kernel.type = KS_GAUSSIAN;
      this->kernel.scal = -0.5;
    };

    /** (Default) destructor */
    ~DistKernelMatrix() {};


    /** Compute Kij according legacy kernel_s<T> */
    T operator () ( TP *itargets, TP *jsources )
    {
      /** Return value */
      T Kij = 0.0;

      /** At this moment we already have the corrdinates on this process */
      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          for ( size_t k = 0; k < d; k++ )
          {
            TP tar = itargets[ k ];
            TP src = jsources[ k ];
            Kij += ( tar - src ) * ( tar - src );
          }
          Kij = exp( kernel.scal * Kij );
          break;
        }
        case KS_LAPLACE:
        {
          //T scale = 0.0;
          //if ( d == 1 ) scale = 1.0;
          //else if ( d == 2 ) scale = ( 0.5 / M_PI );
          //else
          //{
          //  scale = tgamma( 0.5 * d + 1.0 ) / ( d * ( d - 2 ) * pow( M_PI, 0.5 * d ) ); 
          //}
          for ( size_t k = 0; k < d; k++ )
          {
            /** Use high precision */
            TP tar = itargets[ k ];
            TP src = jsources[ k ];
            Kij += ( tar - src ) * ( tar - src );
          }

          /** Deal with the signularity (i.e. r ~ 0.0). */
          if ( Kij < 1E-6 ) Kij = 0.0;
          else
          {
            if ( d == 1 ) Kij = sqrt( Kij );
            else if ( d == 2 ) Kij = ( log( Kij ) / 2.0 );
            else Kij = 1 / sqrt( Kij );
          }
          break;
        }
        case KS_QUARTIC:
        {
          for ( size_t k = 0; k < d; k++ )
          {
            TP tar = itargets[ k ];
            TP src = jsources[ k ];
            Kij += ( tar - src ) * ( tar - src );
          }
          //if ( Kij < sqrt ( 2 ) / 2 ) Kij = 1;
          if ( Kij < 0.5 ) Kij = 1;
          else Kij = 0;
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
      }
      return Kij;
    };

    /** (Overwrittable) Request a single Kij */
    virtual T operator () ( size_t i, size_t j )
    {
      /** Return value */
      if ( is_symmetric )
      {
        return (*this)( sources_user.columndata( i ),
                        sources_user.columndata( j ) );
      }
      else
      {
        return (*this)( targets_user.columndata( i ),
                        sources_user.columndata( j ) );
      }
    }; /** end operator () */


    /** (Overwrittable) ESSENTIAL: return K( I, J ) */
    virtual Data<T> operator() ( vector<size_t> &I, 
                                 vector<size_t> &J )
    {
      Data<T> KIJ = GeometryDistances( I, J );
			/** Early return */
			if ( !I.size() || !J.size() ) return KIJ;

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          /** Apply the scaling factor and exponent */
          #pragma omp parallel for
          for ( size_t i = 0; i < KIJ.size(); i ++ )
          {
            KIJ[ i ] = std::exp( kernel.scal * KIJ[ i ] );
          }
          break;
        }
        case KS_LAPLACE:
        {
          //T scale = 0.0;

          //if ( d == 1 ) scale = 1.0;
          //else if ( d == 2 ) scale = ( 0.5 / M_PI );
          //else
          //{
          //  scale = tgamma( 0.5 * d + 1.0 ) / ( d * ( d - 2 ) * pow( M_PI, 0.5 * d ) ); 
          //}

          #pragma omp parallel for
          for ( size_t i = 0; i < KIJ.size(); i ++ )
          {
            if ( KIJ[ i ] < 1E-6 ) KIJ[ i ] = 0.0;
            else
            {
              if      ( d == 1 ) KIJ[ i ] = sqrt( KIJ[ i ] );
              else if ( d == 2 ) KIJ[ i ] = std::log( KIJ[ i ] ) / 2;
              else               KIJ[ i ] = 1 / sqrt( KIJ[ i ] );
            }
          }
          break;
        }
        case KS_QUARTIC:
        {
          #pragma omp parallel for
          for ( size_t i = 0; i < KIJ.size(); i ++ )
          {
            if ( KIJ[ i ] < 0.5 ) KIJ[ i ] = 1.0;
            else KIJ[ i ] = 0.0;
          }
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
      }

      /** There should be no NAN or INF value. */
      //assert( !KIJ.HasIllegalValue() );
      /** Return submatrix KIJ. */
      return KIJ;
    };


    /** */
    Data<T> GeometryDistances( const vector<size_t> &I, 
                               const vector<size_t> &J )
    {
      Data<T> KIJ( I.size(), J.size() );
			/** Early return if possible. */
			if ( !I.size() || !J.size() ) return KIJ;
			/** Request for coordinates: A (targets), B (sources). */
      Data<TP> A, B;
      /** For symmetry matrices, extract from sources. */
      if ( is_symmetric ) 
      {
        A = sources_user( all_dimensions, I );
        B = sources_user( all_dimensions, J );
      }
      /** For non-symmetry matrices, extract from targets and sources. */
      else
      {
        A = targets_user( all_dimensions, I );
        B = sources_user( all_dimensions, J );
      }
      /** Compute inner products. */
      xgemm( "Transpose", "No-transpose", I.size(), J.size(), d,
        -2.0, A.data(), A.row(),
              B.data(), B.row(),
         0.0, KIJ.data(), KIJ.row() );
      /** Compute square 2-norms. */
      vector<TP> A2( I.size() ), B2( J.size() );

      #pragma omp parallel for
      for ( size_t i = 0; i < I.size(); i ++ )
      {
        A2[ i ] = xdot( d, A.columndata( i ), 1, A.columndata( i ), 1 );
      } /** end omp parallel for */

      #pragma omp parallel for
      for ( size_t j = 0; j < J.size(); j ++ )
      {
        B2[ j ] = xdot( d, B.columndata( j ), 1, B.columndata( j ), 1 );
      } /** end omp parallel for */

      /** Add square norms to inner products to get square distances. */
      #pragma omp parallel for
      for ( size_t j = 0; j < J.size(); j ++ )
        for ( size_t i = 0; i < I.size(); i ++ )
          KIJ( i, j ) += A2[ i ] + B2[ j ];
      /** Return all pair-wise distances. */
      return KIJ;
    }; /** end GeometryDistances() */


    /** Get the diagonal of KII, i.e. diag( K( I, I ) ). */
    Data<T> Diagonal( vector<size_t> &I )
    {
      /** MPI */
      int size = this->Comm_size();
      int rank = this->Comm_rank();

      /** 
       *  return values
       * 
       *  NOTICE: even KIJ can be an 0-by-0 matrix for this MPI rank,
       *  yet early return is not allowed. All MPI process must execute
       *  all collaborative communication routines to avoid deadlock.
       */
      Data<T> DII( I.size(), 1, 0.0 );

      /** at this moment we already have the corrdinates on this process */
      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          for ( size_t i = 0; i < I.size(); i ++ ) DII[ i ] = 1.0;
          break;
        }
        case KS_LAPLACE:
        {
          for ( size_t i = 0; i < I.size(); i ++ ) DII[ i ] = 1.0;
          break;
        }
        case KS_QUARTIC:
        {
          for ( size_t i = 0; i < I.size(); i ++ ) DII[ i ] = 1.0;
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
      }

      return DII;

    }; /** end Diagonal() */


    /** Important sampling */
    pair<T, size_t> ImportantSample( size_t j )
    {
      size_t i = std::rand() % this->col();
      while( !sources_user.HasColumn( i ) ) i = std::rand() % this->col();
      assert( sources_user.HasColumn( i ) );
      pair<T, size_t> sample( 0, i );
      return sample; 
    };

    void ComputeDegree()
    {
      printf( "ComputeDegree(): K * ones\n" );
      assert( is_symmetric );
      Data<T> ones( this->row(), (size_t)1, 1.0 );
      Degree.resize( this->row(), (size_t)1, 0.0 );
      Multiply( 1, Degree, ones );
    };

    void Print()
    {
      for ( size_t j = 0; j < this->col(); j ++ )
      {
        printf( "%8lu ", j );
      }
      printf( "\n" );
      for ( size_t i = 0; i < this->row(); i ++ )
      {
        for ( size_t j = 0; j < this->col(); j ++ )
        {
          printf( "% 3.1E ", (*this)( i, j ) );
        }
        printf( "\n" );
      }
    }; /** end Print() */

    /** return number of attributes */
    size_t dim() { return d; };

    /** flops required for Kab */
    double flops( size_t na, size_t nb ) 
    {
      double flopcount = 0.0;

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          flopcount = na * nb * ( 2.0 * d + 35.0 );
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
      }
      return flopcount; 
    };

    void SendIndices( vector<size_t> ids, int dest, mpi::Comm comm )
    {
      auto param = sources_user( all_dimensions, ids );
      mpi::SendVector(   ids, dest, 90, comm );
      mpi::SendVector( param, dest, 90, comm );
    };

    void RecvIndices( int src, mpi::Comm comm, mpi::Status *status )
    {
      vector<size_t> ids;
      Data<TP> param;
      mpi::RecvVector(   ids, src, 90, comm, status );
      mpi::RecvVector( param, src, 90, comm, status );
      assert( param.size() == ids.size() * dim() );
      param.resize( dim(), param.size() / dim() );
      /** Insert into hash table */
      sources_user.InsertColumns( ids, param );
    };

    /** Bcast cids from sender for K( :, cids ) evaluation. */
    void BcastIndices( vector<size_t> ids, int root, mpi::Comm comm )
    {
      int rank; mpi::Comm_rank( comm, &rank );
      /** Bcast size of cids from root */
      size_t recv_size = ids.size();
      mpi::Bcast( &recv_size, 1, root, comm );
      /** Resize to receive cids and parameters */
      Data<TP> param;
      if ( rank == root )
      {
         param = sources_user( all_dimensions, ids );
      }
      else
      {
         ids.resize( recv_size );
         param.resize( dim(), recv_size );
      }
      /** Bcast cids and parameters from root */
      mpi::Bcast( ids.data(), recv_size, root, comm );
      mpi::Bcast( param.data(), dim() * recv_size, root, comm );
      /** Insert into hash table */
      sources_user.InsertColumns( ids, param );
    };

    void RequestIndices( vector<vector<size_t>> ids )
    {
      auto comm = this->GetComm();
      auto rank = this->GetCommRank();
      auto size = this->GetCommSize();

      assert( ids.size() == size );

      vector<vector<size_t>> recv_ids( size );
      vector<vector<TP>>     send_para( size );
      vector<vector<TP>>     recv_para( size );

      /** Send out cids request to each rank. */
      mpi::AlltoallVector( ids, recv_ids, comm );

      for ( int p = 0; p < size; p ++ )
      {
        Data<TP> para = sources_user( all_dimensions, recv_ids[ p ] );
        send_para[ p ].insert( send_para[ p ].end(), para.begin(), para.end() );
        //send_para[ p ] = sources_user( all_dimensions, recv_ids[ p ] );
      }

      /** Exchange out parameters. */
      mpi::AlltoallVector( send_para, recv_para, comm );

      for ( int p = 0; p < size; p ++ )
      {
        assert( recv_para[ p ].size() == dim() * ids[ p ].size() );
        if ( p != rank && ids[ p ].size() )
        {
          Data<TP> para;
          para.insert( para.end(), recv_para[ p ].begin(), recv_para[ p ].end() );
          para.resize( dim(), para.size() / dim() );
          sources_user.InsertColumns( ids[ p ], para );
        }
      }
    };

  private:

    bool is_symmetric = false;

    size_t d = 0;

    /** Legacy data structure for kernel matrices. */
    kernel_s<T> kernel;

    /** Pointers to user provided data points in block cylic distribution. */
    DistData<STAR, CBLK, TP> *sources = NULL;
    DistData<STAR, CBLK, TP> *targets = NULL;

    /** For local essential tree [LET]. */
    DistData<STAR, USER, TP> sources_user;
    DistData<STAR, USER, TP> targets_user;

    /** [ 0, 1, ..., d-1 ] */
    vector<size_t> all_dimensions;
}; /** end class DistKernelMatrix */

}; /** end namespace hmlp */

#endif /** define KERNELMATRIX_HPP */
