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

#include <hmlp_blas_lapack.h>

/** kernel matrix uses VirtualMatrix<T> as base */
#include <containers/VirtualMatrix.hpp>


namespace hmlp
{

#ifdef HMLP_MIC_AVX512
template<typename T, class Allocator = hbw::allocator<T> >
#else
template<typename T, class Allocator = std::allocator<T> >
#endif
/**
 *  @brief
 */ 
class KernelMatrix : public VirtualMatrix<T, Allocator>, ReadWrite
{
  public:

    /** symmetric kernel matrix */
    template<typename TINDEX>
    KernelMatrix( TINDEX m, TINDEX n, TINDEX d, kernel_s<T> &kernel, 
        Data<T> &sources )
    : sources( sources ), targets( sources ), VirtualMatrix<T>( m, n )
    {
      assert( m == n );
      this->is_symmetric = true;
      this->d = d;
      this->kernel = kernel;
      /** compute square 2-norms for Euclidian family */
      ComputeSquare2Norm();
    };

    /** unsymmetric kernel matrix */
    template<typename TINDEX>
    KernelMatrix( TINDEX m, TINDEX n, TINDEX d, kernel_s<T> &kernel, 
        Data<T> &sources, Data<T> &targets )
    : sources( sources ), targets( targets ), VirtualMatrix<T>( m, n )
    {
      this->is_symmetric = false;
      this->d = d;
      this->kernel = kernel;
      /** compute square 2-norms for Euclidian family */
      ComputeSquare2Norm();
    };


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
    T operator()( std::size_t i, std::size_t j )
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


    /** ESSENTIAL: return K( imap, jmap ) */
    template<typename TINDEX>
    hmlp::Data<T> operator()
		  ( std::vector<TINDEX> &imap, std::vector<TINDEX> &jmap )
    {
      hmlp::Data<T> submatrix( imap.size(), jmap.size() );

      if ( !submatrix.size() ) return submatrix;

      /** Get coordinates of sources and targets */
      hmlp::Data<T> itargets = targets( imap );
      hmlp::Data<T> jsources = sources( jmap );

      assert( itargets.col() == submatrix.row() );
      assert( itargets.row() == d );
      assert( jsources.col() == submatrix.col() );
      assert( jsources.row() == d );

      /** compute inner products */
      xgemm
      (
        "T", "N",
        imap.size(), jmap.size(), d,
        -2.0, itargets.data(),   itargets.row(),
              jsources.data(),   jsources.row(),
         0.0, submatrix.data(), submatrix.row()
      );

      /** compute square norms */
      std::vector<T> itarget_sqnorms( imap.size() );
      std::vector<T> jsource_sqnorms( jmap.size() );


      #pragma omp parallel for
      for ( TINDEX i = 0; i < imap.size(); i ++ )
      {
        if ( target_sqnorms.size() )
        {
          /** if precomputed then directly copy */
          itarget_sqnorms[ i ] = target_sqnorms[ imap[ i ] ];
        }
        else
        {
          /** otherwise compute them now */
          itarget_sqnorms[ i ] = xdot(
              d, itargets.data() + i * d, 1, itargets.data() + i * d, 1 );
        }
      }

      #pragma omp parallel for
      for ( TINDEX j = 0; j < jmap.size(); j ++ )
      {
        if ( source_sqnorms.size() )
        {
          /** if precomputed then directly copy */
          jsource_sqnorms[ j ] = source_sqnorms[ jmap[ j ] ];
        }
        else
        {
          /** otherwise compute them now */
          jsource_sqnorms[ j ] = xdot(
              d, jsources.data() + j * d, 1, jsources.data() + j * d, 1 );
        }
      }

      /** add square norms to inner products to get square distances */
      #pragma omp parallel for
      for ( TINDEX j = 0; j < jmap.size(); j ++ )
      {
        for ( TINDEX i = 0; i < imap.size(); i ++ )
        {
          submatrix( i, j ) += itarget_sqnorms[ i ] + jsource_sqnorms[ j ];
        }
      }

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          /** apply the scaling factor and exponentiate */
          #pragma omp parallel for
          for ( TINDEX i = 0; i < submatrix.size(); i ++ )
          {
            submatrix[ i ] = std::exp( kernel.scal * submatrix[ i ] );
          }

          // gemm: 2 * i * j * d
          // compute sqnorms: 2 * ( i + j ) * d
          // add sqnorms: 2 * i * j
          // scale and exponentiate: 2 * i * j
          //flopcount += 2 * ( imap.size() * jmap.size() + imap.size() + jmap.size() ) * d
          //           + 4 * imap.size() * jmap.size();
          break;
        }
        default:
        {
          printf( "invalid kernel type\n" );
          exit( 1 );
          break;
        }
      }

      return submatrix;
    }; 



    /** get the diagonal of KII, i.e. diag( K( I, I ) ) */
    hmlp::Data<T> Diagonal( std::vector<size_t> &I )
    {
      /** 
       *  return values
       */
      hmlp::Data<T> DII( I.size(), 1, 0.0 );

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
    template<typename TINDEX>
    std::pair<T, TINDEX> ImportantSample( TINDEX j )
    {
      TINDEX i = std::rand() % this->col();
      std::pair<T, TINDEX> sample( (*this)( i, j ), i );
      return sample; 
    };

    void ComputeDegree()
    {
      printf( "ComputeDegree(): K * ones\n" );
      assert( is_symmetric );
      hmlp::Data<T> ones( this->row(), (size_t)1, 1.0 );
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
      hmlp::Data<T> invDw( this->row(), (size_t)1, 0.0 );

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
      std::vector<int> amap( this->row() );
      std::vector<int> bmap( this->col() );
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
        std::vector<T> &u, std::vector<TINDEX> &umap, 
                           std::vector<TINDEX> &amap,
                           std::vector<TINDEX> &bmap,
        std::vector<T> &w, std::vector<TINDEX> &wmap )
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
        std::vector<T> &u,
                           std::vector<TINDEX> &amap,
                           std::vector<TINDEX> &bmap,
        std::vector<T> &w )
    {
      Multiply( nrhs, u, amap, amap, bmap, w, bmap );
    };


    /** u += K * w */
    void Multiply( size_t nrhs, std::vector<T> &u, std::vector<T> &w )
    {
      std::vector<int> amap( this->row() );
      std::vector<int> bmap( this->col() );
      for ( size_t i = 0; i < amap.size(); i ++ ) amap[ i ] = i;
      for ( size_t j = 0; j < bmap.size(); j ++ ) bmap[ j ] = j;
      Multiply( nrhs, u, amap, bmap, w );
    };

    void Multiply( hmlp::Data<T> &u, hmlp::Data<T> &w )
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

    /** return number of attributes */
    std::size_t dim() { return d; };

    /** flops required for Kab */
    template<typename TINDEX>
    double flops( TINDEX na, TINDEX nb ) 
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

    std::size_t d;

    Data<T> &sources;

    Data<T> &targets;

    /** Degree (diagonal matrix) */
    Data<T> Degree;

    std::vector<T> source_sqnorms;

    std::vector<T> target_sqnorms;

    /** legacy data structure */
    kernel_s<T> kernel;

}; /** end class KernelMatrix */







}; /** end namespace hmlp */

#endif /** define KERNELMATRIX_HPP */
