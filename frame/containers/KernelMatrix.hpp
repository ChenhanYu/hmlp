#ifndef KERNEL_HPP
#define KERNEL_HPP

/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif

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
    KernelMatrix( TINDEX m, TINDEX n, TINDEX d, kernel_s<T> &kernel, Data<T> &sources )
    : sources( sources ), targets( sources ), VirtualMatrix<T>( m, n )
    {
      this->d = d;
      this->kernel = kernel;
    };

    /** unsymmetric kernel matrix */
    template<typename TINDEX>
    KernelMatrix( TINDEX m, TINDEX n, TINDEX d, kernel_s<T> &kernel, Data<T> &sources, Data<T> &targets )
    : sources( sources ), targets( targets ), VirtualMatrix<T>( m, n )
    {
      this->d = d;
      this->kernel = kernel;
    };

    ~KernelMatrix() {};

    /** ESSENTIAL: return  K( i, j ) */
    //template<typename TINDEX>
    //T operator()( TINDEX i, TINDEX j )
    //{
    //  T Kij = 0.0;

    //  switch ( kernel.type )
    //  {
    //    case KS_GAUSSIAN:
    //    {
    //      for ( TINDEX k = 0; k < d; k++ )
    //      {
		//				T tar = targets[ i * d + k ];
		//				T src = sources[ j * d + k ];
    //        Kij += ( tar - src ) * ( tar - src );
    //      }
    //      Kij = exp( kernel.scal * Kij );
    //      break;
    //    }
    //    default:
    //    {
    //      printf( "invalid kernel type\n" );
    //      exit( 1 );
    //      break;
    //    }
    //  }

    //  return Kij;
    //};


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
          Kij = exp( kernel.scal * Kij );
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

      // Get coordinates of sources and targets
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
      std::vector<T> target_sqnorms( imap.size() );
      std::vector<T> source_sqnorms( jmap.size() );
      #pragma omp parallel for
      for ( TINDEX i = 0; i < imap.size(); i ++ )
      {
        target_sqnorms[ i ] = xdot
                              (
                                d,
                                itargets.data() + i * d, 1,
                                itargets.data() + i * d, 1
                              );
      }
      #pragma omp parallel for
      for ( TINDEX j = 0; j < jmap.size(); j ++ )
      {
        source_sqnorms[ j ] = xdot
                              (
                                d,
                                jsources.data() + j * d, 1,
                                jsources.data() + j * d, 1
                              );
      }

      /** add square norms to inner products to get pairwise square distances
       * */
      #pragma omp parallel for
      for ( TINDEX j = 0; j < jmap.size(); j ++ )
      {
        for ( TINDEX i = 0; i < imap.size(); i ++ )
        {
          submatrix[ j * imap.size() + i ] += target_sqnorms[ i ] + source_sqnorms[ j ];
        }
      }

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
          {
            // Apply the scaling factor and exponentiate
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

    /** important sampling */
    template<typename TINDEX>
    std::pair<T, TINDEX> ImportantSample( TINDEX j )
    {
      TINDEX i = std::rand() % this->col();
      std::pair<T, TINDEX> sample( (*this)( i, j ), i );
      return sample; 
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
    }; // end Print()

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

    std::size_t d;

    Data<T> &sources;

    Data<T> &targets;

    kernel_s<T> kernel;

}; /** end class KernelMatrix */

}; /** end namespace hmlp */

#endif /** define KERNEL_HPP */
