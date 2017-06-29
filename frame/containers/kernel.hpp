#ifndef KERNEL_HPP
#define KERNEL_HPP

/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif // ifdef HMLP}_MIC_AVX512


namespace hmlp
{

#ifdef HMLP_MIC_AVX512
template<typename T, class Allocator = hbw::allocator<T> >
#else
template<typename T, class Allocator = std::allocator<T> >
#endif
class Kernel : public ReadWrite
{
  public:

    /** symmetric kernel matrix */
    template<typename TINDEX>
    Kernel( TINDEX m, TINDEX n, TINDEX d, kernel_s<T> &kernel, Data<T> &sources )
    : sources( sources ), targets( sources )
    {
      this->m = m;
      this->n = n;
      this->d = d;
      this->kernel = kernel;
    };

    /** unsymmetric kernel matrix */
    template<typename TINDEX>
    Kernel( TINDEX m, TINDEX n, TINDEX d, kernel_s<T> &kernel, Data<T> &sources, Data<T> &targets )
    : sources( sources ), targets( targets )
    {
      this->m = m;
      this->n = n;
      this->d = d;
      this->kernel = kernel;
    };

    ~Kernel() {};

    /** ESSENTIAL: return  K( i, j ) */
    template<typename TINDEX>
    inline T operator()( TINDEX i, TINDEX j )
    {
      T Kij = 0.0;

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          for ( TINDEX k = 0; k < d; k++ )
          {
            Kij += std::pow( targets[ i * d + k] - sources[ j * d + k ], 2 );
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
    inline hmlp::Data<T> operator()( std::vector<TINDEX> &imap, std::vector<TINDEX> &jmap )
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
      TINDEX i = std::rand() % m;
      std::pair<T, TINDEX> sample( (*this)( i, j ), i );
      return sample; 
    };

    void Print()
    {
      for ( size_t j = 0; j < n; j ++ )
      {
        printf( "%8lu ", j );
      }
      printf( "\n" );
      for ( size_t i = 0; i < m; i ++ )
      {
        for ( size_t j = 0; j < n; j ++ )
        {
          printf( "% 3.1E ", (*this)( i, j ) );
        }
        printf( "\n" );
      }
    }; // end Print()

    /** ESSENTIAL: return number of rows */
    std::size_t row() { return m; };

    /** ESSENTIAL: return number of columns */
    std::size_t col() { return n; };

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

    /** ESSENTIAL */
    std::size_t m;

    /** ESSENTIAL */
    std::size_t n;

    std::size_t d;

    Data<T> &sources;

    Data<T> &targets;

    kernel_s<T> kernel;

}; /** end class Kernel */

}; /** end namespace hmlp */

#endif /** define KERNEL_HPP */
