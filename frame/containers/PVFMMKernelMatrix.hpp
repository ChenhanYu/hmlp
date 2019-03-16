#ifndef _PVFMM_HPP_
#define _PVFMM_HPP_

#define PVFMM_HAVE_BLAS 1
#define PVFMM_HAVE_LAPACK 1

// Parameters for memory manager
#define PVFMM_MEM_ALIGN 64
#define PVFMM_GLOBAL_MEM_BUFF 0LL  // in MB
//#ifndef NDEBUG
//#define PVFMM_MEMDEBUG // Enable memory checks.
//#endif

// Memory Manager, Iterators
#include <pvfmm/mem_mgr.hpp>

// Vector
#include <pvfmm/vector.hpp>

// Matrix, Permutation operators
#include <pvfmm/matrix.hpp>

// Template vector intrinsics
#include <pvfmm/intrin_wrapper.hpp>

// Kernel Matrix
#include <pvfmm/kernel_matrix.hpp>


#include <containers/Cache.hpp>
/** PVFMMKernelMatrix<T> uses VirtualMatrix<T> as base */
#include <containers/VirtualMatrix.hpp>
/** For GOFMM compatability */
#include <containers/SPDMatrix.hpp>

namespace hmlp
{

template <typename T> 
class PVFMMKernelMatrix : public VirtualMatrix<T>
{

  public:

    PVFMMKernelMatrix( size_t M, size_t N ) 
			: K( M, N ), VirtualMatrix<T>( M, N )//, cache( M )
	  {
			/** assertion */
      PVFMM_ASSERT( M == N );

			/** now compute and store all diagonal entries */
			D.resize( M, 1 );
			for ( size_t i = 0; i < M; i ++ ) D[ i ] = (*this)( i, i );
    };

    virtual T operator() ( size_t i, size_t j ) 
		{
			//auto KIJ = cache.Read( i, j );
			Data<T> KIJ;
			/** if cache missed */
			if ( !KIJ.size() ) 
			{
				KIJ.resize( 1, 1 );
				double beg = omp_get_wtime();
				KIJ[ 0 ] = K( i, j );
				double KIJ_t = omp_get_wtime() - beg;
        //cache.Write( i, j, std::pair<T, double>( KIJ[ 0 ], KIJ_t ) );
			}
			return KIJ[ 0 ];
    };

    virtual Data<T> operator() ( std::vector<size_t> &I, std::vector<size_t> &J ) 
		{
      Data<T> KIJ( I.size(), J.size() );

      double beg = omp_get_wtime();
      for ( size_t j = 0; j < J.size(); j ++ ) 
			{
        for ( size_t i = 0; i < I.size(); i++ ) 
				{
					if ( I[ i ] == J[ j ] ) 
						KIJ( i, j ) = D[ I[ i ] ];
					else                    
						KIJ( i, j ) = (*this)( I[ i ], J[ j ] );
        }
      }
			double KIJ_t = omp_get_wtime() - beg;

			if ( KIJ_t > 1.0 )
			{
			  printf( "KIJ %lu-by-%lu in %lfs, avg %lfs per element\n", 
				  	I.size(), J.size(), KIJ_t,
					  KIJ_t / ( I.size() * J.size() ) ); fflush( stdout );
			}

      return KIJ;
    };


		virtual Data<T> Diagonal( std::vector<size_t> &I )
		{
			Data<T> DII( I.size(), 1, 0.0 );
			for ( size_t i = 0; i < I.size(); i ++ ) DII[ i ] = D[ I[ i ] ];
			return DII;
		};

  private:

    pvfmm::KernelMatrix<T> K;

		/** precompute and store all diagonal entries */
		Data<T> D;

		//Cache2D<128, 16384, T> cache;
};

}; /** end namespace hmlp */


#endif //_PVFMM_HPP_
