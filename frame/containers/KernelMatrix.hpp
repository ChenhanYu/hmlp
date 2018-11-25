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

/** Using tgamma, M_PI, M_SQRT2 ... */
#include <cmath>
/** BLAS/LAPACK support. */
#include <hmlp_blas_lapack.h>
/** KernelMatrix uses VirtualMatrix<T> as base. */
#include <containers/VirtualMatrix.hpp>
/** DistData is used to store the data points. */
#include <DistData.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{

typedef enum
{
  GAUSSIAN,
  SIGMOID,
  POLYNOMIAL,
  LAPLACE,
  GAUSSIAN_VAR_BANDWIDTH,
  TANH,
  QUARTIC,
  MULTIQUADRATIC,
  EPANECHNIKOV,
  USER_DEFINE
} kernel_type;

template<typename T, typename TP>
struct kernel_s
{
  kernel_type type;

  /** Compute a single inner product. */
  static inline T innerProduct( const TP* x, const TP* y, size_t d )
  {
    T accumulator = 0.0;
    #pragma omp parallel for reduction(+:accumulator)
    for ( size_t i = 0; i < d; i ++ ) accumulator += x[ i ] * y[ i ];
    return accumulator;
  }

  /** Compute all pairwise inner products using GEMM_TN( 1.0, X, Y, 0.0, K ). */
  static inline void innerProducts( const TP* X, const TP* Y, size_t d, T* K, size_t m, size_t n )
  {
    /** This BLAS function is defined in frame/base/blas_lapack.hpp.  */
    xgemm( "Transpose", "No-transpose", (int)m, (int)n, (int)d, (T)1.0, X, (int)d, Y, (int)d, (T)0.0, K, (int)m ); 
  }

  /** Compute a single squared distance. */
  static inline T squaredDistance( const TP* x, const TP* y, size_t d )
  {
    T accumulator = 0.0;
    #pragma omp parallel for reduction(+:accumulator)
    for ( size_t i = 0; i < d; i ++ ) accumulator += ( x[ i ] - y[ i ] ) * ( x[ i ] - y[ i ] );
    return accumulator;
  }

  /** Compute all pairwise squared distances. */
  static inline void squaredDistances( const TP* X, const TP* Y, size_t d, T* K, size_t m, size_t n )
  {
    innerProducts( X, Y, d, K, m, n );
    vector<T> squaredNrmX( m, 0 ), squaredNrmY( n, 0 );
    /** This BLAS function is defined in frame/base/blas_lapack.hpp.  */
    #pragma omp parallel for
    for ( size_t i = 0; i < m; i ++ ) squaredNrmX[ i ] = xdot( d, X + i * d, 1, X + i * d, 1 );
    #pragma omp parallel for
    for ( size_t j = 0; j < n; j ++ ) squaredNrmY[ j ] = xdot( d, Y + j * d, 1, Y + j * d, 1 );
    #pragma omp parallel for collapse(2)
    for ( size_t j = 0; j < n; j ++ )
      for ( size_t i = 0; i < m; i ++ )
        K[ j * m + i ] = squaredNrmX[ i ] - 2 * K[ j * m + i ] + squaredNrmY[ j ]; 
  }

  inline T operator () ( const void* param, const TP* x, const TP* y, size_t d ) const
  {
    switch ( type )
    {
      case GAUSSIAN:
        return exp( scal * squaredDistance( x, y, d ) );
      case SIGMOID:
        return tanh( scal* innerProduct( x, y, d ) + cons );
      case LAPLACE:
        return 0;
      case QUARTIC:
        return 0;
      case USER_DEFINE:
        return user_element_function( param, x, y, d );
      default:
        printf( "invalid kernel type\n" );
        exit( 1 );
    } /** end switch ( type ) */
  };

  /** X should be at least d-by-m, and Y should be at least d-by-n. */
  inline void operator () ( const void* param, const TP* X, const TP* Y, size_t d, T* K, size_t m, size_t n ) const
  {
    switch ( type )
    {
      case GAUSSIAN:
        squaredDistances( X, Y, d, K, m, n );
        #pragma omp parallel for
        for ( size_t i = 0; i < m * n; i ++ ) K[ i ] = exp( scal * K[ i ] );
        break;
      case SIGMOID:
        innerProducts( X, Y, d, K, m, n );
        #pragma omp parallel for
        for ( size_t i = 0; i < m * n; i ++ ) K[ i ] = tanh( scal * K[ i ] + cons );
        break;
      case LAPLACE:
        break;
      case QUARTIC:
        break;
      case USER_DEFINE:
        user_matrix_function( param, X, Y, d, K, m, n );
        break;
      default:
        printf( "invalid kernel type\n" );
        exit( 1 );
    } /** end switch ( type ) */
  };

  T powe = 1;
  T scal = 1;
  T cons = 0;
  T *hi;
  T *hj;
  T *h;
  
  /** User-defined kernel functions. */
  T (*user_element_function)( const void* param, const TP* x, const TP* y, size_t d ) = nullptr;
  void (*user_matrix_function)( const void* param, const T* X, const T* Y, size_t d, T* K,  size_t m, size_t n ) = nullptr;

};


template<typename T, class Allocator = std::allocator<T>>
class KernelMatrix : public VirtualMatrix<T, Allocator>, 
                     public ReadWrite
{
  public:

    /** (Default) constructor for non-symmetric kernel matrices. */
    KernelMatrix( size_t m_, size_t n_, size_t d_, kernel_s<T, T> &kernel_, 
        Data<T> &sources_, Data<T> &targets_ )
      : VirtualMatrix<T>( m_, n_ ), d( d_ ), 
        sources( sources_ ), targets( targets_ ), 
        kernel( kernel_ ), all_dimensions( d_ )
    {
      this->is_symmetric = false;
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
    };

    /** (Default) constructor for symmetric kernel matrices. */
    KernelMatrix( size_t m, size_t n, size_t d, kernel_s<T, T>& kernel, Data<T> &sources )
      : KernelMatrix( m, n, d, kernel, sources, sources )
    {
      assert( m == n );
      this->is_symmetric = true;
    };

    KernelMatrix( Data<T> &sources )
      : sources( sources ), 
        targets( sources ), 
        VirtualMatrix<T>( sources.col(), sources.col() ),
        all_dimensions( sources.row() )
    {
      this->is_symmetric = true;
      this->d = sources.row();
      this->kernel.type = GAUSSIAN;
      this->kernel.scal = -0.5;
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
    };

    /** (Default) destructor. */
    ~KernelMatrix() {};

		/** ESSENTIAL: override the virtual function */
    virtual T operator()( size_t i, size_t j ) override
    {
      return kernel( nullptr, targets.columndata( i ), sources.columndata( j ), d );
		};

    /** (Overwrittable) ESSENTIAL: return K( I, J ) */
    virtual Data<T> operator() ( const vector<size_t>& I, const vector<size_t>& J ) override
    {
      Data<T> KIJ( I.size(), J.size() );
			/** Early return if possible. */
			if ( !I.size() || !J.size() ) return KIJ;
			/** Request for coordinates: A (targets), B (sources). */
      Data<T> X = ( is_symmetric ) ? sources( all_dimensions, I ) : targets( all_dimensions, I );
      Data<T> Y = sources( all_dimensions, J );
      /** Evaluate KIJ using legacy interface. */
      kernel( nullptr, X.data(), Y.data(), d, KIJ.data(), I.size(), J.size() );
      /** Return K( I, J ). */
      return KIJ;
    };


    virtual Data<T> GeometryDistances( const vector<size_t>& I, const vector<size_t>& J ) override
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
        case GAUSSIAN:
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
        case GAUSSIAN:
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

    size_t d = 0;

    Data<T> &sources;

    Data<T> &targets;

    /** legacy data structure */
    kernel_s<T, T> kernel;
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
    : all_dimensions( d ), sources_user( sources ), targets_user( targets ),
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
      kernel_s<T, TP> &kernel, 
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
      kernel_s<T, TP> &kernel,
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
      this->kernel.type = GAUSSIAN;
      this->kernel.scal = -0.5;
    };

    /** (Default) destructor */
    ~DistKernelMatrix() {};

    /** (Overwrittable) Request a single Kij */
    virtual T operator () ( size_t i, size_t j ) override
    {
      TP* x = ( is_symmetric ) ? sources_user.columndata( i ) : targets_user.columndata( i );
      TP* y = sources_user.columndata( j );
      return kernel( nullptr, x, y, d );
    }; /** end operator () */


    /** (Overwrittable) ESSENTIAL: return K( I, J ) */
    virtual Data<T> operator() ( const vector<size_t>& I, const vector<size_t>& J ) override
    {
      Data<T> KIJ( I.size(), J.size() );
			/** Early return if possible. */
			if ( !I.size() || !J.size() ) return KIJ;
			/** Request for coordinates: A (targets), B (sources). */
      Data<T> X = ( is_symmetric ) ? sources_user( all_dimensions, I ) : targets_user( all_dimensions, I );
      Data<T> Y = sources_user( all_dimensions, J );
      kernel( nullptr, X.data(), Y.data(), d, KIJ.data(), I.size(), J.size() );
      return KIJ;
    };


    /** */
    virtual Data<T> GeometryDistances( const vector<size_t>& I, const vector<size_t>& J ) override
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
        case GAUSSIAN:
        {
          for ( size_t i = 0; i < I.size(); i ++ ) DII[ i ] = 1.0;
          break;
        }
        case LAPLACE:
        {
          for ( size_t i = 0; i < I.size(); i ++ ) DII[ i ] = 1.0;
          break;
        }
        case QUARTIC:
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
        case GAUSSIAN:
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

    void RequestIndices( const vector<vector<size_t>>& ids ) override
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
    kernel_s<T, TP> kernel;

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
