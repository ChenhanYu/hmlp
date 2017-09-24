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




#ifndef DISTKERNELMATRIX_HPP
#define DISTKERNELMATRIX_HPP

/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif

#include <hmlp_blas_lapack.h>

/** kernel matrix uses VirtualMatrix<T> as base */
#include <containers/DistVirtualMatrix.hpp>


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
class DistKernelMatrix : public DistVirtualMatrix<T, Allocator>, ReadWrite
{
  public:

    /** symmetric kernel matrix */
    DistKernelMatrix( 
        size_t m, size_t n, size_t d, 
        kernel_s<T> &kernel, 
        /** by default we assume sources are distributed in [STAR, CBLK] */
        DistData<STAR, CBLK, T> &sources, 
        mpi::Comm comm )
    : sources_sqnorms( 1, m, comm ), 
      targets_sqnorms( 1, n, comm ),
      DistVirtualMatrix<T>( m, n, comm )
    {
      assert( m == n );
      this->is_symmetric = true;
      this->d = d;
      this->kernel = kernel;
      this->sources = &sources;
      this->targets = &sources;
      /** compute square 2-norms for Euclidian family */
      ComputeSquare2Norm();
    };

    /** unsymmetric kernel matrix */
    DistKernelMatrix( 
        size_t m, size_t n, size_t d, 
        kernel_s<T> &kernel, 
        /** by default we assume sources are distributed in [STAR, CBLK] */
        DistData<STAR, CBLK, T> &sources, 
        /** by default we assume targets are distributed in [STAR, CBLK] */
        DistData<STAR, CBLK, T> &targets, 
        mpi::Comm comm )
    : sources_sqnorms( 1, m, comm ), 
      targets_sqnorms( 1, n, comm ),
      DistVirtualMatrix<T>( m, n, comm )
    {
      this->is_symmetric = false;
      this->d = d;
      this->kernel = kernel;
      this->sources = &sources;
      this->targets = &targets;
      /** compute square 2-norms for Euclidian family */
      ComputeSquare2Norm();
    };

    ~DistKernelMatrix() {};

    void ComputeSquare2Norm()
    {
      /** MPI */
      int size = this->Comm_size();
      int rank = this->Comm_rank();

      printf( "before ComputeSquare\n" ); fflush( stdout );

      /** compute 2-norm using xnrm2 */
      #pragma omp parallel for
      for ( size_t j = 0; j < sources->col_owned(); j ++ )
      {
        /** 
         *  notice that here we want to compute 2-norms only
         *  for the sources owned locally. In this case, we
         *  cannot use ( i, j ) operator. 
         */
        sources_sqnorms[ j ] = xnrm2( d, sources->data() + j * d, 1 ); 
      } /** end omp parallel for */


      printf( "after ComputeSquare\n" ); fflush( stdout );


      /** compute 2-norms for targets if unsymmetric */
      if ( is_symmetric ) 
      {
        /** directly copy from sources */
        targets_sqnorms = sources_sqnorms;
      }
      else
      {
        /** 
         *  targets are stored as "d-by-m" in the distributed
         *  fashion. We directly access the local data by pointers
         *  but not ( i, j ) operator.
         */
        #pragma omp parallel for
        for ( size_t i = 0; i < targets->col_owned(); i ++ )
        {
          targets_sqnorms[ i ] = xnrm2( d, targets->data() + i * d, 1 );
        } /** end omp paralllel for */
      }
    }; /** end ComputeSquare2Norm() */


	/** 
     *  ESSENTIAL: override the virtual function 
     *
     *  This operator "may" invoke MPI collective communication routines 
     *  and return a value Kij. Thus, all MPI processes in the communicator
     *  must invoke this operator to avoid dead lock.
     */
    T operator()( size_t i, size_t j )
    {
      /** MPI */
      int size = this->Comm_size();
      int rank = this->Comm_rank();

      /** return value */
      T Kij = 0.0;

      /** check if target i and source j are both owned by this MPI process */
      std::vector<std::vector<size_t>> sendrids( size );
      std::vector<std::vector<size_t>> recvrids( size );
      std::vector<std::vector<size_t>> sendcids( size );
      std::vector<std::vector<size_t>> recvcids( size );

      /** request Kij from rank ( j % size ) */
      sendrids[ i % size ].push_back( i );    
      sendcids[ j % size ].push_back( j );    

      /** exchange ids */
      mpi::AlltoallVector( sendrids, recvrids, this->GetComm() );
      mpi::AlltoallVector( sendcids, recvcids, this->GetComm() );

      /** allocate buffer for data */
      //std::vector<hmlp::Data<T>> sendsrcs( size );
      //std::vector<hmlp::Data<T>> recvsrcs( size );
      //std::vector<hmlp::Data<T>> sendtars( size );
      //std::vector<hmlp::Data<T>> recvtars( size );

      std::vector<std::vector<T>> sendsrcs( size );
      std::vector<std::vector<T>> recvsrcs( size );
      std::vector<std::vector<T>> sendtars( size );
      std::vector<std::vector<T>> recvtars( size );



      std::vector<size_t> amap( d );
      for ( size_t k = 0; k < d; k ++ ) amap[ k ] = k;

      for ( size_t p = 0; p < size; p ++ )
      {
        if ( recvcids[ p ].size() )
        {
          sendsrcs[ p ] = (*sources)( amap, recvcids[ p ] );
        }
        if ( recvrids[ p ].size() )
        {
          sendtars[ p ] = (*targets)( amap, recvrids[ p ] );
        }
      };

      /** exchange coordinates */
      mpi::AlltoallVector( sendsrcs, recvsrcs, this->GetComm() );
      mpi::AlltoallVector( sendtars, recvtars, this->GetComm() );

      std::vector<T> itargets( d );
      std::vector<T> jsources( d );

      for ( size_t p = 0; p < size; p ++ )
      {
        if ( recvsrcs[ p ].size() ) jsources = recvsrcs[ p ];
        if ( recvtars[ p ].size() ) itargets = recvtars[ p ];
      }

      /** at this moment we already have the corrdinates on this process */
      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          for ( size_t k = 0; k < d; k++ )
          {
            T tar = itargets[ k ];
            T src = jsources[ k ];
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


    /** ESSENTIAL: return K( I, J ) */
    hmlp::Data<T> operator() 
    ( 
      std::vector<size_t> &I, std::vector<size_t> &J 
    )
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
      hmlp::Data<T> KIJ( I.size(), J.size() );

      /** check if target i and source j are both owned by this MPI process */
      std::vector<std::vector<size_t>> sendrids( size );
      std::vector<std::vector<size_t>> recvrids( size );
      std::vector<std::vector<size_t>> sendcids( size );
      std::vector<std::vector<size_t>> recvcids( size );

      /** record the destination order */
      std::vector<std::vector<size_t>> imaprids( size );
      std::vector<std::vector<size_t>> jmapcids( size );

      /** 
       *  request Xi from rank ( I[ i ] % size ) 
       *  request Xj from rank ( J[ j ] % size ) 
       */
      for ( size_t i = 0; i < I.size(); i ++ )
      {
        size_t rid = I[ i ];
        sendrids[ rid % size ].push_back( rid );    
        imaprids[ rid % size ].push_back(   i );
      }
      for ( size_t j = 0; j < J.size(); j ++ )
      {
        size_t cid = J[ j ];
        sendcids[ cid % size ].push_back( cid );    
        jmapcids[ cid % size ].push_back(   j );
      }

      /** MPI: exchange ids */
      mpi::AlltoallVector( sendrids, recvrids, this->GetComm() );
      mpi::AlltoallVector( sendcids, recvcids, this->GetComm() );

      /** allocate buffer for data */
      std::vector<hmlp::Data<T>> sendsrcs( size );
      std::vector<hmlp::Data<T>> recvsrcs( size );
      std::vector<hmlp::Data<T>> sendtars( size );
      std::vector<hmlp::Data<T>> recvtars( size );

      std::vector<size_t> amap( d );
      for ( size_t k = 0; k < d; k ++ ) amap[ k ] = k;

      for ( size_t p = 0; p < size; p ++ )
      {
        if ( recvcids[ p ].size() )
        {
          sendsrcs[ p ] = (*sources)( amap, recvcids[ p ] );
        }
        if ( recvrids[ p ].size() )
        {
          sendtars[ p ] = (*targets)( amap, recvrids[ p ] );
        }
      };
      
      //printf( "Before AlltoallVector\n" ); fflush( stdout );


      /** exchange coordinates */
      mpi::AlltoallData( d, sendsrcs, recvsrcs, this->GetComm() );
      mpi::AlltoallData( d, sendtars, recvtars, this->GetComm() );

      //printf( "After AlltoallVector\n" ); fflush( stdout );


      /** allocate space for permuted coordinates*/
      hmlp::Data<T> itargets( d, I.size() );
      hmlp::Data<T> jsources( d, J.size() );

      /** TODO: reorder */
      for ( size_t p = 0; p < size; p ++ )
      {
        if ( recvtars[ p ].size() )
        {
          assert( recvtars[ p ].row() == d );
          assert( recvtars[ p ].col() == imaprids[ p ].size() );
        }

        if ( recvsrcs[ p ].size() )
        {
          assert( recvsrcs[ p ].row() == d );
          assert( recvsrcs[ p ].col() == jmapcids[ p ].size() );
        }


        /** use resize() to properly set [ m, n ] */
        //recvtars[ p ].resize( d, imaprids[ p ].size() );
        //recvsrcs[ p ].resize( d, jmapcids[ p ].size() );

        /** permute recvtars to itargets */
        for ( size_t j = 0; j < imaprids[ p ].size(); j ++ )
          for ( size_t i = 0; i < d; i ++ )
            //itargets( i, imaprids[ p ][ j ] ) = recvtars[ p ][ j * d + i ];
            itargets( i, imaprids[ p ][ j ] ) = recvtars[ p ]( i, j );

        /** permute recvsrcs to isources */
        for ( size_t j = 0; j < jmapcids[ p ].size(); j ++ )
          for ( size_t i = 0; i < d; i ++ )
            //jsources( i, jmapcids[ p ][ j ] ) = recvsrcs[ p ][ j * d + i ];
            jsources( i, jmapcids[ p ][ j ] ) = recvsrcs[ p ]( i, j );
      }

      //printf( "Before xgemm\n" ); fflush( stdout );


      /** compute inner products */
      xgemm
      (
        "Transpose", "No-transpose",
        I.size(), J.size(), d,
        -2.0, itargets.data(),   itargets.row(),
              jsources.data(),   jsources.row(),
         0.0,      KIJ.data(),        KIJ.row()
      );

      /** compute square norms */
      std::vector<T> itargets_sqnorms( I.size() );
      std::vector<T> jsources_sqnorms( J.size() );

      //printf( "After xgemm\n" ); fflush( stdout );

      #pragma omp parallel for
      for ( size_t i = 0; i < I.size(); i ++ )
      {
        itargets_sqnorms[ i ] = xnrm2( d, itargets.data() + i * d, 1 );
      } /** end omp parallel for */

      #pragma omp parallel for
      for ( size_t j = 0; j < J.size(); j ++ )
      {
        jsources_sqnorms[ j ] = xnrm2( d, jsources.data() + j * d, 1 );
      }

      /** add square norms to inner products to get square distances */
      #pragma omp parallel for
      for ( size_t j = 0; j < J.size(); j ++ )
        for ( size_t i = 0; i < I.size(); i ++ )
          KIJ( i, j ) += itargets_sqnorms[ i ] + jsources_sqnorms[ j ];

      switch ( kernel.type )
      {
        case KS_GAUSSIAN:
        {
          /** apply the scaling factor and exponentiate */
          #pragma omp parallel for
          for ( size_t i = 0; i < KIJ.size(); i ++ )
          {
            KIJ[ i ] = std::exp( kernel.scal * KIJ[ i ] );
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

      /** return submatrix KIJ */
      return KIJ;
    };


    /** get the diagonal of KII, i.e. diag( K( I, I ) ) */
    hmlp::Data<T> operator () ( std::vector<size_t> &I )
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
      hmlp::Data<T> DII( I.size(), 1, 0.0 );

      /** check if target i and source j are both owned by this MPI process */
      std::vector<std::vector<size_t>> sendrids( size );
      std::vector<std::vector<size_t>> recvrids( size );

      /** record the destination order */
      std::vector<std::vector<size_t>> imaprids( size );

      /** 
       *  request Xi from rank ( I[ i ] % size ) 
       */
      for ( size_t i = 0; i < I.size(); i ++ )
      {
        size_t rid = I[ i ];
        sendrids[ rid % size ].push_back( rid );    
        imaprids[ rid % size ].push_back(   i );
      }

      /** MPI: exchange ids */
      mpi::AlltoallVector( sendrids, recvrids, this->GetComm() );

      /** allocate buffer for data */
      std::vector<hmlp::Data<T>> sendtars( size );
      std::vector<hmlp::Data<T>> recvtars( size );

      std::vector<size_t> amap( d );
      for ( size_t k = 0; k < d; k ++ ) amap[ k ] = k;

      for ( size_t p = 0; p < size; p ++ )
      {
        if ( recvrids[ p ].size() )
        {
          sendtars[ p ] = (*targets)( amap, recvrids[ p ] );
        }
      };

      /** exchange coordinates */
      mpi::AlltoallData( d, sendtars, recvtars, this->GetComm() );

      /** allocate space for permuted coordinates*/
      hmlp::Data<T> itargets( d, I.size() );

      /** TODO: reorder */
      for ( size_t p = 0; p < size; p ++ )
      {
        if ( recvtars[ p ].size() )
        {
          assert( recvtars[ p ].row() == d );
          assert( recvtars[ p ].col() == imaprids[ p ].size() );
        }

        /** permute recvtars to itargets */
        for ( size_t j = 0; j < imaprids[ p ].size(); j ++ )
          for ( size_t i = 0; i < d; i ++ )
            itargets( i, imaprids[ p ][ j ] ) = recvtars[ p ]( i, j );
      }


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





    void Redistribute( 
        std::vector<size_t> &rids, 
        std::vector<size_t> &cids )
    {

    };

  private:

    bool is_symmetric = true;

    std::size_t d;

    /** legacy data structure */
    kernel_s<T> kernel;

    DistData<STAR, CBLK, T> *sources = NULL;

    DistData<STAR, CBLK, T> *targets = NULL;

    DistData<STAR, CBLK, T> sources_sqnorms;

    DistData<STAR, CBLK, T> targets_sqnorms;

    DistData<STAR, CIDS, T> *sources_cids = NULL;

    DistData<STAR, CIDS, T> *targets_cids = NULL;



    //Data<T> &sources;

    //Data<T> &targets;

    //std::vector<T> source_sqnorms;

    //std::vector<T> target_sqnorms;

}; /** end class DistKernelMatrix */
}; /** end namespace hmlp */

#endif /** define DISTKERNELMATRIX_HPP */
