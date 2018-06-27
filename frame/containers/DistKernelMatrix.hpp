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




#ifndef DISTKERNELMATRIX_VER2_HPP
#define DISTKERNELMATRIX_VER2_HPP

#ifdef USE_INTEL
#include <mkl.h>
#endif

/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif

#include <hmlp_blas_lapack.h>
#include <hmlp_runtime.hpp>

/** Use software cache */
#include <containers/Cache.hpp>

/** Use VirtualMatrix<T> as base */
#include <containers/DistVirtualMatrix.hpp>


/** use STL templates */
using namespace std;


namespace hmlp
{

#ifdef HMLP_MIC_AVX512
template<typename T, typename TP, class Allocator = hbw::allocator<TP> >
#else
template<typename T, typename TP, class Allocator = std::allocator<TP> >
#endif
/**
 *  @brief
 */ 
class DistKernelMatrix_ver2 : public DistVirtualMatrix<T, Allocator>, 
                              public ReadWrite
{
  public:

    /** (Default) unsymmetric kernel matrix */
    DistKernelMatrix_ver2
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
    DistKernelMatrix_ver2
    ( 
      size_t m, size_t n, size_t d,
      kernel_s<T> &kernel, 
      /** by default we assume sources are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &sources, 
      /** by default we assume targets are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &targets, 
      mpi::Comm comm 
    )
    : DistKernelMatrix_ver2( m, n, d, sources, targets, comm ) 
    {
      this->kernel = kernel;
    };

    /** (Default) symmetric kernel matrix */
    DistKernelMatrix_ver2
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
    DistKernelMatrix_ver2
    ( 
      size_t n, size_t d, 
      kernel_s<T> &kernel,
      /** by default we assume sources are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, TP> &sources, 
      mpi::Comm comm 
    )
    : DistKernelMatrix_ver2( n, d, sources, comm )
    {
      this->kernel = kernel;
    };





    /** (Default) destructor */
    ~DistKernelMatrix_ver2() {};



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


    virtual Data<T> PairwiseDistances( const vector<size_t> &I, 
                                       const vector<size_t> &J )
    {
      /** 
       *  Return values
       * 
       *  NOTICE: even KIJ can be an 0-by-0 matrix for this MPI rank,
       *  yet early return is not allowed. All MPI process must execute
       *  all collaborative communication routines to avoid deadlock.
       */
      Data<T> KIJ( I.size(), J.size() );

			/** Early return. */
			if ( !I.size() || !J.size() ) return KIJ;

			/** Request for coordinates. */
      Data<TP> itargets, jsources;

      if ( is_symmetric ) 
      {
        itargets = sources_user( all_dimensions, I );
        jsources = sources_user( all_dimensions, J );
      }
      else
      {
        itargets = targets_user( all_dimensions, I );
        jsources = sources_user( all_dimensions, J );
      }

      /** Compute inner products. */
      xgemm ( "Transpose", "No-transpose", I.size(), J.size(), d,
        -2.0, itargets.data(),   itargets.row(),
              jsources.data(),   jsources.row(),
         0.0,      KIJ.data(),        KIJ.row() );

      /** Compute square norms. */
      vector<T> itargets_sqnorms( I.size() );
      vector<T> jsources_sqnorms( J.size() );

      #pragma omp parallel for
      for ( size_t i = 0; i < I.size(); i ++ )
      {
        itargets_sqnorms[ i ] = xdot( 
						d, itargets.data() + i * d, 1,
						   itargets.data() + i * d, 1 );
      } /** end omp parallel for */

      #pragma omp parallel for
      for ( size_t j = 0; j < J.size(); j ++ )
      {
        jsources_sqnorms[ j ] = xdot( 
						d, jsources.data() + j * d, 1, 
					     jsources.data() + j * d, 1 );
      } /** end omp parallel for */

      /** Add square norms to inner products to get square distances */
      #pragma omp parallel for
      for ( size_t j = 0; j < J.size(); j ++ )
        for ( size_t i = 0; i < I.size(); i ++ )
          KIJ( i, j ) += itargets_sqnorms[ i ] + jsources_sqnorms[ j ];

      return KIJ;
    };




    /** (Overwrittable) ESSENTIAL: return K( I, J ) */
    virtual Data<T> operator() ( vector<size_t> &I, 
                                 vector<size_t> &J )
    {
      /** 
       *  Return values
       * 
       *  NOTICE: even KIJ can be an 0-by-0 matrix for this MPI rank,
       *  yet early return is not allowed. All MPI process must execute
       *  all collaborative communication routines to avoid deadlock.
       */
      Data<T> KIJ = PairwiseDistances( I, J );

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
          for ( size_t i = 0; i < KIJ.size(); i ++ )
          {
            //if ( KIJ[ i ] < sqrt( 2 ) / 2 ) KIJ[ i ] = 1.0;
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
    template<typename TINDEX>
    pair<T, TINDEX> ImportantSample( TINDEX j )
    {
      TINDEX i = std::rand() % this->col();
      while( !sources_user.HasColumn( i ) ) i = std::rand() % this->col();
      assert( sources_user.HasColumn( i ) );
      pair<T, TINDEX> sample( 0, i );
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

    virtual void SendColumns( vector<size_t> cids, int dest, mpi::Comm comm )
    {
      int comm_rank; mpi::Comm_rank( comm, &comm_rank );
      //printf( "rank[%d %d] SendColumns\n", comm_rank, this->Comm_rank() ); fflush( stdout );
      Data<TP> send_para = sources_user( all_dimensions, cids );
      mpi::SendVector(      cids, dest, 90, comm );
      mpi::SendVector( send_para, dest, 90, comm );
      //printf( "rank[%d %d] end SendColumns\n", comm_rank, this->Comm_rank() ); fflush( stdout );
    };

    virtual void RecvColumns( int root, mpi::Comm comm, mpi::Status *status )
    {
      int comm_rank; mpi::Comm_rank( comm, &comm_rank );
      //printf( "rank[%d %d] RecvColumns\n", comm_rank, this->Comm_rank() ); fflush( stdout );
      vector<size_t> cids;
      Data<TP> recv_para;
      mpi::RecvVector(      cids, root, 90, comm, status );
      mpi::RecvVector( recv_para, root, 90, comm, status );
      assert( recv_para.size() == cids.size() * dim() );
      recv_para.resize( dim(), recv_para.size() / dim() );
      /** Insert into hash table */
      sources_user.InsertColumns( cids, recv_para );
      //printf( "rank[%d %d] end RecvColumns\n", comm_rank, this->Comm_rank() ); fflush( stdout );
    };

    /** Bcast cids from sender for K( :, cids ) evaluation. */
    virtual void BcastColumns( vector<size_t> cids, int root, mpi::Comm comm )
    {
      int comm_rank; mpi::Comm_rank( comm, &comm_rank );

      /** Bcast size of cids from root */
      size_t recv_size = cids.size();
      mpi::Bcast( &recv_size, 1, root, comm );
      
      //printf( "rank %d, before Bcast root %d\n", comm_rank, root ); fflush( stdout );
      mpi::Barrier( comm );
      /** Resize to receive cids and parameters */
      Data<TP> recv_para;
      if ( comm_rank == root )
      {
         recv_para = sources_user( all_dimensions, cids );
      }
      else
      {
         cids.resize( recv_size );
         recv_para.resize( dim(), recv_size );
      }
      //printf( "rank %d, after Bcast root %d\n", comm_rank, root ); fflush( stdout );
      mpi::Barrier( comm );

      /** Bcast cids and parameters from root */
      mpi::Bcast( cids.data(), recv_size, root, comm );
      mpi::Bcast( recv_para.data(), dim() * recv_size, root, comm );

      /** Insert into hash table */
      sources_user.InsertColumns( cids, recv_para );

    };

    virtual void RequestColumns( vector<vector<size_t>> requ_cids )
    {
      mpi::Comm comm = this->GetComm();
      int comm_rank = this->Comm_rank();
      int comm_size = this->Comm_size();

      assert( requ_cids.size() == comm_size );

      vector<vector<size_t>> recv_cids( comm_size );
      vector<vector<TP>>     send_para( comm_size );
      vector<vector<TP>>     recv_para( comm_size );

      /** Send out cids request to each rank. */
      mpi::AlltoallVector( requ_cids, recv_cids, comm );
      //printf( "Finish all2allv requ_cids\n" ); fflush( stdout );

      for ( int p = 0; p < comm_size; p ++ )
      {
        Data<TP> para = sources_user( all_dimensions, recv_cids[ p ] );
        send_para[ p ].insert( send_para[ p ].end(), para.begin(), para.end() );
      }

      /** Send out parameters */
      mpi::AlltoallVector( send_para, recv_para, comm );
      //printf( "Finish all2allv send_para\n" ); fflush( stdout );

      for ( int p = 0; p < comm_size; p ++ )
      {
        assert( recv_para[ p ].size() == dim() * requ_cids[ p ].size() );
        if ( p != comm_rank && requ_cids[ p ].size() )
        {
          Data<TP> para;
          para.insert( para.end(), recv_para[ p ].begin(), recv_para[ p ].end() );
          para.resize( dim(), para.size() / dim() );
          sources_user.InsertColumns( requ_cids[ p ], para );
        }
      }

    };





    //virtual void Redistribute( bool enforce_ordered, vector<size_t> &cids )
    //{
    //}; /** end Redistribute() */

    

    /** 
     *  P2P exchange parameters (or points) with partners in the local
     *  communicator.
     */
    virtual void RedistributeWithPartner
    (
      vector<size_t> &gids, vector<size_t> &lhs, vector<size_t> &rhs, 
      mpi::Comm comm 
    )
    {
      mpi::Status status;
      int loc_size; mpi::Comm_size( comm, &loc_size );
      int loc_rank; mpi::Comm_rank( comm, &loc_rank );

      //printf( "RedistributeWithPartner loc_rank %d loc_size %d glb_size %d\n", 
      //    loc_rank, loc_size, this->Comm_size() ); fflush( stdout );


      /** 
       *  At the root level, starts from sources<STAR, CBLK>. We use
       *  dynamic_sources as buffer space.
       */
      if ( loc_size == this->Comm_size() )
      {
        dynamic_sources = *sources;
        //printf( "dynamic_sources %lu %lu, sources %lu %lu %lu\n",
        //    dynamic_sources.row(), dynamic_sources.col(), 
        //    sources->row(), sources->col(), sources->col_owned() ); fflush( stdout );
      }

      /** Use cylic order for p2p exchange */
      int par_rank = ( loc_rank + loc_size / 2 ) % loc_size;
      vector<size_t> send_gids, keep_gids, recv_gids;
      Data<TP> Xsend, Xkeep;

      send_gids.reserve( gids.size() );
      recv_gids.reserve( gids.size() );

      /** Split parameters (or points) into lhs and rhs. */
      if ( loc_rank < loc_size / 2 )
      {
        for ( auto it = lhs.begin(); it != lhs.end(); it ++ )
          keep_gids.push_back( gids[ *it ] );
        for ( auto it = rhs.begin(); it != rhs.end(); it ++ )
          send_gids.push_back( gids[ *it ] );
        Xkeep = dynamic_sources( all_dimensions, lhs );
        Xsend = dynamic_sources( all_dimensions, rhs );
      }
      else
      {
        for ( auto it = lhs.begin(); it != lhs.end(); it ++ )
          send_gids.push_back( gids[ *it ] );
        for ( auto it = rhs.begin(); it != rhs.end(); it ++ )
          keep_gids.push_back( gids[ *it ] );
        Xsend = dynamic_sources( all_dimensions, lhs );
        Xkeep = dynamic_sources( all_dimensions, rhs );
      }
      dynamic_sources = Xkeep;



      //mpi::Barrier( comm );
      //if ( loc_rank == 0 )
      //{
      //  printf( "rank %2d RedistributeWithPartner Checkpoint#1\n", 
      //      this->Comm_rank() ); 
      //  fflush( stdout );
      //}


      /** Overwrite Xkeep */
      mpi::ExchangeVector( 
          send_gids, par_rank, 0,
          recv_gids, par_rank, 0, comm, &status );
      mpi::ExchangeVector( 
        Xsend, par_rank, 1, 
        Xkeep, par_rank, 1, comm, &status );

      /** Resize Xkeep to adjust its row() and col(). */
      Xkeep.resize( Xkeep.row(), Xkeep.size() / Xkeep.row() );


      //mpi::Barrier( comm );
      //if ( loc_rank == 0 )
      //{
      //  printf( "rank %2d RedistributeWithPartner Checkpoint#2\n", 
      //      this->Comm_rank() ); 
      //  fflush( stdout );
      //}

      /** Insert received parameters into table */
      sources_user.InsertColumns( recv_gids, Xkeep );

      //mpi::Barrier( comm );
      //if ( loc_rank == 0 )
      //{
      //  printf( "rank %2d RedistributeWithPartner Checkpoint#3\n", 
      //      this->Comm_rank() ); 
      //  fflush( stdout );
      //}

      /** Concatenate */
      dynamic_sources.insert( dynamic_sources.end(), Xkeep.begin(), Xkeep.end() );

      //mpi::Barrier( comm );
      //if ( loc_rank == 0 )
      //{
      //  printf( "rank %2d RedistributeWithPartner Checkpoint#4\n", 
      //      this->Comm_rank() ); 
      //  fflush( stdout );
      //}

    }; /** end RedistributeWithPartner() */


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

    /** Temporary buffer for P2P exchange during tree partition */
    Data<TP> dynamic_sources;


}; /** end class DistKernelMatrix_ver2 */

}; /** end namespace hmlp */

#endif /** define DISTKERNELMATRIX_VER2_HPP */
