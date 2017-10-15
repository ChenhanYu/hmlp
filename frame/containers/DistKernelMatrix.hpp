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

#ifdef USE_INTEL
#include <mkl.h>
#endif

/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif

#include <hmlp_blas_lapack.h>

/** use software cache */
#include <containers/Cache.hpp>

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
    DistKernelMatrix
    ( 
      size_t m, size_t n, size_t d, 
      kernel_s<T> &kernel, 
      /** by default we assume sources are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, T> &sources, 
      mpi::Comm comm 
    )
    : sources_sqnorms( 1, m, comm ), 
      targets_sqnorms( 1, n, comm ),
      all_dimensions( d ),
			cache( d ),
      DistVirtualMatrix<T>( m, n, comm )
    {
      assert( m == n );
      this->is_symmetric = true;
      this->d = d;
      this->kernel = kernel;
      this->sources = &sources;
      this->targets = &sources;
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
      /** compute square 2-norms for Euclidian family */
      ComputeSquare2Norm();
    };

    /** unsymmetric kernel matrix */
    DistKernelMatrix
    ( 
      size_t m, size_t n, size_t d, 
      kernel_s<T> &kernel, 
      /** by default we assume sources are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, T> &sources, 
      /** by default we assume targets are distributed in [STAR, CBLK] */
      DistData<STAR, CBLK, T> &targets, 
      mpi::Comm comm 
    )
    : sources_sqnorms( 1, m, comm ), 
      targets_sqnorms( 1, n, comm ),
      all_dimensions( d ),
			cache( d ),
      DistVirtualMatrix<T>( m, n, comm )
    {
      this->is_symmetric = false;
      this->d = d;
      this->kernel = kernel;
      this->sources = &sources;
      this->targets = &targets;
      for ( size_t i = 0; i < d; i ++ ) all_dimensions[ i ] = i;
      /** compute square 2-norms for Euclidian family */
      ComputeSquare2Norm();
    };

    ~DistKernelMatrix() {};

    void ComputeSquare2Norm()
    {
      /** MPI */
      int size = this->Comm_size();
      int rank = this->Comm_rank();

      //printf( "before ComputeSquare\n" ); fflush( stdout );

      /** compute 2-norm using xnrm2 */
      #pragma omp parallel for
      for ( size_t j = 0; j < sources->col_owned(); j ++ )
      {
        /** 
         *  notice that here we want to compute 2-norms only
         *  for the sources owned locally. In this case, we
         *  cannot use ( i, j ) operator. 
         */
        sources_sqnorms[ j ] = xdot( 
						d, sources->data() + j * d, 1,
						   sources->data() + j * d, 1 ); 
      } /** end omp parallel for */


      //printf( "after ComputeSquare\n" ); fflush( stdout );


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
          targets_sqnorms[ i ] = xdot( 
							d, targets->data() + i * d, 1,
							   targets->data() + i * d, 1 );
        } /** end omp paralllel for */
      }
    }; /** end ComputeSquare2Norm() */


		/**
		 *  @brief This function collects 
		 */ 
    hmlp::Data<T> RequestSources( std::vector<size_t>& J )
    {
      /** MPI */
      int size = this->Comm_size();
			int rank = this->Comm_rank();


      //printf( "%d RequestSources %lu\n", rank, J.size() ); fflush( stdout );

			/** return value */
      hmlp::Data<T> XJ( d, J.size() );

			/** check if source j are both owned by this MPI process */
			std::vector<std::vector<size_t>> sendcids( size );

      /** record the destination order */
      std::vector<std::vector<size_t>> jmapcids( size );

			/** check if source j is cached */
      std::vector<size_t> cacheids;
			std::vector<size_t> jmapline;

			for ( size_t j = 0; j < J.size(); j ++ )
			{
        size_t cid = J[ j ];

        /** source( cid ) is stored locally <STAR, CBLK> */
        if ( cid % size == rank )
        {
					for ( size_t i = 0; i < d; i ++ ) XJ( i, j ) = (*sources)( i, cid );
          continue;
        }

        if ( sources_cids )
        {
          /** source( cid ) is stored locally <STAR, CIDS> */
          if ( sources_cids->HasColumn( cid ) )
          {
            T *XJ_cids = sources_cids->columndata( cid );
					  for ( size_t i = 0; i < d; i ++ ) XJ( i, j ) = XJ_cids[ i ];
            continue;
          }
        }

				/** try to read from cache (critical region) */
				auto line = cache.Read( cid );

        //printf( "%d fetching cache line cid %lu, return %lu\n", rank, cid, line.size() ); fflush( stdout );

				if ( line.size() != d )
				//if ( 1 )
				{
          sendcids[ cid % size ].push_back( cid );    
          jmapcids[ cid % size ].push_back(   j );
				}
				else
				{
					/** source( cid ) is cached */
					for ( size_t i = 0; i < d; i ++ ) XJ( i, j ) = line[ i ];
				}
			}

			/** double buffer for XJp */
			hmlp::Data<T> buffer[ 2 ];

			/** started to request coordinates */
			for ( size_t pp = 0; pp < size; pp ++ )
			{
				/** use Round-Robin order to interleave requests */
        size_t p = ( pp + rank ) % size;

				auto &Jp = sendcids[ p ];

				if ( !Jp.size() ) continue;

				/** get the interleaving buffer for XJp */
        hmlp::Data<T> &XJp = buffer[ pp % 2 ];

        if ( p == rank )
				{
					XJp = (*sources)( all_dimensions, Jp );
				}
				else
				{
					/** resize buffer for receving from p */
          XJp.resize( d, Jp.size() );
					/** tag 128 for sources */
          int tag = omp_get_thread_num() + 128;
          int max_tag = MPI_TAG_UB;
          //printf( " %d Sending tag %d (MAX %d) to %lu\n", rank, tag, max_tag, p ); fflush( stdout );

          //#pragma omp critical
					{
						/** issue a request to dest for source( j ) */
						//mpi::Send( Jp.data(), Jp.size(), p, tag, this->GetRecvComm() );
						//printf( "%d Expect to receive %lu tag %d from %lu\n",
						//    rank, XJp.size(), tag, p ); fflush( stdout );



						/** issue a request to dest for source( j ) */
						int test_flag = false;
		        mpi::Request request;
						mpi::Isend( Jp.data(), Jp.size(), p, tag, this->GetRecvComm(), &request );
						//printf( "%d Expect to receive %lu tag %d from %lu\n",
						//    rank, XJp.size(), tag, p ); fflush( stdout );

						/** wait and cache */
						if ( pp )
						{
              int previous = ( pp - 1 + rank ) % size;

              for ( size_t j = 0; j < sendcids[ previous ].size(); j ++ )
							{
						    mpi::Test( &request, &test_flag, MPI_STATUS_IGNORE );
								if ( test_flag ) break;
					      /** write to cache */
                hmlp::Data<T> line( d, (size_t)1 );					
                for ( size_t i = 0; i < d; i ++ ) 
									line[ i ] = buffer[ ( pp - 1 ) % 2 ]( i, j );
					      /** write to cache */
                cache.Write( sendcids[ previous ][ j ], line ); 
							}
						}
						else
						{
						  while ( !test_flag )
						  {
						    mpi::Test( &request, &test_flag, MPI_STATUS_IGNORE );
							}
						}

						/** wait to recv the coorinates */
						mpi::Status status;
						mpi::Recv( XJp.data(), XJp.size(), p, tag - 128, 
								this->GetSendComm(), &status );
					}
          //printf( "%d Done receive %lu tag %d from %lu\n",
          //    rank, XJp.size(), tag - 128, p ); fflush( stdout );
				}

				/** fill-in XJ and write to cache */
        for ( size_t j = 0; j < jmapcids[ p ].size(); j ++ )
				{
					/** write to cache */
          //hmlp::Data<T> line( d, (size_t)1 );					
					/** fill-in XJ */
          for ( size_t i = 0; i < d; i ++ )
					{
						//line[ i ] = XJp( i, j );
            XJ( i, jmapcids[ p ][ j ] ) = XJp( i, j );
					}

					/** write to cache */
          //cache.Write( Jp[ j ], line ); 
				}
			}

			return XJ;

    }; /** end RequestSources() */









		/**
		 *  @brief This function collects KIJ from myself
		 */ 
    hmlp::Data<T> RequestColumns( std::vector<size_t>& I, hmlp::Data<T> &XJ )
    {
      /** MPI */
      int size = this->Comm_size();
			int rank = this->Comm_rank();

      /** assertion */
      assert( XJ.row() == d );

			/** return value */
      hmlp::Data<T> KIJ( I.size(), XJ.col() );

			/** check if source j are both owned by this MPI process */
			std::vector<std::vector<size_t>> sendcids( size );

      /** record the destination order */
      std::vector<std::vector<size_t>> jmapcids( size );

			/** check if source j is cached */
      std::vector<size_t> cacheids;
			std::vector<size_t> jmapline;

			for ( size_t i = 0; i < I.size(); i ++ )
			{
        size_t cid = I[ i ];

        std::vector<size_t> cid_query( 1, cid );

        /** source( cid ) is stored locally <STAR, CBLK> */
        if ( cid % size == rank )
        {
					auto XI = (*sources)( all_dimensions, cid_query );
					for ( size_t j = 0; j < KIJ.col(); j ++ )
            KIJ( i, j ) = (*this)( XI.data(), XJ.columndata( j ) );
          continue;
        }

        if ( sources_cids )
        {
          /** source( cid ) is stored locally <STAR, CIDS> */
          if ( sources_cids->HasColumn( cid ) )
          {
					  auto XI = (*sources_cids)( all_dimensions, cid_query );
					  for ( size_t j = 0; j < KIJ.col(); j ++ )
              KIJ( i, j ) = (*this)( XI.data(), XJ.columndata( j ) );
            continue;
          }
          else
          {
            //printf( "RequestColumn: cid not found\n" ); fflush( stdout );
          }
        }

				/** try to read from cache (critical region) */
				auto line = cache.Read( cid );

				if ( line.size() != d )
				{
          sendcids[ cid % size ].push_back( cid );    
          jmapcids[ cid % size ].push_back(   i );
				}
				else
				{
					for ( size_t j = 0; j < KIJ.col(); j ++ )
            KIJ( i, j ) = (*this)( line.data(), XJ.columndata( j ) );
				}
			}


			/** started to request values of the column */
			for ( size_t pp = 0; pp < size; pp ++ )
			{
				/** use Round-Robin order to interleave requests */
        size_t p = ( pp + rank ) % size;

				auto &Jp = sendcids[ p ];

				if ( !Jp.size() ) continue;

        /** use tags larger than 1024 */
        int tag = omp_get_thread_num() + 1024;

			  /** issue a request to notify rank p */
				mpi::Send(   Jp.data(),   Jp.size(), p, tag, this->GetRecvComm() );
        /** send rank p XJ for K( XI, XJ ) */
				//mpi::Send(   XJ.data(),   XJ.size(), p, tag, this->GetRecvComm() );
			  mpi::SendVector( XJ, p, tag, this->GetRecvComm() );



        /** wait to recv K( XI, XJ ) */
				mpi::Status status;
        Data<T> KIJp( Jp.size(), XJ.col() );
				mpi::Recv( KIJp.data(), KIJp.size(), p, tag, this->GetSendComm(), &status );

        for ( size_t i = 0; i < jmapcids[ p ].size(); i ++ )
        {
					for ( size_t j = 0; j < KIJ.col(); j ++ )
            KIJ( jmapcids[ p ][ i ], j ) = KIJp( i, j );
        }
      }

      return KIJ;

    }; /** end RequestColumns() */





    T operator () ( T *itargets, T *jsources )
    {
      /** return value */
      T Kij = 0.0;

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






    T operator () ( size_t i, size_t j )
    {
      /** MPI */
      int size = this->Comm_size();
      int rank = this->Comm_rank();

      /** return value */
      T Kij = 0.0;
 
			std::vector<size_t> I( 1, i );
			std::vector<size_t> J( 1, j );

      /** get coordinates */
      hmlp::Data<T> itargets = RequestSources( I );
      hmlp::Data<T> jsources = RequestSources( J );

      /** at this moment we already have the corrdinates on this process */
//      switch ( kernel.type )
//      {
//        case KS_GAUSSIAN:
//          {
//            for ( size_t k = 0; k < d; k++ )
//            {
//              T tar = itargets[ k ];
//              T src = jsources[ k ];
//              Kij += ( tar - src ) * ( tar - src );
//            }
//            Kij = exp( kernel.scal * Kij );
//            break;
//          }
//        default:
//          {
//            printf( "invalid kernel type\n" );
//            exit( 1 );
//            break;
//          }
//      }
//      return Kij;

      return (*this)( itargets.data(), jsources.data() );

    }; /** end operator () */









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
			Data<T> itargets, jsources;

			/** early return */
			if ( !I.size() || !J.size() ) return KIJ;

      /** 
       *  Request form column values directly, this access pattern happends only
       *  during the tree partitioning. We need to provide the parter rank.
       *
       */
      if ( J.size() < 3 )
      {
        if ( I.size() == dynamic_sources.col() ) 
        {
          itargets = dynamic_sources;
          jsources = RequestSources( J );
        }
        else
        {
          auto XJ = RequestSources( J );
          return RequestColumns( I, XJ );
        }
      }
      else
      {
			  if ( &I == &J )
			  {
          itargets = RequestSources( I );
			    jsources = itargets;
			  }
			  else
			  {
          itargets = RequestSources( I );
          jsources = RequestSources( J );
			  }
      }

			/** request for coordinates */
      //hmlp::Data<T> itargets = RequestSources( I );
      //hmlp::Data<T> jsources = RequestSources( J );

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
    hmlp::Data<T> Diagonal( std::vector<size_t> &I )
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

    }; /** end Diagonal() */





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


    virtual void BackGroundProcess( bool *do_terminate )
    {
#ifdef USE_INTEL
			mkl_set_num_threads_local( 1 );
#endif

      /** reset the termination flag */
      this->ResetTerminationFlag();

      /** MPI */
      int size = this->Comm_size();
      int rank = this->Comm_rank();

      /** Iprobe flag */
      int probe_flag = 0;

      /** Iprobe and Recv status */
      mpi::Status status;

      /** buffer for I and J */
      size_t buff_size = 1048576;
      std::vector<size_t> I;

      /** reserve space for I and J */
      I.reserve( buff_size );

      /** recving XJ for evaluating KIJ */
      hmlp::Data<T> XJ;
			XJ.reserve( d, (size_t)8192 );



      //printf( "Enter DistKernelMatrix::BackGroundProcess\n" ); fflush( stdout );


			size_t idle_counter = 0;

			/** keep probing for messages */
			while ( 1 ) 
			{
        /** info from mpi::Status */
        int recv_src;
        int recv_tag;
        int recv_cnt;

        /** only on thread will probe and recv message at a time */
        #pragma omp critical
        {
          mpi::Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, 
            this->GetRecvComm(), &probe_flag, &status );

          /** if receive any message, then handle it */
          if ( probe_flag )
          {
            /** extract info from status */
            recv_src = status.MPI_SOURCE;
            recv_tag = status.MPI_TAG;

            if ( this->IsBackGroundMessage( recv_tag ) )
            {
              /** get I object count */
              mpi::Get_count( &status, HMLP_MPI_SIZE_T, &recv_cnt );

              /** recv (typeless) I by matching SOURCE and TAG */
              I.resize( recv_cnt );
              mpi::Recv( I.data(), recv_cnt, recv_src, recv_tag, 
                  this->GetRecvComm(), &status );

              if ( recv_tag >= 1024 )
              {
                /** recv XJ that contains multiple coordinates */
							  mpi::RecvVector( XJ, recv_src, recv_tag, this->GetRecvComm(), &status );
                XJ.resize( d, XJ.size() / d );
                //mpi::Recv( XJ.data(), d, recv_src, recv_tag, this->GetRecvComm(), &status );

              }
            }
            else probe_flag = 0;
          }
        }

        if ( probe_flag )
        {
          if ( recv_tag >= 1024 )
          {
            Data<T> KIJ( I.size(), XJ.col() );
              
            for ( size_t i = 0; i < I.size(); i ++ )
            {
              std::vector<size_t> query( 1, I[ i ] );
              auto XI = (*sources)( all_dimensions, query );
							for ( size_t j = 0; j < KIJ.col(); j ++ )
                KIJ( i, j ) = (*this)( XI.data(), XJ.columndata( j ) );
            }

            /** blocking send */
            mpi::Send( KIJ.data(), KIJ.size(), recv_src, 
                recv_tag, this->GetSendComm() );
          }
          else
          {
            auto XI = (*sources)( all_dimensions, I );
            /** blocking send */
            mpi::Send( XI.data(), XI.size(), recv_src, 
                (recv_tag - 128), this->GetSendComm() );
          }

          /** reset flag to zero */
          probe_flag = 0;
        }


				/** nonblocking consensus for termination */
				if ( *do_terminate ) 
				{
          /** while reaching both global and local concensus, exit */
          if ( this->IsTimeToTerminate() ) break;
        }

      }; /** end while ( 1 ) */


      //printf( "Exit DistKernelMatrix::BackGroundProcess\n" ); fflush( stdout );

    }; /** end BackGroundProcess() */


    virtual void Redistribute( bool enforce_ordered, std::vector<size_t> &cids )
    {
			mpi::Comm comm = this->GetComm();
			size_t n = this->col();

			/** delete the previous redistribution */
			if ( sources_cids ) 
      {
        //printf( "delete sources_cids\n" ); fflush( stdout );
        delete sources_cids;
      }

			/** allocation */
      if ( enforce_ordered )
      {
			  sources_cids = new DistData<STAR, CIDS, T>( d, n, cids, comm );
			  *sources_cids = *sources;
      }
      else
      {
			  sources_cids = new DistData<STAR, CIDS, T>( d, n, cids, dynamic_sources, comm );

        for ( size_t i = 0; i < cids.size(); i ++ )
        {
          assert( sources_cids->HasColumn( cids[ i ] ) );
        }
      }

    }; /** end Redistribute() */

   

    virtual void RedistributeWithPartner(
        std::vector<size_t> &gids,
        std::vector<size_t> &lhs, 
        std::vector<size_t> &rhs, mpi::Comm comm )
    {
      mpi::Status status;
      int loc_size, loc_rank;
      mpi::Comm_size( comm, &loc_size );
      mpi::Comm_rank( comm, &loc_rank );

      /** at the root level, starts from CBLK */
      if ( loc_size == this->Comm_size() )
      {
        dynamic_sources = *sources;
        //printf( "dynamic_sources %lu %lu, sources %lu %lu %lu\n",
        //    dynamic_sources.row(), dynamic_sources.col(), 
        //    sources->row(), sources->col(), sources->col_owned() ); fflush( stdout );
      }

      //if ( loc_size == 1 )
      //{
			//  /** delete the previous redistribution */
			//  if ( sources_cids ) delete sources_cids;

			//  /** allocation */
			//  sources_cids = new DistData<STAR, CIDS, T>( d, this->cids, gids, this->GetComm() );

      //  *sources_cids = dynamic_sources;

      //  return;
      //}

      auto Xlhs = dynamic_sources( all_dimensions, lhs );
      auto Xrhs = dynamic_sources( all_dimensions, rhs );
      Data<T> Xpar;

      if ( loc_rank < loc_size / 2 )
      {
        int par_rank = loc_rank + loc_size / 2;
        mpi::ExchangeVector( 
            Xrhs, par_rank, 0, 
            Xpar, par_rank, 0, comm, &status );
        Xlhs.insert( Xlhs.end(), Xpar.begin(), Xpar.end() );
        Xlhs.resize( d, Xlhs.size() / d );
        dynamic_sources = Xlhs;
      }
      else
      {
        int par_rank = loc_rank - loc_size / 2;
        mpi::ExchangeVector( 
            Xlhs, par_rank, 0, 
            Xpar, par_rank, 0, comm, &status );
        Xrhs.insert( Xrhs.end(), Xpar.begin(), Xpar.end() );
        Xrhs.resize( d, Xrhs.size() / d );
        dynamic_sources = Xrhs;
      }

    }; /** end RedistributeWithPartner() */




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

    /** */
    std::vector<size_t> all_dimensions;

		Cache1D<4096, 256, T> cache;

    //DistData<STAR, CIDS, T> dynamic_sources;
    Data<T> dynamic_sources;


}; /** end class DistKernelMatrix */
}; /** end namespace hmlp */

#endif /** define DISTKERNELMATRIX_HPP */
