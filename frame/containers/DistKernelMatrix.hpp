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
    DistKernelMatrix( 
        size_t m, size_t n, size_t d, 
        kernel_s<T> &kernel, 
        /** by default we assume sources are distributed in [STAR, CBLK] */
        DistData<STAR, CBLK, T> &sources, 
        mpi::Comm comm )
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
     *  ESSENTIAL: override the virtual function 
     *
     *  This operator "may" invoke MPI collective communication routines 
     *  and return a value Kij. Thus, all MPI processes in the communicator
     *  must invoke this operator to avoid dead lock.
     */
//    T operator()( size_t i, size_t j )
//    {
//      /** MPI */
//      int size = this->Comm_size();
//      int rank = this->Comm_rank();
//
//      /** return value */
//      T Kij = 0.0;
//
//      /** check if target i and source j are both owned by this MPI process */
//      std::vector<std::vector<size_t>> sendrids( size );
//      std::vector<std::vector<size_t>> recvrids( size );
//      std::vector<std::vector<size_t>> sendcids( size );
//      std::vector<std::vector<size_t>> recvcids( size );
//
//      /** request Kij from rank ( j % size ) */
//      sendrids[ i % size ].push_back( i );    
//      sendcids[ j % size ].push_back( j );    
//
//      /** exchange ids */
//      mpi::AlltoallVector( sendrids, recvrids, this->GetComm() );
//      mpi::AlltoallVector( sendcids, recvcids, this->GetComm() );
//
//      /** allocate buffer for data */
//      //std::vector<hmlp::Data<T>> sendsrcs( size );
//      //std::vector<hmlp::Data<T>> recvsrcs( size );
//      //std::vector<hmlp::Data<T>> sendtars( size );
//      //std::vector<hmlp::Data<T>> recvtars( size );
//
//      std::vector<std::vector<T>> sendsrcs( size );
//      std::vector<std::vector<T>> recvsrcs( size );
//      std::vector<std::vector<T>> sendtars( size );
//      std::vector<std::vector<T>> recvtars( size );
//
//
//
//      std::vector<size_t> amap( d );
//      for ( size_t k = 0; k < d; k ++ ) amap[ k ] = k;
//
//      for ( size_t p = 0; p < size; p ++ )
//      {
//        if ( recvcids[ p ].size() )
//        {
//          sendsrcs[ p ] = (*sources)( amap, recvcids[ p ] );
//        }
//        if ( recvrids[ p ].size() )
//        {
//          sendtars[ p ] = (*targets)( amap, recvrids[ p ] );
//        }
//      };
//
//      /** exchange coordinates */
//      mpi::AlltoallVector( sendsrcs, recvsrcs, this->GetComm() );
//      mpi::AlltoallVector( sendtars, recvtars, this->GetComm() );
//
//      std::vector<T> itargets( d );
//      std::vector<T> jsources( d );
//
//      for ( size_t p = 0; p < size; p ++ )
//      {
//        if ( recvsrcs[ p ].size() ) jsources = recvsrcs[ p ];
//        if ( recvtars[ p ].size() ) itargets = recvtars[ p ];
//      }
//
//      /** at this moment we already have the corrdinates on this process */
//      switch ( kernel.type )
//      {
//        case KS_GAUSSIAN:
//        {
//          for ( size_t k = 0; k < d; k++ )
//          {
//            T tar = itargets[ k ];
//            T src = jsources[ k ];
//            Kij += ( tar - src ) * ( tar - src );
//          }
//          Kij = exp( kernel.scal * Kij );
//          break;
//        }
//        default:
//        {
//          printf( "invalid kernel type\n" );
//          exit( 1 );
//          break;
//        }
//      }
//      return Kij;
//    };


    //int GetCacheLine( size_t &request )
    //{
    //  int lineid = -1;
    //  if ( id2cacheline.count( request ) ) 
		//		lineid = id2cacheline[ request ]; 
    //  return lineid;
    //}; /** end GetCacheLine() */



		/**
		 *  @brief This function collects 
		 *
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




      ///** destination rank in CBLK (Round-Robin) distribution */
      //int dest = j % size;

      //std::vector<size_t> J( 1, j );

      ///** if j is on this rank [STAR, CBLK] */
      //if ( dest == rank )
      //{
      //  return (*sources)( all_dimensions, J );
      //}
      //else
      //{
      //  /** check if j is cached */
      //  auto lineids = GetCacheLine( J );
      //  if ( lineids[ 0 ] != -1 )
      //  {
      //    std::vector<size_t> lines( 1, lineids[ 0 ] );
      //    return cached_sources( all_dimensions, lines );
      //  }
      //  else
      //  {
      //    hmlp::Data<T> XJ( d, 1 );
      //    int tag = omp_get_thread_num() + 128;
      //    /** issue a request to dest for source( j ) */
      //    mpi::Send( &j, 1, dest, tag, this->GetComm() );
      //    /** wait to recv the coorinates */
      //    mpi::Status status;
      //    mpi::Recv( XJ.data(), XJ.size(), dest, tag, this->GetComm(), &status );
      //    return XJ;
      //  }
      //}
    }; /** end GetCoordinates() */



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

			/** early return */
			if ( !I.size() || !J.size() ) return KIJ;

			/** request for coordinates */
      hmlp::Data<T> itargets = RequestSources( I );
      hmlp::Data<T> jsources = RequestSources( J );

			assert( itargets.row() == d );
			assert( itargets.col() == I.size() );
			assert( itargets.size() == d * I.size() );
			assert( jsources.row() == d );
			assert( jsources.col() == J.size() );
			assert( jsources.size() == d * J.size() );
			assert( KIJ.row() == I.size() );
			assert( KIJ.col() == J.size() );
			assert( KIJ.size() == I.size() * J.size() );

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


    virtual void BackGroundProcess( bool *do_terminate )
    {
      /** MPI */
      int size = this->Comm_size();
      int rank = this->Comm_rank();

      /** Iprobe flag */
      int probe_flag = 0;

      /** Iprobe and Recv status */
      mpi::Status status;

			/** Ibarrier flag */
			bool has_Ibarrier = false;

			/** Ibarrier request */
		  mpi::Request request;

			/** Test flag */
			int test_flag = 0;

      /** buffer for I and J */
      size_t buff_size = 8192;
      std::vector<size_t> I;

      /** reserve space for I and J */
      I.reserve( buff_size );

      //printf( "Enter DistKernelMatrix::BackGroundProcess\n" ); fflush( stdout );


			size_t idle_counter = 0;

			/** keep probing for messages */
			while ( 1 ) 
			{
				int err = mpi::Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, 
						this->GetRecvComm(), &probe_flag, &status );

				if ( err != MPI_SUCCESS )
				{
					printf( "Iprobe error %d\n", err ); fflush( stdout );
					continue;
				}

				/** if receive any message, then handle it */
				if ( probe_flag )
				{
					/** extract info from status */
					int recv_src = status.MPI_SOURCE;
					int recv_tag = status.MPI_TAG;
					int recv_cnt;

          //printf( "Probe with tag %d\n", recv_tag ); fflush( stdout );

					if ( this->IsBackGroundMessage( recv_tag ) )
					{
						idle_counter = 0;
            //printf( "Matching message with tag %d\n", recv_tag ); fflush( stdout );

						/** get I object count */
						mpi::Get_count( &status, HMLP_MPI_SIZE_T, &recv_cnt );

            //printf( "%d Matching message with tag %d count %d from %d\n", 
            //    rank, recv_tag, recv_cnt, recv_src ); fflush( stdout );


						/** recv (typeless) I by matching SOURCE and TAG */
						I.resize( recv_cnt );
						mpi::Recv( I.data(), recv_cnt, recv_src, recv_tag, 
								this->GetRecvComm(), &status );

            //printf( "%d access XI\n", rank ); fflush( stdout );


						auto XI = (*sources)( all_dimensions, I );

            //printf( "%d Send %lu to %d tag %d\n", 
						//		rank, XI.size(), recv_src, recv_tag - 128 ); fflush( stdout );

						/** blocking send */
						mpi::Send( XI.data(), XI.size(), recv_src, 
								(recv_tag - 128), this->GetSendComm() );
					}
          else
          {
						idle_counter ++;

						if ( idle_counter > 100 )
						{
							if ( idle_counter % 100 == 0 )
                printf( "%d idle %lu to\n", rank, idle_counter ); fflush( stdout );
						};
            //printf( "no incoming message\n" ); fflush( stdout );
          }

          /** reset flag to zero */
          probe_flag = 0;
        }

				/** nonblocking consensus for termination */
				if ( *do_terminate ) 
				{
					if ( !has_Ibarrier ) 
					{
						mpi::Ibarrier( this->GetComm(), &request );
						has_Ibarrier = true;
					}
					if ( test_flag ) 
					{
						break;
					}
					else
					{
						mpi::Test( &request, &test_flag, MPI_STATUS_IGNORE );
					}
				}
      };


      //printf( "Exit DistKernelMatrix::BackGroundProcess\n" ); fflush( stdout );

    }; /** end BackGroundProcess() */












    void Redistribute( std::vector<size_t> &cids )
    {
			mpi::Comm comm = this->GetComm();
			size_t m = this->row();
			size_t n = this->col();

			/** allocation */
			sources_cids = new DistData<STAR, CBLK, T>( m, n, cids, comm );

			/** redistribute */
			*sources_cids = *sources;
      
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

    /** map gids to cacahed_coordinates */
    std::map<size_t, size_t> id2cacheline;

    /** d-by-cachesize */
    Data<T> cached_sources;

    /** */
    std::vector<size_t> all_dimensions;


		Cache1D<512, 128, T> cache;


    //Data<T> &sources;

    //Data<T> &targets;

    //std::vector<T> source_sqnorms;

    //std::vector<T> target_sqnorms;

}; /** end class DistKernelMatrix */
}; /** end namespace hmlp */

#endif /** define DISTKERNELMATRIX_HPP */
