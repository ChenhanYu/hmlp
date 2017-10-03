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




#ifndef DISTVIRTUALMATRIX_HPP
#define DISTVIRTUALMATRIX_HPP

/** -lmemkind */
#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#include <hbw_allocator.h>
#endif

/** */
#include <hmlp_mpi.hpp>


namespace hmlp
{

#ifdef HMLP_MIC_AVX512
template<typename T, class Allocator = hbw::allocator<T> >
#else
template<typename T, class Allocator = std::allocator<T> >
#endif
/**
 *  @brief DistVirtualMatrix is the abstract base class for matrix-free
 *         access and operations. Most of the public functions will
 *         be virtual. To inherit DistVirtualMatrix, you "must" implement
 *         the evaluation operator. Otherwise, the code won't compile.
 */ 
class DistVirtualMatrix : public mpi::MPIObject
{
  public:

    DistVirtualMatrix() {};

    DistVirtualMatrix( size_t m, size_t n, mpi::Comm comm )
      : mpi::MPIObject( comm )
    {
      this->m = m;
      this->n = n;
    };

    /** ESSENTIAL: return number of coumns */
    virtual size_t row() 
    { 
      return m; 
    };

    /** ESSENTIAL: return number of rows */
    virtual size_t col() 
    { 
      return n; 
    };

    /** ESSENTIAL: this is an abstract function  */
    virtual T operator()( size_t i, size_t j ) = 0; 

    /** ESSENTIAL: return a submatrix */
    virtual hmlp::Data<T> operator()
    ( 
      std::vector<size_t> &I, std::vector<size_t> &J 
    ) = 0;


		bool IsBackGroundMessage( int tag )
		{
			return ( tag >= background_tag_offset );
		};

		virtual hmlp::Data<T> RequestKIJ
		( 
		  std::vector<size_t> &I, std::vector<size_t> &J, int p
	  )
		{
			/** return values */
      Data<T> KIJ( I.size(), J.size() );

			int Itag = omp_get_thread_num() + 1 * background_tag_offset;
			int Jtag = omp_get_thread_num() + 2 * background_tag_offset;
			int Ktag = omp_get_thread_num();

			/** issue a request to dest for I */
			mpi::Send(   I.data(),   I.size(), p, Jtag, this->GetRecvComm() );
			/** issue a request to dest for J */
			mpi::Send(   J.data(),   J.size(), p, Jtag, this->GetRecvComm() );
			/** wait to recv KIJ */
			mpi::Status status;
			mpi::Recv( KIJ.data(), KIJ.size(), p, Ktag, this->GetSendComm(), &status );

			return KIJ;

		}; /** end RequestKIJ() */


    /**
     *  this routine is executed by the serving worker
     *  wait and receive message from any source
     */ 
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
      std::vector<size_t> J;

      /** reserve space for I and J */
      I.reserve( buff_size );
      J.reserve( buff_size );



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


					if ( this->IsBackGroundMessage( recv_tag ) )
					{
						/** get I object count */
						mpi::Get_count( &status, HMLP_MPI_SIZE_T, &recv_cnt );

						/** recv (typeless) I by matching SOURCE and TAG */
						I.resize( recv_cnt );
						mpi::Recv( I.data(), recv_cnt, recv_src, recv_tag, 
								this->GetRecvComm(), &status );

						/** blocking Probe the message that contains J */
						mpi::Probe( recv_src, recv_tag + 128, this->GetComm(), &status );

						/** get J object count */
						mpi::Get_count( &status, HMLP_MPI_SIZE_T, &recv_cnt );

						/** recv (typeless) J by matching SOURCE and TAG */
						J.resize( recv_cnt );
						mpi::Recv( J.data(), recv_cnt, recv_src, recv_tag + 128, 
								this->GetRecvComm(), &status );

						/** 
						 *  this invoke the operator () to get K( I, J )  
						 *
						 *  notice that operator () can invoke MPI routines,
						 *  but limited to one-sided routines without blocking.
						 */
						auto KIJ = (*this)( I, J );

						/** blocking send */
						mpi::Send( KIJ.data(), KIJ.size(), recv_src, 
								( recv_tag - 128 ), this->GetComm() );
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

			} /** end while */

    }; /** end BackGroundProcess() */

		

  private:

    size_t m = 0;

    size_t n = 0;

		const int background_tag_offset = 128;

}; /** end class DistVirtualMatrix */

}; /** end namespace hmlp */

#endif /** define DISTVIRTUALMATRIX_HPP */
