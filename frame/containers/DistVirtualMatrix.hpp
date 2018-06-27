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


using namespace std;
using namespace hmlp;


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
 *
 *         Two virtual functions must be implemented:
 *
 *         T operator () ( size_t i, size_t j ), and
 *         Data<T> operator () ( vector<size_t> &I, vector<size_t> &J ).
 *
 *         These two functions can involve nonblocking MPI routines, but
 *         blocking collborative communication routines are not allowed.
 *         
 *         DistVirtualMatrix inherits mpi::MPIObject, which is initalized
 *         with the provided comm. MPIObject duplicates comm into
 *         sendcomm and recvcomm, which allows concurrent multi-threaded 
 *         nonblocking send/recv. 
 *
 *         For example, RequestKIJ( I, J, p ) sends I and J to rank-p,
 *         requesting the submatrix. MPI process rank-p has at least
 *         one thread will execute BackGroundProcess( do_terminate ),
 *         waiting for incoming requests.
 *         Rank-p then invoke K( I, J ) locally, and send the submatrix
 *         back to the clients. Overall the pattern usually looks like
 *
 *         Data<T> operator () ( vector<size_t> &I, vector<size_t> &J ).
 *         {
 *           for each submatrix KAB entirely owned by p
 *             KAB = RequestKIJ( A, B )
 *             pack KAB back to KIJ
 *
 *           return KIJ 
 *         }
 *         
 */ 
class DistVirtualMatrix : public mpi::MPIObject
{
  public:

    /** empty constructor */
    DistVirtualMatrix() {};

    /** default constructor  */
    DistVirtualMatrix( size_t m, size_t n, mpi::Comm comm )
      : mpi::MPIObject( comm )
    {
      this->m = m;
      this->n = n;
    };

    /** return number of columns */
    virtual size_t row() { return m; };

    /** return number of rows */
    virtual size_t col() { return n; };



    /** 
     *  ESSENTIAL: this is an abstract function  
     */
    virtual T operator()( size_t i, size_t j ) = 0; 

    /** 
     *  ESSENTIAL: return a submatrix.
     *
     *  hmlp::Data<T> inherits std::vector<T>. Use constructor
     *  hmlp::Data<T> KIJ( I.size(), J.size(), 0 ) to initialize
     *  a zero output matrix. 
     *
     *  hmlp::Data<T>::operator [] is the same as std::vector<T> operator [].
     *  The memory is stored in column-major contiguously.
     *
     *  T &hmlp::Data<T>::operator () ( size_t i, size_t j ) returns
     *  the reference of K( i, j ).
     *
     */
    virtual Data<T> operator() ( vector<size_t> &I, vector<size_t> &J ) = 0;



    virtual void SendColumns( vector<size_t> cids, int dest, mpi::Comm comm )
    {
    };

    virtual void RecvColumns( int root, mpi::Comm comm, mpi::Status *status )
    {
    };


    /** Bcast cids from sender for K( :, cids ) evaluation. */
    virtual void BcastColumns( vector<size_t> cids, int root, mpi::Comm comm )
    {
    }; /** end BcastColumns() */


    virtual void RequestColumns( vector<vector<size_t>> requ_cids )
    {
    };



    /**
     *  ESSENTIAL: redistribute the physical memory distribution
     *             to reduce the communication cost. gids provide
     *             information on how row and column indices are
     *             pysically distributed. Users may choose to 
     *             follow the guidance of gids, but after invoking
     *             Redistribute( gids ) the two virtual functions
     *             above cannot involve any blocking communication.
     */ 
    //virtual void Redistribute( bool enforce_ordered, vector<size_t> &cids )
    //{

    //}; /** end Redistribute() */


    virtual void RedistributeWithPartner( 
        vector<size_t> &lhs, 
        vector<size_t> &rhs, mpi::Comm comm )
    {

    }; /** end RedistributeWithPartner() */



    void ProvideImportantSamples( vector<size_t> &I, vector<size_t> &J, Data<T> &KIJ )
    {
      tuple<map<size_t, size_t>, map<size_t, size_t>, Data<T>> candidate;
      
      for ( size_t i = 0; i < I.size(); i ++ )
        get<0>( candidate )[ I[ i ] ] = i;
      for ( size_t j = 0; j < J.size(); j ++ )
        get<1>( candidate )[ J[ j ] ] = j;

      get<2>( candidate ) = KIJ;

      samples.push_back( candidate );
    };





    /** 
     *  (Clients) request KIJ from rank-p 
     */
		virtual Data<T> RequestKIJ
		( 
		  vector<size_t> &I, vector<size_t> &J, int p
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
     *  (Servers) wait and receive message from any client
     */ 
//    virtual void BackGroundProcess( bool *do_terminate )
//    {
//      /** MPI */
//      int size = this->Comm_size();
//      int rank = this->Comm_rank();
//
//      /** Iprobe flag */
//      int probe_flag = 0;
//
//      /** Iprobe and Recv status */
//      mpi::Status status;
//
//      /** buffer for I and J */
//      size_t buff_size = 1048576;
//      std::vector<size_t> I;
//      std::vector<size_t> J;
//
//      /** reserve space for I and J */
//      I.reserve( buff_size );
//      J.reserve( buff_size );
//
//
//      /** keep probing for messages */
//      while ( 1 ) 
//      {
//        /** info from mpi::Status */
//        int recv_src;
//        int recv_tag;
//        int recv_cnt;
//
//        /** only on thread will probe and recv message at a time */
//        #pragma omp critical
//        {
//          mpi::Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, 
//              this->GetRecvComm(), &probe_flag, &status );
//
//          /** if receive any message, then handle it */
//          if ( probe_flag )
//          {
//            /** extract info from status */
//            recv_src = status.MPI_SOURCE;
//            recv_tag = status.MPI_TAG;
//
//            if ( this->IsBackGroundMessage( recv_tag ) )
//            {
//              /** get I object count */
//              mpi::Get_count( &status, HMLP_MPI_SIZE_T, &recv_cnt );
//
//              /** recv (typeless) I by matching SOURCE and TAG */
//              I.resize( recv_cnt );
//              mpi::Recv( I.data(), recv_cnt, recv_src, recv_tag, 
//                  this->GetRecvComm(), &status );
//
//              /** blocking Probe the message that contains J */
//              mpi::Probe( recv_src, recv_tag + 128, this->GetComm(), &status );
//
//              /** get J object count */
//              mpi::Get_count( &status, HMLP_MPI_SIZE_T, &recv_cnt );
//
//              /** recv (typeless) J by matching SOURCE and TAG */
//              J.resize( recv_cnt );
//              mpi::Recv( J.data(), recv_cnt, recv_src, recv_tag + 128, 
//                  this->GetRecvComm(), &status );
//            }
//            else probe_flag = 0;
//          }
//        } /** end pragma omp critical */
//
//
//        if ( probe_flag )
//        {
//          /** 
//           *  this invoke the operator () to get K( I, J )  
//           *
//           *  notice that operator () can invoke MPI routines,
//           *  but limited to one-sided routines without blocking.
//           */
//          auto KIJ = (*this)( I, J );
//
//          /** blocking send */
//          mpi::Send( KIJ.data(), KIJ.size(), recv_src, 
//              ( recv_tag - 128 ), this->GetComm() );
//
//          /** reset flag to zero */
//          probe_flag = 0;
//        }
//
//
//        /** nonblocking consensus for termination */
//        if ( *do_terminate ) 
//        {
//          /** while reaching both global and local concensus, exit */
//          if ( this->IsTimeToTerminate() ) break;
//        }
//
//      } /** end while ( 1 ) */
//
//    }; /** end BackGroundProcess() */
//
//
//
//
//
//
//
//    /** check if this tag is for one-sided communication */
//    bool IsBackGroundMessage( int tag )
//    {
//      return ( tag >= background_tag_offset );
//
//    }; /** end IsBackGroundMessage() */
//
//
//    /**
//     *  @brief The termination flag is reset by the first omp thread
//     *         who execute the function.
//     */ 
//    void ResetTerminationFlag()
//    {
//      test_flag = 0;
//      has_Ibarrier = false;
//      do_terminate = false;
//
//    }; /** end ResetTerminationFlag () */
//
//
//    bool IsTimeToTerminate()
//    {
//      #pragma omp critical
//      {
//        if ( !has_Ibarrier )
//        {
//          mpi::Ibarrier( this->GetComm(), &request );
//          has_Ibarrier = true;
//        }
//
//        if ( !test_flag )
//        {
//          /** while test_flag = 1, MPI request got reset */
//          mpi::Test( &request, &test_flag, MPI_STATUS_IGNORE );
//          if ( test_flag ) do_terminate = true;
//        }
//      }
//
//      /** if this is not the mater thread, just return the flag */
//      return do_terminate;
//
//    }; /** end IsTimeToTerminate() */
//

  private:

    /** this is an m-by-n virtual matrix */
    size_t m = 0;
    size_t n = 0;

    /** we use tags >= 128 for */
    const int background_tag_offset = 128;

    /** for mpi::Ibarrier */
    mpi::Request request;
    int test_flag = 0;
    bool has_Ibarrier = false;

    /** whether to terminate the background process */
    bool do_terminate = false;

    /** 
     *  Importannt samples:
     *
     *  The idea of important samples in GOFMM is to collect submatrices
     *  that may be accessed in advance to prevent hugh latency.
     *
     *  Each stored submatrix has row and column maps, and the values
     *  are stored in hmlp::Data<T> format.
     *   
     */
    vector<tuple<map<size_t, size_t>, map<size_t, size_t>, Data<T>>> samples;


}; /** end class DistVirtualMatrix */

}; /** end namespace hmlp */

#endif /** define DISTVIRTUALMATRIX_HPP */
