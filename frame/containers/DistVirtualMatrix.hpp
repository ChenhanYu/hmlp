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

//    {
//      hmlp::Data<T> submatrix( imap.size(), jmap.size() );
//      #pragma omp parallel for
//      for ( size_t j = 0; j < jmap.size(); j ++ )
//        for ( size_t i = 0; i < imap.size(); i ++ )
//          submatrix[ j * imap.size() + i ] = 
//						(*this)( imap[ i ], jmap[ j ], comm );
//      return submatrix;
//		};

    //virtual std::pair<T, size_t> ImportantSample( size_t j, hmlp::mpi::Comm comm )
    //{
    //  size_t i = std::rand() % m;
    //  std::pair<T, size_t> sample( (*this)( i, j, comm ), i );
    //  return sample; 
    //}; /** end ImportantSample() */

    //virtual std::pair<T, int> ImportantSample( int j, hmlp::mpi::Comm comm )
    //{
    //  int i = std::rand() % m;
    //  std::pair<T, int> sample( (*this)( i, j, comm ), i );
    //  return sample; 
    //}; /** end ImportantSample() */


    /**
     *  this routine is executed by the serving worker
     *  wait and receive message from any source
     */ 
    virtual void BusyWaiting()
    {
      /** Iprobe flag */
      int flag = 0;

      /** Iprobe and Recv status */
      mpi::Status status;

      /** buffer for I and J */
      size_t buff_size = 8192;
      std::vector<size_t> I;
      std::vector<size_t> J;

      /** reserve space for I and J */
      I.reserve( buff_size );
      J.reserve( buff_size );

      while( mpi::Iprobe( MPI_ANY_SOURCE, MPI_ANY_TAG, this->GetComm(), &flag, &status ) )
      {
        /** if receive any message, then handle it */
        if ( flag )
        {
          /** extract info from status */
          int recv_src = status.MPI_SOURCE;
          int recv_tag = status.MPI_TAG;
          int recv_cnt;

          /** get I object count */
          mpi::Get_count( &status, HMLP_MPI_SIZE_T, &recv_cnt );

          /** recv (typeless) I by matching SOURCE and TAG */
          I.resize( recv_cnt );
          mpi::Recv( I.data(), recv_cnt, recv_src, recv_tag, 
              this->GetComm(), &status );

          /** blocking Probe the message that contains J */
          mpi::Probe( recv_src, recv_tag, this->GetComm(), &status );

          /** get J object count */
          mpi::Get_count( &status, HMLP_MPI_SIZE_T, &recv_cnt );

          /** recv (typeless) I by matching SOURCE and TAG */
          J.resize( recv_cnt );
          mpi::Recv( J.data(), recv_cnt, recv_src, recv_tag, 
              this->GetComm(), &status );

          /** 
           *  this invoke the operator () to get K( I, J )  
           *
           *  notice that operator () can invoke MPI routines,
           *  but limited to one-sided routines without blocking.
           */
          auto KIJ = (*this)( I, J );

          /** blocking send */
          mpi::Send( KIJ.data(), KIJ.size(), recv_src, recv_tag, this->GetComm() );
        }
      };

    }; /** end BusyWaiting() */


  private:

    size_t m = 0;

    size_t n = 0;


}; /** end class DistVirtualMatrix */

}; /** end namespace hmlp */

#endif /** define DISTVIRTUALMATRIX_HPP */
