/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2018, The University of Texas at Austin
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

/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;

/** 
 *  @brief In this example, we explain how you can compute
 *         approximate all-nearest neighbors (ANN) using MPIGOFMM. 
 */ 
int main( int argc, char *argv[] )
{
  try
  {
    /** Use float as data type. */
    using T = float;
    /** [Required] Problem size. */
    size_t n = 5000;

    /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support. */
    int  provided = 0;
    mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    if ( provided != MPI_THREAD_MULTIPLE ) exit( 1 );
    /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
    mpi::Comm CommGOFMM;
    mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );
    /** [Step#0] HMLP API call to initialize the runtime. */
    HANDLE_ERROR( hmlp_init( &argc, &argv, CommGOFMM ) );
    /** [Step#1] Get MPI rank and size. */
    int comm_rank, comm_size;
    mpi::Comm_size( CommGOFMM, &comm_size );
    mpi::Comm_rank( CommGOFMM, &comm_rank );

    /** [Step#2] Create distributed point clouds with random 6D data. */
    size_t d = 6;
    /** Method#1: directly construct a d-by-n matrix with distributed columns. */
    DistData<STAR, CBLK, T> X1( d, n, CommGOFMM ); X1.randn();
    /** Method#2: construct from p local data. */
    size_t d_loc = d;
    size_t n_loc = n / comm_size;
    size_t n_cut = n % comm_size;
    if ( comm_rank < n_cut ) n_loc ++;
    Data<T> X2_local( d_loc, n_loc ); X2_local.randn();
    DistData<STAR, CBLK, T> X2( d, n, X2_local, CommGOFMM );
    /** Method#3: construct from p local std::vector. */
    vector<T> X3_local( d_loc * n_loc );
    DistData<STAR, CBLK, T> X3( d, n, X3_local, CommGOFMM );

    /** [Step#3] Create a n-by-nrhs matrix with distributed rows. */
    size_t nrhs = 64;
    /** Method#1: */
    DistData<RBLK, STAR, T> Y1( n, nrhs, CommGOFMM );
    /** Method#2: */
    Data<T> Y2_local( n_loc, nrhs );
    DistData<RBLK, STAR, T> Y2( n, nrhs, Y2_local, CommGOFMM );
    /** Method#3: */
    vector<T> Y3_local( n_loc * nrhs );
    DistData<RBLK, STAR, T> Y3( n, nrhs, Y3_local, CommGOFMM );

    /** [Step#4] HMLP API call to terminate the runtime. */
    hmlp_finalize();
    /** Finalize Message Passing Interface. */
    mpi::Finalize();
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}; /** end main() */
