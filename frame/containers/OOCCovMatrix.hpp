#ifndef OOCCOVMATRIX_HPP
#define OOCCOVMATRIX_HPP

#include <exception>

/** BLAS/LAPACK support */
#include <base/blas_lapack.hpp>
/** GEMM task support */
#include <primitives/gemm.hpp>
/** MLPGaussNewton uses VirtualMatrix<T> as base */
#include <containers/VirtualMatrix.hpp>
/** For GOFMM compatability */
#include <containers/SPDMatrix.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{


template<typename T>
class CovTask : public Task
{
  public:

    vector<OOCData<T>> *arg = NULL;

    vector<size_t> ids;
    vector<size_t> I;    
    vector<size_t> J;

    Data<T> *KIJ = NULL;


    int *count = NULL;

    hmlpError_t Set( vector<OOCData<T>> *user_arg, 
        const vector<size_t> user_ids, 
        const vector<size_t> user_I, 
        const vector<size_t> user_J, Data<T> *user_KIJ, int *user_count )
    {
      arg  = user_arg;
      ids  = user_ids;
      I    = user_I;
      J    = user_J;
      KIJ  = user_KIJ;
      count = user_count;
      return HMLP_ERROR_SUCCESS;
    };

    /** Directly enqueue. */
    hmlpError_t DependencyAnalysis() 
    {
      this->TryEnqueue(); 
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t Execute( Worker* user_worker )
    {
      try
      {
        Data<T> C( I.size(), J.size(), 0 );
        assert( arg && KIJ );
        assert( KIJ->row() == I.size() && KIJ->col() == J.size() );

        for ( auto id : ids )
        {
          assert( id < arg->size() );
          OOCData<T> &X = (*arg)[ id ];
          /** Allocate temporary buffers. */
          Data<T> A( X.row(), I.size() );
          Data<T> B( X.row(), J.size() );

          for ( size_t j = 0; j < I.size(); j ++ )
            for ( size_t i = 0; i < X.row(); i ++ )
              A( i, j ) = X( i, I[ j ] );

          for ( size_t j = 0; j < J.size(); j ++ )
            for ( size_t i = 0; i < X.row(); i ++ )
              B( i, j ) = X( i, J[ j ] );

          /** Further create subtasks. */
          //gemm::xgemm( HMLP_OP_T, HMLP_OP_N, (T)1.0, A, B, (T)1.0, C );
          xgemm( "T", "N", I.size(), J.size(), X.row(),
              1.0, A.data(), A.row(), 
                   B.data(), B.row(), 
              1.0, C.data(), C.row() );
        }
     
        assert( KIJ->row() == I.size() && KIJ->col() == J.size() );
        for ( size_t j = 0; j < J.size(); j ++ )
          for ( size_t i = 0; i < I.size(); i ++ )
            #pragma omp atomic update
            (*KIJ)( i, j ) += C( i, j );
      }
      catch ( exception & e )
      {
        HANDLE_EXCEPTION( e );
      }
      return HMLP_ERROR_SUCCESS;
    };

};


template<typename T>
class CovReduceTask : public Task
{
  public:

    vector<CovTask<T>*> subtasks;

    int count = 0;

    const size_t batch_size = 32;

    hmlpError_t Set( vector<OOCData<T>> *arg, const vector<size_t> I, const vector<size_t> J, Data<T> *KIJ )
    {
      name = string( "CovReduce" );
      vector<size_t> ids;

      /** Create subtasks for each OOCData<T> */
      for ( size_t i = 0; i < arg->size(); i ++ )
      {
        ids.push_back( i );
        if ( ids.size() == batch_size )
        {
          subtasks.push_back( new CovTask<T>() );
          subtasks.back()->Submit();
          subtasks.back()->Set( arg, ids, I, J, KIJ, &count );
          ids.clear();
        }
      }      
      if ( ids.size() )
      {
        subtasks.push_back( new CovTask<T>() );
        subtasks.back()->Submit();
        subtasks.back()->Set( arg, ids, I, J, KIJ, &count );
        ids.clear();
      }
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t DependencyAnalysis()
    {
      for ( auto task : subtasks ) 
      {
        Scheduler::DependencyAdd( task, this );
        task->DependencyAnalysis();
      }
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t Execute( Worker* user_worker ) 
    {
      return HMLP_ERROR_SUCCESS;
    };
};
  
  
template<typename T>
class OOCCovMatrix : public VirtualMatrix<T>
{
  public:

    OOCCovMatrix( size_t d, size_t n, size_t nb, string filename )
    :
    VirtualMatrix<T>( d, d )
    {
      this->nb = nb;
      for ( int i = 0; i < n; i += nb )
      {
        int ib = min( nb, n - i );
        Samples.resize( Samples.size() + 1 );
        printf( "ib %d d %lu\n", ib, d );
        Samples.back().initFromFile( ib, d, filename + to_string( i ) );
        //X.Set( ib, d, filename + to_string( i ) );
      }

      int comm_rank; mpi::Comm_rank( MPI_COMM_WORLD, &comm_rank );
      int comm_size; mpi::Comm_size( MPI_COMM_WORLD, &comm_size );

      D.resize( d, 0 );
      #pragma omp parallel for
      for ( size_t i = comm_rank; i < d; i += comm_size ) 
      {
        D[ i ] = (*this)( i, i );
        if ( D[ i ] <= 0 ) D[ i ] = 1.0;
      }
      auto Dsend = D;
      mpi::Allreduce( Dsend.data(), D.data(), D.size(), MPI_SUM, MPI_COMM_WORLD );
      printf( "Finish diagonal\n" ); fflush( stdout );
    };


    /** Need additional support for diagonal evaluation */
    Data<T> Diagonal( const vector<size_t> &I )
    {
      Data<T> DII( I.size(), 1 );
      for ( auto i = 0; i < I.size(); i ++ ) DII[ i ] = D[ I[ i ] ];
      return DII;
    };

    /** ESSENTIAL: this is an abstract function  */
    virtual T operator()( size_t i, size_t j )
    {
      auto KIJ = (*this)( vector<size_t>( 1, i ), vector<size_t>( 1, j ) );
      return KIJ[ 0 ];
    };

    /** ESSENTIAL: return a submatrix */
    virtual Data<T> operator()
		( 
      const vector<size_t> &I, 
      const vector<size_t> &J 
    )
    {
      Data<T> KIJ( I.size(), J.size(), 0 );

      if ( !I.size() || !J.size() ) return KIJ;

      for ( auto &i : I ) assert( i < this->row() ); 
      for ( auto &j : J ) assert( j < this->col() ); 

      double beg = omp_get_wtime();
      double ooc_time = 0;

      //if ( hmlp_is_in_epoch_session() )
      if ( 0 )
      {
        auto *task = new CovReduceTask<T>();
        task->Set( &Samples, I, J, &KIJ );
        task->Submit();
        task->DependencyAnalysis();
        task->CallBackWhileWaiting();
      }
      else
      {
        for ( auto &X : Samples )
        {
          Data<T> A( X.row(), I.size() );
          Data<T> B( X.row(), J.size() );

          double ooc_beg = omp_get_wtime();

          for ( size_t j = 0; j < I.size(); j ++ )
            for ( size_t i = 0; i < X.row(); i ++ )
              A( i, j ) = X( i, I[ j ] );

          for ( size_t j = 0; j < J.size(); j ++ )
            for ( size_t i = 0; i < X.row(); i ++ )
              B( i, j ) = X( i, J[ j ] );

          ooc_time += omp_get_wtime() - ooc_beg;

          assert( !A.HasIllegalValue() );
          assert( !B.HasIllegalValue() );
          //printf( "I %lu J %lu X.row() %lu X.col() %lu\n", I[ 0 ], J[ 0 ], X.row(), X.col() );
          //for ( size_t i = 0; i < all_rows.size(); i ++ ) all_rows[ i ] = i;
          //auto A = X( all_rows, I );
          //printf( "I %lu J %lu X.row() %lu X.col() %lu Finish A\n", I[ 0 ], J[ 0 ], X.row(), X.col() );
          //auto B = X( all_rows, J );
          //printf( "I %lu J %lu X.row() %lu X.col() %lu Finish B\n", I[ 0 ], J[ 0 ], X.row(), X.col() );
          //gemm::xgemm( HMLP_OP_T, HMLP_OP_N, (T)1.0, A, B, (T)1.0, KIJ );
          {
            xgemm( "T", "N", I.size(), J.size(), X.row(),
                1.0, A.data(), A.row(), 
                     B.data(), B.row(), 
                1.0, KIJ.data(), KIJ.row() );
          }
        }
      }
      //assert( !KIJ.HasIllegalValue() );

      double KIJ_time = omp_get_wtime() - beg;

      if ( !reported && I.size() >= 512 && J.size() >= 512  )
      {
        printf( "KIJ %lu %lu in %lfs ooc %lfs\n", I.size(), J.size(), KIJ_time, ooc_time );
        reported = true;
      }


      return KIJ;
    };


    template<typename TINDEX>
    double flops( TINDEX na, TINDEX nb ) 
    {
      return 0.0;
    };

  private:

    bool reported = false;

    size_t n_samples = 0;

    size_t nb = 512;

    vector<T> D;

    /** n_samples / nb files, each with n_samples */
    vector<OOCData<T>> Samples;

}; /** end class OOCCovMatrix */
}; /** end namespace hmlp */

#endif /** define OOCCOVMATRIX_HPP */
