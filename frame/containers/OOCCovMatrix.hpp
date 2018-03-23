#ifndef OOCCOVMATRIX_HPP
#define OOCCOVMATRIX_HPP

#include <exception>

/** BLAS/LAPACK support */
#include <hmlp_blas_lapack.h>
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

    Lock *lock = NULL;


    void Set( vector<OOCData<T>> *user_arg, 
        const vector<size_t> user_ids, 
        const vector<size_t> user_I, 
        const vector<size_t> user_J, Data<T> *user_KIJ, Lock *user_lock )
    {
      arg  = user_arg;
      ids  = user_ids;
      I    = user_I;
      J    = user_J;
      KIJ  = user_KIJ;
      lock = user_lock;
    };

    /** Directly enqueue. */
    void DependencyAnalysis() { this->TryEnqueue(); };

    void Execute( Worker* user_worker )
    {
      try
      {
        Data<T> C( I.size(), J.size(), 0 );
        assert( arg && KIJ && lock );
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
     
        lock->Acquire();
        {
          assert( KIJ->row() == I.size() && KIJ->col() == J.size() );
          for ( size_t j = 0; j < J.size(); j ++ )
            for ( size_t i = 0; i < I.size(); i ++ )
              (*KIJ)( i, j ) += C( i, j );
        }
        lock->Release();
      }
      catch ( exception & e )
      {
        cout << "Standard execption: " << e.what() << endl;
      }
    };

};


template<typename T>
class CovReduceTask : public Task
{
  public:

    vector<CovTask<T>*> subtasks;

    Lock lock;

    const size_t batch_size = 32;

    void Set( vector<OOCData<T>> *arg,
        const vector<size_t> I, 
        const vector<size_t> J, Data<T> *KIJ )
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
          subtasks.back()->Set( arg, ids, I, J, KIJ, &lock );
          ids.clear();
        }
      }      
      if ( ids.size() )
      {
        subtasks.push_back( new CovTask<T>() );
        subtasks.back()->Submit();
        subtasks.back()->Set( arg, ids, I, J, KIJ, &lock );
        ids.clear();
      }
    };

    void DependencyAnalysis()
    {
      for ( auto task : subtasks ) 
      {
        Scheduler::DependencyAdd( task, this );
        task->DependencyAnalysis();
      }
      this->TryEnqueue();
    };

    void Execute( Worker* user_worker ) { /** Place holder. */ };
};
  
  
template<typename T>
class OOCCovMatrix : public VirtualMatrix<T>,
                     public SPDMatrixMPISupport<T>
{
  public:

    OOCCovMatrix( size_t d, size_t n, string filename )
    :
    VirtualMatrix<T>( d, d )
    {
      for ( int i = 0; i < n; i += nb )
      {
        int ib = min( nb, n - i );
        Samples.resize( Samples.size() + 1 );
        printf( "ib %d d %lu\n", ib, d );
        Samples.back().Set( ib, d, filename + to_string( i ) );
        //X.Set( ib, d, filename + to_string( i ) );
      }

      D.resize( d );
      #pragma omp parallel for
      for ( size_t i = 0; i < d; i ++ ) 
      {
        D[ i ] = (*this)( i, i );
        if ( D[ i ] <= 0 ) D[ i ] = 1.0;
      }
      //for ( size_t i = 0; i < d; i ++ ) D[ i ] = 10000; 
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

      //if ( hmlp_is_in_epoch_session() && hmlp_is_nested_queue_empty() )
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

          for ( size_t j = 0; j < I.size(); j ++ )
            for ( size_t i = 0; i < X.row(); i ++ )
              A( i, j ) = X( i, I[ j ] );

          for ( size_t j = 0; j < J.size(); j ++ )
            for ( size_t i = 0; i < X.row(); i ++ )
              B( i, j ) = X( i, J[ j ] );

          //printf( "I %lu J %lu X.row() %lu X.col() %lu\n", I[ 0 ], J[ 0 ], X.row(), X.col() );
          //for ( size_t i = 0; i < all_rows.size(); i ++ ) all_rows[ i ] = i;
          //auto A = X( all_rows, I );
          //printf( "I %lu J %lu X.row() %lu X.col() %lu Finish A\n", I[ 0 ], J[ 0 ], X.row(), X.col() );
          //auto B = X( all_rows, J );
          //printf( "I %lu J %lu X.row() %lu X.col() %lu Finish B\n", I[ 0 ], J[ 0 ], X.row(), X.col() );
          //gemm::xgemm( HMLP_OP_T, HMLP_OP_N, (T)1.0, A, B, (T)1.0, KIJ );
          xgemm( "T", "N", I.size(), J.size(), X.row(),
              1.0, A.data(), A.row(), 
                   B.data(), B.row(), 
              1.0, KIJ.data(), KIJ.row() );
        }
      }

      return KIJ;
    };

    virtual Data<T> PairwiseDistances
    ( 
      const vector<size_t> &I, 
      const vector<size_t> &J 
    )
    {
      return (*this)( I, J );
    };

  private:

    size_t n_samples = 0;

    size_t nb = 512;

    vector<T> D;

    /** n_samples / nb files, each with n_samples */
    vector<OOCData<T>> Samples;
    //OOCData<T> X;

}; /** end class OOCCovMatrix */
}; /** end namespace hmlp */

#endif /** define OOCCOVMATRIX_HPP */
