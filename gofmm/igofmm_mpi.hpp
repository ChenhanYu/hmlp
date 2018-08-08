#ifndef IGOFMM_MPI_HPP
#define IGOFMM_MPI_HPP

/** MPI support. */
#include <hmlp_mpi.hpp>
/** Shared-memory Inv-GOFMM templates. */
#include <igofmm.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;


namespace hmlp
{
namespace mpigofmm
{

template<typename NODE, typename T>
class DistSetupFactorTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "PSF" );
      label = to_string( arg->treelist_id );
    };

    void DependencyAnalysis() { arg->DependOnChildren( this ); };

    void Execute( Worker *user_worker )
    {
      auto *node   = arg;
      auto &data   = node->data;
      auto *setup  = node->setup;

      /** Get node MPI size and rank. */
      auto comm = node->GetComm();
      int  size = node->GetCommSize();
      int  rank = node->GetCommRank();

      if ( size == 1 ) return gofmm::SetupFactor<NODE, T>( arg );

      size_t n, nl, nr, s, sl, sr;
      bool issymmetric, do_ulv_factorization;
  
      issymmetric = setup->IsSymmetric();
      do_ulv_factorization = setup->do_ulv_factorization;
      n  = node->n;
      nl = 0;
      nr = 0;
      s  = data.skels.size();
      sl = 0;
      sr = 0;

      mpi::Bcast( &n, 1, 0, comm );
      mpi::Bcast( &s, 1, 0, comm );

      if ( rank < size / 2 )
      {
        nl = node->child->n;
        sl = node->child->data.s;
      }
      else
      {
        nr = node->child->n;
        sr = node->child->data.s;
      }

      mpi::Bcast( &nl, 1,        0, comm );
      mpi::Bcast( &nr, 1, size / 2, comm );
      mpi::Bcast( &sl, 1,        0, comm );
      mpi::Bcast( &sr, 1, size / 2, comm );

      data.SetupFactor( issymmetric, do_ulv_factorization,
        node->isleaf, !node->l, n, nl, nr, s, sl, sr );

      //printf( "n %lu nl %lu nr %lu s %lu sl %lu sr %lu\n",
      //    n, nl, nr, s, sl, sr ); fflush( stdout );
    };
};



template<typename NODE, typename T>
class DistFactorizeTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "PULVF" );
      label = to_string( arg->treelist_id );
      /** We don't know the exact cost here */
      cost = 5.0;
    };

    void DependencyAnalysis() { arg->DependOnChildren( this ); };

    void Execute( Worker *user_worker )
    {
      auto *node   = arg;
      auto &data   = node->data;
      auto *setup  = node->setup;
      auto &K      = *setup->K;

      /** Get node MPI size and rank. */
      auto comm = node->GetComm();
      int  size = node->GetCommSize();
      int  rank = node->GetCommRank();
      mpi::Status status;

      if ( size == 1 ) return gofmm::Factorize<NODE, T>( node );

      if ( rank == 0 )
      {
        /** Recv Ur from my sibling.*/
        Data<T> Ur, &Ul = node->child->data.U;
        mpi::RecvData( Ur, size / 2, comm );
        //printf( "Ur %lux%lu\n", Ur.row(), Ur.col() ); fflush( stdout );
        /** TODO: implement symmetric ULV factorization. */
        Data<T> Vl, Vr;
        /** Recv Zr from my sibing. */
        View<T> &Zl = node->child->data.Zbr;
        Data<T>  Zr;
        mpi::RecvData( Zr, size / 2, comm );
        View<T> Zrv( Zr );
        /** Get the skeleton rows and columns. */
        auto &amap = node->child->data.skels;
        vector<size_t> bmap( data.sr );
        mpi::Recv( bmap.data(), bmap.size(), size / 2, 20, comm, &status );

        /** Get the skeleton row and column basis. */
        data.Crl = K( bmap, amap );

        if ( node->l )
        {
          data.Telescope( false, data.U, data.proj, Ul, Ur );
          data.Orthogonalization();
        }
        data.PartialFactorize( Zl, Zrv, Ul, Ur, Vl, Vr );
      }

      if ( rank == size / 2 )
      {
        /** Send Ur to my sibling. */
        auto &Ur = node->child->data.U;
        mpi::SendData( Ur, 0, comm );
        /** Send Zr to my sibing. */
        auto  Zr = node->child->data.Zbr.toData();
        mpi::SendData( Zr, 0, comm );
        /** Evluate the skeleton rows and columns. */
        auto &bmap = node->child->data.skels;
        mpi::Send( bmap.data(), bmap.size(), 0, 20, comm );
      }
    };
}; /** end class DistFactorizeTask */



template<typename NODE, typename T>
class DistFactorTreeViewTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "PTV" );
      label = to_string( arg->treelist_id );
      /** We don't know the exact cost here */
      cost = 5.0;
    };

    void DependencyAnalysis() { arg->DependOnParent( this ); };

    void Execute( Worker *user_worker )
    {
      auto *node   = arg;
      auto &data   = node->data;
      auto *setup  = node->setup;
      auto &input  = *(setup->input);
      auto &output = *(setup->output);

      /** Get node MPI size and rank. */
      int size = node->GetCommSize();
      int rank = node->GetCommRank();

      /** Create contigious matrix view for output at root level. */
      data.bview.Set( output );

      /** Case: distributed leaf node. */
      if ( size == 1 ) return gofmm::SolverTreeView( arg );

      if ( rank == 0 )
      {
        /** Allocate working buffer for ULV solve. */
        data.B.resize( data.sl + data.sr, input.col() );
        /** Partition B = [ Bf; Bc ] with matrix view. */
        data.Bv.Set( data.B );
        data.Bv.Partition2x1( data.Bf,
                              data.Bc,  data.s, BOTTOM );
        /** Bv = [ cdata.Bp; Bsibling ], and receive Bsiling. */
        auto &cdata = node->child->data;
        data.Bv.Partition2x1( cdata.Bp,
                               data.Bsibling, data.sl, TOP );
      }

      if ( rank == size / 2 )
      {
        /** Allocate working buffer for mpi::send(). */
        data.B.resize( data.sr, input.col() );
        /** Directly map cdata.Bp to parent's data.B. */
        auto &cdata = node->child->data;
        cdata.Bp.Set( data.B );
      }

    };
}; /** end class DistSolverTreeViewTask */




template<typename NODE, typename T>
class DistULVForwardSolveTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "PULVS1" );
      label = to_string( arg->treelist_id );
      /** We don't know the exact cost here */
      cost = 5.0;
      /** "High" priority */
      priority = true;
    };

    void DependencyAnalysis() { arg->DependOnChildren( this ); };

    void Execute( Worker *user_worker )
    {
      //printf( "[BEG] level-%lu\n", arg->l ); fflush( stdout );
      auto *node = arg;
      auto &data = node->data;
      /** Get node MPI size and rank. */
      auto comm = node->GetComm();
      int  size = node->GetCommSize();
      int  rank = node->GetCommRank();
      mpi::Status status;

      /** Perform the forward step locally on distributed leaf nodes. */
      if ( size == 1 ) return data.ULVForward();

      if ( rank == 0 )
      {
        auto Br = data.Bsibling.toData();
        /** Receive Br from my sibling. */
        mpi::Recv( Br.data(), Br.size(), size / 2, 0, comm, &status );
        /** Copy valus from temporary buffer to the view of my B. */
        data.Bsibling.CopyValuesFrom( Br );
        /** Perform the forward step locally. */
        data.ULVForward();
      }

      if ( rank == size / 2 )
      {
        auto &Br = data.B;
        /** Send Br from my sibling. */
        mpi::Send( Br.data(), Br.size(), 0, 0, comm );
      }
      //printf( "[END] level-%lu\n", arg->l ); fflush( stdout );
    };
}; /** end class DistULVForwardSolveTask */


template<typename NODE, typename T>
class DistULVBackwardSolveTask : public Task
{
  public:

    NODE *arg = NULL;

    void Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "PULVS2" );
      label = to_string( arg->treelist_id );
      /** We don't know the exact cost here */
      cost = 5.0;
      /** "High" priority */
      priority = true;
    };

    void DependencyAnalysis() { arg->DependOnParent( this ); };

    void Execute( Worker *user_worker )
    {
      //printf( "[BEG] level-%lu\n", arg->l ); fflush( stdout );
      auto *node = arg;
      auto &data = node->data;
      /** Get node MPI size and rank. */
      auto comm = node->GetComm();
      int  size = node->GetCommSize();
      int  rank = node->GetCommRank();
      mpi::Status status;

      /** Perform the forward step locally on distributed leaf nodes. */
      if ( size == 1 ) return data.ULVBackward();

      if ( rank == 0 )
      {
        /** Perform the backward step locally. */
        data.ULVBackward();
        /** Pack Br using matrix view Bsibling. */
        auto Br = data.Bsibling.toData();
        /** Send Br to my sibling. */
        mpi::Send( Br.data(), Br.size(), size / 2, 0, comm );
      }

      if ( rank == size / 2 )
      {
        auto &Br = data.B;
        /** Receive Br from my sibling. */
        mpi::Recv( Br.data(), Br.size(), 0, 0, comm, &status );
      }
      //printf( "[END] level-%lu\n", arg->l ); fflush( stdout );
    };

}; /** end class DistULVBackwardSolveTask */







/** @biref Top-level factorization routine. */ 
template<typename T, typename TREE>
void DistFactorize( TREE &tree, T lambda )
{
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;

  /** setup the regularization parameter lambda */
  tree.setup.lambda = lambda;
  /** setup factorization type */
  tree.setup.do_ulv_factorization = true;

  /** Traverse the tree upward to prepare thr ULV factorization. */
  gofmm::SetupFactorTask<NODE, T> seqSETUPFACTORtask; 
  DistSetupFactorTask<MPINODE, T> parSETUPFACTORtask;

  mpi::PrintProgress( "[BEG] DistFactorize setup ...\n", tree.GetComm() ); 
  tree.DependencyCleanUp();
  tree.LocaTraverseUp( seqSETUPFACTORtask );
  tree.DistTraverseUp( parSETUPFACTORtask );
  tree.ExecuteAllTasks();
  mpi::PrintProgress( "[END] DistFactorize setup ...\n", tree.GetComm() ); 

  mpi::PrintProgress( "[BEG] DistFactorize ...\n", tree.GetComm() ); 
  gofmm::FactorizeTask<   NODE, T> seqFACTORIZEtask; 
  DistFactorizeTask<MPINODE, T> parFACTORIZEtask; 
  tree.LocaTraverseUp( seqFACTORIZEtask );
  tree.DistTraverseUp( parFACTORIZEtask );
  tree.ExecuteAllTasks();
  mpi::PrintProgress( "[END] DistFactorize ...\n", tree.GetComm() ); 

}; /** end DistFactorize() */


template<typename T, typename TREE>
void DistSolve( TREE &tree, Data<T> &input )
{
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;

  /** Attach the pointer to the tree structure. */
  tree.setup.input  = &input;
  tree.setup.output = &input;

  /** All factorization tasks. */
  gofmm::SolverTreeViewTask<NODE> seqTREEVIEWtask;
  DistFactorTreeViewTask<MPINODE, T>   parTREEVIEWtask;
  gofmm::ULVForwardSolveTask<NODE, T> seqFORWARDtask;
  DistULVForwardSolveTask<MPINODE, T>   parFORWARDtask;
  gofmm::ULVBackwardSolveTask<NODE, T> seqBACKWARDtask;
  DistULVBackwardSolveTask<MPINODE, T>   parBACKWARDtask;

  tree.DistTraverseDown( parTREEVIEWtask );
  tree.LocaTraverseDown( seqTREEVIEWtask );
  tree.LocaTraverseUp( seqFORWARDtask );
  tree.DistTraverseUp( parFORWARDtask );
  tree.DistTraverseDown( parBACKWARDtask );
  tree.LocaTraverseDown( seqBACKWARDtask );
  mpi::PrintProgress( "[PREP] DistSolve ...\n", tree.GetComm() ); 
  tree.ExecuteAllTasks();
  mpi::PrintProgress( "[DONE] DistSolve ...\n", tree.GetComm() ); 

}; /** end DistSolve() */


/**
 *  @brief Compute the average 2-norm error. That is given
 *         lambda and weights, 
 */ 
template<typename TREE, typename T>
void ComputeError( TREE &tree, T lambda, Data<T> weights, Data<T> potentials )
{
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;

  auto comm = tree.GetComm();
  auto size = tree.GetCommSize();
  auto rank = tree.GetCommRank();


  size_t n    = weights.row();
  size_t nrhs = weights.col();

  /** Shift lambda and make it a column vector. */
  Data<T> rhs( n, nrhs );
  for ( size_t j = 0; j < nrhs; j ++ )
    for ( size_t i = 0; i < n; i ++ )
      rhs( i, j ) = potentials( i, j ) + lambda * weights( i, j );

  /** Solver. */
  DistSolve( tree, rhs );


  /** Compute relative error = sqrt( err / nrm2 ) for each rhs. */
  if ( rank == 0 )
  {
    printf( "========================================================\n" );
    printf( "Inverse accuracy report\n" );
    printf( "========================================================\n" );
    printf( "#rhs,  max err,        @,  min err,        @,  relative \n" );
    printf( "========================================================\n" );
  }

  size_t ntest = 10;
  T total_err  = 0.0;
  T total_nrm  = 0.0;
  T total_max  = 0.0;
  T total_min  = 0.0;
  for ( size_t j = 0; j < std::min( nrhs, ntest ); j ++ )
  {
    /** Counters */
    T nrm2 = 0.0, err2 = 0.0;
    T max2 = 0.0, min2 = std::numeric_limits<T>::max(); 

    for ( size_t i = 0; i < n; i ++ )
    {
      T sse = rhs( i, j ) - weights( i, j );
      assert( rhs( i, j ) == rhs( i, j ) );
      sse = sse * sse;
      nrm2 += weights( i, j ) * weights( i, j );
      err2 += sse;
      max2 = std::max( max2, sse );
      min2 = std::min( min2, sse );
    }

    mpi::Allreduce( &err2, &total_err, 1, MPI_SUM, comm );
    mpi::Allreduce( &nrm2, &total_nrm, 1, MPI_SUM, comm );
    mpi::Allreduce( &max2, &total_max, 1, MPI_MAX, comm );
    mpi::Allreduce( &min2, &total_min, 1, MPI_MIN, comm );

    total_err += std::sqrt( total_err / total_nrm );
    total_max  = std::sqrt( total_max );
    total_min  = std::sqrt( total_min );

    if ( rank == 0 )
    {
      printf( "%4lu,  %3.1E,  %7lu,  %3.1E,  %7lu,   %3.1E\n", 
          j, total_max, (size_t)0,  total_min, (size_t)0, total_err );
    }
  }
  if ( rank == 0 )
  {
    printf( "========================================================\n" );
  }
};


}; /** end namespace mpigofmm */
}; /** end namespace hmlp */

#endif /** define IGOFMM_MPI_HPP */
