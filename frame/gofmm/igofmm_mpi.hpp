#ifndef IGOFMM_MPI_HPP
#define IGOFMM_MPI_HPP

#include <hmlp_mpi.hpp>
#include <gofmm/igofmm.hpp>

using namespace std;
using namespace hmlp;


namespace hmlp
{
namespace mpigofmm
{

//template<typename T>
//class DistFactor : public Factor<T>
//{
//  public:
//  private:
//}; /** end class DistFactor */


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

      /** We don't know the exact cost here */
      cost = 5.0;
      /** "High" priority */
      priority = true;
    };

    void DependencyAnalysis() { arg->DependOnChildren( this ); };

    void Execute( Worker *user_worker )
    {
      auto *myself = arg;
      NODE *parent = (NODE*)myself->parent;
      auto *lchild = myself->lchild;
      auto *rchild = myself->rchild;
      auto *xchild = myself->child;

      mpi::Comm comm = myself->GetComm();
      int rank = myself->GetCommRank();
      int size = myself->GetCommSize();

      size_t n, nl, nr, s, sl, sr;
      bool issymmetric, do_ulv_factorization, isleft;
  
      issymmetric = myself->setup->issymmetric;
      do_ulv_factorization = myself->setup->do_ulv_factorization;
      isleft = false;
      n  = myself->n;
      nl = 0;
      nr = 0;
      /** TODO: may need to BCAST s from 0. */
      s  = myself->data.skels.size();
      sl = 0;
      sr = 0;


      if ( myself->GetCommSize() == 1 )
      {
        if ( !myself->isleaf )
        {
          nl = lchild->n;
          nr = rchild->n;
          sl = lchild->data.skels.size();
          sr = rchild->data.skels.size();
        }
      }
      else /** */
      {
        if ( myself->GetCommRank() < myself->GetCommSize() / 2 )
        {
          nl = xchild->n;
          sl = xchild->data.s;
        }
        else
        {
          nr = xchild->n;
          sr = xchild->data.s;
        }
        mpi::Bcast( &nl, 1,        0, comm );
        mpi::Bcast( &nr, 1, size / 2, comm );
        mpi::Bcast( &sl, 1,        0, comm );
        mpi::Bcast( &sr, 1, size / 2, comm );
      }

      if ( parent )
      {
        if ( parent->GetCommRank() < parent->GetCommSize() / 2 ) isleft = true;
      }

      myself->data.SetupFactor
      (
        issymmetric, do_ulv_factorization,
        isleft, myself->isleaf, !myself->l,
        n, nl, nr,
        s, sl, sr 
      );

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
      /** "High" priority */
      priority = true;
    };


    void DependencyAnalysis() { arg->DependOnChildren( this ); };


    void Execute( Worker *user_worker )
    {
      auto *myself = arg;
      NODE *parent = (NODE*)myself->parent;
      auto *lchild = myself->lchild;
      auto *rchild = myself->rchild;
      auto *xchild = myself->child;

      mpi::Status status;
      mpi::Comm comm = myself->GetComm();
      int rank = myself->GetCommRank();
      int size = myself->GetCommSize();

      auto &data = myself->data;
      auto &setup = myself->setup;
      auto &K = *setup->K;
      auto &proj = data.proj;

      if ( myself->GetCommSize() == 1 )
      {
        gofmm::Factorize<NODE, T>( myself );
      }
      else
      {
        if ( myself->GetCommRank() == 0 )
        {
          /** Recv Ur from my sibling.*/
          auto &Ul = xchild->data.U;
          Data<T> Ur( data.sr, data.s );
          mpi::RecvVector( Ur, size / 2,  0, comm, &status );
          /** TODO: implement symmetric ULV factorization. */
          Data<T> Vl, Vr;
          /** Recv Zr from my sibing. */
          auto &Zl = xchild->data.Zbr;
          Data<T> Zr( data.sr, data.sr );
          mpi::RecvVector( Zr, size / 2, 10, comm, &status );
          View<T> Zrv( Zr );

          /** Get the skeleton rows and columns. */
          auto &amap = xchild->data.skels;
          vector<size_t> bmap( data.sr );
          mpi::RecvVector( bmap, size / 2, 20, comm, &status );
          myself->data.Crl = K( bmap, amap );

          if ( parent )
          {
            data.Telescope( false, data.U, proj, Ul, Ur );
            data.Orthogonalization();
          }
          data.PartialFactorize( Zl, Zrv, Ul, Ur, Vl, Vr );
        }

        if ( myself->GetCommRank() == myself->GetCommSize() / 2 )
        {
          /** Send Ur to my sibling. */
          auto &Ur = xchild->data.U;
          mpi::SendVector( Ur, 0,  0, comm );
          /** Send Zr to my sibing. */
          auto  Zr = xchild->data.Zbr.toData();
          mpi::SendVector( Zr, 0, 10, comm );
          /** Evluate the skeleton rows and columns. */
          auto &bmap = xchild->data.skels;
          mpi::SendVector( bmap, 0, 20, comm );
        }
      }
    };
};



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
      /** "High" priority */
      priority = true;
    };

    void DependencyAnalysis() { arg->DependOnParent( this ); };

    void Execute( Worker *user_worker )
    {
    };
};




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
    };
};


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
    };
};







/**
 *  @biref Top level factorization routine.
 */ 
template<typename T, typename TREE>
void DistFactorize( TREE &tree, T lambda )
{
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;

  /** setup the regularization parameter lambda */
  tree.setup.lambda = lambda;
  /** setup the symmetric type */
  tree.setup.issymmetric = true;
  /** setup factorization type */
  tree.setup.do_ulv_factorization = true;

  /** Traverse the tree upward to prepare thr ULV factorization. */
  gofmm::SetupFactorTask<NODE, T> seqSETUPFACTORtask; 
  DistSetupFactorTask<MPINODE, T> parSETUPFACTORtask;

  tree.DependencyCleanUp();
  tree.LocaTraverseUp( seqSETUPFACTORtask );
  tree.DistTraverseUp( parSETUPFACTORtask );
  hmlp_run();
  printf( "Finish SETUPFACTOR\n" ); fflush( stdout );

  gofmm::FactorizeTask<   NODE, T> seqFACTORIZEtask; 
  DistFactorizeTask<MPINODE, T> parFACTORIZEtask; 
  tree.DependencyCleanUp();
  tree.LocaTraverseUp( seqFACTORIZEtask );
  tree.DistTraverseUp( parFACTORIZEtask );
  hmlp_run();
  printf( "Finish FACTORIZE\n" ); fflush( stdout );

}; /** end DistFactorize() */


template<typename T, typename TREE>
void DistSolve( TREE &tree, Data<T> &input )
{
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;

  /** Attach the pointer to the tree structure. */
  tree.setup.input  = &input;

  /** clean up all dependencies on tree nodes */
  gofmm::TreeViewTask<NODE> seqTREEVIEWtask;
  DistFactorTreeViewTask<MPINODE, T>   parTREEVIEWtask;
  gofmm::ULVForwardSolveTask<NODE, T> seqFORWARDtask;
  DistULVForwardSolveTask<MPINODE, T>   parFORWARDtask;
  gofmm::ULVBackwardSolveTask<NODE, T> seqBACKWARDtask;
  DistULVBackwardSolveTask<MPINODE, T>   parBACKWARDtask;

  tree.DependencyCleanUp();
  tree.DistTraverseDown( parTREEVIEWtask );
  tree.LocaTraverseDown( seqTREEVIEWtask );
  tree.LocaTraverseUp( seqFORWARDtask );
  tree.DistTraverseUp( parFORWARDtask );
  tree.DistTraverseDown( parBACKWARDtask );
  tree.LocaTraverseDown( seqBACKWARDtask );
  hmlp_run();

}; /** end DistSolve() */


/**
 *  @brief Compute the average 2-norm error. That is given
 *         lambda and weights, 
 */ 
template<typename TREE, typename T>
void ComputeError( TREE &tree, T lambda, 
    Data<T> weights, 
    Data<T> potentials )
{
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;

  size_t n    = weights.row();
  size_t nrhs = weights.col();

  printf( "Shift lambda\n" ); fflush( stdout );
  /** shift lambda and make it a column vector */
  Data<T> rhs( n, nrhs );
  for ( size_t j = 0; j < nrhs; j ++ )
    for ( size_t i = 0; i < n; i ++ )
      rhs( i, j ) = potentials( i, j ) + lambda * weights( i, j );

  /** Solver */
  DistSolve( tree, rhs );
  printf( "Finish distributed solve\n" ); fflush( stdout );


  /** Compute relative error = sqrt( err / nrm2 ) for each rhs. */
  printf( "========================================================\n" );
  printf( "Inverse accuracy report\n" );
  printf( "========================================================\n" );
  printf( "#rhs,  max err,        @,  min err,        @,  relative \n" );
  printf( "========================================================\n" );
  size_t ntest = 10;
  T total_err  = 0.0;
  for ( size_t j = 0; j < std::min( nrhs, ntest ); j ++ )
  {
    /** Counters */
    T nrm2 = 0.0, err2 = 0.0;
    T max2 = 0.0, min2 = std::numeric_limits<T>::max(); 
    /** indecies */
    size_t maxi = 0, mini = 0;

    for ( size_t i = 0; i < n; i ++ )
    {
      T sse = rhs( i, j ) - weights( i, j );
      assert( rhs( i, j ) == rhs( i, j ) );
      sse = sse * sse;

      nrm2 += weights( i, j ) * weights( i, j );
      err2 += sse;

      //printf( "%lu %3.1E\n", i, sse );


      if ( sse > max2 ) { max2 = sse; maxi = i; }
      if ( sse < min2 ) { min2 = sse; mini = i; }
    }
    total_err += std::sqrt( err2 / nrm2 );

    printf( "%4lu,  %3.1E,  %7lu,  %3.1E,  %7lu,   %3.1E\n", 
        j, std::sqrt( max2 ), maxi, std::sqrt( min2 ), mini, 
        std::sqrt( err2 / nrm2 ) );
  }
  printf( "========================================================\n" );
  printf( "                             avg over %2lu rhs,   %3.1E \n",
      std::min( nrhs, ntest ), total_err / std::min( nrhs, ntest ) );
  printf( "========================================================\n\n" );




};


}; /** end namespace mpigofmm */
}; /** end namespace hmlp */

#endif /** define IGOFMM_MPI_HPP */
