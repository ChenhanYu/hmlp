#ifndef IGOFMM_HPP
#define IGOFMM_HPP

#include <gofmm/igofmm.hpp>


namespace hmlp
{
namespace gofmm
{

template<typename T>
class DistFactor : public Factor<T>
{
  public:




    /**
     *  @brief Distributed factorization has two different implementations:
     *         ULV or SMW. ULV requires O( N ) work, however, the 
     *         parallelism diminishes because Ul and Ur have size 2s-by-s.
     *         In this case, only rank 0 and size / 2 in each communicator
     *         will participate in the factorization.
     *         On the other hand, SMW takes O( NlogN ) work, but the sizes
     *         of Ul and Ur are N-by-s. All MPI ranks will join the 
     *         factorization while creating the reduced system. 
     */ 
    template<bool SYMMETRIC>
    void Factorize
    ( 
      /** Ul,  nl-by-sl */
      hmlp::Data<T> &Ul, 
      /** Ur,  nr-by-sr */
      hmlp::Data<T> &Ur, 
      /** Vl,  nl-by-sr */
      hmlp::Data<T> &Vl,
      /** Vr,  nr-by-sr */
      hmlp::Data<T> &Vr
    )
    {
      /**
       *  leaf nodes will always be factorized using non-distributed Factor
       */ 
      assert( !isleaf );

      /** even SYMMETRIC this routine uses LU factorization */
      if ( this->IsSymmetric() )
      {
        /** the skeleton matrix Crl should be provided */
        assert( Crl.row() == sr ); assert( Crl.col() == sl );
      }
      else
      {
        /** both skeleton matrices Crl and Clr should be provided */
        assert( Clr.row() == sl ); assert( Clr.col() == sr );
        assert( Crl.row() == sr ); assert( Crl.col() == sl );
      }

      /** 
       *  clean up and begin with Z = eye( sl + sr ) =     | sl  sr
       *                                                ------------
       *                                                sl | Zrl Ztr
       *                                                sr | Zbl Zbr 
       */
      Z.resize( 0, 0 );
      Z.resize( sl + sr, sl + sr, 0.0 );
      for ( size_t i = 0; i < sl + sr; i ++ ) Z( i, i ) = 1.0;


      /**
       *  While doing ULV factorization, 
       *
       */ 
      if ( this->DoULVFactorization() )
      {
        /**
         *  Z = I + UR * C * VR' = [                 I  URl * Clr * VRr'
         *                            URr * Crl * VRl'                 I ]
         **/
        if ( this->IsSymmetric() ) /** Cholesky */
        {
          /** Zbl = URr * Crl * VRl' */
          hmlp::Data<T> Zbl = Crl;

          /** trmm */
          xtrmm
          ( 
            "Right", "Upper", "Transpose", "Non-unit",
            Zbl.row(), Zbl.col(),
            1.0,  Ul.data(),  Ul.row(),
                 Zbl.data(), Zbl.row()
          );

          /** trmm */
          xtrmm
          ( 
            "Left", "Upper", "Non-transpose", "Non-unit",
            Zbl.row(), Zbl.col(),
            1.0,  Ur.data(),  Ur.row(),
                 Zbl.data(), Zbl.row()
          );
          //printf( "Ur.row() %lu Zbl.row() %lu Zbl.col() %lu\n",
          //    Ur.row(), Zbl.row(), Zbl.col() );

          /** Zbl */
          for ( size_t j = 0; j < sl; j ++ )
          {
            for ( size_t i = 0; i < sr; i ++ )
            {
              Z( sl + i,      j ) = Zbl( i, j );
              Z(      j, sl + i ) = Zbl( i, j );
            }
          }

          /** LL' = potrf( Z ) */
          xpotrf( "Lower", Z.row(), Z.data(), Z.row() );
        }
        else /** TODO: no LU implementation for ULV */
        {
          printf( "no unsymmetric ULV implementation\n" );
          exit( 1 );
          /** pivoting row indices */
          ipiv.resize( Z.row(), 0 );
        }
      }
      else /** Sherman-Morrison-Woodbury */
      {



      }





    }; /** end Factorize() */

    /**
     *  @brief b - U * inv( Z ) * C * V' * b 
     */
    template<bool TRANS>
    void Solve( hmlp::View<T> &bl, hmlp::View<T> &br ) 
    {
      size_t nrhs = bl.col();





    }; /** end Solve() */


    /**
     *  @brief here x is the return value
     */ 
    void ULVForwardSolve( hmlp::View<T> &x )
    {

    }; /** end ULVForwardSolve() */



    void ULVBackwardSolve( hmlp::View<T> &x )
    {
      /** get the view of right hand sides */
      auto &b = bview;

    }; /** end ULVBackwardSolve() */




    /** RIGHT: V = [ P(:, 0:st-1) * Vl , P(:,st:st+sb-1) * Vr ] 
     *  LEFT:  U = [ Ul * P(:, 0:st-1)'; Ur * P(:,st:st+sb-1) ] */
    void Telescope
    ( 
      bool DO_INVERSE,
      /** n-by-s */
      hmlp::Data<T> &Pa,
      /** s-by-(sl+sr) */
      hmlp::Data<T> &Palr,
      /** nl-by-sl */
      hmlp::Data<T> &Pl,
      /** nr-by-sr */
      hmlp::Data<T> &Pr
    ) 
    {
    }; /** end Telescope() */




  private:

}; /** end class DistFactor */









/**
 *  @biref Top level factorization routine.
 */ 
template<typename NODE, typename T, typename TREE>
void DistFactorize( bool do_ulv_factorization, TREE &tree, T lambda )
{
  const bool AUTO_DEPENDENCY = true;
  const bool USE_RUNTIME = true;


  using NULLTASK           = hmlp::NULLTask<NODE>;
  using MPINULLTASK        = hmlp::NULLTask<MPINODE>;


  /** all task instances */
  NULLTASK nulltask;
  MPINULLTASK mpinulltask;
  SetupFactorTask<NODE, T> setupfactortask; 
  DistFactorizeTask<MPINODE, T> distfactorizetask;
  FactorizeTask<NODE, T> factorizetask; 

  /** setup the regularization parameter lambda */
  tree.setup.lambda = lambda;

  /** setup the symmetric type */
  tree.setup.issymmetric = true;

  /** setup factorization type */
  tree.setup.do_ulv_factorization = do_ulv_factorization;





  /** setup  */
  //tree.template TraverseUp<AUTO_DEPENDENCY, USE_RUNTIME>( setupfactortask );
  //tree.template ParTraverseUp<true>( );
  
  
  if ( USE_RUNTIME ) hmlp_run();
  //printf( "Execute setupfactortask\n" ); fflush( stdout );

  /** factorization */
  tree.template TraverseUp<AUTO_DEPENDENCY, USE_RUNTIME>( factorizetask );
  //printf( "Create factorizetask\n" ); fflush( stdout );
  if ( USE_RUNTIME ) hmlp_run();
  //printf( "Execute factorizetask\n" ); fflush( stdout );

}; /** end Factorize() */







}; /** end namespace gofmm */
}; /** end namespace hmlp */

#endif /** define IGOFMM_HPP */
