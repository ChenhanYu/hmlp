#ifndef GEMM_HPP
#define GEMM_HPP

#include <containters/view.hpp>


namespace hmlp
{
namespace gemm
{


template<typename T>
void xgemmTask(
    T alpha, hmlp::View<T> &A, 
             hmlp::View<T> &B, 
    T beta,  hmlp::View<T> &C )
{
}; /** end */


/**
 *  @brief
 */ 
template<size_t NB = 512, typename T>
void xgemm_var1( 
    T alpha, hmlp::View<T> &A, 
             hmlp::View<T> &B, 
    T beta,  hmlp::View<T> &C )
{
  /** all subviews */
  hmlp::View<T> AL, AR, 
                A0, A1, A2;
  hmlp::View<T> BT, BB, 
                B0, B1, B2;
  
  
  A.partition1x2( AL, AR, 0, LEFT );
  B.partition2x1( BT,
                  BB,     0, TOP  ); 

  while ( AL.col() < A.col() )
  {
    size_t b = std::min( AR.col(), NB );

    /** repartition A */
    Repartition1x2To1x3( AL,      AR,
                         /** **** */
                         A0,  A1, A2, b, RIGHT );
    /** repartition B */
    Repartition2x1To3x1( BT, /**/ B0,
                             /**/ B1,
                         BB, /**/ B2, b, BOTTOM );

    /** --------------------------------------------------- */
    xgemmTask( alpha, A, B, beta, C );
    /** --------------------------------------------------- */

    /** merge A */
    ContinueWith1x3To1x2( AL,      AR,
                          /** **** */
                          A0,  A1, A2, LEFT );
    /** merge B */
    ContinueWith3x1To2x1( BT, /**/ B0,
                              /**/ B1,
                          BB, /**/ B2,  TOP );

  } /** end while */

}; /** end xgemm_var1() */


/**
 *  @brief [ A * BL + CL, A * BR + CR ] 
 */ 
template<size_t NB = 512, typename T>
void xgemm_var2( 
    T alpha, hmlp::View<T> &A, 
             hmlp::View<T> &B, 
    T beta,  hmlp::View<T> &C )
{
  /** all subviews */
  hmlp::View<T> CL, CR, 
                C0, C1, C2;
  hmlp::View<T> BL, BR, 
                B0, B1, B2;
  
  C.partition1x2( CL, CR, 0, LEFT );
  B.partition1x2( BL, BR, 0, LEFT );

  while ( BL.col() < B.col() )
  {
    size_t b = std::min( BR.col(), NB );

    /** repartition C */
    Repartition1x2To1x3( CL,      CR,
                         /** **** */
                         C0,  C1, C2, b, RIGHT );
    /** repartition B */
    Repartition1x2To1x3( BL,      BR,
                         /** **** */
                         B0,  B1, B2, b, RIGHT );

    /** --------------------------------------------------- */
    xgemm_var1( alpha, A, B1, beta, C1 );
    /** --------------------------------------------------- */

    /** merge C */
    ContinueWith1x3To1x2( CL,      CR,
                          /** **** */
                          C0,  C1, C2, LEFT );
    /** merge B */
    ContinueWith1x3To1x2( BL,      BR,
                          /** **** */
                          B0,  B1, B2, LEFT );

  } /** end while */

}; /** end xgemm_var2() */


/**
 *  @brief [ AT * B + CT; AB * B + CB ] 
 */ 
template<size_t NB = 512, typename T>
void xgemm_var3( 
    T alpha, hmlp::View<T> &A, 
             hmlp::View<T> &B, 
    T beta,  hmlp::View<T> &C )
{
  /** all subviews */
  hmlp::View<T> AT, A0, CT, C0, 
                AB, A1, CB, C1,
                    A2,     C2;

  A.partition2x1( AT,
                  AB,     0, TOP  ); 
  C.partition2x1( CT,
                  CB,     0, TOP  ); 

  while ( AT.row() < A.row() )
  {
    size_t b = std::min( AB.row(), NB );

    /** repartition A */
    Repartition2x1To3x1( AT, /**/ A0,
                             /**/ A1,
                         AB, /**/ A2, b, BOTTOM );
    /** repartition B */
    Repartition2x1To3x1( CT, /**/ C0,
                             /**/ C1,
                         CB, /**/ C2, b, BOTTOM );

    /** --------------------------------------------------- */
    xgemm_var2( alpha, A1, B, beta, C1 );
    /** --------------------------------------------------- */

    /** merge A */
    ContinueWith3x1To2x1( AT, /**/ A0,
                              /**/ A1,
                          AB, /**/ A2,  TOP );
    /** merge C */
    ContinueWith3x1To2x1( CT, /**/ C0,
                              /**/ C1,
                          CB, /**/ C2,  TOP );
  }; /** end while */

}; /** end xgemm_var3() */



}; /** end namespace gemm */
}; /** end namespace hmlp */


#endif /** define GEMM_HPP */
