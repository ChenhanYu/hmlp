#ifndef HFAMILY_HPP
#define HFAMILY_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <random>


#include <hmlp.h>
#include <hmlp_blas_lapack.h>
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>
#include <tree.hpp>
#include <skel.hpp>

#define DEBUG_IASKIT 1

namespace hmlp
{
namespace hfamily
{


typedef enum 
{
  HODLR,
  PHSS,
  HSS,
  ASKIT
} HFamilyType;

typedef enum 
{
  UV,
  COLUMNID,
  ROWID,
  CUR,
  LOWRANKPLUSSPARSE
} OffdiagonalType;

//template<HFamilyType TH, OffdiagonalType TOFFDIAG, typename T>
template<typename T>
class Factor
{
  public:

    Factor() {};
    
    void Setup
    ( 
      bool isleaf, bool isroot,
      std::size_t n, 
      std::size_t n1,  std::size_t n2,
      std::size_t s12, std::size_t s21
    )
    {
      this->isleaf = isleaf;
      this->isroot = isroot;
      this->n = n;
      this->n1 = n1;
      this->n2 = n2;
      this->s12 = s12;
      this->s21 = s21;
    };

    void Setup
    (
      bool isleaf, bool isroot,
      std::size_t n, 
      std::size_t n1,  std::size_t n2,
      std::size_t s12, std::size_t s21,
      hmlp::Data<T> &U,
      hmlp::Data<T> &V
    )
    {
      Setup( isleaf, isroot, n, n1, n2, s12, s21 );

      //assert( U.row() == n );
      //assert( U.col() == n );
    };

    void Factorize( hmlp::Data<T> &K ) 
    {
      assert( isleaf );
      assert( K.row() == n );
      assert( K.col() == n );

      /** initialize */
      Z = K;
      ipiv.resize( Z.row(), 0 );

      /** LU factorization */
      xgetrf( n, n, Z.data(), n, ipiv.data() );

      /** inv( K ) * U */
      xgetrs( "N", n, s12, Z.data(), n, ipiv.data(), U.data(), n );
    };

    /** Also compute inv( K ) * P' */
    void Factorize( hmlp::Data<T> &K, hmlp::Data<T> &P ) 
    {
      assert( P.row() == s12 );
      assert( P.col() == n );

      /** inv( K )* U = inv( K ) * P' */
      U.resize( n, s12 );
      for ( size_t j = 0; j < n; j ++ )
        for ( size_t i = 0; i < s12; i ++ )
          U[ i * n + j ] = P[ j * s12 + i ];

      /** K = LU */
      Factorize( K );
    };


    void Factorize( hmlp::Data<T> &Ul, hmlp::Data<T> &Vl,
                    hmlp::Data<T> &Ur, hmlp::Data<T> &Vr )
    {
      assert( !isleaf );
    };

    void Factorize( hmlp::Data<T> &Ul, hmlp::Data<T> &Vl,
                    hmlp::Data<T> &Ur, hmlp::Data<T> &Vr,
                    hmlp::Data<T> &P )
    {
      /** initialize Z = zero( s12 + s21 ) */
      Z.resize( s12 + s21, s12 + s21, 0.0 );

      /** VrUr */
      xgemm( "N", "N", s21, s12, n2,  )
    };


    void Factorize( hmlp::Data<T> $Ul, hmlp::Data<T> &Slr, hmlp::Data<T> $Ur )
    {
      assert( !isleaf );
    };

    void Solve( hmlp::Data<T> &b ) 
    {
    };

    void Telescope( hmlp::Data<T> &parent ) 
    {
    };

  private:

    bool isleaf;

    bool isroot;

    size_t n;

    size_t n1;

    size_t n2;

    size_t s12;

    size_t s21;

    // Reduced system Z = [ I  VU   if ( HODLR || p-HSS )
    //                      VU  I ]
    //                Z = [ 
    hmlp::Data<T> Z;

    /** pivoting rows */
    std::vector<int> ipiv;
    

    /** Low-rank */
    hmlp::Data<T> U; 
    hmlp::Data<T> V; 

    /** r21 * r12 */
    hmlp::Data<T> Sigma12; 

    /** r12 * r21, if ( SYMMETROC ) Sigma21 = Sigma12' */
    hmlp::Data<T> Sigma21;

    /** n1 * n2 */
    //hmlp::CSC<false, T> S12;

    /** n2 * n1, if ( SYMMETRIC ) S21 = S12' */
    //hmlp::CSC<false, T> S21; 
};


//template<typename NODE>
//void Factorize( NODE *node )
//{
//  if ( !node ) return;
//
//  auto &kernel = node->setup->kernel;
//  auto &X = node->setup->X;
//
//  auto &data = node->data; 
//  auto *lchild = node->lchild;
//  auto *rchild = node->rchild;
//
//  if ( node->isleaf )
//  {
//    auto lids = node->lids;
//    auto A = X( lids );
//    data.VU = kernel( A, A );
//
//  }
//  else
//  {
//  }
//}; // end Factorize()
//









}; // end namespace iaskit
}; // end namespace hmlp

#endif // define IASKIT_HPP
