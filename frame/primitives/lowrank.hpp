#ifndef LOWRANK_HPP
#define LOWRANK_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <random>


#include <hmlp.h>
#include <hmlp_blas_lapack.h>
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>
#include <containers/data.hpp>

//#define DEBUG_SKEL 1

namespace hmlp
{
namespace lowrank
{






  
/** TODO: this fixed rank id will be deprecated */ 
//template<typename T>
//void id
//(
//  int m, int n, int maxs,
//  std::vector<T> A,
//  std::vector<size_t> &skels, hmlp::Data<T> &proj, std::vector<int> &jpvt
//)
//{
//  int nb = 512;
//  int lwork = 2 * n  + ( n + 1 ) * nb;
//  std::vector<T> work( lwork );
//  std::vector<T> tau( std::min( m, n ) );
//  std::vector<T> S, Z;
//  std::vector<T> A_tmp = A;
//
//  // Early return
//  if ( n <= maxs )
//  {
//    skels.resize( n );
//    proj.resize( n, n, 0.0 );
//    for ( int i = 0; i < n; i ++ )
//    {
//      skels[i] = i;
//      proj[ i * proj.row() + i ] = 1.0;
//    }
//    return;
//  }
//
//  // Initilize jpvt to zeros. Otherwise, GEQP3 will permute A.
//  jpvt.resize( n, 0 );
//
//  // Traditional pivoting QR (GEQP3)
//  hmlp::xgeqp3
//  (
//    m, n, 
//    A_tmp.data(), m,
//    jpvt.data(), 
//    tau.data(),
//    work.data(), lwork
//  );
//  //printf( "end xgeqp3\n" );
//
//  jpvt.resize( maxs );
//  skels.resize( maxs );
//
//  // Now shift jpvt from 1-base to 0-base index.
//  for ( int j = 0; j < jpvt.size(); j ++ ) 
//  {
//    jpvt[ j ] = jpvt[ j ] - 1;
//    skels[ j ] = jpvt[ j ];
//  }
//
//  // TODO: Here we only need several things to get rid of xgels.
//  //
//  // 0. R11 = zeros( s )
//  // 1. get R11 = up_tiangular( A_tmp( 1:s, 1:s ) )
//  // 2. get proj( 1:s, jpvt( 1:n ) ) = A_tmp( 1:s 1:n )
//  // 3. xtrsm( "L", "U", "N", "N", s, n, 1.0, R11.data(), s, proj.data(), s )
//  
//
//
//
//
//  Z.resize( m * jpvt.size() );
//  
//  for ( int j = 0; j < jpvt.size(); j ++ )
//  {
//    // reuse Z
//    for ( int i = 0; i < m; i ++ )
//    {
//      Z[ j * m + i ] = A[ jpvt[ j ] * m + i ];
//    }
//  }
//  auto A_skel = Z;
//
//
//  S = A;
//  // P (overwrite S) = pseudo-inverse( Z ) * S
//  hmlp::xgels
//  ( 
//    "N", 
//    m, jpvt.size(), n, 
//       Z.data(), m, 
//       S.data(), m, 
//    work.data(), lwork 
//  );
//
//  
//  // Fill in proj
//  proj.resize( jpvt.size(), n );
//  for ( int j = 0; j < n; j ++ )
//  {
//    for ( int i = 0; i < jpvt.size(); i ++ )
//    {
//      proj[ j * jpvt.size() + i ] = S[ j * m + i ];
//    }
//  }
//
//  
//
//#ifdef DEBUG_SKEL
//  double nrm = hmlp_norm( m, n, A.data(), m );
//
//  hmlp::xgemm
//  ( 
//    "N", "N", 
//    m, n, jpvt.size(), 
//    -1.0,  A_skel.data(), m, 
//                S.data(), m, 
//     1.0,       A.data(), m 
//   );
//
//  double err = hmlp_norm( m, n, A.data(), m );
//  printf( "m %d n %d k %lu absolute l2 error %E related l2 error %E\n", 
//      m, n, jpvt.size(),
//      err, err / nrm );
//#endif
//
//}; // end id()
//




/**
 *
 *
 */ 
template<bool ADAPTIVE, bool LEVELRESTRICTION, typename T>
void id
(
  int m, int n, int maxs, T stol,
  hmlp::Data<T> A,
  std::vector<size_t> &skels, hmlp::Data<T> &proj, std::vector<int> &jpvt
)
{
  int s;
  int nb = 512;
  int lwork = 2 * n  + ( n + 1 ) * nb;
  std::vector<T> work( lwork );
  std::vector<T> tau( std::min( m, n ) );
  hmlp::Data<T> S, Z;
  hmlp::Data<T> A_tmp = A;

  /** sample rows must be larger than columns */
  assert( m >= n );

  // Initilize jpvt to zeros. Otherwise, GEQP3 will permute A.
  jpvt.clear();
  jpvt.resize( n, 0 );

  /** Traditional pivoting QR (GEQP3) */
#ifdef HMLP_USE_CUDA
  auto *dev = hmlp_get_device( 0 );
  cublasHandle_t &handle = 
      reinterpret_cast<hmlp::gpu::Nvidia*>( dev )->gethandle( 0 );
  hmlp::xgeqp3
  (
    handle,
    m, n, 
    A_tmp.data(), m,
    jpvt.data(), 
    tau.data(),
    work.data(), lwork
  );
#else
  hmlp::xgeqp4
  (
    m, n, 
    A_tmp.data(), m,
    jpvt.data(), 
    tau.data(),
    work.data(), lwork
  );
#endif
  //printf( "end xgeqp3\n" );

  /** shift jpvt from 1-base to 0-base index. */
  for ( int j = 0; j < jpvt.size(); j ++ ) jpvt[ j ] = jpvt[ j ] - 1;

  /** search for rank 1 <= s <= maxs that satisfies the error tolerance */
  for ( s = 1; s < n; s ++ )
  {
    if ( s > maxs || std::abs( A_tmp[ s * m + s ] ) < stol ) break;
    //if ( s > maxs || std::abs( A_tmp[ s * m + s ] ) / std::abs( A_tmp[ 0 ] ) < stol ) break;
  }

  /** if using fixed rank */
  if ( !ADAPTIVE ) s = std::min( maxs, n );

  /** failed to satisfy error tolerance */
  if ( s > maxs )
  {
    if ( LEVELRESTRICTION ) /** abort */
    {
      skels.clear();
      proj.resize( 0, 0 );
      jpvt.resize( 0 );
      return;
    }
    else /** Continue with rank maxs */
    {
      s = maxs;
    }
  }

  /** now #skeleton has been decided, resize skels to fit */
  skels.resize( s );
  for ( int j = 0; j < skels.size(); j ++ ) skels[ j ] = jpvt[ j ]; 


  // TODO: Here we only need several things to get rid of xgels.
  //
  // 0. R11 = zeros( s )
  // 1. get R11 = up_tiangular( A_tmp( 1:s, 1:s ) )
  // 2. get proj( 1:s, jpvt( 1:n ) ) = A_tmp( 1:s 1:n )
  // 3. xtrsm( "L", "U", "N", "N", s, n, 1.0, R11.data(), s, proj.data(), s )


  /** extract proj. It will be computed in Interpolate. */
  if ( true ) 
  {
    /** fill in proj */
    proj.clear();
    proj.resize( s, n, 0.0 );

    for ( int j = 0; j < n; j ++ )
    {
      for ( int i = 0; i < s; i ++ )
      {
        if ( j < s )
        {
          if ( j >= i ) proj[ j * s + i ] = A_tmp[ j * m + i ];
          else          proj[ j * s + i ] = 0.0;
        }
        else
        {
	      proj[ j * s + i ] = A_tmp[ j * m + i ];
	    }
	  }
	}
  }
  else /** in the old version we use xgels, which is expensive */
  {
    Z.resize( m, skels.size() );
 
    for ( int j = 0; j < skels.size(); j ++ )
    {
      for ( int i = 0; i < m; i ++ )
      {
        Z[ j * m + i ] = A[ skels[ j ] * m + i ];
      }
    }
    auto A_skel = Z;

    S = A;
    // P (overwrite S) = pseudo-inverse( Z ) * S
    hmlp::xgels
    ( 
      "N", 
      m, skels.size(), n, 
         Z.data(), m, 
         S.data(), m, 
      work.data(), lwork 
    );
 
    // Fill in proj
    proj.resize( skels.size(), n );
    for ( int j = 0; j < n; j ++ )
    {
      for ( int i = 0; i < skels.size(); i ++ )
      {
        proj[ j * skels.size() + i ] = S[ j * m + i ];
      }
    }

#ifdef DEBUG_SKEL
  double nrm = hmlp_norm( m, n, A.data(), m );

  hmlp::xgemm
  ( 
    "N", "N", 
    m, n, skels.size(), 
    -1.0,  A_skel.data(), m, 
                S.data(), m, 
     1.0,       A.data(), m 
   );

  double err = hmlp_norm( m, n, A.data(), m );
  printf( "m %d n %d k %lu absolute l2 error %E related l2 error %E\n", 
      m, n, skels.size(),
      err, err / nrm );
#endif

  }

}; // end id()
  


/**
 *  @brief
 */ 
template<bool ONESHOT = false,typename T>
void nystrom( size_t m, size_t n, size_t r, 
    std::vector<T> &A, std::vector<T> &C, 
    std::vector<T> &U, std::vector<T> &V )
{
  /** if C is not initialized, then sample from A */
  if ( C.size() != r * r )
  {
    /** uniform sampling */
    
  }
  else
  {
    /** compute the pseudo-inverse of C using SVD */
  }

  /** output an approximate  */
  if ( ONESHOT )
  {
  }


}; /** end nystrom() */





template<typename T>
void pmid
(
  int m,
  int n,
  int maxs,
  std::vector<T> A,
  std::vector<int> &jpiv, std::vector<T> &P
)
{
  int rank = maxs + 10;
  int lwork = 512 * n;

  printf( "maxs %d\n", maxs );

  if ( rank > n ) rank = n;


  std::vector<T> S = A;
  std::vector<T> O( n * rank );
  std::vector<T> Z( m * rank );
  std::vector<T> tau( std::min( m, n ), 0.0 );
  std::vector<T> work( lwork, 0.0 );

  std::default_random_engine generator;
  std::normal_distribution<T> gaussian( 0.0, 1.0 );
  
  // generate O n-by-(maxs+10) random matrix (need to be Gaussian samples)
  #pragma omp parallel for
  for ( int i = 0; i < n * rank; i ++ )
  {
    O[ i ] = gaussian( generator );
  }

#ifdef DEBUG_SKEL
  printf( "O\n" );
  hmlp_printmatrix( n, rank, O.data(), n );
  printf( "A\n" );
  hmlp_printmatrix( m, n, A.data(), m );
#endif


  // Z = 0.0 * Z + 1.0 * A * O
  hmlp::xgemm
  ( 
    "N", "N", 
    m, rank, n, 
    1.0, A.data(), m, 
         O.data(), n, 
    0.0, Z.data(), m
  ); 
  printf( "here xgemm\n" );


#ifdef DEBUG_SKEL
  printf( "Z\n" );
  hmlp_printmatrix( m, rank, Z.data(), m );
#endif






  // [Q,~] = qr(Z,0), so I need the orthogonal matrix
  hmlp::xgeqrf
  ( 
    m, rank, 
       Z.data(), m, 
     tau.data(), 
    work.data(), lwork 
  );
  printf( "here xgeqrf\n" );

#ifdef DEBUG_SKEL
  printf( "Z\n" );
  hmlp_printmatrix( m, rank, Z.data(), m );
#endif

  // S =  Q' * A
  hmlp::xormqr
  ( 
    "L", "T", 
    m, n, rank, 
       Z.data(), m, 
     tau.data(), 
       S.data(), m, 
    work.data(), lwork 
  );
  printf( "here xormqr\n" );


  for ( int i = 0; i < rank; i ++ )
  {
    for ( int j = 0; j < n; j ++ )
    {
      S[ j * m + i ] = fabs( S[ j * m + i ] );
    }
  }






  // abs( S(1:rank,1:n) ) and select the largest entry per row.  
  while ( jpiv.size() < maxs )
  {
    for ( int i = 0; i < rank; i ++ )
    {
      std::pair<T,int> pivot( 0.0, -1 );

      for ( int j = 0; j < n; j ++ )
      {
        if ( S[ j * m + i ] > pivot.first )
        {
          pivot = std::make_pair( S[ j * m + i ], j );
        }
      }
      if ( pivot.second != -1 )
      {
        jpiv.push_back( pivot.second );
      }
    }

    std::sort( jpiv.begin(), jpiv.end() );
    auto last = std::unique( jpiv.begin(), jpiv.end() );
    jpiv.erase( last, jpiv.end() );

    printf( "Total %lu pivots\n", jpiv.size() );

    // zero out S
    for ( int j = 0; j < jpiv.size(); j ++ )
    {
      for ( int i = 0; i < rank; i ++ )
      {
        S[ jpiv[ j ] * m + i ] = 0.0;
      }
    }
  }

  jpiv.resize( maxs );

#ifdef DEBUG_SKEL
  printf( "jpjv:\n" );
  for ( int j = 0; j < jpiv.size(); j ++ )
  {
    printf( "%12d ", jpiv[ j ] );
  }
#endif
  // std::sort( ipiv.begin(), ipiv.end() );
  // auto last = std::unique( ipiv.begin(), ipiv.end() );
  // ipiv.erase( last, ipiv.end() );

  // printf( "Total %lu pivots\n", ipiv.size() );





  Z.resize( m * jpiv.size() );
  
  for ( int j = 0; j < jpiv.size(); j ++ )
  {
    // reuse Z
    for ( int i = 0; i < m; i ++ )
    {
      Z[ j * m + i ] = A[ jpiv[ j ] * m + i ];
    }
  }
  auto A_skel = Z;
  //P.resize( ipiv.size() * n );


#ifdef DEBUG_SKEL
  printf( "jpjv:\n" );
  for ( int j = 0; j < jpiv.size(); j ++ )
  {
    printf( "%12d ", jpiv[ j ] );
  }
  printf( "\n" );
  printf( "Z = [\n" );
  hmlp_printmatrix( m, jpiv.size(), Z.data(), m );
#endif


  S = A;
  // P (overwrite S) = pseudo-inverse( Z ) * S
  hmlp::xgels
  ( 
    "N", 
    m, jpiv.size(), n, 
       Z.data(), m, 
       S.data(), m, 
    work.data(), lwork 
  );


#ifdef DEBUG_SKEL
  printf( "S\n" );
  hmlp_printmatrix<true, true>( m, n, S.data(), m );

  double nrm = hmlp_norm( m, n, A.data(), m );

  hmlp::xgemm
  ( 
    "N", "N", 
    m, n, jpiv.size(), 
    -1.0,  A_skel.data(), m, 
                S.data(), m, 
     1.0,       A.data(), m 
   );

  double err = hmlp_norm( m, n, A.data(), m );

  printf( "absolute l2 error %E related l2 error %E\n", err, err / nrm );
#endif


#ifdef DEBUG_SKEL
  printf( "A\n" );
  hmlp_printmatrix<true, true>( m, n, A.data(), m );
#endif

}; // end pmid()

template<class Node>
void skeletonize( Node *node )
{
  auto lchild = node->lchild;
  auto rchild = node->rchild;

  // random sampling or important sampling for rows.
  std::vector<size_t> amap;

  std::vector<size_t> bmap;

  //bmap = lchild


  printf( "id %d l %d n %d\n", node->treelist_id, node->l, node->n );

}; // end skeletonize()




//template<typename CONTEXT>
//class Task : public hmlp::Task
//{
//  public:
//    
//    /* function ptr */
//    void (*function)(Task<CONTEXT>*);
//
//    /* argument ptr */
//    CONTEXT *arg;
//
//    void Set( CONTEXT *user_arg )
//    {
//      name = std::string( "Skeletonization" );
//      arg = user_arg;
//    }
//
//    void Execute( Worker* user_worker )
//    {
//      printf( "SkeletonizeTask Execute 2\n" );
//    }
//
//  private:
//
//}; // end class Task

template<class Node>
class Task : public hmlp::Task
{
  public:

    Node *arg;

    void Set( Node *user_arg )
    {
      name = std::string( "Skeletonization" );
      arg = user_arg;
    };

    void Execute( Worker* user_worker )
    {
      //printf( "SkeletonizeTask Execute 2\n" );
      skeletonize( arg );
    };

  private:
};







}; /** end namespace lowrank */
}; /** end namespace hmlp */

#endif // define LOWRANK_HPP
