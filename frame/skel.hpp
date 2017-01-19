#ifndef SKEL_HPP
#define SKEL_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <random>


#include <hmlp.h>
#include <hmlp_blas_lapack.h>
#include <hmlp_util.hpp>
#include <hmlp_thread.hpp>
#include <hmlp_runtime.hpp>

//#define DEBUG_SKEL 1

namespace hmlp
{
namespace skel
{






  

template<typename T>
void id
(
  int m, int n, int maxs,
  std::vector<T> A,
  std::vector<size_t> &skels, hmlp::Data<T> &proj
)
{
  int nb = 512;
  int lwork = 2 * n  + ( n + 1 ) * nb;
  std::vector<T> work( lwork );
  std::vector<T> tau( std::min( m, n ) );
  std::vector<T> S, Z;
  std::vector<T> A_tmp = A;

  // Initilize jpvt to zeros. Otherwise, GEQP3 will permute A.
  std::vector<int> jpvt( n, 0 );

  // Traditional pivoting QR (GEQP3)
  hmlp::xgeqp3
  (
    m, n, 
    A_tmp.data(), m,
    jpvt.data(), 
    tau.data(),
    work.data(), lwork
  );
  //printf( "end xgeqp3\n" );

  jpvt.resize( maxs );
  skels.resize( maxs );

  // Now shift jpvt from 1-base to 0-base index.
  for ( int j = 0; j < jpvt.size(); j ++ ) 
  {
    jpvt[ j ] = jpvt[ j ] - 1;
    skels[ j ] = jpvt[ j ];
  }

  Z.resize( m * jpvt.size() );
  
  for ( int j = 0; j < jpvt.size(); j ++ )
  {
    // reuse Z
    for ( int i = 0; i < m; i ++ )
    {
      Z[ j * m + i ] = A[ jpvt[ j ] * m + i ];
    }
  }
  auto A_skel = Z;


  S = A;
  // P (overwrite S) = pseudo-inverse( Z ) * S
  hmlp::xgels
  ( 
    "N", 
    m, jpvt.size(), n, 
       Z.data(), m, 
       S.data(), m, 
    work.data(), lwork 
  );

  
  // Fill in proj
  proj.resize( jpvt.size(), n );
  for ( int j = 0; j < n; j ++ )
  {
    for ( int i = 0; i < jpvt.size(); i ++ )
    {
      proj[ j * jpvt.size() + i ] = S[ j * m + i ];
    }
  }

  

#ifdef DEBUG_SKEL
  double nrm = hmlp_norm( m, n, A.data(), m );

  hmlp::xgemm
  ( 
    "N", "N", 
    m, n, jpvt.size(), 
    -1.0,  A_skel.data(), m, 
                S.data(), m, 
     1.0,       A.data(), m 
   );

  double err = hmlp_norm( m, n, A.data(), m );
  printf( "m %d n %d k %lu absolute l2 error %E related l2 error %E\n", 
      m, n, jpvt.size(),
      err, err / nrm );
#endif

}; // end id()
  





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







}; // end namespace skel
}; // end namespace hmlp

#endif // define SKEL_HPP
