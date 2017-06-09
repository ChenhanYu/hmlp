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
#include <view.hpp>

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
    
    void SetupFactor
    ( 
      bool isleaf, bool isroot,
      /** n == nl + nr (left + right) */
      std::size_t n, std::size_t nl, std::size_t nr,
      /** s <= sl + sr */
      std::size_t s, std::size_t sl, std::size_t sr
    )
    {
      this->isleaf = isleaf;
      this->isroot = isroot;
      this->n = n; this->nl = nl; this->nr = nr;
      this->s = s; this->sl = sl; this->sr = sr;
    };

    void SetupFactor
    (
      bool isleaf, bool isroot,
      std::size_t n, std::size_t nl, std::size_t nr,
      std::size_t s, std::size_t sl, std::size_t sr,
      /** n-by-?; its rank depends on mu sibling */
      hmlp::Data<T> &U,
      /** ?-by-n; its rank depends on my sibling */
      hmlp::Data<T> &V
    )
    {
      Setup( isleaf, isroot, n, nl, nr, s, sl, sr );
    };


    template<bool SYMMETRIC>
    void Factorize( hmlp::Data<T> &Kaa ) 
    {
      assert( isleaf );
      assert( Kaa.row() == n ); assert( Kaa.col() == n );

      /** initialize */
      Z = Kaa;
      if ( !SYMMETRIC ) ipiv.resize( n, 0 );

      /** LU or Cholesky factorization */
      if ( SYMMETRIC ) xpotrf( "L", n, Z.data(), n );
      else             xgetrf(   n, n, Z.data(), n, ipiv.data() );
    };


    /** also compute inv( K ) * P' */
    //void Factorize( hmlp::Data<T> &K, hmlp::Data<T> &P ) 
    //{
    //  /** s-by-n */
    //  assert( P.row() == s ); assert( P.col() == n );

    //  /** n-by-s; inv( K )* U = inv( K ) * P' */
    //  U.resize( n, s );
    //  for ( size_t j = 0; j < n; j ++ )
    //    for ( size_t i = 0; i < s; i ++ )
    //      U[ i * n + j ] = P[ j * s + i ];

    //  /** K = LU, and compute inv( K ) * P' */
    //  Factorize( K );
    //};


    /** generate symmetric factorization I + UCU' = ( I + UXU' )( I + UXU' )' */
    void Factorize
    (
      /** Ul,  nl-by-sl */
      hmlp::Data<T> &Ul, 
      /** Ur,  nr-by-sr */
      hmlp::Data<T> &Ur
    )
    {
      assert( !isleaf );
      assert( Ul.row() == nl ); assert( Ul.col() == sl );
      assert( Ur.row() == nr ); assert( Ur.col() == sr );

      /** skeleton rows and columns of lower triangular  */
      assert( Crl.row() == sr ); assert( Crl.col() == sl );

      /**  inv( L' ) * ( M - I ) * inv( L ) */
      Z.resize( sl + sr, sl + sr, 0.0 );

      /** pivoting row indices */
      ipiv.resize( Z.row(), 0 );

      for ( size_t j = 0; j < sl + sr; j ++ )
      {
        for ( size_t i = 0; i < sl + sr; i ++ ) 
        {
          if ( i == j ) Z[ j * Z.row() + i ] = 1.0;
        }
      }

      /** U'U */
      std::vector<T> UltUl( sl * sl, 0.0 );
      std::vector<T> UrtUr( sr * sr, 0.0 );

      /** Lr * Crl * Ll' */
      std::vector<T> LrCrlLlt = Crl;

      /** UltUl, TODO: syrk */
      xgemm( "T", "N", sl, sl, nl, 
          1.0, Ul.data(), nl, Ul.data(), nl, 
          0.0, UltUl.data(), sl );
      /** UrtUr, TODO: syrk */
      xgemm( "T", "N", sr, sr, nr, 
          1.0, Ur.data(), nr, Ur.data(), nr, 
          0.0, UrtUr.data(), sr );
      /** Ll = POTRF( UltUl ) */
      xpotrf( "L", sl, UltUl.data(), sl );
      /** Lr = POTRF( UrtUr ) */
      xpotrf( "L", sr, UrtUr.data(), sr );
      /** Lr * Crl */
      xtrmm( "L", "L", "N", "N", sr, sl, 
          1.0, UrtUr.data(), sr, LrCrlLlt.data(), sr );
      /** Crl * Ll' */
      xtrmm( "R", "L", "T", "N", sr, sl, 
          1.0, UltUl.data(), sl, LrCrlLlt.data(), sr );
      /** Z = I */
      for ( size_t i = 0; i < sl + sr; i ++ ) 
            Z[ i * Z.row() + i ] = 1.0;
      /** Zrl = LrCrlLlt */
      for ( size_t j = 0; j < sl; j ++ )
        for ( size_t i = sl; i < sl + sr; i ++ ) 
          Z[ j * Z.row() + i ] = LrCrlLlt( i - sl, j );
      /** M = POTRF( Z ) */
      xpotrf( "L", sl + sr, Z.data(), sl + sr );
      /** Z = M - I */
      for ( size_t i = 0; i < sl + sr; i ++ ) 
            Z[ i * Z.row() + i ] -= 1.0;
      /** M = Z, Z = identity  */
      hmlp::Data<T> M = Z;
      for ( size_t i = 0; i < ( sl + sr ) * ( sl + sr ); i ++ ) 
            Z[ i * Z.row() + i ] = 0.0;
      for ( size_t i = 0; i < sl + sr; i ++ ) 
            Z[ i * Z.row() + i ] = 1.0;
      /** Z = inv( M ) */
      xtrsm( "L", "L", "N", "N", sl + sr, sl + sr, 
          1.0, M.data(), sl + sr, Z.data(),                   sl + sr );
      /** Z = Z + identity */
      for ( size_t i = 0; i < sl + sr; i ++ ) 
            Z[ i * Z.row() + i ] += 1.0;
      /** L * Z */
      xtrmm( "L", "L", "N", "N", sl, sl + sr, 
          1.0, UltUl.data(), sl, Z.data(),                    sl + sr );
      xtrmm( "L", "L", "N", "N", sr, sl + sr, 
          1.0, UrtUr.data(), sr, Z.data() + sl,               sl + sr );
      /** Z * L' */
      xtrmm( "R", "L", "T", "N", sl + sr, sl, 
          1.0, UltUl.data(), sl, Z.data(),                    sl + sr );
      xtrmm( "R", "L", "T", "N", sl + sr, sr, 
          1.0, UrtUr.data(), sr, Z.data() + ( sl + sr ) * sl, sl + sr );
      /** LU( Z ) */
      xgetrf( sl + sr, sl + sr, Z.data(), sl + sr, ipiv.data() );

    }; /** end Factorize() */


    /**    
     *   two-sided UCVt   one-sided UBt
     *   
     *      | sl   sr        | sl   sr
     *   -------------    -------------  
     *   nl | Ul          nl | Ul  
     *   nr |      Ur     nr |      Ur
     *
     *      | sl   sr
     *   -------------  
     *   sl |     Clr
     *   sr | Crl      
     *
     *      | nl   nr        | nl   nr
     *   -------------    -------------
     *   sl | Vlt         sl |      Brt
     *   sr |      Vrt    sr | Blt
     *
     *
     **/
    template<bool TWOSIDED, bool SYMMETRIC>
    void Factorize
    ( 
      /** Ul,  nl-by-sl */
      hmlp::Data<T> &Ul, 
      /** Ur,  nr-by-sr */
      hmlp::Data<T> &Ur, 
      /** Vl,  nl-by-sl (or 0-by-0) */
      hmlp::Data<T> &Vl,
      /** Vr,  nr-by-sr (or 0-by-0) */
      hmlp::Data<T> &Vr,
      /** Vl,  nl-by-sr (or 0-by-0) */
      hmlp::Data<T> &Bl,
      /** Vr,  nr-by-sl (or 0-by-0) */
      hmlp::Data<T> &Br
    )
    {
      assert( !isleaf );
      assert( Ul.row() == nl ); assert( Ul.col() == sl );
      assert( Ur.row() == nr ); assert( Ur.col() == sr );

      if ( TWOSIDED )
      {
        assert( Vl.row() == nl ); assert( Vl.col() == sl );
        assert( Vr.row() == nr ); assert( Vr.col() == sr );
        if ( SYMMETRIC )
        {
          assert( Clr.row() == sl ); assert( Clr.col() == sr );
        }
        else
        {
          assert( Clr.row() == sl ); assert( Clr.col() == sr );
          assert( Crl.row() == sr ); assert( Crl.col() == sl );
        }
      }
      else
      {
        assert( Bl.row() == nl ); assert( Bl.col() == sr );
        assert( Br.row() == nr ); assert( Br.col() == sl );
      }
     
      /**  I + CVtU =  [        I  ClrVrtUr
        *                CrlVltUl         I ] 
        *  I +  BtU =  [        I     BrtUr
        *                   BltUl         I ] */
      Z.resize( sl + sr, sl + sr, 0.0 );

      /** pivoting row indices */
      ipiv.resize( Z.row(), 0 );
      
      /**     | sl  sr
       *   -----------
       *   sl | Zrl Ztr
       *   sr | Zbl Zbr   
       *
       *   I + CVtU =  [        I  ClrVrtUr
       *                 CrlVltUl         I ] 
       *   I +  BtU =  [        I     BrtUr
       *                    BltUl         I ] */
      for ( size_t j = 0; j < sl + sr; j ++ )
      {
        for ( size_t i = 0; i < sl + sr; i ++ ) 
        {
          if ( i == j ) Z[ j * Z.row() + i ] = 1.0;
        }
      }

      if ( TWOSIDED )
      {
        std::vector<T> VltUl( sl * sl, 0.0 );
        std::vector<T> VrtUr( sr * sr, 0.0 );

        /** VltUl */
        xgemm( "T", "N", sl, sl, nl, 
            1.0, Vl.data(), nl, Ul.data(), nl, 
            0.0, VltUl.data(), sl );

        /** VrtUr */
        xgemm( "T", "N", sr, sr, nr, 
            1.0, Vr.data(), nr, Ur.data(), nr, 
            0.0, VrtUr.data(), sr );

        /** ClrVrtUr */
        xgemm( "N", "N", sl, sr, sr,
            1.0, Clr.data(), sl, VrtUr.data(), sr, 
            0.0, Z.data() + sl, sl + sr );

        if ( SYMMETRIC )
        {
          /** Clr'VltUl */
          xgemm( "T", "N", sr, sl, sl,
              1.0, Clr.data(), sl, VltUl.data(), sl, 
              0.0, Z.data() + Z.row() * sl, sl + sr );
        }
        else
        {
          /** CrlVltUl */
          xgemm( "N", "N", sr, sl, sl, 
              1.0, Crl.data(), sr, VltUl.data(), sl, 
              0.0, Z.data() + Z.row() * sl, sl + sr );
        }
      }
      else
      {
        /** BltUl */
        xgemm( "T", "N", sr, sl, nl, 
            1.0, Bl.data(), nl, Ul.data(), nl, 
            0.0, Z.data() + sl, sl + sr );
        /** BrtUr */
        xgemm( "T", "N", sl, sr, nr, 
            1.0, Br.data(), nr, Ur.data(), nr, 
            0.0, Z.data() + Z.row() * sl, sl + sr );
      }

      /** LU factorization */
      xgetrf( Z.row(), Z.col(), Z.data(), Z.row(), ipiv.data() );

      /** record points of children factors */
      this->Ul = &Ul;
      this->Ur = &Ur;
      this->Vl = &Vl;
      this->Vr = &Vr;
      this->Bl = &Bl;
      this->Br = &Br;
    }; /** end Factorize() */


    /**
     *
     */ 
    void Solve( hmlp::View<T> &b ) 
    {
      if ( isleaf )
      {
        /** LU solver */
        xgetrs
        ( 
          "N", b.row(), b.col(), 
          Z.data(), Z.row(), ipiv.data(), 
          b.data(), b.ld() 
        );
      }
      else
      {
        /** SMW solver, b - U * inv( Z ) * C * V' * b */
        hmlp::Data<T> x( sl + sr, b.col() );
        hmlp::Data<T> bl( sl, b.col() );
        hmlp::Data<T> br( sr, b.col() );

        /** Vl' * bl */
        xgemm( "T", "N", sl, b.col(), nl,
            1.0, Vl->data(), nl, 
                   b.data(), b.ld(), 
            0.0,  bl.data(), sl ); 
        /** Vr' * br */
        xgemm( "T", "N", sr, b.col(), nr,
            1.0, Vr->data(), nr, 
                   b.data(), b.ld(), 
            0.0,  br.data(), sr );
        /** Clr * Vr' * br */
        xgemm( "N", "N", sl, br.col(), sr,
            1.0, Clr.data(), sl, 
                  br.data(), sr, 
            0.0,   x.data(), sl + sr );
        /** Clr' * Vl' * bl */
        xgemm( "T", "N", sr, bl.col(), sl,
            1.0, Clr.data(), sl, 
                  bl.data(), sl, 
            0.0,   x.data() + sl, sl + sr );
        /** inv( Z ) * x */
        xgetrs
        ( 
          "N", x.row(), x.col(), 
          Z.data(), Z.row(), ipiv.data(), 
          x.data(), x.row() 
        );
        /** bl - Ul * xl */
        xgemm( "N", "N", nl, b.col(), sl,
           -1.0, Ul->data(), nl, 
                   x.data(), sl + sr, 
            1.0,   b.data(), b.ld() ); 
        /** br - Ur * xr */
        xgemm( "N", "N", nr, b.col(), sr,
           -1.0, Ur->data(), nr, 
                   x.data() + sr, sl + sr, 
            1.0,   b.data() + nl, b.ld() );
      }
    };


    /** RIGHT: V = [ P(:, 0:st-1) * Vl , P(:,st:st+sb-1) * Vr ] 
     *  LEFT:  U = [ Ul * P(:, 0:st-1)'; Ur * P(:,st:st+sb-1) ] */
    template<bool SYMMETRIC, bool DO_INVERSE>
    void Telescope
    ( 
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
      Pa.resize( n, s, 0.0 );
      if ( isleaf )
      {
        assert( Palr.row() == s ); assert( Palr.col() == n );
        /** Pa = Palr' */
        for ( size_t j = 0; j < s; j ++ )
          for ( size_t i = 0; i < n; i ++ )
            Pa[ j * n + i ] = Palr( j, i );
      }
      else
      {
        /** Pa( 0:nl-1, : ) = Pl * Palr( :, 0:sl-1 )' */
        xgemm( "N", "T", nl, s, sl, 
            1.0, Pl.data(), nl, Palr.data(), s, 
            0.0, Pa.data(), n );
        /** Pa( nl:n-1, : ) = Pr * Palr( :, sl:sl+sr-1 )' */
        xgemm( "N", "T", nr, s, sr, 
            1.0, Pr.data(), nr, Palr.data() + s * sl, s, 
            0.0, Pa.data() + nl, n );
      }
      /** Pa = inv( I + UCV' ) * Pa */
      if ( DO_INVERSE )
      {
        if ( isleaf )
        {
          if ( SYMMETRIC )
            /** triangular solver */
            xtrsm( "L", "L", "N", "N", n, s, 1.0, Z.data(), n, Pa.data(), n );
          else             
            /** LU solver */
            xgetrs( "N", n, s, Z.data(), n, ipiv.data(), Pa.data(), n );
        }
        else
        {
          hmlp::Data<T> b( sl + sr, s );
          hmlp::Data<T> xl( sl, s );
          hmlp::Data<T> xr( sr, s );

          if ( SYMMETRIC )
          {
            /** ( I - V * inv( Z ) * V' ) * Pa */

            /** xl = Vlt * Pa( 0:nl-1, : ) */
            xgemm( "T", "N", sl, s, nl, 
                1.0, Vl->data(), nl, Pa.data(), n, 
                0.0, xl.data(), sl );
            /** xr = Vrt * Pa( nl:n-1, : ) */
            xgemm( "T", "N", sr, s, nr, 
                1.0, Vr->data(), nr, Pa.data() + nl, n, 
                0.0, xr.data(), sr );
          }
          else
          {
            /** xl = Vlt * Pa( 0:nl-1, : ) */
            xgemm( "T", "N", sl, s, nl, 
                1.0, Vl->data(), nl, Pa.data(), n, 
                0.0, xl.data(), sl );
            /** xr = Vrt * Pa( nl:n-1, : ) */
            xgemm( "T", "N", sr, s, nr, 
                1.0, Vr->data(), nr, Pa.data() + nl, n, 
                0.0, xr.data(), sr );
            /** b = [ Clr * xr; Clr' * xl ] */
            xgemm( "N", "N", sl, s, sr, 
                1.0, Clr.data(), sl, xr.data(), sr, 
                0.0, b.data(), sl + sr );
            xgemm( "T", "N", sr, s, sl, 
                1.0, Clr.data(), sl, xl.data(), sl, 
                0.0, b.data() + sl, sl + sr );
            /** b = inv( Z ) * b */
            xgetrs
              ( 
               "N", b.row(), b.col(), 
               Z.data(), Z.row(), ipiv.data(), 
               b.data(), b.row() 
              );
            /** Pa( 0:nl-1, : ) -= Ul * b( 0:sl-1, : ) */
            xgemm( "N", "N", nl, s, sl, 
                -1.0, Ul->data(), nl, b.data(), sl + sr, 
                1.0, Pa.data(), n );
            /** Pa( nl:n-1, : ) -= Ur * b( sl:sl+sr-1, : ) */
            xgemm( "N", "N", nr, s, sr, 
                -1.0, Ur->data(), nr, b.data() + sl, sl + sr, 
                1.0, Pa.data() + nl, n );
          }
        }
      }
    };



    bool isleaf;

    bool isroot;

    size_t n;

    size_t nl;

    size_t nr;

    size_t s;

    size_t sl;

    size_t sr;

    /** Reduced system Z = [ I  VU   if ( HODLR || p-HSS )
     *                       VU  I ] */
    hmlp::Data<T> Z;

    /** pivoting rows */
    std::vector<int> ipiv;
    
    /** U, n-by-s */
    hmlp::Data<T> U;

    /** U, n-by-s (or 0-by-0) */
    hmlp::Data<T> V; 

    /** U, n-by-? (or 0-by-0) */
    hmlp::Data<T> B;

    /** Clr, sl-by-sr (or 0-by-0) */
    hmlp::Data<T> Clr;

    /** Crl, sr-by-sl (or 0-by-0) */
    hmlp::Data<T> Crl;

    /** pointers to children's factors */
    hmlp::Data<T> *Ul = NULL;
    hmlp::Data<T> *Ur = NULL;
    hmlp::Data<T> *Vl = NULL;
    hmlp::Data<T> *Vr = NULL;
    hmlp::Data<T> *Bl = NULL;
    hmlp::Data<T> *Br = NULL;

};


/**
 *
 */ 
template<typename NODE, typename T>
void SetupFactor( NODE *node )
{
  size_t n, nl, nr, s, sl, sr;

  n  = node->n;
  nl = 0;
  nr = 0;
  s  = node->data.skels.size();
  sl = 0;
  sr = 0;

  if ( !node->isleaf )
  {
    nl = node->lchild->n;
    nr = node->rchild->n;
    sl = node->lchild->data.skels.size();
    sr = node->rchild->data.skels.size();
  }

  node->data.SetupFactor
  (
    node->isleaf, !node->l,
    n, nl, nr,
    s, sl, sr 
  );

}; /** end void SetupFactor() */


/**
 *  @brief
 */ 
template<typename NODE, typename T>
class SetupFactorTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "sf" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();
      // Need an accurate cost model.
      cost = 1.0;

      //printf( "Set treelist_id %lu\n", arg->treelist_id ); fflush( stdout );
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    void DependencyAnalysis()
    {
      /** remove all previous read/write records */
      arg->DependencyCleanUp();
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      }
      else
      {
        this->Enqueue();
      }
    };

    void Execute( Worker* user_worker )
    {
      SetupFactor<NODE, T>( arg );
    };

}; /** end class SetupFactorTask */



template<typename NODE, typename T>
void Solve( NODE *node, hmlp::View<T> &b )
{
  auto &data = node->data;
  data.Solve( b );
}; /** end Solve() */


/**
 *  @brief
 */ 
template<typename NODE, typename T>
class SolveTask : public hmlp::Task
{
  public:

    NODE *arg;

    hmlp::View<T> b;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "sl" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();
      // Need an accurate cost model.
      cost = 1.0;

      //printf( "Set treelist_id %lu\n", arg->treelist_id ); fflush( stdout );
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      }
    };

    void Execute( Worker* user_worker )
    {
      Solve<NODE, T>( arg, b );
    };

}; /** end class SolveTask */


template<typename NODE, typename T, typename TREE>
void Solve( TREE &tree, hmlp::Data<T> &b )
{
  const bool AUTO_DEPENDENCY = true;
  const bool USE_RUNTIME = true;

  SolveTask<NODE, T> solvetask;

  /** attach the pointer to the tree structure */
  tree.setup.b = &b;

  /** permute weights into w_leaf */
  //printf( "Forward permute ...\n" ); fflush( stdout );
  //beg = omp_get_wtime();
  //int n_nodes = ( 1 << tree.depth );
  //auto level_beg = tree.treelist.begin() + n_nodes - 1;
  //#pragma omp parallel for
  //for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
  //{
  //  auto *node = *(level_beg + node_ind);
  //  weights.GatherColumns<true>( node->lids, node->data.w_leaf );
  //}





  tree.template TraverseUp<AUTO_DEPENDENCY, USE_RUNTIME>( solvetask );



}; /** end Solve() */








/**
 *  @brief Factorizarion using LU and SMW
 */ 
template<typename NODE, typename T>
void Factorize( NODE *node )
{
  /** SYMMETRIC */
  const bool SYMMETRIC = true;

  auto &data = node->data;
  auto &setup = node->setup;
  auto &K = *setup->K;
  auto &proj = data.proj;

  /** we use this to replace those nop arguments */
  hmlp::Data<T> dummy;


  if ( node->isleaf )
  {
    auto lambda = setup->lambda;
    auto &amap = node->lids;
    /** evaluate the diagonal block */
    auto Kaa = K( amap, amap );
    /** apply the regularization */
    for ( size_t i = 0; i < Kaa.row(); i ++ ) 
      Kaa[ i * Kaa.row() + i ] += lambda;
    /** LU factorization */
    data.Factorize<SYMMETRIC>( Kaa );
    /** U = inv( Kaa ) * proj' */
    data.Telescope<SYMMETRIC,  true>( data.U, proj, dummy, dummy );
    /** V = proj' */
    data.Telescope<SYMMETRIC, false>( data.V, proj, dummy, dummy );
      
    printf( "Factorize %lu\n", node->treelist_id );
  }
  else
  {
    auto &Ul = node->lchild->data.U;
    auto &Ur = node->rchild->data.U;
    auto &Vl = node->lchild->data.V;
    auto &Vr = node->rchild->data.V;
    auto &Bl = dummy;
    auto &Br = dummy;
    /** evluate the skeleton rows and columns */
    auto &amap = node->lchild->data.skels;
    auto &bmap = node->rchild->data.skels;
    node->data.Clr = K( amap, bmap );
    /** SMW factorization */
    data.Factorize<true, true>( Ul, Ur, Vl, Vr, Bl, Br );
    /** telescope U and V */
    if ( !node->data.isroot )
    {
      /** U = inv( I + UCV' ) * [ Ul; Ur ] * proj' */
      data.Telescope<SYMMETRIC,  true>( data.U, proj, Ul, Ur );
      /** V = [ Vl; Vr ] * proj' */
      data.Telescope<SYMMETRIC, false>( data.V, proj, Vl, Vr );
    }
  }
}; /** end void Factorize() */



/**
 *  @brief
 */ 
template<typename NODE, typename T>
class FactorizeTask : public hmlp::Task
{
  public:

    NODE *arg;

    void Set( NODE *user_arg )
    {
      std::ostringstream ss;
      arg = user_arg;
      name = std::string( "fa" );
      //label = std::to_string( arg->treelist_id );
      ss << arg->treelist_id;
      label = ss.str();
      // Need an accurate cost model.
      cost = 1.0;

      //printf( "Set treelist_id %lu\n", arg->treelist_id ); fflush( stdout );
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    void DependencyAnalysis()
    {
      arg->DependencyAnalysis( hmlp::ReadWriteType::RW, this );

      if ( !arg->isleaf )
      {
        arg->lchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
        arg->rchild->DependencyAnalysis( hmlp::ReadWriteType::R, this );
      }
    };

    void Execute( Worker* user_worker )
    {
      Factorize<NODE, T>( arg );
    };

}; /** end class FactorizeTask */




/**
 *
 */ 
template<typename NODE, typename T, typename TREE>
void Solve( TREE &tree )
{
  const bool AUTO_DEPENDENCY = true;
  const bool USE_RUNTIME = true;

  SolveTask<NODE, T> solvetask;
  tree.template TraverseUp<AUTO_DEPENDENCY, USE_RUNTIME>( solvetask );
  printf( "SolveTask\n" ); fflush( stdout );

  if ( USE_RUNTIME ) hmlp_run();
  printf( "Execute Solve\n" ); fflush( stdout );


}; /** end Solve() */



template<typename NODE, typename T, typename TREE>
void Factorize( TREE &tree, T lambda )
{
  const bool AUTO_DEPENDENCY = true;
  const bool USE_RUNTIME = true;

  SetupFactorTask<NODE, T> setupfactortask; 
  FactorizeTask<NODE, T> factorizetask; 

  /** setup the regularization parameter lambda */
  tree.setup.lambda = lambda;

  tree.template TraverseUp<AUTO_DEPENDENCY, USE_RUNTIME>( setupfactortask );
  printf( "SetupFactorTask\n" ); fflush( stdout );
  tree.template TraverseUp<AUTO_DEPENDENCY, USE_RUNTIME>( factorizetask );
  printf( "FactorTask\n" ); fflush( stdout );

  if ( USE_RUNTIME ) hmlp_run();
  printf( "Factorize\n" ); fflush( stdout );

}; /** end Factorize() */







}; // end namespace hfamily
}; // end namespace hmlp

#endif // define HFAMILY_HPP
