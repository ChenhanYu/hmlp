/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  



#ifndef IGOFMM_HPP
#define IGOFMM_HPP


/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;




namespace hmlp
{
namespace gofmm
{


/**
 *
 * for each level 
 *   for each alpha
 *     
 *     if ( leaf ) 
 *       
 *       LL' = Chol( Kaa )
 *       U   = inv( L ) * proj' or
 *       U'  = proj * inv( L' )
 *       QR  = qr( U ) or 
 *       LQ' = lq( U' )
 *
 *     else
 *       
 *       LL' = Chol( I + k.R * C * k.R' )
 *
 *       if ( not root )
 *
 *         U   = inv( L ) * [ l.R        * proj' or
 *                               r.R ]
 *         U'  = proj' *  [ l.R        * inv( L' )
 *                             r.R ]
 *         QR  = qr( U ) or
 *         LQ' = lq( U' )
 *
 **/ 


template<typename T>
class Factor
{
  public:

    Factor() {};
    
    void SetupFactor
    (
      bool issymmetric, bool do_ulv_factorization,
      bool is_leaf, bool isroot,
      /** n == nl + nr (left + right) */
      size_t n, size_t nl, size_t nr,
      /** s <= sl + sr */
      size_t s, size_t sl, size_t sr
    )
    {
      this->issymmetric = issymmetric;
      this->do_ulv_factorization = do_ulv_factorization;
      this->is_leaf_ = is_leaf;
      this->isroot = isroot;
      this->n = n; this->nl = nl; this->nr = nr;
      this->s = s; this->sl = sl; this->sr = sr;
    };

    void SetupFactor
    (
      bool issymmetric, bool do_ulv_factorization,
      bool is_leaf, bool isroot,
      size_t n, size_t nl, size_t nr,
      size_t s, size_t sl, size_t sr,
      /** n-by-?; its rank depends on mu sibling */
      Data<T> &U,
      /** ?-by-n; its rank depends on my sibling */
      Data<T> &V
    )
    {
      SetupFactor( issymmetric, do_ulv_factorization, 
          is_leaf, isroot, n, nl, nr, s, sl, sr );
    };

    bool DoULVFactorization()
    {
      return do_ulv_factorization;
    };

    bool IsSymmetric() { return issymmetric; };





    void CheckCondition()
    {
      assert( do_ulv_factorization && issymmetric );
      T max_diag = 0.0;
      T min_diag = 0.0;

      for ( size_t i = 0; i < Z.row(); i ++ )
      {
        T abs_diag =  std::abs( Z( i, i ) );

        if ( !i )
        {
          max_diag = abs_diag;
          min_diag = abs_diag;
        }

        if ( abs_diag > max_diag ) max_diag = abs_diag;
        if ( abs_diag < min_diag ) min_diag = abs_diag;
      }

      printf( "condiditon( Z ): min_diag %3.1E max_diag %3.1E ratio %3.1E\n",
          min_diag, max_diag, min_diag / max_diag );
    };

    void Factorize( Data<T> &Kaa ) 
    {
      assert( isLeaf() );
      assert( Kaa.row() == n ); assert( Kaa.col() == n );

      /** Initialize with Kaa. */
      Z = Kaa;

      /** Record the partial pivoting order. */
      ipiv.resize( n, 0 );

      /** Compute 1-norm of Z. */
      T nrm1 = 0.0;
      for ( auto &z : Z ) nrm1 += z;

      /** Pivoted LU factorization. */
      xgetrf( n, n, Z.data(), n, ipiv.data() );

      /** Compute 1-norm condition number. */
      T rcond1 = 0.0;
      Data<T> work( Z.row(), 4 );
      vector<int> iwork( Z.row() );
      xgecon( "1", Z.row(), Z.data(), Z.row(), nrm1, 
          &rcond1, work.data(), iwork.data() );
      if ( 1.0 / rcond1 > 1E+6 )
        printf( "Warning! large 1-norm condition number %3.1E, nrm1( Z ) %3.1E\n", 
            1.0 / rcond1, nrm1 );
    }; /** end Factorize() */


    /**
     *  Kaa = [ P     [ L11      [ I     [ U11 U12  
     *            I ]   L21  I ]     C ]         I ]
     */ 
    void partialLU(const hmlp::Data<T> & A)
    {
      /* Similar transformation ( Q' * Z * Q ). */
      Z = A;
      ChangeBasis( Z );
      /* Create matrix views for Z. */
      Zv.Set( false, Z );
      Zv.Partition2x2( Ztl, Ztr,
                       Zbl, Zbr, s, s, BOTTOMRIGHT );
      /* Initialize pivoting rows. */
      ipiv.resize( Ztl.row(), 0 );
      /* [Ztl, Ztr] = PLU */
      xgetrf( Ztl.row(), Z.col(), Z.data(), Z.row(), ipiv.data() );
      /* Zbl * U^{-1} */
      xtrsm( "Right", "Upper", "No transpose", "Non-unit", Zbl.row(), Zbl.col(),
          1.0,  Ztl.data(), Ztl.ld(), Zbl.data(), Zbl.ld() );
      /* Update Schur complement Zbr. */
      xgemm( "No transpose", "No transpose", Zbr.row(), Zbr.col(), Ztl.col(),
          -1.0, Zbl.data(), Zbl.ld(),
                Ztr.data(), Ztr.ld(),
           1.0, Zbr.data(), Zbr.ld() );
    }; 




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
    void Factorize
    ( 
      /** Ul,  nl-by-sl */
      Data<T> &Ul, 
      /** Ur,  nr-by-sr */
      Data<T> &Ur, 
      /** Vl,  nl-by-sr */
      Data<T> &Vl,
      /** Vr,  nr-by-sr */
      Data<T> &Vr
    )
    {
      assert( !isLeaf() );
      //assert( Ul.row() == nl ); assert( Ul.col() == sl );
      //assert( Ur.row() == nr ); assert( Ur.col() == sr );
      //assert( Vl.row() == nl ); assert( Vl.col() == sl );
      //assert( Vr.row() == nr ); assert( Vr.col() == sr );

      /** even SYMMETRIC this routine uses LU factorization */
      if ( issymmetric )
      {
        assert( Crl.row() == sr ); assert( Crl.col() == sl );
      }
      else
      {
        assert( Clr.row() == sl ); assert( Clr.col() == sr );
        assert( Crl.row() == sr ); assert( Crl.col() == sl );
      }
     
      /** 
       *  clean up and begin with Z = eye( sl + sr ) =     | sl  sr
       *                                                ------------
       *                                                sl | Zrl Ztr
       *                                                sr | Zbl Zbr 
       **/
      Z.resize( 0, 0 );
      Z.resize( sl + sr, sl + sr, 0.0 );
      for ( size_t i = 0; i < sl + sr; i ++ ) Z[ i * Z.row() + i ] = 1.0;

      /** pivoting row indices */
      ipiv.resize( Z.row(), 0 );

      /**    
       *  Z = I + CVtU =  [        I  ClrVrtUr
       *                    CrlVltUl         I ] 
       **/  
      std::vector<T> VltUl( sl * sl, 0.0 );
      std::vector<T> VrtUr( sr * sr, 0.0 );

      /** VltUl */
      xgemm( "T", "N", sl, sl, nl, 
          1.0,    Vl.data(), nl, 
          Ul.data(), nl, 
          0.0, VltUl.data(), sl );

      /** VrtUr */
      xgemm( "T", "N", sr, sr, nr, 
          1.0,    Vr.data(), nr, 
          Ur.data(), nr, 
          0.0, VrtUr.data(), sr );

      /** CrlVltUl */
      xgemm( "N", "N", sr, sl, sl,
          1.0,   Crl.data(), sr, 
          VltUl.data(), sl, 
          0.0,     Z.data() + sl, sl + sr );


      if ( issymmetric )
      {
        /** Crl'VrtUr */
        xgemm( "T", "N", sl, sr, sr,
            1.0,   Crl.data(), sr, 
            VrtUr.data(), sr, 
            0.0,     Z.data() + ( sl + sr ) * sl, sl + sr );
      }
      else
      {
        printf( "bug\n" ); exit( 1 );
        /** ClrVrtUr */
        xgemm( "N", "N", sl, sr, sr,
            1.0,   Clr.data(), sl, 
            VrtUr.data(), sr, 
            0.0,     Z.data() + ( sl + sr ) * sl, sl + sr );
      }

      /** compute 1-norm of Z */
      T nrm1 = 0.0;
      for ( size_t i = 0; i < Z.size(); i ++ ) 
        nrm1 += std::abs( Z[ i ] );

      /** LU factorization */
      xgetrf( Z.row(), Z.col(), Z.data(), Z.row(), ipiv.data() );

      /** record points of children factors */
      this->Ul = &Ul;
      this->Ur = &Ur;
      this->Vl = &Vl;
      this->Vr = &Vr;

      /** compute 1-norm condition number */
      T rcond1 = 0.0;
      hmlp::Data<T> work( Z.row(), 4 );
      std::vector<int> iwork( Z.row() );
      xgecon( "1", Z.row(), Z.data(), Z.row(), nrm1, &rcond1, work.data(), iwork.data() );
      if (1.0 / rcond1 > 1E+6)
      {
        printf("Warning! large 1-norm condition number %3.1E\n", 1.0 / rcond1); 
        fflush(stdout);
      }
    }; /** end Factorize() */


    void partialLU( 
      /** Zl,  nl-by-nl,  Zr,  nr-by-nr */
      View<T> &Zl, View<T> &Zr,
      /** Ul,  nl-by-sl,  Ur,  nr-by-sr */
      Data<T> &Ul, Data<T> &Ur, 
      /** Vl,  nl-by-sr,  Vr,  nr-by-sr */
      Data<T> &Vl, Data<T> &Vr )
    {
      Z.clear();
      Z.resize(sl + sr, sl + sr, 0.0);

      /* Create a matrix views for Z. */
      Zv.Set( false, Z );
      Zv.Partition2x2(Ztl, Ztr,
                      Zbl, Zbr, sl, sl, TOPLEFT);

      //printf( "Ztl %lux%lu Ztr %lux%lu\n", Ztl.row(), Ztl.col(), Ztr.row(), Ztr.col() ); fflush( stdout );
      //printf( "Zbl %lux%lu Zbr %lux%lu\n", Zbl.row(), Zbl.col(), Zbr.row(), Zbr.col() ); fflush( stdout );


      Zbl.CopyValuesFrom( Crl );
      /** trmm */
      xtrmm( "Right", "Upper", "Transpose", "Non-unit", Zbl.row(), Zbl.col(),
        1.0,  Ul.data(),  Ul.row(), Zbl.data(), Zbl.ld() );
      /** trmm */
      xtrmm(  "Left", "Upper", "Non-transpose", "Non-unit", Zbl.row(), Zbl.col(),
        1.0,  Ur.data(),  Ur.row(), Zbl.data(), Zbl.ld() );

      Ztl.CopyValuesFrom( Zl );
      Zbr.CopyValuesFrom( Zr );

      for ( size_t j = 0; j < sl; j ++ )
        for ( size_t i = 0; i < sr; i ++ )
          Ztr( j, i ) = Zbl( i, j );

      partialLU(Z);

    }; 



    /** */
    void Multiply( View<T> &bl, View<T> &br )
    {
      assert( !isLeaf() && bl.col() == br.col() );
    
      size_t nrhs = bl.col();

      std::vector<T> ta( ( sl + sr ) * nrhs );
      std::vector<T> tl(      sl * nrhs );
      std::vector<T> tr(      sr * nrhs );

      /** Vl' * bl */
      xgemm( "T", "N", sl, nrhs, nl,
          1.0, Vl->data(), nl, 
                bl.data(), bl.ld(), 
          0.0,  tl.data(), sl );
      /** Vr' * br */
      xgemm( "T", "N", sr, nrhs, nr,
          1.0, Vr->data(), nr, 
                br.data(), br.ld(), 
          0.0,  tr.data(), sr );

      /** Crl * Vl' * bl */
      xgemm( "N", "N", sr, nrhs, sl,
          1.0, Crl.data(), sr, 
                tl.data(), sl, 
          0.0,  ta.data() + sl, sl + sr );

      if ( issymmetric )
      {
        /** Crl' * Vr' * br */
        xgemm( "T", "N", sl, nrhs, sr,
            1.0, Crl.data(), sr, 
                  tr.data(), sr, 
            0.0,  ta.data(), sl + sr );
      }
      else
      {
        printf( "bug here !!!!!\n" ); fflush( stdout ); exit( 1 );
        /** Clr * Vr' * br */
        xgemm( "N", "N", sl, nrhs, sr,
            1.0, Clr.data(), sl, 
                  tr.data(), sr, 
            0.0,  ta.data(), sl + sr );
      }

      /** bl += Ul * xl */
      xgemm( "N", "N", nl, nrhs, sl,
        -1.0, Ul->data(), nl, 
               ta.data(), sl + sr, 
         1.0,  bl.data(), bl.ld() );

      /** br += Ur * xr */
      xgemm( "N", "N", nr, nrhs, sr,
         -1.0, Ur->data(), nr, 
                ta.data() + sl, sl + sr, 
          1.0,  br.data(), br.ld() );
    };

    /**
     *  @brief Solver for leaf nodes
     */
    void Solve( View<T> &rhs ) 
    {
      /** assure this is a leaf node */
      assert( isLeaf() );
      assert( !do_ulv_factorization );
      assert( rhs.data() && Z.data() );
      assert( ipiv.data() );

      //rhs.Print();

      size_t nrhs = rhs.col();

      /** LU solver */
      xgetrs( "Non-transpose", rhs.row(), nrhs, 
          Z.data(), Z.row(), ipiv.data(), 
          rhs.data(), rhs.ld() );

    }; /** end Solve() */



    /**
     *  @brief b - U * inv( Z ) * C * V' * b 
     */
    void Solve( View<T> &bl, View<T> &br ) 
    {
      size_t nrhs = bl.col();

      //bl.Print();
      //br.Print();

      /** assertion */
      assert( !do_ulv_factorization );
      assert( bl.col() == br.col() );
      assert( bl.row() == nl );
      assert( br.row() == nr );
      assert( Ul && Ur && Vl && Vr );

      /** buffer */
//      hmlp::Data<T> ta( sl + sr, nrhs );
//      hmlp::Data<T> tl(      sl, nrhs );
//      hmlp::Data<T> tr(      sr, nrhs );

      vector<T> ta( ( sl + sr ) * nrhs );
      vector<T> tl(      sl * nrhs );
      vector<T> tr(      sr * nrhs );


      ///** views of buffer */
      //hmlp::View<T> xa( ta ), xl, xr;

      ///** xa = [ xl; xr; ] */
      //xa.Partition2x1
      //( 
      //  xl, 
      //  xr, sl
      //);


        /** Vl' * bl */
        xgemm( "T", "N", sl, nrhs, nl,
            1.0, Vl->data(), nl, 
                  bl.data(), bl.ld(), 
            0.0,  tl.data(), sl );
        /** Vr' * br */
        xgemm( "T", "N", sr, nrhs, nr,
            1.0, Vr->data(), nr, 
                  br.data(), br.ld(), 
            0.0,  tr.data(), sr );


        /** Crl * Vl' * bl */
        xgemm( "N", "N", sr, nrhs, sl,
            1.0, Crl.data(), sr, 
            tl.data(), sl, 
            0.0,  ta.data() + sl, sl + sr );

        if ( issymmetric )
        {
          /** Crl' * Vr' * br */
          xgemm( "T", "N", sl, nrhs, sr,
              1.0, Crl.data(), sr, 
              tr.data(), sr, 
              0.0,  ta.data(), sl + sr );
        }
        else
        {
          printf( "bug here !!!!!\n" ); fflush( stdout ); exit( 1 );
          /** Clr * Vr' * br */
          xgemm( "N", "N", sl, nrhs, sr,
              1.0, Clr.data(), sl, 
              tr.data(), sr, 
              0.0,  ta.data(), sl + sr );
        }

        /** inv( Z ) * x */
        xgetrs( "N", sl + sr, nrhs, 
            Z.data(), Z.row(), ipiv.data(), 
            ta.data(), sl + sr );

      /** bl -= Ul * xl */
      xgemm( "N", "N", nl, nrhs, sl,
          -1.0, Ul->data(), nl, 
          ta.data(), sl + sr, 
          1.0,  bl.data(), bl.ld() );

      /** br -= Ur * xr */
      xgemm( "N", "N", nr, nrhs, sr,
          -1.0, Ur->data(), nr, 
          ta.data() + sl, sl + sr, 
          1.0,  br.data(), br.ld() );

    }; /** end Solve() */




    void Telescope
    (
      bool DO_INVERSE,
      /** n-by-s */
      Data<T> &Pa,
      /** s-by-(sl+sr) */
      Data<T> &Palr 
    )
    {
      assert( isLeaf() ); 
      /** Initialize Pa */
      Pa.resize( n, s, 0.0 );

      /** create view and subviews for Pa */
      //hmlp::View<T> Xa;

      //Xa.Set( Pa ); 

      assert( Palr.row() == s ); assert( Palr.col() == n );

      /** Pa = Palr' */
      for ( size_t j = 0; j < Pa.col(); j ++ )
        for ( size_t i = 0; i < Pa.row(); i ++ )
          Pa( i, j ) = Palr( j, i );

      if ( DO_INVERSE )
      {
        if ( do_ulv_factorization )
        {
          xtrsm( "Left", "Lower", "No transpose", "Non-unit", 
              Pa.row(), Pa.col(), 
              1.0,  Z.data(),  Z.row(), Pa.data(), Pa.row() );
        }
        else
        {
          assert( ipiv.size() );
          /** LU solver */
          xgetrs( "Non-transpose", 
              n, s, Z.data(), n, ipiv.data(), Pa.data(), n );
        }
      }


      //printf( "call solve from telescope\n" ); fflush( stdout );
      //if ( DO_INVERSE ) Solve<true>( Xa );
      //printf( "call solve from telescope (exist)\n" ); fflush( stdout );

    }; /** end Telescope() */


    /** RIGHT: V = [ P(:, 0:st-1) * Vl , P(:,st:st+sb-1) * Vr ] 
     *  LEFT:  U = [ Ul * P(:, 0:st-1)'; Ur * P(:,st:st+sb-1) ] */
    void Telescope
    ( 
      bool DO_INVERSE,
      /** n-by-s */
      Data<T> &Pa,
      /** s-by-(sl+sr) */
      Data<T> &Palr,
      /** nl-by-sl */
      Data<T> &Pl,
      /** nr-by-sr */
      Data<T> &Pr
    ) 
    {
      assert( !isLeaf() );
      assert( n == nl + nr );
      assert( Pl.col() == sl );
      assert( Pr.col() == sr );
      assert( Palr.row() == s  ); 
      
      if( Palr.col() != ( sl + sr ) )
      {
        printf( "Palr.col %lu sl %lu sr %lu\n", Palr.col(), sl, sr );
      }
      
      assert( Palr.col() == ( sl + sr ) );

      /** Initialize Pa */
      Pa.resize( 0, 0 );

      /** create view and subviews for Pa */
      //hmlp::View<T> Xa;

      //Xa.Set( Pa ); 
      //assert( Xa.row() == Pa.row() ); 
      //assert( Xa.col() == Pa.col() ); 

      if ( do_ulv_factorization )
      {
        Pa.resize( sl + sr, s, 0.0 );

        /** Pa = Palr' */
        for ( size_t j = 0; j < Pa.col(); j ++ )
          for ( size_t i = 0; i < Pa.row(); i ++ )
            Pa[ j * Pa.row() + i ] = Palr[ i * Palr.row() + j ];

        /** Pa( 0:sl-1, : ) = Pl * Palr( :, 0:sl-1 )' */
        xtrmm( "Left", "Upper", "No Transpose", "Non-unit", sl, s,
           1.0,   Pl.data(), Pl.row(), 
                  Pa.data(), Pa.row() );
        //  printf( "Pl.row() %lu Pa.row() %lu Pa.col() %lu\n",
        //      Pl.row(), Pa.row(), Pa.col() );

        /** Pa( sl:sl+sr-1, : ) = Pr * Palr( :, sl:sl+sr-1 )' */
        xtrmm( "Left", "Upper", "No Transpose", "Non-unit", sr, s,
           1.0,   Pr.data()     , Pr.row(), 
                  Pa.data() + sl, Pa.row() ); 
        //  printf( "Pr.row() %lu Pa.row() %lu Pa.col() %lu\n",
        //      Pr.row(), Pa.row(), Pa.col() );

        /** inv( L ) * Pa */
        if ( DO_INVERSE )
        {
          if ( 1 )
          {
            xtrsm( "Left", "Lower", "No transpose", "Non-unit", 
                Pa.row(), Pa.col(), 
                1.0,  Z.data(),  Z.row(), 
                     Pa.data(), Pa.row() ); 
          }
          else
          {
            xlaswp( Pa.col(), Pa.data(), Pa.row(), 
                1, Pa.row(), ipiv.data(), 1 );
            xtrsm( "Left", "Lower", "No transpose", "Unit", Pa.row(), Pa.col(), 
                1.0,  Z.data(),  Z.row(), 
                     Pa.data(), Pa.row() ); 
          }
          //printf( "Z.row() %lu Z.col() %lu\n", Z.row(), Z.col() );
        }

      }
      else /** Shernman-Morrison-Woodbury */
      {
        Pa.resize( nl + nr, s, 0.0 );

        ///** */
        //hmlp::View<T> Xl, Xr;

        ///** Xa = [ Xl; Xr; ] */
        //Xa.Partition2x1
        //( 
        //  Xl, 
        //  Xr, nl
        //);

        //assert( Xl.row() == nl );
        //assert( Xr.row() == nr );
        //assert( Xl.col() == s );
        //assert( Xr.col() == s );
        

        /** Pa( 0:nl-1, : ) = Pl * Palr( :, 0:sl-1 )' */
        xgemm( "N", "T", nl, s, sl, 
            1.0,   Pl.data(), nl, 
                 Palr.data(), s, 
            0.0,   Pa.data(), n );
        /** Pa( nl:n-1, : ) = Pr * Palr( :, sl:sl+sr-1 )' */
        xgemm( "N", "T", nr, s, sr, 
            1.0,   Pr.data(), nr, 
                 Palr.data() + s * sl, s, 
            0.0,   Pa.data() + nl, n );

        

        //if ( DO_INVERSE ) Solve<true>( Xl, Xr );
        //printf( "end inner solve from telescope\n" ); fflush( stdout );

        if ( DO_INVERSE )
        {
            Data<T> x( sl + sr, s );
            Data<T> xl( sl, s );
            Data<T> xr( sr, s );

            /** xl = Vlt * Pa( 0:nl-1, : ) */
            xgemm( "T", "N", sl, s, nl, 
                1.0, Vl->data(), nl, 
                Pa.data(), n, 
                0.0,  xl.data(), sl );
            /** xr = Vrt * Pa( nl:n-1, : ) */
            xgemm( "T", "N", sr, s, nr, 
                1.0, Vr->data(), nr, 
                Pa.data() + nl, n, 
                0.0,  xr.data(), sr );

            /** b = [ Crl' * xr;
             *        Crl  * xl; ] */
            xgemm( "T", "N", sl, s, sr, 
                1.0, Crl.data(), sr, 
                xr.data(), sr, 
                0.0,   x.data(), sl + sr );
            xgemm( "N", "N", sr, s, sl, 
                1.0, Crl.data(), sr, 
                xl.data(), sl, 
                0.0,   x.data() + sl, sl + sr );

            /** b = inv( Z ) * b */
            xgetrs( "N", x.row(), x.col(), 
                Z.data(), Z.row(), ipiv.data(), 
                x.data(), x.row() );

            /** Pa( 0:nl-1, : ) -= Ul * b( 0:sl-1, : ) */
            xgemm( "N", "N", nl, s, sl, 
                -1.0, Ul->data(), nl, 
                x.data(), sl + sr, 
                1.0,  Pa.data(), n );
            /** Pa( nl:n-1, : ) -= Ur * b( sl:sl+sr-1, : ) */
            xgemm( "N", "N", nr, s, sr, 
                -1.0, Ur->data(), nr, 
                x.data() + sl, sl + sr, 
                1.0,  Pa.data() + nl, n );
        } /** end if ( DO_INVERSE ) */
      } /** end if ( do_ulv_factorization )*/
    };

    /** */
    void Orthogonalization()
    {
      /** Initialize householder reflectors "tau". */
      tau.resize( std::min( U.row(), U.col() ) );
      /** Initialize work space for xgeqrf. */
      Data<T> work( U.col() * 512, 1 );
      /** QR factorization without column pivoting. */
      xgeqrf( U.row(), U.col(), U.data(), U.row(),
          tau.data(), work.data(), work.size() );
      /** Copy U to Q to generate the full orthonormal basis. */
      Q = U;
      /** Increase the rank of Q to full rank. */
      Q.resize( U.row(), U.row() );
      /** Generate the full orthonormal basis Q. */
      xorgqr( Q.row(), Q.col(), U.col(), Q.data(), Q.row(), tau.data(), 
          work.data(), work.size() );



      /** Create views Qv = [Q1, Q2] for Q. */
      Qv.Set( false, Q );
      Qv.Partition1x2( Q1, Q2, tau.size(), LEFT );
      /** Sanity check for Q1'Q1 and Q2'Q2 and Q1'Q2. */
      Data<T> C = Q;
      Data<T> D = Q;

      xgemm( "Transpose", "No Transpose", C.row(), C.col(), Q.row(),
          1.0, Q.data(), Q.row(), 
               Q.data(), Q.row(),
          0.0, C.data(), C.row() );

      xgemm( "No Transpose", "Transpose", D.row(), D.col(), Q.row(),
          1.0, Q.data(), Q.row(), 
               Q.data(), Q.row(),
          0.0, D.data(), D.row() );

      for ( size_t j = 0; j < Q.col(); j ++ )
      {
        for ( size_t i = 0; i < Q.row(); i ++ )
        {
          if ( i == j ) assert( std::fabs( C( i, j ) - 1 ) < 1E-5 );
          else          assert( std::fabs( C( i, j ) - 0 ) < 1E-5 );
        }
      }
      for ( size_t j = 0; j < Q.col(); j ++ )
      {
        for ( size_t i = 0; i < Q.row(); i ++ )
        {
          if ( i == j ) assert( std::fabs( D( i, j ) - 1 ) < 1E-5 );
          else          assert( std::fabs( D( i, j ) - 0 ) < 1E-5 );
        }
      }

    };




    /** [Q2 Q1]' * B or B * [Q2 Q1] */
    void ChangeBasis( SideType side, hmlp::Data<T> & B)
    {
      /** Early return if Q does not exist. */
      if ( !Q.size() ) return;

      /** Create a deep copy of B. */
      hmlp::Data<T> A = B;

      /** Create matrix views for A and B. */
      hmlp::View<T> Av( false, A );
      hmlp::View<T> Bv( false, B );
      hmlp::View<T> Bl, Br, Bt, Bb;
     
      /** Enumerate case "LEFT", "RIGHT", and execptions. */
      switch ( side )
      {
        case LEFT:
        {
          /** Partition Bv = [ Bt; Bb ]. */
          Bv.Partition2x1( Bt,
                           Bb,     Q2.col(),  TOP );
          //printf("Bt %lux%lu Bb %lux%lu\n", Bt.row(), Bt.col(), 
          //                                  Bb.row(), Bb.col() ); fflush( stdout );

          /** Bt = Q2' * A */
          xgemm( "Transpose", "No Transpose", Bt.row(), Bt.col(), Q2.row(),
              1.0, Q2.data(), Q2.ld(),
                   Av.data(), Av.ld(),
              0.0, Bt.data(), Bt.ld() );
          /** Bb = Q1' * A */
          xgemm( "Transpose", "No Transpose", Bb.row(), Bb.col(), Q1.row(),
              1.0, Q1.data(), Q1.ld(),
                   Av.data(), Av.ld(),
              0.0, Bb.data(), Bb.ld() );
          break;
        }
        case RIGHT:
        {
          /** Partition Bv = [ Bl, Br ]. */
          Bv.Partition1x2( Bl, Br, Q2.col(), LEFT );

          //printf("Bl %lux%lu Br %lux%lu\n", Bl.row(), Bl.col(), 
          //                                  Br.row(), Br.col() ); fflush( stdout );

          /** Bl = A * Q2 */
          xgemm( "No Transpose", "No Transpose", Bl.row(), Bl.col(), Q2.row(),
              1.0, Av.data(), Av.ld(),
                   Q2.data(), Q2.ld(),
              0.0, Bl.data(), Bl.ld() );
          /** Br = A * Q1 */
          xgemm( "No Transpose", "No Transpose", Br.row(), Br.col(), Q1.row(),
              1.0, Av.data(), Av.ld(),
                   Q1.data(), Q1.ld(),
              0.0, Br.data(), Br.ld() );
          break;
        }
        default:
        {
          /** Do nothing and throw exception. */
          throw "Value of (SideType) side is not recognized.";
        }
      }
      //printf( "end ChangeBasis\n" ); fflush( stdout );
    }; /** changeBasis() */


    /** [Q2 Q1]' * A * [Q2 Q1] */
    void ChangeBasis( Data<T> &A )
    {
      ChangeBasis(  LEFT, A );
      ChangeBasis( RIGHT, A );
    }; /** changeBasis() */



    void ULVForward()
    {
      /** For internal nodes, B has been initialized by children. */
      if ( isLeaf() ) B = bview.toData();
      /** B = Q' * B */
      ChangeBasis( LEFT, B );
      /** P * Bf */
      xlaswp( Bf.col(), Bf.data(), Bf.ld(), 1, Bf.row(), ipiv.data(), 1 );
      /** Lff^{-1} * P * Bf, where Lff is the lower-triangular part of Ztl. */
      xtrsm( "Left", "Lower", "No transpose", "Unit", Bf.row(), Bf.col(), 
          1.0, Ztl.data(), Ztl.ld(), Bf.data(), Bf.ld() );
      /** Bc -= Lcf * Bf, where Lcf is Zbl. */
      xgemm( "No Transpose", "No Transpose", Bc.row(), Bc.col(), Bf.row(),
          -1.0, Zbl.data(), Zbl.ld(), Bf.data(), Bf.ld(), 1.0, Bc.data(), Bc.ld() );
      //printf( "Bc %lux%lu Bp %lux%lu\n", Bc.row(), Bc.col(), Bp.row(), Bp.col() ); fflush( stdout );
      /** Copy Bc to Bp (subview of parent's B). */
      Bp.CopyValuesFrom( Bc );
    }; /** end ULVForward() */


    void ULVBackward()
    {
      /** Copy Bp (subview of parent's B) to Bc. */
      Bc.CopyValuesFrom( Bp );
      /** Bf -= Ufc * Bc, where Ufc is Ztr. */
      xgemm( "No Transpose", "No Transpose", Bf.row(), Bf.col(), Bc.row(),
          -1.0, Ztr.data(), Ztr.ld(), Bc.data(), Bc.ld(), 1.0, Bf.data(), Bf.ld() );
      /** Lff^{-1} * P * Bf, where Lff is the lower-triangular part of Ztl. */
      xtrsm( "Left", "Upper", "No transpose", "Non-unit", Bf.row(), Bf.col(), 
          1.0, Ztl.data(), Ztl.ld(), Bf.data(), Bf.ld() );
      if ( Q.size() )
      {
        /** Create a temporary buffer for projection Q2 * Bf + Q1 * Bc. */
        Data<T> A = B;
        xgemm( "No Transpose", "No Transpose", A.row(), A.col(), Bf.row(),
            1.0, Q2.data(), Q2.ld(), Bf.data(), Bf.ld(), 0.0, A.data(), A.row() );
        xgemm( "No Transpose", "No Transpose", A.row(), A.col(), Bc.row(),
            1.0, Q1.data(), Q1.ld(), Bc.data(), Bc.ld(), 1.0, A.data(), A.row() );
        /** Copy A back to B. */
        if ( isLeaf() ) bview.CopyValuesFrom( A );
        else Bv.CopyValuesFrom( A );
      }
    }; /** end ULVBackward() */


    bool isLeaf() const noexcept
    {
      return is_leaf_;
    }




    bool is_leaf_ = false;

    bool isroot = false;

    size_t n = 0;

    size_t nl = 0;

    size_t nr = 0;

    size_t s = 0;

    size_t sl = 0;

    size_t sr = 0;


    /** Reduced system Z = [ I  VU   if ( HODLR || p-HSS )
     *                       VU  I ] */
    Data<T> Z;
    View<T> Zv;
    View<T> Ztl, Ztr, Zbl, Zbr;

    /** Partial pivoting order (used in GETRF). */
    vector<int> ipiv;

    /** n-by-s (SMW) or (sl+sr)-by-s (ULV) */
    Data<T> U, V;

    /** sr-by-sl and sl-by-sr, skeleton row and column basis. */
    Data<T> Crl, Clr;

    /** A correspinding view of the right hand side of this node. */
    View<T> bview;

    /** Pointers to children's factors */
    Data<T> *Ul = NULL;
    Data<T> *Ur = NULL;
    Data<T> *Vl = NULL;
    Data<T> *Vr = NULL;

    /** Q, (sl+sr)-by-s (ULV) */
    Data<T> Q;
    View<T> Qv, Q1, Q2;

    /** tau, sl+sr (used in xgeqrf( U ) of ULV) */
    vector<T> tau;

    /** Temporary buffer for the solve. */
    Data<T> B;
    View<T> Bv, Bp, Bsibling, Bf, Bc;

  protected: /** this class will be public inherit by gofmm::Data<T> */

    bool issymmetric = true;

    bool do_ulv_factorization = false;

}; /** end class Factor */


/**
 *  @brief 
 */ 
template<typename NODE, typename T>
void SetupFactor( NODE *node )
{
  size_t n, nl, nr, s, sl, sr;
  bool issymmetric, do_ulv_factorization;
  

#ifdef DEBUG_IGOFMM
  printf( "begin SetupFactor %lu\n", node->treelist_id ); fflush( stdout );
#endif

  issymmetric = node->setup->IsSymmetric();
  do_ulv_factorization = node->setup->do_ulv_factorization;
  n  = node->n;
  nl = 0;
  nr = 0;
  s  = node->data.skels.size();
  sl = 0;
  sr = 0;

  if ( !node->isLeaf() )
  {
    nl = node->lchild->n;
    nr = node->rchild->n;
    sl = node->lchild->data.skels.size();
    sr = node->rchild->data.skels.size();
  }


  node->data.SetupFactor( issymmetric, do_ulv_factorization,
    node->isLeaf(), !node->getGlobalDepth(), n, nl, nr, s, sl, sr );

#ifdef DEBUG_IGOFMM
  printf( "end SetupFactor %lu\n", node->treelist_id ); fflush( stdout );
#endif

}; /** end void SetupFactor() */


/**
 *  @brief
 */ 
template<typename NODE, typename T>
class SetupFactorTask : public Task
{
  public:

    NODE *arg = NULL;

    hmlpError_t Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "sf" );
      label = to_string( arg->treelist_id );
      cost = 1.0;
      return HMLP_ERROR_SUCCESS;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    hmlpError_t DependencyAnalysis()
    {
      arg->DependencyAnalysis( W, this );
      this->TryEnqueue();
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t Execute( Worker* user_worker )
    {
      SetupFactor<NODE, T>( arg );
      return HMLP_ERROR_SUCCESS;
    };

}; /** end class SetupFactorTask */




template<typename NODE>
void SolverTreeView( NODE *node )
{
  auto &data   = node->data;
  auto *setup  = node->setup;
  auto &input  = *(setup->input);
  auto &output = *(setup->output);
  /** Allocate working buffer for ULV solve. */
  if ( node->isLeaf() ) data.B.resize( data.n, input.col() );
  else data.B.resize( data.sl + data.sr, input.col() );

  /** Partition B = [ Bf; Bc ] with matrix view. */
  data.Bv.Set( data.B );
  data.Bv.Partition2x1( data.Bf,
                        data.Bc,  data.s, BOTTOM );

  /** Create contigious matrix view for output at root level. */
  if ( !node->parent ) data.bview.Set( output );

  /** Hierarchical tree view. */
  if ( !node->isLeaf() )
  {
    auto &ldata = node->lchild->data;
    auto &rdata = node->rchild->data;
    /** Partition b = [ bl; br; ] with matrix view. */
    data.bview.Partition2x1( ldata.bview, 
                             rdata.bview, data.nl, TOP );
    data.Bv.Partition2x1( ldata.Bp,
                          rdata.Bp, data.sl, TOP );
  }
}; /** end SolverTreeView() */




/** @brief Creates an hierarchical tree view for a matrix. */
template<typename NODE>
class SolverTreeViewTask : public Task
{
  public:

    NODE *arg = NULL;

    hmlpError_t Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "TreeView" );
      label = to_string( arg->treelist_id );
      cost = 1.0;
      return HMLP_ERROR_SUCCESS;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    /** Preorder dependencies (with a single source node) */
    hmlpError_t DependencyAnalysis() 
    { 
      arg->DependOnParent( this ); 
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t Execute( Worker* user_worker ) 
    { 
      SolverTreeView( arg ); 
      return HMLP_ERROR_SUCCESS;
    };

}; /** end class TreeViewTask */



/**
 *  @brief doward traversal to create matrix views, at the leaf
 *         level execute explicit permutation.
 */ 
template<bool FORWARD, typename NODE>
class MatrixPermuteTask : public hmlp::Task
{
  public:

    NODE *arg;

    hmlpError_t Set( NODE *user_arg )
    {
      name = std::string( "MatrixPermutation" );
      arg = user_arg;
      cost = 1.0;
      return HMLP_ERROR_SUCCESS;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    /** depends on previous task */
    hmlpError_t DependencyAnalysis()
    {
      if ( FORWARD )
      {
        arg->DependencyAnalysis( RW, this );
      }
      else
      {
        this->Enqueue();
      }
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t Execute( Worker* user_worker )
    {
      //printf( "PermuteMatrix %lu\n", arg->treelist_id );
      auto *node   = arg;
      auto &gids   = node->gids;
      auto &input  = *(node->setup->input);
      auto &output = *(node->setup->output);
      auto &A      = node->data.bview;

      assert( A.row() == gids.size() );
      assert( A.col() == input.col() );

      //for ( size_t i = 0; i < gids.size(); i ++ )
      //  printf( "%lu ", gids[ i ] );
      //printf( "\n" );

      /** perform permutation and output */
      for ( size_t j = 0; j < input.col(); j ++ )
        for ( size_t i = 0; i < gids.size(); i ++ )
          /** foward  permutation */
          if ( FORWARD ) A( i, j ) = input( gids[ i ], j );
          /** inverse permutation */
          else           input( gids[ i ], j ) = A( i, j );

      //for ( size_t j = 0; j < 1; j ++ )
      //  for ( size_t i = 0; i < gids.size(); i ++ )
      //    printf( "%E ", A( i, j ) );
      //printf( "\n" );

      //printf( "end PermuteMatrix %lu\n", arg->treelist_id );
      return HMLP_ERROR_SUCCESS;
    };

}; /** end class MatrixPermuteTask */



/**
 *  @brief
 */ 
template<typename NODE, typename T>
void Apply( NODE *node )
{
  auto &data = node->data;
  auto &setup = node->setup;
  auto &K = *setup->K;

  if ( node->isLeaf() )
  {
    auto lambda = setup->lambda;
    auto &amap = node->gids;
    /** evaluate the diagonal block */
    auto Kaa = K( amap, amap );
    /** apply the regularization */
    for ( size_t i = 0; i < Kaa.row(); i ++ ) 
      Kaa[ i * Kaa.row() + i ] += lambda;
  }
  else
  {
    auto &bl = node->lchild->data.bview;
    auto &br = node->rchild->data.bview;
    data.Apply<true>( bl, br );
  }
}; /** end Apply() */



//template<typename NODE, typename T>
//void ULVForwardSolve( NODE *node ) { node->data.ULVForward(); };



template<typename NODE, typename T>
class ULVForwardSolveTask : public Task
{
  public:

    NODE *arg = nullptr;

    hmlpError_t Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "ulvforward" );
      label = to_string( arg->treelist_id );
      cost = 1.0;
      return HMLP_ERROR_SUCCESS;
    };

    //void DependencyAnalysis()
    //{      
    //  arg->DependencyAnalysis( RW, this );
    //  /** depend on two children */
    //  if ( !arg->isLeaf() )
    //  {
    //    arg->lchild->DependencyAnalysis( R, this );
    //    arg->rchild->DependencyAnalysis( R, this );
    //  }
    //  /** dispatch the task if there is no dependency */
    //  this->TryEnqueue();
    //};

    hmlpError_t DependencyAnalysis() 
    { 
      arg->DependOnChildren( this ); 
      return HMLP_ERROR_SUCCESS;
    };


    hmlpError_t Execute( Worker* user_worker ) 
    { 
      arg->data.ULVForward(); 
      return HMLP_ERROR_SUCCESS;
    };
    
}; /** end class ULVForwardSolveTask */




template<typename NODE, typename T>
class ULVBackwardSolveTask : public Task
{
  public:

    NODE *arg = nullptr;

    hmlpError_t Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "ulvbackward" );
      label = std::to_string( arg->treelist_id );
      cost = 1.0;
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t DependencyAnalysis() 
    { 
      arg->DependOnParent( this ); 
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t Execute( Worker* user_worker ) 
    { 
      arg->data.ULVBackward(); 
      return HMLP_ERROR_SUCCESS;
    };
    
}; /** end class ULVBackwardSolveTask */


















/**
 *  @brief 
 */ 
template<typename NODE, typename T>
void Solve( NODE *node )
{
  
  auto &data = node->data;
  auto &setup = node->setup;
  auto &K = *setup->K;


  //printf( "%lu beg Solve\n", node->treelist_id ); fflush( stdout );

  /** TODO: need to decide to use LU or not */
  if ( node->isLeaf() )
  {
    auto &b = data.bview;
    data.Solve( b );
    //printf( "Solve %lu, m %lu n %lu\n", node->treelist_id, b.row(), b.col() );
  }
  else
  {
    auto &bl = node->lchild->data.bview;
    auto &br = node->rchild->data.bview;
    data.Solve( bl, br );
    //printf( "Solve %lu, m %lu n %lu\n", node->treelist_id, bl.row(), bl.col() );
  }

  //printf( "%lu end Solve\n", node->treelist_id ); fflush( stdout );

}; /** end Solve() */


/**
 *  @brief
 */ 
template<typename NODE, typename T>
class SolveTask : public Task
{
  public:

    NODE *arg = nullptr;

    hmlpError_t Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "sl" );
      label = to_string( arg->treelist_id );
      cost = 1.0;
      return HMLP_ERROR_SUCCESS;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    hmlpError_t DependencyAnalysis()
    {
      arg->DependencyAnalysis( RW, this );
      if ( !arg->isLeaf() )
      {
        arg->lchild->DependencyAnalysis( R, this );
        arg->rchild->DependencyAnalysis( R, this );
      }
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t Execute( Worker* user_worker )
    {
      Solve<NODE, T>( arg );
      return HMLP_ERROR_SUCCESS;
    };

}; /** end class SolveTask */


/**
 *
 */ 
template<typename T, typename TREE>
hmlpError_t Solve( TREE &tree, Data<T> &input )
{
  using NODE = typename TREE::NODE;

  const bool AUTO_DEPENDENCY = true;
  const bool USE_RUNTIME     = true;

  /** copy input to output */
  auto *output = new Data<T>( input.row(), input.col() );

  SolverTreeViewTask<NODE>             treeviewtask;
  MatrixPermuteTask<true,  NODE> forwardpermutetask;
  MatrixPermuteTask<false, NODE> inversepermutetask;
  /** Sherman-Morrison-Woodbury */
  SolveTask<NODE, T>      solvetask1;
  /** ULV */
  ULVForwardSolveTask<NODE, T>   ulvforwardsolvetask;
  ULVBackwardSolveTask<NODE, T>  ulvbackwardsolvetask;

  /** attach the pointer to the tree structure */
  tree.setup.input  = &input;
  tree.setup.output = output;

  if ( tree.setup.do_ulv_factorization )
  {
    /** clean up all dependencies on tree nodes */
    RETURN_IF_ERROR( tree.dependencyClean() );
    tree.traverseDown( treeviewtask );
    tree.traverseLeafs( forwardpermutetask );
    tree.traverseUp( ulvforwardsolvetask );
    tree.traverseDown( ulvbackwardsolvetask );
    if ( USE_RUNTIME ) hmlp_run();

    /** clean up all dependencies on tree nodes */
    RETURN_IF_ERROR( tree.dependencyClean() );
    tree.traverseLeafs( inversepermutetask );
    if ( USE_RUNTIME ) hmlp_run();
  }
  else
  {
    /** clean up all dependencies on tree nodes */
    RETURN_IF_ERROR( tree.dependencyClean() );
    tree.traverseDown( treeviewtask );
    tree.traverseLeafs( forwardpermutetask );
    tree.traverseUp( solvetask1 );
    if ( USE_RUNTIME ) hmlp_run();
    /** clean up all dependencies on tree nodes */
    RETURN_IF_ERROR( tree.dependencyClean() );
    tree.traverseLeafs( inversepermutetask );
    if ( USE_RUNTIME ) hmlp_run();
  }

  /** delete buffer space */
  delete output;

  return HMLP_ERROR_SUCCESS;
}; /** end Solve() */





/**
 *  @brief Compute relative Forbenius error for two-sided 
 *  interpolative decomposition.
 */ 
template<typename NODE, typename T>
void LowRankError( NODE *node )
{
  auto &data = node->data;
  auto &setup = node->setup;
  auto &K = *setup->K;

  if ( !node->isLeaf() )
  {
    auto Krl = K( node->rchild->gids, node->lchild->gids );

    auto nrm2 = hmlp_norm( Krl.row(),  Krl.col(), 
                           Krl.data(), Krl.row() ); 


    hmlp::Data<T> VrCrl( data.nr, data.sl );

    /** VrCrl = Vr * Crl */
    xgemm( "N", "N", data.nr, data.sl, data.sr,
        1.0, data.Vr->data(), data.nr,
             data.Crl.data(), data.sr,
        0.0, VrCrl.data(), data.nr );

    /** Krl - VrCrlVl' */
    xgemm( "N", "T", data.nr, data.nl, data.sl,
       -1.0, VrCrl.data(), data.nr,
             data.Vl->data(), data.nl,
        1.0, Krl.data(), data.nr );

    auto err = hmlp_norm( Krl.row(),  Krl.col(), 
                          Krl.data(), Krl.row() ); 

    printf( "%4lu ||Krl -VrCrlVl|| %3.1E\n", 
        node->treelist_id, std::sqrt( err / nrm2 ) );
  }

}; /** end LowRankError() */



/**
 *  @brief Factorizarion using LU and SMW
 */ 
template<typename NODE, typename T>
hmlpError_t Factorize(NODE *node)
{
  auto &data = node->data;
  auto &setup = node->setup;
  auto &K = *setup->K;
  auto &proj = data.proj;

  auto do_ulv_factorization = setup->do_ulv_factorization;

  if ( node->isLeaf() )
  {
    auto lambda = setup->lambda;
    auto &amap = node->gids;

    /** Evaluate the diagonal block. */
    Data<T> Kaa = K( amap, amap );

    /** Apply the regularization */
    for ( size_t i = 0; i < Kaa.row(); i ++ ) Kaa( i, i ) += lambda;

    if ( do_ulv_factorization )
    {
      /** U = proj */
      data.Telescope( false, data.U, proj );
      /** QR factorization */
      data.Orthogonalization();
      /** LU factorization */
      data.partialLU( Kaa );
    }
    else
    {
      /** LU factorization */
      data.Factorize( Kaa );
      /** U = inv( Kaa ) * proj' */
      data.Telescope( true, data.U, proj );
      /** V = proj' */
      data.Telescope( false, data.V, proj );
    }
  }
  else
  {
    auto &Ul = node->lchild->data.U;
    auto &Vl = node->lchild->data.V;
    auto &Zl = node->lchild->data.Zbr;
    auto &Ur = node->rchild->data.U;
    auto &Vr = node->rchild->data.V;
    auto &Zr = node->rchild->data.Zbr;

    /** Evluate the skeleton rows and columns. */
    auto &amap = node->lchild->data.skels;
    auto &bmap = node->rchild->data.skels;

    /* Get the skeleton rows and columns */
    node->data.Crl = K(bmap, amap);

    if ( do_ulv_factorization )
    {
      if ( !node->data.isroot )
      {
        //printf( "here level %lu\n", node->l );
        data.Telescope( false, data.U, proj, Ul, Ur );
        data.Orthogonalization();
      }
      data.partialLU( Zl, Zr, Ul, Ur, Vl, Vr );
    }
    else
    {
      /** SMW factorization (LU or Cholesky) */
      data.Factorize( Ul, Ur, Vl, Vr );
      /** telescope U and V */
      if ( !node->data.isroot )
      {
        /** U = inv( I + UCV' ) * [ Ul; Ur ] * proj' */
        data.Telescope(  true, data.U, proj, Ul, Ur );
        /** V = [ Vl; Vr ] * proj' */
        data.Telescope( false, data.V, proj, Vl, Vr );
      }
    }
  }

  return HMLP_ERROR_SUCCESS;
}; /** end void Factorize() */



/**
 *  @brief
 */ 
template<typename NODE, typename T>
class FactorizeTask : public Task
{
  public:

    NODE *arg = nullptr;

    hmlpError_t Set( NODE *user_arg )
    {
      arg = user_arg;
      name = string( "fa" );
      label = to_string( arg->treelist_id );
      // Need an accurate cost model.
      cost = 1.0;
      return HMLP_ERROR_SUCCESS;
    };

    void GetEventRecord()
    {
      double flops = 0.0, mops = 0.0;
      event.Set( label + name, flops, mops );
    };

    hmlpError_t DependencyAnalysis() 
    { 
      arg->DependOnChildren( this ); 
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t Execute( Worker* user_worker ) 
    { 
      return Factorize<NODE, T>( arg ); 
    };

}; /** end class FactorizeTask */













/** @biref Top-level factorization routine. */
template<typename T, typename TREE>
hmlpError_t Factorize( TREE &tree, T lambda )
{
  using NODE = typename TREE::NODE;

  /** Clean up all dependencies on tree nodes. */
  RETURN_IF_ERROR( tree.dependencyClean() );

  /** Regularization parameter lambda. */
  tree.setup.lambda = lambda;

  /** Perform ULV factorization. */
  tree.setup.do_ulv_factorization = true;

  /** Setup  */
  SetupFactorTask<NODE, T> setupfactortask; 
  tree.traverseUp( setupfactortask );
  tree.ExecuteAllTasks();

  /** Factorization */
  FactorizeTask<NODE, T> factorizetask; 
  tree.traverseUp( factorizetask );
  tree.ExecuteAllTasks();

  return HMLP_ERROR_SUCCESS;

}; /** end Factorize() */



/**
 *  @brief Compute the average 2-norm error. That is given
 *         lambda and weights, 
 */ 
template<typename TREE, typename T>
void ComputeError( TREE &tree, T lambda, Data<T> weights, Data<T> potentials )
{
  using NODE = typename TREE::NODE;
  

  /** assure the dimension matches */
  assert( weights.row() == potentials.row() );
  assert( weights.col() == potentials.col() );

  size_t n    = weights.row();
  size_t nrhs = weights.col();

  /** shift lambda and make it a column vector */
  Data<T> rhs( n, nrhs );
  for ( size_t j = 0; j < nrhs; j ++ )
    for ( size_t i = 0; i < n; i ++ )
      rhs( i, j ) = potentials( i, j ) + lambda * weights( i, j );

  /** potentials = inv( K + lambda * I ) * potentials */
  Solve( tree, rhs );


  /** Compute relative error = sqrt( err / nrm2 ) for each rhs */
  printf( "========================================================\n" );
  printf( "Inverse accuracy report\n" );
  printf( "========================================================\n" );
  printf( "#rhs,  max err,        @,  min err,        @,  relative \n" );
  printf( "========================================================\n" );
  size_t ntest = 10;
  T total_err  = 0.0;
  for ( size_t j = 0; j < std::min( nrhs, ntest ); j ++ )
  {
    /** counters */
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

}; /** end ComputeError() */








}; /** end namespace gofmm */
}; /** end namespace hmlp */

#endif /** define IGOFMM_HPP */
