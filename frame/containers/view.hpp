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




#ifndef VIEW_HPP
#define VIEW_HPP

#include <containers/data.hpp>

using namespace std;


namespace hmlp
{

typedef enum { TRANSPOSE, NOTRANSPOSE } TransposeType;
/** 1x2, 2x1, 1x3, 3x1 */
typedef enum { LEFT, RIGHT, BOTTOM, TOP } SideType;
/** 2x2, 3x3 */
typedef enum { TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT } QuadrantType;


template<typename T>
class View : public ReadWrite
{
  public:

    /** empty constructor */
    View() {};

    /** constructor for the buffer */
    View( hmlp::Data<T> &buff ) { Set( buff ); };

    View( bool TRANS, hmlp::Data<T> &buff )
    { 
      Set( TRANS, buff );
    };


    /** destructor */
    //~View() { printf( "~View()\n" ); fflush( stdout ); };

    /** base case setup */
    void Set( bool TRANS, hmlp::Data<T> &buff )
    {
      this->trans = TRANS;
      if ( trans )
      {
        /** hmlp::Data<T> is stored in column major */
        this->m = buff.col();
        this->n = buff.row();
      }
      else
      {
        this->m = buff.row();
        this->n = buff.col();
      }
      this->offm  = 0;
      this->offn  = 0;
      this->base  = this;
      this->buff  = &buff;
    };

    void Set( Data<T> &buff )
    {
      /** default is none transpose  */
      Set( false, buff );
    };

    /** non-base case setup */
    void Set( size_t m, size_t n, size_t offm, size_t offn, hmlp::View<T> *base )
    {
      this->trans = base->trans;
      if ( trans )
      {
        assert( offm <= base->buff->col() );
        assert( offn <= base->buff->row() );
      }
      else
      {
        assert( offm <= base->buff->row() );
        assert( offn <= base->buff->col() );
      }
      this->m     = m;
      this->n     = n;
      this->offm  = offm;
      this->offn  = offn;
      this->base  = base;
      this->buff  = base->buff;
    };

    /** subview operator */
    template<typename TINDEX>
    T & operator () ( TINDEX i, TINDEX j )
    {
      size_t offset = 0;
      if ( trans )
      {
        assert( offm + i < buff->col() );
        assert( offn + j < buff->row() );
        offset = i * ld() + j;
      }
      else
      {
        assert( offm + i < buff->row() );
        assert( offn + j < buff->col() );
        offset = j * ld() + i;
      }
      return *( data() + offset );
    };

    void CopyValuesFrom( View<T> & A )
    {
      if ( !m || !n ) return;
      assert( A.row() == m && A.col() == n );
      for ( size_t j = 0; j < n; j ++ )
        for ( size_t i = 0; i < m; i ++ )
          (*this)( i, j ) = A( i, j );
    };

    void CopyValuesFrom( Data<T> & A )
    {
      if ( !m || !n ) return;
      assert( A.row() == m && A.col() == n );
      for ( size_t j = 0; j < n; j ++ )
        for ( size_t i = 0; i < m; i ++ )
          (*this)( i, j ) = A( i, j );
    };

    /** Return a data copy of the subview */
    Data<T> toData()
    {
      Data<T> A( m, n );
      for ( size_t j = 0; j < n; j ++ )
        for ( size_t i = 0; i < m; i ++ )
          A( i, j ) = (*this)( i, j );
      return A;
    };

    /** [ A1;   = A 
     *    A2; ]     */
    void Partition2x1
    (
      hmlp::View<T> &A1,
      hmlp::View<T> &A2, size_t mb, SideType side 
    )
    {
      /** readjust mb */
      if ( mb > m ) mb = m;
      if ( side == BOTTOM ) mb = m - mb;
      /** setup A1 */
      A1.Set(     mb, this->n, this->offm,      this->offn, this->base );
      /** setup A2 */
      A2.Set( m - mb, this->n, this->offm + mb, this->offn, this->base );
    };


    /** A = [ A1; 
     *        A2; ] */
    void ContinueWith2x1
    (
      hmlp::View<T> &A1,
      hmlp::View<T> &A2
    )
    {
      if ( A1.row() && A2.row() ) assert( A1.col() == A2.col() );
      (*this) = A1;
      assert( A2.HasTheSameBuffer( this->buff ) );
      this->m = A1.row() + A2.row();
    };

    /** [ A1, A2; ] = A */
    void Partition1x2
    (
      hmlp::View<T> &A1, hmlp::View<T> &A2, size_t nb, SideType side
    )
    {
      /** readjust mb */
      if ( nb > n ) nb = n;
      if ( side == RIGHT ) nb = n - nb;
      /** setup A1 */
      A1.Set( this->m,     nb, this->offm, this->offn,      this->base );
      /** setup A2 */
      A2.Set( this->m, n - nb, this->offm, this->offn + nb, this->base );
    };

    /** A = [ A1, A2; ] */
    void ContinueWith1x2
    (
      hmlp::View<T> &A1, hmlp::View<T> &A2
    )
    {
      if ( A1.col() && A2.col() ) assert( A1.row() == A2.row() );
      (*this) = A1;
      assert( A2.HasTheSameBuffer( this->buff ) );
      this->n = A1.col() + A2.col();
    };

    /** A = [ A11, A12; A21, A22; ]; */
    void Partition2x2
    (
      hmlp::View<T> &A11, hmlp::View<T> &A12,
      hmlp::View<T> &A21, hmlp::View<T> &A22,
      size_t mb, size_t nb, QuadrantType quadrant
    )
    {
      if ( mb > m ) mb = m;
      if ( nb > n ) nb = n;

      switch ( quadrant )
      {
        case TOPLEFT:
        {
          break;
        }
        case TOPRIGHT:
        {
          nb = n - nb;
          break;
        }
        case BOTTOMLEFT:
        {
          mb = m - mb;
          break;
        }
        case BOTTOMRIGHT:
        {
          mb = m - mb;
          nb = n - nb;
          break;
        }
        default:
        {
          printf( "invalid quadrant\n" );
          break;
        }
      }

      /** setup A11 */
      A11.Set(     mb,     nb, offm     , offn     , this->base );
      /** setup A12 */
      A12.Set(     mb, n - nb, offm     , offn + nb, this->base );
      /** setup A21 */
      A21.Set( m - mb,     nb, offm + mb, offn     , this->base );
      /** setup A22 */
      A22.Set( m - mb, n - nb, offm + mb, offn + nb, this->base );

    };


    void ContinueWith2x2
    (
      hmlp::View<T> &A11, hmlp::View<T> &A12,
      hmlp::View<T> &A21, hmlp::View<T> &A22
    )
    {
       assert( A11.row() == A12.row() );
       assert( A11.col() == A21.col() );
       assert( A22.row() == A21.row() );
       assert( A22.col() == A12.col() );
       (*this) = A11;
       this->m = A11.row() + A21.row();
       this->n = A11.col() + A12.col();
    };

    bool IsTransposed()
    {
      return trans;
    };

    bool HasTheSameBuffer( hmlp::Data<T> *target )
    {
      return ( target == buff );
    };

    bool HasUniformBlockSize()
    {
      return has_uniform_block_size;
    };

    void CreateLeafMatrixBlocks( size_t mb, size_t nb )
    {
      /** only the base view can have leaf r/w blocks */
      if ( base == this )
      {
        if ( !rwblocks.HasBeenSetup() )
        {
          this->mb = mb;
          this->nb = nb;
          rwblocks.Setup(
              (size_t)std::ceil( (double)m / mb ),
              (size_t)std::ceil( (double)n / nb ) );
        }
        assert( this->mb == mb );
        assert( this->nb == nb );
      }
      else
      {
        base->CreateLeafMatrixBlocks( mb, nb );
      }
    };

    size_t GetRowBlockSize()
    {
      if ( base == this ) return mb;
      else                return base->GetRowBlockSize();
    };

    size_t GetColumnBlockSize()
    {
      if ( base == this ) return nb;
      else                return base->GetColumnBlockSize();
    };

    bool HasLeafReadWriteBlocks()
    {
      if ( base == this ) return rwblocks.HasBeenSetup();
      else                return base->HasLeafReadWriteBlocks();
    };

    /**
     *  @brief  If leaf r/w blocks were created, then the r/w dependency
     *          applies to all leaf r/w blocks covered by this view.
     *          Otherwise, the r/w dependency only applies to this view.
     */ 
    void DependencyAnalysis( ReadWriteType type, hmlp::Task *task )
    {
      if ( HasLeafReadWriteBlocks() )
      {
        for ( size_t j = 0; j < n; j += GetColumnBlockSize() )
          for ( size_t i = 0; i < m; i += GetRowBlockSize() )
            DependencyAnalysis( offm + i, offn + j, type, task );
      }
      else
      {
        ReadWrite::DependencyAnalysis( type, task );
      }
    };

    void DependencyAnalysis( size_t i, size_t j, ReadWriteType type, hmlp::Task *task )
    {
      if ( base == this )
      {
        rwblocks.DependencyAnalysis( i / mb, j / nb, type, task );
      }
      else
      {
        base->DependencyAnalysis( i, j, type, task );
      }    
    };

    void DependencyCleanUp()
    {
      if ( base == this ) rwblocks.DependencyCleanUp();
      else base->DependencyCleanUp();
    };

    /** return the row size of the current view */
    size_t row() { return m; };

    /** return the col size of the current view */
    size_t col() { return n; };

    /** return leading dimension of the buffer */
    size_t ld()  { return buff->row(); };

    /** return the pointer of the current view in the buffer */
    T *data()
    {
      assert( buff );
      size_t offset;
      if ( trans ) offset = offm * ld() + offn;
      else         offset = offn * ld() + offm;
      return ( buff->data() + offset );
    };

    /** print out all information */
    void Print()
    {
      if ( trans )
      {
        printf( "[ %5lu+%5lu:%5lu ][ %5lu+%5lu:%5lu ]\n",
            offm, m, buff->col(), offn, n, buff->row() );
      }
      else
      {
        printf( "[ %5lu+%5lu:%5lu ][ %5lu+%5lu:%5lu ]\n",
            offm, m, buff->row(), offn, n, buff->col() );
      }
    }; 

  private:

    /** whether this is a transpose view? */
    bool trans = false;

    size_t m = 0;

    size_t n = 0;

    size_t offm = 0;

    size_t offn = 0;

    hmlp::View<T> *base = NULL;

    hmlp::Data<T> *buff = NULL;

    /** we can only have one kind of mb and nb */
    size_t mb = 0;

    size_t nb = 0;

    bool has_uniform_block_size = true;

    MatrixReadWrite rwblocks;

}; /** end class View */


/**
 *  @brief
 */ 
template<typename T>
void Partition1x2
( 
  View<T> &A, View<T> &A1, View<T> &A2,
  size_t nb, SideType side 
)
{
  A.Partition1x2( A1, A2, nb, side );
}; /** end Partition1x2() */


/**
 *  @brief
 */ 
template<typename T>
void Partition2x1
(
  View<T> &A, View<T> &A1, 
              View<T> &A2, 
  size_t mb, SideType side 
)
{
  A.Partition2x1( A1, 
                  A2, mb, side );
}; /** end Partition2x1() */


/**
 *  @brief
 */ 
template<typename T>
void Partition2x2
(
  hmlp::View<T> &A, View<T> &A11, View<T> &A12,
                    View<T> &A21, View<T> &A22,
  size_t mb, size_t nb, QuadrantType quadrant 
)
{
  A.Partition2x2( A11, A12,
                  A21, A22, mb, nb, quadrant );
}; /** end Partition2x2() */ 


/**
 *  @brief
 */ 
template<typename T>
void Repartition1x2To1x3
(
  View<T> &AL,              View<T> &AR,
  View<T> &A0, View<T> &A1, View<T> &A2,
  size_t nb, SideType side 
)
{
  switch ( side )
  {
    case LEFT: 
    {
      AL.Partition1x2( A0, A1, nb, RIGHT );
      A2 = AR;
      break;
    }
    case RIGHT:
    {
      A0 = AL;
      AR.Partition1x2( A1, A2, nb, LEFT );
      break;
    }
    default:
    {
      printf( "invalid side\n" );
      break;
    }
  } 

}; /** end Repartition1x2To1x3()*/


template<typename T>
void ContinueWith1x3To1x2
(
  View<T> &AL,              View<T> &AR,
  View<T> &A0, View<T> &A1, View<T> &A2,
  SideType side 
)
{
  switch ( side )
  {
    case LEFT: 
    {
      AL.ContinueWith1x2( A0, A1 );
      AR = A2;
      break;
    }
    case RIGHT:
    {
      AL = A0;
      AR.ContinueWith1x2( A1, A2 );
      break;
    }
    default:
    {
      printf( "invalid side\n" );
      break;
    }
  }
}; /** end ContinueWith1x3To1x2() */




template<typename T>
void Repartition2x1To3x1
(
  View<T> &AT, View<T> &A0,
               View<T> &A1, 
  View<T> &AB, View<T> &A2,
  size_t mb, SideType side 
)
{
  switch ( side )
  {
    case TOP: 
    {
      AT.Partition2x1( A0,
                       A1, mb, BOTTOM );
      A2 = AB;
      break;
    }
    case BOTTOM:
    {
      A0 = AT;
      AB.Partition2x1( A1,
                       A2, mb, TOP );
      break;
    }
    default:
    {
      printf( "invalid side\n" );
      break;
    }
  }
}; /** end Repartition2x1To3x1()*/


template<typename T>
void ContinueWith3x1To2x1
(
  View<T> &AT, View<T> &A0,
               View<T> &A1, 
  View<T> &AB, View<T> &A2,
  SideType side 
)
{
  switch ( side )
  {
    case TOP: 
    {
      AT.ContinueWith2x1( A0,
                          A1 );
      AB = A2;
      break;
    }
    case BOTTOM:
    {
      AT = A0;
      AB.ContinueWith2x1( A1,
                          A2 );
      break;
    }
    default:
    {
      printf( "invalid side\n" );
      break;
    }
  }
}; /** end ContinueWith3x1To2x1() */


/**
 *  @brief 
 */ 
template<typename T>
void Repartition2x2To3x3
(
  View<T> &ATL, View<T> &ATR, View<T> &A00, View<T> &A01, View<T> &A02,
                              View<T> &A10, View<T> &A11, View<T> &A12,
  View<T> &ABL, View<T> &ABR, View<T> &A20, View<T> &A21, View<T> &A22,
  size_t mb, size_t nb, QuadrantType quadrant 
)
{
  switch ( quadrant )
  {
    case TOPLEFT:
    {
      ATL.Partition2x2( A00, A01,
                        A10, A11, mb, nb, BOTTOMRIGHT );
      ATR.Partition2x1( A02,
                        A12,      mb,     BOTTOM      );
      ABL.Partition1x2( A20, A21,     nb,       RIGHT );
      A22 = ABR;
      break;
    }
    case TOPRIGHT:
    {
      ATL.Partition2x1( A00,
                        A10,      mb,     BOTTOM     );
      ATR.Partition2x2( A01, A02,
                        A11, A12, mb, nb, BOTTOMLEFT );
      A20 = ABL;
      ABR.Partition1x2( A21, A22,     nb,       LEFT );
      break;
    }
    case BOTTOMLEFT:
    {
      ATL.Partition1x2( A00, A01,     nb,    RIGHT );
      A02 = ATR;
      ABL.Partition2x2( A10, A11,
                        A20, A21, mb, nb, TOPRIGHT );
      ABR.Partition2x1( A12,
                        A22,      mb,     TOP      );
      break;
    }
    case BOTTOMRIGHT:
    {
      A00 = ATL;
      ATR.Partition1x2( A01, A02,     nb,    LEFT );
      ABL.Partition2x1( A10,
                        A20,      mb,     TOP     );
      ABR.Partition2x2( A11, A12,
                        A21, A22, mb, nb, TOPLEFT );
      break;
    }
    default:
    {
      printf( "invalid quadrant\n" );
      break;
    }
  }
}; /** end Repartition2x2To3x3() */ 



/**
 *  @brief
 */ 
template<typename T>
void ContinueWith3x3To2x2
(
  View<T> &ATL, View<T> &ATR, View<T> &A00, View<T> &A01, View<T> &A02,
                              View<T> &A10, View<T> &A11, View<T> &A12,
  View<T> &ABL, View<T> &ABR, View<T> &A20, View<T> &A21, View<T> &A22,
  QuadrantType quadrant 
)
{
  switch ( quadrant )
  {
    case TOPLEFT:
    {
      ATL.ContinueWith2x2( A00, A01,
                           A10, A11 );
      ATR.ContinueWith2x1( A02,
                           A12      );
      ABL.ContinueWith1x2( A20, A21 );
      ABR = A22;
      break;
    }
    case TOPRIGHT:
    {
      ATL.ContinueWith2x1( A00,
                           A10      );
      ATR.ContinueWith2x2( A01, A02,
                           A11, A12 );
      ABL = A20;
      ABR.ContinueWith1x2( A21, A22 );
      break;
    }
    case BOTTOMLEFT:
    {
      ATL.ContinueWith1x2( A00, A01 );
      ATR = A02;
      ABL.ContinueWith2x2( A10, A11,
                           A20, A21 );
      ABR.ContinueWith2x1( A12,
                           A22      );
      break;
    }
    case BOTTOMRIGHT:
    {
      ATL = A00;
      ATR.ContinueWith1x2( A01, A02 );
      ABL.ContinueWith2x1( A10,
                           A20      );
      ABR.ContinueWith2x2( A11, A12,
                           A21, A22 );
      break;
    }
    default:
    {
      printf( "invalid quadrant\n" );
      break;
    }
  }
}; /** end ContinueWith3x3To2x2() */



}; /** end namespace hmlp */

#endif /** define VIEW_HPP */
