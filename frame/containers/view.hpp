#ifndef VIEW_HPP
#define VIEW_HPP

#include <containers/data.hpp>

namespace hmlp
{

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

    /** destructor */
    //~View() { printf( "~View()\n" ); fflush( stdout ); };

    /** base case setup */
    template<bool TRANS = false>
    void Set( hmlp::Data<T> &buff )
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
      A1.Set(     mb, this->n, this->offm,      this->offn, this );
      /** setup A2 */
      A2.Set( m - mb, this->n, this->offm + mb, this->offn, this );
    };


    /** A = [ A1; 
     *        A2; ] */
    void ContinueWith2x1
    (
      hmlp::View<T> &A1,
      hmlp::View<T> &A2
    )
    {
      assert( A1.col() == A2.col() );
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
      A1.Set( this->m,     nb, this->offm, this->offn,      this );
      /** setup A2 */
      A2.Set( this->m, n - nb, this->offm, this->offn + nb, this );
    };

    /** A = [ A1, A2; ] */
    void ContinueWith1x2
    (
      hmlp::View<T> &A1, hmlp::View<T> &A2
    )
    {
      assert( A1.row() == A2.row() );
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
      A11.Set(     mb,     nb, offm     , offn     , this );
      /** setup A12 */
      A12.Set(     mb, n - nb, offm     , offn + nb, this );
      /** setup A21 */
      A21.Set( m - mb,     nb, offm + mb, offn     , this );
      /** setup A22 */
      A21.Set( m - mb, n - nb, offm + mb, offn + nb, this );

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
  A.Partition1x2( A, A1, A2, nb, side );
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
  A.Partition2x1( A, A1, 
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
      Partition1x2( AL, A0, A1, nb, RIGHT );
      A2 = AR;
      break;
    }
    case RIGHT:
    {
      A0 = AL;
      Partition2x1( AL, A0, A1, nb, LEFT );
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
      AT.ContinueWith( A0,
                       A1 );
      AB = A2;
      break;
    }
    case BOTTOM:
    {
      AT = A0;
      AB.ContinueWith( A1,
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
