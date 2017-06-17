#ifndef VIEW_HPP
#define VIEW_HPP

#include <data.hpp>

namespace hmlp
{

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
    void Set( hmlp::Data<T> &buff )
    {
      this->m    = buff.row();
      this->n    = buff.col();
      this->offm = 0;
      this->offn = 0;
      this->base = this;
      this->buff = &buff;
    };

    /** non-base case setup */
    void Set( size_t m, size_t n, size_t offm, size_t offn, hmlp::View<T> *base )
    {
      assert( offm <= base->buff->row() );
      assert( offn <= base->buff->col() );
      this->m    = m;
      this->n    = n;
      this->offm = offm;
      this->offn = offn;
      this->base = base;
      this->buff = base->buff;
    };

    /** subview operator */
    template<typename TINDEX>
    T & operator () ( TINDEX i, TINDEX j )
    {
      assert( offm + i < buff->row() );
      assert( offn + j < buff->col() );
      return *( data() + j * ld() + i );
    };


    /** A = [ A1; 
     *        A2; ] */
    void Partition2x1
    (
      hmlp::View<T> &A1,
      hmlp::View<T> &A2, size_t mb 
    )
    {
      /** readjust mb */
      if ( mb > m ) mb = m;
      /** setup A1 */
      A1.Set(     mb, this->n, this->offm,      this->offn, this );
      /** setup A2 */
      A2.Set( m - mb, this->n, this->offm + mb, this->offn, this );
    };

    /** A = [ A1, A2; ] */
    void Partition1x2
    (
      hmlp::View<T> &A1, hmlp::View<T> &A2, size_t nb 
    )
    {
      /** readjust mb */
      if ( nb > n ) nb = n;
      /** setup A1 */
      A1.Set( this->m,     nb, this->offm, this->offn,      this );
      /** setup A2 */
      A2.Set( this->m, n - nb, this->offm, this->offn + nb, this );
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
      return ( buff->data() + offn * buff->row() + offm );
    };

    /** print out all information */
    void Print()
    {
      printf( "[ %5lu+%5lu:%5lu ][ %5lu+%5lu:%5lu ]\n",
          offm, m, buff->row(), offn, n, buff->col() );
    }; 

  private:

    size_t m = 0;

    size_t n = 0;

    size_t offm = 0;

    size_t offn = 0;

    hmlp::View<T> *base = NULL;

    hmlp::Data<T> *buff = NULL;

}; /** end class View */

}; /** end namespace hmlp */

#endif /** define VIEW_HPP */
