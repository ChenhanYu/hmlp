#ifndef VIEW_HPP
#define VIEW_HPP

#include <data.hpp>

namespace hmlp
{

template<typename T>
class View : public ReadWrite
{
  public:

    View() {};

    /** constructor for the buffer */
    View( size_t m, size_t n, hmlp::Data<T> *buff )
    {
      this->m = m;
      this->n = n;
      this->buff = buff;

      assert( m == buff->row() );
      assert( n == buff->col() );
    };

    void Set( size_t m, size_t n, size_t mb, size_t nb, size_t offm, size_t offn, hmlp::View<T> *base )
    {
      this->m = m;
      this->n = n;
      this->mb = mb;
      this->nb = nb;
      this->offm = offm;
      this->offn = offn;
      this->base = base;
      this->buff = base->buff;
    };

    void Partition2x1
    (
      hmlp::View<T> &A1,
      hmlp::View<T> &A2,
      size_t mb 
    )
    {
      if ( mb > m ) mb = m;


    };


    /** return the row size of the current view */
    size_t row() { return m };

    /** return the col size of the current view */
    size_t col() { return n };

    /** return leading dimension of the buffer */
    size_t ld()  { return buff->row() };

    /** return the pointer of the current view in the buffer */
    T *data()
    {
      assert( buff );
      return ( buff->data() + offn * buff->row() + offm );
    };

  private:

    size_t m = 0;

    size_t n = 0;

    size_t mb = 0;

    size_t nb = 0;

    size_t offm = 0;

    size_t offn = 0;

    hmlp::View<T> *base = NULL;

    hmlp::Data<T> *buff = NULL;

}; /** end class View */

}; /** end namespace hmlp */

#endif /** define VIEW_HPP */
