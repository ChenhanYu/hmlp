
#ifndef SPDMATRIX_HPP
#define SPDMATRIX_HPP

#include<Data.hpp>
#include<VirtualMatrix.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{


/**
 *  @brief This class does not need to inherit hmlp::Data<T>, 
 *         but it should
 *         support two interfaces for data fetching.
 */ 
template<typename T>
class SPDMatrix : public VirtualMatrix<T>
{
  public:

    SPDMatrix() : VirtualMatrix<T>() {};

    SPDMatrix( size_t m, size_t n ) : VirtualMatrix<T>( m, n )
    {
      K.resize( m, n );
    };

    void resize( size_t m, size_t n )
    {
      VirtualMatrix<T>::resize( m, n );
      K.resize( m, n );
    };

    template<bool USE_LOWRANK=true>
    void randspd( T a, T b ) { K.randspd<USE_LOWRANK>( a, b ); };

    void read( size_t m, size_t n, string &filename ) 
    { 
      K.read( m, n, filename ); 
    };

    T operator()( size_t i, size_t j ) { return K( i, j ); };

    Data<T> operator() ( const vector<size_t> &I, 
                         const vector<size_t> &J )
    {
      return K( I, J );
    };

    T* data() noexcept { return K.data(); };

    const T* data() const noexcept { return K.data(); };

  private:

    Data<T> K;

}; /** end class SPDMatrix */




template<typename T>
class OOCSPDMatrix : public OOCData<T>, public SPDMatrixMPISupport<T>
{
  public:

    OOCSPDMatrix( size_t m, size_t n, string filename )
    :
    OOCData<T>( m, n, filename ), SPDMatrixMPISupport<T>()
    {
      assert( m == n );
      D.resize( n );
      for ( size_t i = 0; i < n; i ++ ) D[ i ] = (*this)( i, i );
    };

    /** Need additional support for diagonal evaluation */
    Data<T> Diagonal( const vector<size_t> &I )
    {
      Data<T> DII( I.size(), 1 );
      for ( auto i = 0; i < I.size(); i ++ ) DII[ i ] = D[ I[ i ] ];
        //DII[ i ] = (*this)( I[ i ], I[ i ] );
      return DII;
    };


 private:

    vector<T> D;

}; /** end class OOCSPDMatrix */

}; /** end namespace hmlp */


#endif /** define SPDMATRIX_HPP */
