
#ifndef SPDMATRIX_HPP
#define SPDMATRIX_HPP

#include<Data.hpp>
#include<VirtualMatrix.hpp>

//using namespace hmlp;

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

    SPDMatrix( uint64_t height, uint64_t width ) 
      : VirtualMatrix<T>( height, width ) 
    { 
      K_.resize( height, width );
    };

    SPDMatrix( uint64_t height, uint64_t width, const std::string & filename )
      : VirtualMatrix<T>( height, width ) 
    {
      try
      {
        K_.resize( height, width );
        HANDLE_ERROR( K_.readBinaryFile( height, width, filename ) );
      }
      catch ( const std::exception & e )
      {
        HANDLE_EXCEPTION( e );
      }
    };

    void resize( uint64_t height, uint64_t width )
    {
      VirtualMatrix<T>::resize( height, width );
      K_.resize( height, width );
    };

    template<bool USE_LOWRANK=true>
    void randspd( T a, T b ) 
    { 
      K_.randspd( a, b ); 
    };

    hmlpError_t readBinaryFile( uint64_t height, uint64_t width, const std::string & filename ) 
    { 
      return K_.readBinaryFile( height, width, filename ); 
    };

    T operator()( uint64_t i, uint64_t j )
    { 
      return K_( i, j ); 
    };

    Data<T> operator() ( const std::vector<uint64_t> &I, 
                         const std::vector<uint64_t> &J )
    {
      return K_( I, J );
    };

    T* data() noexcept 
    { 
      return K_.data(); 
    };

    const T* data() const noexcept 
    { 
      return K_.data(); 
    };

  protected:

    Data<T> K_;

}; /* end class SPDMatrix */




template<typename T>
class OOCSPDMatrix : public VirtualMatrix<T>
{
  public:

    OOCSPDMatrix( uint64_t height, uint64_t width, const std::string & filename )
      : VirtualMatrix<T>( height, width ) 
    {
      K_.initFromFile( height, width, filename );
    };

    T operator()( uint64_t i, uint64_t j ) 
    { 
      return K_( i, j ); 
    };

    Data<T> operator() ( const std::vector<uint64_t> & I, 
                         const std::vector<uint64_t> & J )
    {
      return K_( I, J );
    };

 protected:

    OOCData<T> K_;

}; /* end class OOCSPDMatrix */

}; /* end namespace hmlp */


#endif /** define SPDMATRIX_HPP */
