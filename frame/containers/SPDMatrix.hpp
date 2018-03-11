
#ifndef SPDMATRIX_HPP
#define SPDMATRIX_HPP

#include<containers/data.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{

template<typename T>
class SPDMatrixMPISupport
{
  public:

    virtual void BackGroundProcess( bool *do_terminate ) {};

    virtual void SendColumns( vector<size_t> cids, int dest, mpi::Comm comm ) {};

    virtual void RecvColumns( int root, mpi::Comm comm, mpi::Status *status ) {};

    virtual void BcastColumns( vector<size_t> cids, int root, mpi::Comm comm ) {};

    virtual void RequestColumns( vector<vector<size_t>> requ_cids ) {};

    virtual void Redistribute( bool enforce_ordered, vector<size_t> &cids ) {};

    virtual void RedistributeWithPartner( vector<size_t> &gids, 
        vector<size_t> &lhs, vector<size_t> &rhs, mpi::Comm comm ) {};

}; /** end class SPDMatrixMPISupport */


/**
 *  @brief This class does not need to inherit hmlp::Data<T>, 
 *         but it should
 *         support two interfaces for data fetching.
 */ 
template<typename T>
class SPDMatrix : public Data<T>, SPDMatrixMPISupport<T>
{
  public:

    /** Need additional support for diagonal evaluation */
    Data<T> Diagonal( const vector<size_t> &I )
    {
      Data<T> DII( I.size(), 1 );
      for ( size_t i = 0; i < I.size(); i ++ )
        DII[ i ] = (*this)( I[ i ], I[ i ] );
      return DII;
    };

    /** Need additional support for pairwise distances. */
    Data<T> PairwiseDistances( const vector<size_t> &I, 
                               const vector<size_t> &J )
    {
      return (*this)( I, J )
    };

}; /** end class SPDMatrix */




template<typename T>
class OOCSPDMatrix : public OOCData<T>, public SPDMatrixMPISupport<T>
{
  public:

    OOCSPDMatrix( size_t m, size_t n, string filename )
    :
    OOCData<T>( m, n, filename ), SPDMatrixMPISupport<T>()
    {
    };

    /** Need additional support for diagonal evaluation */
    Data<T> Diagonal( const vector<size_t> &I )
    {
      Data<T> DII( I.size(), 1 );
      for ( size_t i = 0; i < I.size(); i ++ )
        DII[ i ] = (*this)( I[ i ], I[ i ] );
      return DII;
    };

    /** Need additional support for pairwise distances. */
    Data<T> PairwiseDistances( const vector<size_t> &I, 
                               const vector<size_t> &J )
    {
      return (*this)( I, J );
    };

}; /** end class OOCSPDMatrix */

}; /** end namespace hmlp */


#endif /** define SPDMATRIX_HPP */
