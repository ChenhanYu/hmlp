#ifndef DISTSPDMATRIX_HPP
#define DISTSPDMATRIX_HPP

#include <mpi/DistData.hpp>
#include <containers/DistVirtualMatrix.hpp>


namespace hmlp
{


template<typename T>
class DistSPDMatrix : public DistData<STAR, CBLK, T>, DistVirtualMatrix<T>
{
  public:


	  DistSPDMatrix( size_t m, size_t n, mpi::Comm comm ) 
			: DistData<STAR, CBLK, T>( m, n, comm ), DistVirtualMatrix<T>( m, n, comm )
		{
		};



    /** ESSENTIAL: this is an abstract function  */
    virtual T operator()( size_t i, size_t j )
		{
      /** MPI */
      int size = this->Comm_size();
      int rank = this->Comm_rank();

			if ( j % size == rank )
			{
				return DistData<STAR, CBLK, T>::operator () ( i, j );
			}
			else
			{
				std::vector<size_t> I( 1, i );
				std::vector<size_t> J( 1, j );
				auto KIJ = DistVirtualMatrix<T>::RequestKIJ( I, J, j % rank );
				return KIJ[ 0 ];
			}
		};


    /** ESSENTIAL: return a submatrix */
    virtual hmlp::Data<T> operator()
    ( 
      std::vector<size_t> &I, std::vector<size_t> &J 
    )
		{
			hmlp::Data<T> KIJ( I.size(), J.size() );

			return KIJ;
		};


  private:


}; /** end class DistSPDMatrix */



}; /** end namespace hmlp */

#endif /** define DISTSPDMATRIX_HPP */
