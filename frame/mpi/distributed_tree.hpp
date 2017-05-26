#ifndef DISTRIBUTED_TREE_HPP
#define DISTRIBUTED_TREE_HPP

#include <mpi.h>

#include <tree.hpp>

namespace hmlp
{
namespace distributed_tree
{


template<typename SETUP, int N_CHILDREN, typename NODEDATA, typename T>
class Node : public hmlp::tree::Node<SETUP, N_CHILDREN, NODEDATA, T>
{
  public:

    /** inherit all parameters from hmlp::tree::Node */


    Node
    (
      SETUP *user_setup,
      size_t n, size_t l, 
      Node *parent 
    ) : hmlp::tree::Node( setup, )
    {
    };

    void Split()
    {

      child = new Node( setup,  );
    };

  private:

    Node *child = NULL;

    /** initialize with all processes */
    MPI_Comm comm = MPI_COMM_WORLD;

    /** subcommunicator size */
    int size = 0;

    /** subcommunicator rank */
    int rank = 0;

}; /** end class Node */



/**
 *  @brief This distributed tree inherits the shared memory tree
 *         with some additional MPI data structure and function call.
 */ 
template<class SETUP, class NODE, int N_CHILDREN, typename T>
class Tree : public hmlp::tree::Tree<SETUP, NODE, N_CHILDREN, T>
{
  public:

    /** inherit parameters n, m, and depth;
     *  local treelists and treequeue.
     **/

    /** distribued tree (a list of tree nodes) */
    std::vector<NODE*> mpitreelists;

    /** mpi size, rank and error message */
    int worldsize = 0;
    int worldrank = 0;
    int ierr = 0;

    /** inherit constructor */
    Tree() : hmlp::tree::Tree::Tree() 
    {
      ierr = MPI_Comm_size( worldcomm, &worldsize );
      ierr = MPI_Comm_rank( worldcomm, &worldrank );

    };

    /** inherit deconstructor */
    ~Tree() : hmlp::tree::Tree::~Tree() {};

    /** */
    void Allocate( NODE *node )
    {
      int mysize, myrank, mycolor;

      ierr = MPI_Comm_size( node->comm, &mysize );
      ierr = MPI_Comm_rank( node->comm, &myrank );

      assert( mysize % 2 == 0 );

      if ( mysize > 1 )
      {

        node->child = new NODE();
       
        mycolor = ( myrank < size / 2 ) ? 0 : 1;
        ierr = MPI_Comm_split( node->comm, my_color, myrank, &(node->child->comm) );

      }
    };


    void TreePartition( std::vector<std::size_t> &gids ) 
    {
      /** decide the number of distributed tree level according to mpi size */
      int sub_comm_size = size;

      mpitreelists.insert( new NODE() );
      while ( sub_comm_size > 1 )
      {
        /** create the child */

        
      }


      /** local tree */
    };

    template<bool SORTED, typename KNNTASK>
    hmlp::Data<std::pair<T, std::size_t>> AllNearestNeighbor
    (
      std::size_t n_tree,
      std::size_t k, std::size_t max_depth,
      std::vector<std::size_t> &gids,
      std::vector<std::size_t> &lids,
      std::pair<T, std::size_t> initNN,
      KNNTASK &dummy
    )
    {
    };
    
  private:

    /** global communicator */
    MPI_Comm worldcomm = MPI_COMM_WORLD;

}; /** end class Tree */


}; /** end namespace distributed_tree */
}; /** end namespace hmlp */

#endif /** define DISTRIBUTED_TREE_HPP */
