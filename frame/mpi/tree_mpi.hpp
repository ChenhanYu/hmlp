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


#ifndef TREE_MPI_HPP
#define TREE_MPI_HPP

#include <mpi.h>
#include <tree.hpp>

namespace hmlp
{
namespace tree
{
/** Obviously, this is the distributed version with mpi. */
namespace mpi
{


/**
 *
 */ 
template<typename SETUP, int N_CHILDREN, typename NODEDATA, typename T>
class Node : public hmlp::tree::Node<SETUP, N_CHILDREN, NODEDATA, T>
{
  public:

    /** inherit all parameters from hmlp::tree::Node */


    //Node
    //(
    //  SETUP *user_setup,
    //  size_t n, size_t l, 
    //  Node *parent 
    //) : hmlp::tree::Node( user_setup, n, l, parent )
    //{
    //};

    //void Split()
    //{

    //  child = new Node( setup,  );
    //};

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
    Tree() 
    {
      ierr = MPI_Comm_size( worldcomm, &worldsize );
      ierr = MPI_Comm_rank( worldcomm, &worldrank );

    };

    /** inherit deconstructor */
    ~Tree() {};

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
      /** k-by-N */
      hmlp::Data<std::pair<T, std::size_t>> NN( k, lids.size(), initNN );



      return NN;

    }; /** end AllNearestNeighbor() */

    
  private:

    /** global communicator */
    MPI_Comm worldcomm = MPI_COMM_WORLD;

}; /** end class Tree */


}; /** end namespace mpi */
}; /** end namespace tree */
}; /** end namespace hmlp */

#endif /** define TREE_MPI_HPP */
