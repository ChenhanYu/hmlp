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


#ifndef MPITREE_HPP
#define MPITREE_HPP

#include <hmlp_runtime_mpi.hpp>
#include <containers/tree.hpp>

namespace hmlp
{
namespace mpitree
{


/**
 *
 */ 
template<typename SETUP, int N_CHILDREN, typename NODEDATA, typename T>
class Node : public hmlp::tree::Node<SETUP, N_CHILDREN, NODEDATA, T>
{
  public:

    /** inherit all parameters from hmlp::tree::Node */


    Node( SETUP *setup, size_t n, size_t l, Node *parent, hmlp::mpi::Comm comm ) 
      : hmlp::tree::Node<SETUP, N_CHILDREN, NODEDATA, T>( setup, n, l, parent ) 
    {
      this->comm = comm;
      
      /** get size and rank */
      hmlp::mpi::Comm_size( comm, &size );
      hmlp::mpi::Comm_rank( comm, &rank );
    };

    void SetupChild( class Node *child )
    {
      this->child = child;
    };

    void Split()
    {

      //child = new Node( setup,  );
    };

  private:

    Node *child = NULL;

    /** initialize with all processes */
    hmlp::mpi::Comm comm = HMLP_MPI_COMM_WORLD;

    /** subcommunicator size */
    int size = 0;

    /** subcommunicator rank */
    int rank = 0;

}; /** end class Node */






/**
 *  @brief This distributed tree inherits the shared memory tree
 *         with some additional MPI data structure and function call.
 */ 
template<class SETUP, class NODEDATA, int N_CHILDREN, typename T>
class Tree : public hmlp::tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>
{
  public:

    /** inherit parameters n, m, and depth;
     *  local treelists and treequeue.
     **/

    /** define our tree node type as NODE */
    typedef Node<SETUP, N_CHILDREN, NODEDATA, T> NODE;

    /** distribued tree (a list of tree nodes) */
    std::vector<NODE*> mpitreelists;

    /** inherit constructor */
    Tree() : hmlp::tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>::Tree()
    {
      ierr = hmlp::mpi::Comm_size( comm, &size );
      ierr = hmlp::mpi::Comm_rank( comm, &rank );
    };

    /** inherit deconstructor */
    ~Tree() 
    {
      hmlp::tree::Tree<SETUP, NODEDATA, N_CHILDREN, T>::~Tree();
      for ( size_t i = 0; i < mpitreelists.size(); i ++ )
        delete mpitreelists[ i ];
    };

    /** 
     *  @breif allocate distributed tree node
     *
     *
     * */
    //void Allocate( NODE *node )
    //{
    //  int mysize, myrank, mycolor;

    //  ierr = hmlp::mpi::Comm_size( node->comm, &mysize );
    //  ierr = hmlp::mpi::Comm_rank( node->comm, &myrank );




    //  assert( mysize % 2 == 0 );

    //  if ( mysize > 1 )
    //  {

    //    node->child = new NODE();
    //   
    //    mycolor = ( myrank < size / 2 ) ? 0 : 1;
    //    ierr = MPI_Comm_split( node->comm, my_color, myrank, &(node->child->comm) );

    //  }
    //};


    /**
     *  @brief partition 
     *
     */ 
    void TreePartition( std::vector<std::size_t> &gids ) 
    {
      /** decide the number of distributed tree level according to mpi size */
      int mycomm  = comm;
      int mysize  = size;
      int myrank  = rank;
      int mycolor = 0;
      int mylevel = 0;

      /** root( setup, n = 0, l = 0, parent = NULL ) */
      auto *root = new NODE( &(this->setup), 
          (size_t)0, (size_t)mylevel, (NODE*)NULL, mycomm );

      /** push to the mpi treelist */
      mpitreelists.push_back( root );

      while ( mysize > 1 )
      {
        hmlp::mpi::Comm childcomm;

        /** increase level */
        mylevel += 1;

        /** left color = 0, right color = 1 */
        mycolor = ( myrank < mysize / 2 ) ? 0 : 1;

        /** get the subcommunicator for children */
        ierr = hmlp::mpi::Comm_split( mycomm, mycolor, myrank, &(childcomm) );

        /** update mycomm */
        mycomm = childcomm;

        /** create the child */
        auto *parent = mpitreelists.back();
        auto *child  = new NODE( &(this->setup), 
            (size_t)0, (size_t)mylevel, parent, mycomm );

        /** setup parent's children */
        parent->SetupChild( child );
        
        /** push to the mpi treelist */
        mpitreelists.push_back( child );

        /** update communicator size */
        hmlp::mpi::Comm_size( mycomm, &mysize );

        /** update myrank in the subcommunicator */
        hmlp::mpi::Comm_rank( mycomm, &myrank );
      }


      /** local tree */
    };

    //template<bool SORTED, typename KNNTASK>
    //hmlp::Data<std::pair<T, std::size_t>> AllNearestNeighbor
    //(
    //  std::size_t n_tree,
    //  std::size_t k, std::size_t max_depth,
    //  std::vector<std::size_t> &gids,
    //  std::vector<std::size_t> &lids,
    //  std::pair<T, std::size_t> initNN,
    //  KNNTASK &dummy
    //)
    //{
    //  /** k-by-N */
    //  hmlp::Data<std::pair<T, std::size_t>> NN( k, lids.size(), initNN );



    //  return NN;

    //}; /** end AllNearestNeighbor() */

    
  private:

    /** global communicator */
    hmlp::mpi::Comm comm = HMLP_MPI_COMM_WORLD;

    /** mpi size, rank and error message */
    int size = 0;

    int rank = 0;

    int ierr = 0;


}; /** end class Tree */


}; /** end namespace mpitree */
}; /** end namespace hmlp */

#endif /** define MPITREE_HPP */
