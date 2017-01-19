#ifndef TREE_HPP
#define TREE_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <vector>
#include <deque>
#include <iostream>
#include <hmlp.h>

#include <mkl.h>

#include <hmlp_runtime.hpp>
#include <data.hpp>

//#define DEBUG_TREE 1


namespace hmlp
{
namespace tree
{


/**
 *  @brief Compute the mean values.
 *
 */ 
template<typename T>
std::vector<T> Mean( int d, int n, std::vector<T> &X, std::vector<std::size_t> &lids )
{
  assert( lids.size() == n );
  int n_split = omp_get_max_threads();
  std::vector<T> mean( d, 0.0 );
  std::vector<T> temp( d * n_split, 0.0 );

  //printf( "n_split %d\n", n_split );

  #pragma omp parallel for num_threads( n_split )
  for ( int j = 0; j < n_split; j ++ )
  {
    for ( int i = j; i < n; i += n_split )
    {
      for ( int p = 0; p < d; p ++ )
      {
        temp[ j * d + p ] += X[ lids[ i ] * d + p ];
      }
    }
  }

  for ( int j = 0; j < n_split; j ++ )
  {
    for ( int p = 0; p < d; p ++ )
    {
      mean[ p ] += temp[ j * d + p ];
    }
  }

  for ( int p = 0; p < d; p ++ )
  {
    mean[ p ] /= n;
    //printf( "%5.2lf ", mean[ p ] );
  }
  //printf( "\n" );


  return mean;
}; // end Mean()


/**
 *  
 */ 
template<typename T>
std::vector<T> Mean( int d, int n, std::vector<T> &X )
{
  std::vector<std::size_t> lids( n );
  for ( int i = 0; i < n; i ++ ) lids[ i ] = i;
  return Mean( d, n, X, lids );
};


/**
 *  @brief
 *
 */ 
template<typename T>
T Select( int n, int k, std::vector<T> &x )
{
  assert( k <= n );
  std::vector<T> mean = Mean( 1, n, x );
  std::vector<T> lhs, rhs;
  lhs.reserve( n );
  rhs.reserve( n );

  for ( int i = 0; i < n; i ++ )
  {
    if ( x[ i ] > mean[ 0 ] ) rhs.push_back( x[ i ] );
    else                      lhs.push_back( x[ i ] );
  }

#ifdef DEBUG_TREE
  printf( "n %d k %d lhs %d rhs %d mean %lf\n", 
      n, k, (int)lhs.size(), (int)rhs.size(), mean[ 0 ] );
#endif


  // TODO: Here lhs.size() == k seems to be buggy.
  if ( lhs.size() == n || lhs.size() == k || n == 1 ) 
  {
    //printf( "lrh[ %d - 1 ] %lf n %d\n", k, lhs[ k - 1 ], n );
    //return lhs[ k - 1 ];
    return mean[ 0 ];
  }
  else if ( lhs.size() > k )
  {
    return Select( lhs.size(), k, lhs );
  }
  else
  {
    return Select( rhs.size(), k - lhs.size(), rhs );
  }
};


template<int N_SPLIT, typename T>
struct centersplit
{
  // closure
  hmlp::Data<T> *Coordinate;

  inline std::vector<std::vector<std::size_t> > operator()
  ( 
    std::vector<std::size_t>& gids,
    std::vector<std::size_t>& lids
  ) const 
  {
    assert( N_SPLIT == 2 );

    hmlp::Data<T> &X = *Coordinate;
    size_t d = X.dim();
    size_t n = lids.size();

    T rcx0 = 0.0, rx01 = 0.0;
    std::size_t x0, x1;
    std::vector<std::vector<std::size_t> > split( N_SPLIT );


    std::vector<T> centroid = Mean( d, n, X, lids );
    std::vector<T> direction( d );
    std::vector<T> projection( n, 0.0 );

    //printf( "After Mean\n" );

    // Compute the farest x0 point from the centroid
    for ( int i = 0; i < n; i ++ )
    {
      T rcx = 0.0;
      for ( int p = 0; p < d; p ++ )
      {
        T tmp = X[ lids[ i ] * d + p ] - centroid[ p ];
        rcx += tmp * tmp;
        //printf( "%5.2lf ", X[ lids[ i ] * d + p  ] );
      }
      //printf( "\n" );
      //printf( "rcx %lf rcx0 %lf lids %d\n", rcx, rcx0, (int)lids[ i ] );
      if ( rcx > rcx0 ) 
      {
        rcx0 = rcx;
        x0 = i;
      }
    }

    //printf( "After Farest\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //  printf( "%5.2lf ", X[ lids[ x0 ] * d + p ] );
    //}
    //printf( "\n" );

    // Compute the farest point x1 from x0
    for ( int i = 0; i < n; i ++ )
    {
      T rxx = 0.0;
      for ( int p = 0; p < d; p ++ )
      {
        T tmp = X[ lids[ i ] * d + p ] - X[ lids[ x0 ] * d + p ];
        rxx += tmp * tmp;
      }
      if ( rxx > rx01 )
      {
        rx01 = rxx;
        x1 = i;
      }
    }

    //printf( "After Nearest\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //  printf( "%5.2lf ", X[ lids[ x1 ] * d + p ] );
    //}
    //printf( "\n" );


    // Compute direction
    for ( int p = 0; p < d; p ++ )
    {
      direction[ p ] = X[ lids[ x1 ] * d + p ] - X[ lids[ x0 ] * d + p ];
    }

    //printf( "After Direction\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //  printf( "%5.2lf ", direction[ p ] );
    //}
    //printf( "\n" );
    //exit( 1 );



    // Compute projection
    projection.resize( n, 0.0 );
    for ( int i = 0; i < n; i ++ )
      for ( int p = 0; p < d; p ++ )
        projection[ i ] += X[ lids[ i ] * d + p ] * direction[ p ];

    //printf( "After Projetion\n" );
    //for ( int p = 0; p < d; p ++ )
    //{
    //  printf( "%5.2lf ", projec[ p ] );
    //}
    //printf( "\n" );



    // Parallel median search
    T median = Select( n, n / 2, projection );

    //printf( "After Select\n" );


    for ( int i = 0; i < n; i ++ )
    {
      if ( projection[ i ] > median ) split[ 1 ].push_back( i );
      else                            split[ 0 ].push_back( i );
    }


    //printf( "split median %lf left %d right %d\n", 
    //    median,
    //    (int)split[ 0 ].size(), (int)split[ 1 ].size() );

    //if ( split[ 0 ].size() > 0.6 * n ||
    //     split[ 1 ].size() > 0.6 * n )
    //{
    //  for ( int i = 0; i < n; i ++ )
    //  {
    //    printf( "%E ", projection[ i ] );
    //  } 
    //  printf( "\n" );
    //}


    return split; 
  };
};


/**
 *  @brief 
 */ 
template<typename SETUP, //typename SPLITTER, 
  int N_CHILDREN, typename NODEDATA, typename T>
class Node
{
  public:

    Node
    (
      SETUP *user_setup,
      size_t n, size_t l, 
      Node *parent 
    )
    {
      this->setup = user_setup;
      this->n = n;
      this->l = l;
      this->treelist_id = -1;
      this->gids.resize( n );
      this->lids.resize( n );
      this->isleaf = false;
      this->parent = parent;
      this->lchild = NULL;
      this->rchild = NULL;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };

    Node
    ( 
      SETUP *user_setup,
      int n, int l, 
      std::vector<std::size_t> gids,
      std::vector<std::size_t> lids,
      Node *parent 
    )
    {
      this->setup = user_setup;
      this->n = n;
      this->l = l;
      this->treelist_id = -1;
      this->gids = gids;
      this->lids = lids;
      this->isleaf = false;
      this->parent = parent;
      this->lchild = NULL;
      this->rchild = NULL;
      for ( int i = 0; i < N_CHILDREN; i++ ) kids[ i ] = NULL;
    };



    ~Node() {};

    void Split( int m, int max_depth )
    {
      if ( n > m && l < max_depth )
      {
        auto split = setup->splitter( gids, lids );

        // TODO: Can be parallelized
        for ( int i = 0; i < N_CHILDREN; i ++ )
        {
          int nchild = split[ i ].size();
         
          //kids[ i ] = new Node( setup, nchild, l + 1, X, this );
          kids[ i ] = new Node( setup, nchild, l + 1, this );

          // TODO: Can be parallelized
          for ( int j = 0; j < nchild; j ++ )
          {
            kids[ i ]->gids[ j ] = gids[ split[ i ][ j ] ];
            kids[ i ]->lids[ j ] = lids[ split[ i ][ j ] ];
          }
        }

        // Facilitate binary tree
        if ( N_CHILDREN > 1  )
        {
          lchild = kids[ 0 ];
          rchild = kids[ 1 ];
        }
      }
      else
      {
        isleaf = true;
      }
    };


    // This is the call back pointer to the shared data.
    SETUP *setup;

    // Per node private data.
    NODEDATA data;

    // number of points in this node.
    size_t n;

    // level in the tree
    size_t l;

    int treelist_id; // In top-down topology order.

    std::vector<std::size_t> gids;

    std::vector<std::size_t> lids;

    Node *parent;

    Node *kids[ N_CHILDREN ];

    Node *lchild; // make it easy

    Node *rchild;

    bool isleaf;

    //SPLITTER splitter;

  private:

};


/**
 *  @brief Data and setup that are shared with all nodes.
 *
 */ 
template<typename SPLITTER, typename T>
class Setup
{
  public:

    hmlp::Data<T> X;

    SPLITTER splitter;
};


//template<typename SPLITTER, int N_CHILDREN, typename DATA, typename T>
template<class SETUP, class NODE, int N_CHILDREN, typename T>
class Tree
{
  public:

    SETUP setup;

    int n;

    int m;

    int depth;

    std::vector<NODE*> treelist;

    std::deque<NODE*> treequeue;

    Tree() : n( 0 ), m( 0 ), depth( 0 )
    {};

    void TreePartition
    (
      int leafsize, int max_depth,
      std::vector<std::size_t> &gids,
      std::vector<std::size_t> &lids
    )
    {
      assert( N_CHILDREN == 2 );

      n = lids.size();

      m = leafsize;

      std::deque<NODE*> treequeue;
      
      treelist.reserve( ( n / m ) * N_CHILDREN );

      auto *root = new NODE( &setup, n, 0, gids, lids, NULL );
   
      treequeue.push_back( root );
    
      depth = max_depth;

      // TODO: there is parallelism to be exploited here.
      while ( auto *node = treequeue.front() )
      {
        node->treelist_id = treelist.size();
        node->Split( m, max_depth );
        for ( int i = 0; i < N_CHILDREN; i ++ )
        {
          treequeue.push_back( node->kids[ i ] );
        }


        //node->data.fakes = size_t( ( (double) std::rand() / RAND_MAX ) * setup.s );


        treelist.push_back( node );
        treequeue.pop_front();
      }

      // All tree nodes were created. Now decide tree depth.
      if ( treelist.size() )
      {
        treelist.shrink_to_fit();
        depth = treelist.back()->l;
      }
    };


    template<class TASK>
    void PostOrder( NODE *node )
    {
      #pragma omp parallel
      #pragma omp single nowait
      {
        for ( int i = 0; i < N_CHILDREN; i ++ )
        {
          if ( node->kids[ i ] )
          {
            #pragma omp task
            PostOrder<TASK>( node->kids[ i ] );
          }
        }
        #pragma omp taskwait
        {
          auto *task = new TASK();
          task->Set( node );
          task->Execute( NULL );
          delete task;
        }
      }
    };


    template<bool LEVELBYLEVEL, class TASK>
    void TraverseUp()
    {
      assert( N_CHILDREN == 2 );
      //printf( "TraverseUp()\n" );

      std::vector<TASK*> tasklist;
      if ( !LEVELBYLEVEL ) tasklist.resize( treelist.size() );

      for ( int l = depth; l >= 0; l -- )
      {
        int n_nodes = 1 << l;
        auto level_beg = treelist.begin() + n_nodes - 1;

        if ( LEVELBYLEVEL )
        {
          int nthd_glb = omp_get_max_threads();
          
          if ( n_nodes >= nthd_glb || n_nodes == 1 )
          {
            #pragma omp parallel for if ( n_nodes > nthd_glb / 2 )
            for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
            {
              auto *node = *(level_beg + node_ind);
              auto *task = new TASK();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
          }
          else
          {
            int nthd_loc = nthd_glb / n_nodes;

            mkl_set_dynamic( 0 );
            mkl_set_num_threads( nthd_loc );
            omp_set_nested( 1 );
            omp_set_max_active_levels( 2 );
            #pragma omp parallel for if ( n_nodes )
            for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
            {
              omp_set_num_threads( nthd_loc );
              auto *node = *(level_beg + node_ind);
              auto *task = new TASK();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
            mkl_set_dynamic( 1 );
            mkl_set_num_threads( nthd_glb );
            omp_set_num_threads( nthd_glb );
            omp_set_nested( 0 );
            omp_set_max_active_levels( 1 );
          }
        }
        else // using dynamic scheduling
        {
          for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            // Create tasks
            tasklist[ node->treelist_id ] = new TASK();
            auto *task = tasklist[ node->treelist_id ];
            task->Submit();
            task->Set( node );

            // Setup dependencies
            if ( node->kids[ 0 ] )
            {
#ifdef DEBUG_TREE
              printf( "DependencyAdd %d -> %d\n", node->kids[ 0 ]->treelist_id, node->treelist_id );
              printf( "DependencyAdd %d -> %d\n", node->kids[ 1 ]->treelist_id, node->treelist_id );
#endif
              Scheduler::DependencyAdd( tasklist[ node->kids[ 0 ]->treelist_id ], task );
              Scheduler::DependencyAdd( tasklist[ node->kids[ 1 ]->treelist_id ], task );
            }
            else // leafnodes, directly enqueue
            {
              task->Enqueue();
            }
          }
        }
      }
    }; // end TraverseUp()

    template<bool LEVELBYLEVEL, class TASK>
    void TraverseDown()
    {
      std::vector<TASK*> tasklist;
      if ( !LEVELBYLEVEL ) tasklist.resize( treelist.size() );

      for ( int l = 0; l < depth; l ++ )
      {
        if ( LEVELBYLEVEL )
        {
          int n_nodes = 1 << l;
          auto level_beg = treelist.begin() + n_nodes - 1;

          if ( LEVELBYLEVEL )
          {
            #pragma omp parallel for 
            for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
            {
              auto *node = *(level_beg + node_ind);
              auto *task = new TASK();
              task->Set( node );
              task->Execute( NULL );
              delete task;
            }
          }
        }
        else // using dynamic scheduling
        {
          for ( int node_ind = 0; node_ind < n_nodes; node_ind ++ )
          {
            auto *node = *(level_beg + node_ind);
            // Create tasks
            tasklist[ node->treelist_id ] = new TASK();
            auto *task = tasklist[ node->treelist_id ];
            task->Submit();
            task->Set( node );

            if ( node->parent )
            {
#ifdef DEBUG_TREE
              printf( "DependencyAdd %d -> %d\n", node->parent->treelist_id, node->treelist_id );
#endif
              Scheduler::DependencyAdd( tasklist[ node->parent->treelist_id ], task );
            }
            else // root, directly enqueue
            {
              task->Enqueue();
            }
          }
        }
      }
    }; // end TraverseDown()

    void Summary()
    {

    };
};

}; // end namespace tree 




}; // end namespace hmlp

#endif // define TREE_HPP
