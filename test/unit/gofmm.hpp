#ifndef HMLP_TEST_GOFMM_HPP
#define HMLP_TEST_GOFMM_HPP

/* Public headers. */
#include <hmlp.h>
/* Internal headers. */
#include <gofmm.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>

namespace hmlp
{
namespace test
{

void all_nearest_neighbor()
{
  /** Use float as data type. */
  using T = float;
  /** [Required] Problem size. */
  size_t n = 5000;
  /** Maximum leaf node size (not used in neighbor search). */
  size_t m = 128;
  /** [Required] Number of nearest neighbors. */
  size_t k = 64;
  /** Maximum off-diagonal rank (not used in neighbor search). */
  size_t s = 128;
  /** Approximation tolerance (not used in neighbor search). */
  T stol = 1E-5;
  /** The amount of direct evaluation (not used in neighbor search). */
  T budget = 0.01;

  /** [Step#0] HMLP API call to initialize the runtime. */
  //HANDLE_ERROR( hmlp_init( &argc, &argv ) );
  HANDLE_ERROR( hmlp_init() );
  /** [Step#1] Create a configuration for generic SPD matrices. */
  gofmm::Configuration<T> config1( ANGLE_DISTANCE, n, m, k, s, stol, budget );
  /** [Step#2] Create a dense random SPD matrix. */
  SPDMatrix<T> K1( n, n );
  K1.randspd( 0.0, 0.01 );
  /** SPDMatrix<T> wraps hmlp::Data<T> which inherits std::vector<T>. */
  cout << "Number of rows: " << K1.row() << " number of columns: " << K1.col() << endl;
  cout << "K(0,0) " << K1( 0, 0 ) << " K(1,2) " << K1( 1, 2 ) << endl;
  /** Acquire the raw pointer of K1. */
  T* K1_ptr = K1.data();
  /** [Step#3] Create a randomized splitter. */
  gofmm::randomsplit<SPDMatrix<T>, 2, T> rkdtsplitter1( K1 );
  /** [Step#4] Perform the iterative neighbor search. */
  auto neighbors1 = gofmm::FindNeighbors( K1, rkdtsplitter1, config1 );
  /** Neighbors are stored as Data<pair<T,size_t>> in k-by-n. */
  cout << "Number of neighboprs: " << neighbors1.row() << " number of queries: " << neighbors1.col() << endl;
  /** Access entries using operator () with 2 indices. */
  neighbors1.Print();
  for ( int i = 0; i < std::min( k, (size_t)10 ); i ++ )
    printf( "[%E,%5lu]\n", neighbors1( i, 0 ).first, neighbors1( i, 0 ).second );
  /** Access entries using operator [] with 1 index (inherited from std::vector<T>). */
  for ( int i = 0; i < std::min( k, (size_t)10 ); i ++ )
    printf( "[%E,%5lu]\n", neighbors1[ i ].first, neighbors1[ i ].second );

  /** [Step#1] Create a configuration for kernel matrices. */
  gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget );
  /** [Step#2] Create a Gaussian kernel matrix with random 6D data. */
  size_t d = 6;
  Data<T> X( d, n ); X.randn();
  KernelMatrix<T> K2( X );
  cout << "Number of rows: " << K2.row() << " number of columns: " << K2.col() << endl;
  cout << "K(0,0) " << K2( 0, 0 ) << " K(1,2) " << K2( 1, 2 ) << endl;
  /** [Step#3] Create a randomized splitter. */
  gofmm::randomsplit<KernelMatrix<T>, 2, T> rkdtsplitter2( K2 );
  /** [Step#4] Perform the iterative neighbor search. */
  auto neighbors2 = gofmm::FindNeighbors( K2, rkdtsplitter2, config2 );
  cout << "Number of neighboprs: " << neighbors2.row() << " number of queries: " << neighbors2.col() << endl;
  for ( int i = 0; i < std::min( k, (size_t)10 ); i ++ )
    printf( "[%E,%5lu]\n", neighbors2( i, 0 ).first, neighbors2( i, 0 ).second );

  /** [Step#5] HMLP API call to terminate the runtime. */
  HANDLE_ERROR( hmlp_finalize() );
};

void fast_matvec_solver()
{
  /** Use float as data type. */
  using T = float;
  /** [Required] Problem size. */
  size_t n = 5000;
  /** Maximum leaf node size (not used in neighbor search). */
  size_t m = 128;
  /** [Required] Number of nearest neighbors. */
  size_t k = 64;
  /** Maximum off-diagonal rank (not used in neighbor search). */
  size_t s = 128;
  /** Approximation tolerance (not used in neighbor search). */
  T stol = 1E-5;
  /** The amount of direct evaluation (not used in neighbor search). */
  T budget = 0.01;
  /** Whether are not to stop compressing when failing to achieve target accuracy. */
  bool secure_accuracy = false;
  /** Number of right-hand sides. */
  size_t nrhs = 10;
  /** Regularization for the system (K+lambda*I). */
  T lambda = 1.0;

  /** [Step#0] HMLP API call to initialize the runtime. */
  //HANDLE_ERROR( hmlp_init( &argc, &argv ) );
  HANDLE_ERROR( hmlp_init() );

  /** [Step#1] Create a configuration for generic SPD matrices. */
  gofmm::Configuration<T> config1( ANGLE_DISTANCE, n, m, k, s, stol, budget, secure_accuracy );
  /** [Step#2] Create a dense random SPD matrix. */
  SPDMatrix<T> K1( n, n ); 
  K1.randspd( 0.0, 1.0 );
  /** [Step#3] Create randomized and center splitters. */
  gofmm::randomsplit<SPDMatrix<T>, 2, T> rkdtsplitter1( K1 );
  gofmm::centersplit<SPDMatrix<T>, 2, T> splitter1( K1 );
  /** [Step#4] Perform the iterative neighbor search. */
  auto neighbors1 = gofmm::FindNeighbors( K1, rkdtsplitter1, config1 );
  /** [Step#5] Compress the matrix with an algebraic FMM. */
  auto* tree_ptr1 = gofmm::Compress( K1, neighbors1, splitter1, rkdtsplitter1, config1 );
  auto& tree1 = *tree_ptr1;
  /** [Step#6] Compute an approximate MATVEC. */
  Data<T> w1( n, nrhs ); w1.randn();
  auto u1 = gofmm::Evaluate( tree1, w1 );
  if ( secure_accuracy )
  {
    /** [Step#7] Factorization (HSS using ULV). */
    gofmm::Factorize( tree1, lambda ); 
    /** [Step#8] Solve (K+lambda*I)w = u approximately with HSS. */
    auto x1 = u1;
    gofmm::Solve( tree1, x1 ); 
  }

  /** [Step#1] Create a configuration for kernel matrices. */
  gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget, secure_accuracy );
  /** [Step#2] Create a Gaussian kernel matrix with random 6D data. */
  size_t d = 6;
  Data<T> X( d, n ); X.randn();
  KernelMatrix<T> K2( X );
  /** [Step#3] Create randomized and center splitters. */
  gofmm::randomsplit<KernelMatrix<T>, 2, T> rkdtsplitter2( K2 );
  gofmm::centersplit<KernelMatrix<T>, 2, T> splitter2( K2 );
  /** [Step#4]Perform the iterative neighbor search. */
  auto neighbors2 = gofmm::FindNeighbors( K2, rkdtsplitter2, config2 );
  /** [Step#5] Compress the matrix with an algebraic FMM. */
  auto* tree_ptr2 = gofmm::Compress( K2, neighbors2, splitter2, rkdtsplitter2, config2 );
  auto& tree2 = *tree_ptr2;
  /** [Step#6] Compute an approximate MATVEC. */
  Data<T> w2( n, nrhs ); w2.randn();
  auto u2 = gofmm::Evaluate( tree2, w2 );
  if ( secure_accuracy )
  {
    /** [Step#7] Factorization (HSS using ULV). */
    gofmm::Factorize( tree2, lambda ); 
    /** [Step#8] Solve (K+lambda*I)w = u approximately with HSS. */
    auto x2 = u2;
    gofmm::Solve( tree2, x2 ); 
  }

  /** [Step#9] HMLP API call to terminate the runtime. */
  HANDLE_ERROR( hmlp_finalize() );
};

//void custom_kernel()
//{
//  /** Use float as data type. */
//  using T = float;
//  /** [Required] Problem size. */
//  size_t n = 5000;
//  /** Maximum leaf node size (not used in neighbor search). */
//  size_t m = 128;
//  /** [Required] Number of nearest neighbors. */
//  size_t k = 64;
//  /** Maximum off-diagonal rank (not used in neighbor search). */
//  size_t s = 128;
//  /** Approximation tolerance (not used in neighbor search). */
//  T stol = 1E-5;
//  /** The amount of direct evaluation (not used in neighbor search). */
//  T budget = 0.01;
//  /** Whether are not to stop compressing when failing to achieve target accuracy. */
//  bool secure_accuracy = false;
//  /** Number of right-hand sides. */
//  size_t nrhs = 10;
//  /** Regularization for the system (K+lambda*I). */
//  T lambda = 1.0;
//
//  /** HMLP API call to initialize the runtime. */
//  HANDLE_ERROR( hmlp_init() );
//
//  /** [Step#1] Create a configuration for kernel matrices. */
//  gofmm::Configuration<T> config2( GEOMETRY_DISTANCE, n, m, k, s, stol, budget, secure_accuracy );
//  /** [Step#2] Create a costomized kernel matrix with random 6D data. */
//  size_t d = 6;
//  Data<T> X( d, n ); X.randn();
//  kernel_s<T, T> kernel;
//  kernel.type = USER_DEFINE;
//  kernel.user_element_function = element_relu<T, T>;
//  kernel.user_matrix_function = matrix_relu<T, T>;
//  KernelMatrix<T> K2( n, n, d, kernel, X );
//  //KernelMatrix<T> K2( X );
//  /** [Step#3] Create randomized and center splitters. */
//  gofmm::randomsplit<KernelMatrix<T>, 2, T> rkdtsplitter2( K2 );
//  gofmm::centersplit<KernelMatrix<T>, 2, T> splitter2( K2 );
//  /** [Step#4]Perform the iterative neighbor search. */
//  auto neighbors2 = gofmm::FindNeighbors( K2, rkdtsplitter2, config2 );
//  /** [Step#5] Compress the matrix with an algebraic FMM. */
//  auto* tree_ptr2 = gofmm::Compress( K2, neighbors2, splitter2, rkdtsplitter2, config2 );
//  auto& tree2 = *tree_ptr2;
//  /** [Step#6] Compute an approximate MATVEC. */
//  Data<T> w2( n, nrhs ); w2.randn();
//  auto u2 = gofmm::Evaluate( tree2, w2 );
//  /** [Step#7] Factorization (HSS using ULV). */
//  gofmm::Factorize( tree2, lambda );
//  if ( secure_accuracy )
//  {
//    /** [Step#8] Solve (K+lambda*I)w = u approximately with HSS. */
//    auto x2 = u2;
//    gofmm::Solve( tree2, x2 ); 
//    /** HMLP API call to terminate the runtime. */
//    HANDLE_ERROR( hmlp_finalize() );
//  }
//};

}; /* end namespace test */
}; /* end namespace hmlp */

TEST(gofmm, all_nearest_neighbor)
{
  try
  {
    hmlp::test::all_nearest_neighbor();
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
  }
}

TEST(gofmm, fast_matvec_solver)
{
  try
  {
    hmlp::test::fast_matvec_solver();
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
  }
}

/* Put all tests involving MPI here. */
#ifdef HMLP_USE_MPI
#endif /* ifdef HMLP_USE_MPI */

#endif /* define HMLP_TEST_GOFMM_HPP */
