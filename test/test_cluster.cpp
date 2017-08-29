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






#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <hmlp_util.hpp>

#include <primitives/cluster.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define NUM_POINTS 10240
#define TOLERANCE 1E-13
#define GFLOPS 1073741824

using namespace hmlp;


template<typename T>
void test_cluster( int d, int n, int ncluster, int niter, T tol, 
    std::string &user_points_filename, 
    std::string &user_labels_filename, int nclass )
{
  /** data points */
  hmlp::Data<T> X( d, n ); 
  
  if ( user_points_filename.size() )
  {
    /** read data points from files */
    X.read( d, n, user_points_filename );
  }
  else
  {
    /** random initialization */
    X.rand();
  }
 
  /** if true labels provided */
  hmlp::Data<int> Y;

  if ( user_labels_filename.size() )
  {
    assert( nclass );

    /** labels in type T */
    hmlp::Data<T> TY( 1, n, 0.0 );

    /** read data points from files */
    TY.read( 1, n, user_labels_filename );

    /** allocate labels in type int */
    Y.resize( 1, n, 0.0 );

    /** exam the label range */
    int min_label = nclass;
    int max_label = 0;
    for ( size_t i = 0; i < n; i ++ ) 
    {
      Y[ i ] = (int)TY[ i ] - 1;
      if ( Y[ i ] > max_label ) max_label = Y[ i ];
      if ( Y[ i ] < min_label ) min_label = Y[ i ];
    }

    printf( "label range [ %d %d ]\n", min_label, max_label );

    if ( min_label != 0 )
    {
      for ( size_t i = 0; i < n; i ++ ) Y[ i ] = Y[ i ] - min_label;
    }

  }
  else
  {
    nclass = 0;
  }

  /** cluster assignments */
  std::vector<int> assignments( n );

  /** initialize assignment (Round Robin) */
  for ( size_t i = 0; i < n; i ++ ) 
    assignments[ i ] = i % ncluster;

  /** cluster class */
  Cluster<T> cluster( d, n, ncluster, &X, &assignments );

  /** provide true labels to the object */
  if ( nclass ) cluster.ProvideLabels( nclass, &Y );


  /** */
  T h = 0.05;
  kernel_s<T> kernel;
  kernel.type = KS_GAUSSIAN;
  /** h = 1.0, Gaussian bandwidth (scal = -1 / 2h^2) */
  kernel.scal = -1.0 / ( 2.0 * h * h );
  //cluster.KernelKmeans( kernel, niter, tol );
  //cluster.NMI();

  /** try linear Kmeans */
  cluster.Kmeans( niter, tol );

  /** Check confusion */
  cluster.NMI();

  /** try spectral clustering with power methods */
  cluster.Spectral( kernel );



}; /** end test_cluster() */


int main( int argc, char *argv[] )
{
  using T = double;

  int d = 0, n = 0, ncluster = 0, nclass = 0, niter = 0;
  float tol = 0.0;

  /** input file names */
  std::string user_points_filename;
  std::string user_labels_filename;

  sscanf( argv[ 1 ], "%d", &d );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%d", &ncluster );
  sscanf( argv[ 4 ], "%d", &niter );
  sscanf( argv[ 5 ], "%f", &tol );

  if ( argc > 6 )
  {
    /** user provides data points (otherwise random point clouds) */
    user_points_filename = argv[ 6 ];
  }

  if ( argc > 7 )
  {
    /** user provides true labels (NMI available) */
    user_labels_filename = argv[ 7 ];
  sscanf( argv[ 8 ], "%d", &nclass );
  }

  /** */
  test_cluster<T>( 
      d, n, ncluster, niter, (T)tol, 
      user_points_filename,
      user_labels_filename, nclass );

  return 0;
};
