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
#include <fstream>
#include <omp.h>
#include <math.h>
#include <hmlp.h>
#include <hmlp_util.hpp>

#include <primitives/regression.hpp>

#ifdef HMLP_MIC_AVX512
#include <hbwmalloc.h>
#endif

#define NUM_POINTS 10240
#define TOLERANCE 1E-13
#define GFLOPS 1073741824

using namespace hmlp;


template<typename T>
void test_regression( int d, int n, T h, int niter,
    std::string &user_points_filename, 
    std::string &user_labels_filename )
{
  /** data points */
  hmlp::Data<T> X( d, n ); 
 
  T beg, spectral_t, gofmm_kkmeans_t, gofmm_spectral_t;

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
  hmlp::Data<double> Y( 1, n );

  if ( user_labels_filename.size() )
  {
    Y.read( 1, n, user_labels_filename );
  }
  else
  {
  }

  /** cluster class */
  Regression<T> regression( d, n, &X, &Y );

  kernel_s<T> kernel;
  kernel.type = KS_GAUSSIAN;
  /** h = 1.0, Gaussian bandwidth (scal = -1 / 2h^2) */
  kernel.scal = -1.0 / ( 2.0 * h * h );

  //auto W = regression.Solve( kernel, niter );
  auto W = regression.SoftMax( kernel, 9, niter );

}; /** end test_regression() */


int main( int argc, char *argv[] )
{
  using T = double;

  int d = 0, n = 0, niter = 0;
  float h = 1.0;

  /** input file names */
  std::string user_points_filename;
  std::string user_labels_filename;

  sscanf( argv[ 1 ], "%d", &d );
  sscanf( argv[ 2 ], "%d", &n );
  sscanf( argv[ 3 ], "%f", &h );
  sscanf( argv[ 4 ], "%d", &niter );
  
  user_points_filename = argv[ 5 ];
  user_labels_filename = argv[ 6 ];

  printf( "%d %d %s %s\n",
      d, n, user_points_filename.data(), user_labels_filename.data() );

  hmlp_init();

  /** */
  test_regression<T>( 
      d, n, (T)h, niter,
      user_points_filename,
      user_labels_filename );

  hmlp_finalize();

  return 0;
};
