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




#include <gofmm/gofmm.hpp>

using namespace std;
using namespace hmlp;


namespace hmlp
{
namespace gofmm
{



/** Evaluate routines with different interfaces **/ 
Data<double> Evaluate( dTree_t *tree, Data<double> *weights )
{
  return Evaluate<true, false, true, true, dTree_t, double>( *tree, *weights );
};

Data<float>  Evaluate( sTree_t *tree, Data<float> *weights )
{
  return Evaluate<true, false, true, true, sTree_t, float> ( *tree, *weights );
};


/**
 *  Compress routines with different interfaces
 **/ 
dTree_t *Compress( dSPDMatrix_t *K, double stol, double budget )
{
	return Compress<double>( *K, stol, budget );
};
sTree_t *Compress( sSPDMatrix_t *K,  float stol,  float budget )
{
	return Compress<float >( *K, stol, budget );
};
dTree_t *Compress( dSPDMatrix_t *K, double stol, double budget, size_t m, size_t k, size_t s )
{
	return Compress<double>( *K, stol, budget, m, k, s );
};
sTree_t *Compress( sSPDMatrix_t *K,  float stol,  float budget, size_t m, size_t k, size_t s )
{
	return Compress<float >( *K, stol, budget, m, k, s );
};

/**
 *  ComputeError routines with different interfaces
 **/ 
double ComputeError( dTree_t *tree, size_t gid, Data<double> *potentials )
{
  return ComputeError( *tree, gid, *potentials );
};

float  ComputeError( sTree_t *tree, size_t gid, Data<float > *potentials )
{
  return ComputeError( *tree, gid, *potentials );
};






}; /** end namespace gofmm */
}; /** end namespace hmlp */
