#include <gofmm/gofmm.hpp>


namespace hmlp
{
namespace gofmm
{



/**
 *  Evaluate routines with different interfaces
 **/ 
hmlp::Data<double> Evaluate( dTree_t *tree, hmlp::Data<double> *weights )
{
  return Evaluate<true, false, true, true, true, dTree_t, double> 
		( *tree, *weights );
};

hmlp::Data<float>  Evaluate( sTree_t *tree, hmlp::Data<float> *weights )
{
  return Evaluate<true, false, true, true, true, sTree_t, float> 
		( *tree, *weights );
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
double ComputeError( dTree_t *tree, size_t gid, hmlp::Data<double> *potentials )
{
  return ComputeError( *tree, gid, *potentials );
};

float  ComputeError( sTree_t *tree, size_t gid, hmlp::Data<float > *potentials )
{
  return ComputeError( *tree, gid, *potentials );
};











}; /** end namespace gofmm */
}; /** end namespace hmlp */
