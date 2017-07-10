#include <gofmm/gofmm.hpp>


namespace hmlp
{
namespace gofmm
{



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


dTree_t *Compress( dSPDMatrix_t *K, double stol, double budget )
{
	printf( "In *Compress\n" );
	return Compress<double>( *K, stol, budget );
};

sTree_t *Compress( sSPDMatrix_t *K,  float stol,  float budget )
{
	return Compress<float >( *K, stol, budget );
};

double ComputeError( dTree_t *tree, size_t gid, hmlp::Data<double> *potentials )
{
  return ComputeError( *tree, gid, *potentials );
};

float  ComputeError( sTree_t *tree, size_t gid, hmlp::Data<float > *potentials )
{
  return ComputeError( *tree, gid, *potentials );
};










///**
// *  @brielf A "double" instance of the simple gofmm::Compress()
// */ 
//hmlp::tree::Tree<
//  hmlp::gofmm::Setup<SPDMatrix<double>, 
//    centersplit<SPDMatrix<double>, 2, double, SPLIT_ANGLE>, 
//    double>, 
//  hmlp::gofmm::Data<double>,
//  2,
//  double
//  > 
//Compress( SPDMatrix<double> &K, double stol, double budget );
//
//
///**
// *  @brielf A "float" instance of the simple gofmm::Compress()
// */ 
//hmlp::tree::Tree<
//  hmlp::gofmm::Setup<SPDMatrix<float>, 
//    centersplit<SPDMatrix<float>, 2, float, SPLIT_ANGLE>, 
//    float>, 
//  hmlp::gofmm::Data<float>,
//  2,
//  float
//  > 
//Compress( SPDMatrix<float> &K, float stol, float budget );


}; /** end namespace gofmm */
}; /** end namespace hmlp */
