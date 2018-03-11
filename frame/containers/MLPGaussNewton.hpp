#ifndef MLPGAUSSNEWTON_HPP
#define MLPGAUSSNEWTON_HPP

/** BLAS/LAPACK support */
#include <hmlp_blas_lapack.h>
/** GEMM task support */
#include <primitives/gemm.hpp>
/** MLPGaussNewton uses VirtualMatrix<T> as base */
#include <containers/VirtualMatrix.hpp>
/** For GOFMM compatability */
#include <containers/SPDMatrix.hpp>

using namespace std;
using namespace hmlp;

namespace hmlp
{

typedef enum { INPUT, FC, CONV2D, POOLING } LayerType;



class LayerBase
{
  public:

    LayerBase( size_t user_N, size_t user_B )
    :
    N( user_N ), B( user_B )
    {
    };

    /** Number of neurons */
    size_t NeuronSize() { return N; };

    size_t BatchSize() { return B; };

    virtual size_t ParameterSize() = 0;

    /** The input and output types are subjected to change. */
    virtual void Forward() = 0;

  private:

    size_t N = 0;

    size_t B = 0;

};

template<LayerType LAYERTYPE, typename T>
class Layer : public LayerBase {};

template<typename T>
class Layer<INPUT, T> : public LayerBase
{
  public:

    Layer( size_t N, size_t B ) : public LayerBase( N, B )
    {
      x.resize( B, N );
    };

    size_t ParameterSize() { return 0; };

    void Forward() { /** NOP */ };

    Data<T> & GetValues() { return x; };

  private:

    Data<T> x; 
};


template<typename T>
class Layer<FC, T> : public LayerBase
{
  public:

    Layer( size_t N, size_t B, const LayerBase &user_prev ) 
    : 
    public LayerBase( N, B ), prev( user_prev )
    {
      x.resize( B, N );
      w.resize( N, prev.NeuronSize() );
      /** Still We need to Initialize w */
    };

    size_t ParameterSize() { return w.size(); };

    /** x = prev.x * Transpose( w ) + bias */
    void Forward() 
    {
      Data<T> &A = prev.GetValues();
      Data<T> &B = w;
      Data<T> &C = x;
      /** C = AB^{T} */
      gemm::xgemm( HMLP_OP_N, HMLP_OP_T, (T)1.0, A, B, (T)0.0, C );
    };

    /** B-by-#perburation */
    Data<T> ForwardPerturbation
    ( 
      const vector<tuple<size_t, size_t, size_t>> &Perturbation 
    )
    {
      Data<T> C( this->BatchSize(), Perturbation.size() );

      for ( size_t i = 0; i < Perturbation.size(); i ++ )
      {
        //C( :, )
      }

      return C;
    };

    Data<T> & GetValues() { return x; };

    Data<T> & GetParameters() { return w; };
    
  private:

    /** B-by-N */
    Data<T> x;

    /** N-by-prev.N */
    Data<T> w;

    Data<T> bias;

    LayerBase &prev;
};


template<typename T>
class MLPGaussNewton : public VirtualMatrix<T>,
                       public SPDMatrixMPISupport<T>
{
  public:

    MLPGaussNewton()
    {
    };

    void AppendInputLayer( Layer<INPUT, T> &layer )
    {
      this->in = &layer;
    }

    void AppendFCLayer( Layer<FC, T> &layer )
    {
      size_t N = this->row() + layer.ParameterSize();
      this->resize( N, N );
      net.push_back( &layer );
      index_range.push_back( N );
    };

    void Update( const Data<T> &data )
    {
      /** Feed forward */
      for ( auto layer : net ) layer->Forward();
      /** Compute product for fast perturbation */
      int q = net.size() - 1;
      product.resize( q );
      for ( int i = q - 1; i >= 0; i -- )
      {
        Data<T> &C = product[ i ];
        if ( i == q - 1 ) C = net[ i + 1 ].GetParameters();
        else
        {
          Data<T> &A = product[ i + 1 ];
          Data<T> &B = net[ i + 1 ].GetParameters();
          C.resize( A.row(), B.col() );
          /** (Wq*...*Wi+2)*Wi+1 */
          gemm::xgemm( HMLP_OP_N, HMLP_OP_N, (T)1.0, A, B, (T)0.0, C );
        }
      }
    };

    /** ESSENTIAL: this is an abstract function  */
    virtual T operator()( size_t i, size_t j )
    {
      Data<T> KIJ = (*this)( vector<size_t>( 1, i ), vector<size_t>( 1, j ) );
      return KIJ[ 0 ];
    };

    /** ESSENTIAL: return a submatrix */
    virtual Data<T> operator()
		( 
      const vector<size_t> &I, const vector<size_t> &J 
    )
    {
      Data<T> KIJ( I.size(), J.size() );
      //Data<T> A = Jacobian( I );
      //Data<T> B = Jacobian( J );
      ///** KIJ = A^{T}B */
      //gemm::xgemm( HMLP_OP_T, HMLP_OP_N, (T)1.0, A, B, (T)0.0, KIJ );
      return KIJ;
    };

    virtual Data<T> PairwiseDistances
    ( 
      const vector<size_t> &I, const vector<size_t> &J 
    )
    {
      return (*this)( I, J );
    };


  private:

    Layer<INPUT, T> *in = NULL;

    /** [ L0, L1, ..., Lq ], q + 1 layers */
    vector<Layer<FC, T>*> net;

    /** [ #W1, #W1+#W1, ..., #W1+...+#Wq], sorted q + 1 numbers */
    vector<size_t> index_range;

    /** [ Wq*...*W1, Wq*...*W2, ..., Wq, I ], q products */
    vector<Data<T>> product;

    /** Convert global parameter index to local layer index. */
    vector<vector<tuple<size_t, size_t, size_t>>> Index2Layer( const vector<size_t> &I )
    {
      size_t n_layer = net.size();
      vector<vector<tuple<size_t, size_t, size_t>>> ret( n_layer );

      for ( size_t i = 0; i < I.size(); i ++ )
      {
        auto upper = upper_bound( index_range.begin(), index_range.end(), I[ i ] );
        assert( upper != index_range.end() );
        size_t layer = distance( index_range.begin(), upper );

        /** Compute local index within the layer */
        size_t lid = I[ i ];
        if ( layer ) lid -= index_range[ layer - 1 ];
        Data<T> &w = net[ layer ]->GetParameters();
        size_t lidi = lid % w.row();
        size_t lidj = lid / w.row();
        ret[ layer ].push_back( make_tuple( i, lidi, lidj ) );
      }

      return ret;
    };

    vector<pair<size_t, size_t>> Index2Neuron( size_t l, 
        const vector<size_t> &Iatl, const vector<size_t> &I )
    {
    }


    /** JN is LastLayerNeuronSize-by-I.size(), JB is BatchSize-by-I.size() */
    void Jacobian( const vector<size_t> &I, Data<T> &JN, Data<T> &JB )
    {
      size_t B = net.back()->BatchSize();
      size_t N = net.back()->NeuronSize();

      JN.resize( N, I.size(), 0 );
      JB.resize( B, I.size(), 0  );

      /** */
      auto Iatl = Index2Layer( I );

      /** Compute Jacobian perburbation for each layer. */
      for ( size_t l = 0; l < Iatl.size(); l ++ )
      {
        Data<T> JBl = net[ l ]->ForwardPerturbation( Iatl[ l ] );
        /** Compute the rest of the feed forward network. */  
        for ( size_t j = 0; j < Iatl[ l ].size(); j ++ )
        {
          size_t lid  = get<0>( Iatl[ l ][ j ] );
          size_t lidi = get<1>( Iatl[ l ][ j ] );
          for ( size_t b = 0; b < B; b ++ ) JB( b, lid ) = JBl( b, j );
          for ( size_t i = 0; i < N; i ++ ) JN( i, lid ) = product[ l ]( i, lidi );
        }
        
      }
    };




};





}; /** end namespace hmlp */

#endif /** define MLPGAUSSNEWTON_HPP */
