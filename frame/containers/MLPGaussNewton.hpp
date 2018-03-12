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


template<typename T>
class LayerBase
{
  public:

    LayerBase( size_t user_N, size_t user_B )
    :
    N( user_N ), B( user_B )
    {};

    /** Number of neurons */
    size_t NeuronSize() { return N; };

    size_t BatchSize() { return B; };

    virtual size_t ParameterSize() = 0;

    /** The input and output types are subjected to change. */
    virtual void Forward() = 0;
    virtual void TemporaryForward() = 0;

    virtual Data<T> & GetValues() = 0;
    virtual Data<T> & GetTemporaryValues() = 0;

  private:

    size_t N = 0;

    size_t B = 0;

};

template<LayerType LAYERTYPE, typename T>
class Layer : public LayerBase<T> {};

template<typename T>
class Layer<INPUT, T> : public LayerBase<T>
{
  public:

    Layer( size_t N, size_t B ) : LayerBase<T>( N, B )
    {
      x.resize( N, B );
    };

    size_t ParameterSize() { return 0; };

    void Forward() { /** NOP */ };

    void TemporaryForward() { /** NOP */ };

    void SetValues( const Data<T> &user_x )
    {
      assert( user_x.row() = x.row() && user_x.col() == x.col() );
      x = user_x;
    };

    Data<T> & GetValues() { return x; };

    void SetTemporaryValues( const Data<T> &user_tmp ) { tmp = user_tmp; };

    Data<T> & GetTemporaryValues() { return tmp; };

  private:

    Data<T> x;

    Data<T> tmp;
};


template<typename T>
class Layer<FC, T> : public LayerBase<T>
{
  public:

    Layer( size_t N, size_t B, LayerBase<T> &user_prev ) 
    : 
    LayerBase<T>( N, B ), prev( user_prev )
    {
      x.resize( N, B );
      w.resize( N, prev.NeuronSize() );
      /** Initialize w as normal( 0, 1 ) */
      w.randn( 0.0, 1.0 );
    };

    size_t ParameterSize() { return w.size(); };

    /** x = w  * prev.x + bias */
    void Forward() 
    {
      Data<T> &A = w;
      Data<T> &B = prev.GetValues();
      Data<T> &C = x;
      /** C = AB^{T} */
      gemm::xgemm( HMLP_OP_N, HMLP_OP_N, (T)1.0, A, B, (T)0.0, C );
      /** RELU activation function */
      for ( auto c : C ) c = std::max( c, (T)0 );
    };

    void TemporaryForward()
    {
      Data<T> &A = w;
      Data<T> &B = prev.GetTemporaryValues();
      Data<T> &C = tmp;
      tmp.resize( A.row(), B.col() );
      gemm::xgemm( HMLP_OP_N, HMLP_OP_N, (T)1.0, A, B, (T)0.0, C );
      /** RELU activation function */
      for ( auto c : C ) c = std::max( c, (T)0 );
    };

    /** The tuple contains ( lid in I, idi of W, idj of W ) */
    Data<T> ForwardPerturbation
    ( 
      const vector<tuple<size_t, size_t, size_t>> &Perturbation 
    )
    {
      /** B-by-#perburation */
      Data<T> C( this->BatchSize(), Perturbation.size() );

      /** Severin: C( :, j ) = delta * prev.x( get<2>( Perturbation[ j ] ), :  ) */


      return C;
    };

    Data<T> & GetValues() { return x; };

    Data<T> & GetParameters() { return w; };

    void SetTemporaryValues( const Data<T> &user_tmp ) { tmp = user_tmp; };

    Data<T> & GetTemporaryValues() { return tmp; };

  private:

    /** N-by-B */
    Data<T> x;

    /** N-by-prev.N */
    Data<T> w;

    Data<T> bias;

    LayerBase<T> &prev;

    Data<T> tmp;
};


template<typename T>
class MLPGaussNewton : public VirtualMatrix<T>,
                       public SPDMatrixMPISupport<T>
{
  public:

    MLPGaussNewton() {};

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
      printf( "Layer %3lu N %8lu\n", net.size(), N ); fflush( stdout );
    };

    void Update( const Data<T> &data )
    {
      /** Feed forward */
      for ( auto layer : net ) layer->Forward();
      /** Compute product for fast perturbation */
      int q = net.size();
      product.resize( q );

      /** The product of the last layer is I. */
      for ( int i = 0; i < q; i ++ )
      {
        size_t N = net[ i ]->NeuronSize();
        Data<T> identity( N, N, 0 );
        for ( size_t k = 0; k < N; k ++ ) identity( k, k ) = 1;
        net[ i ]->SetTemporaryValues( identity );
        /** (RELU(Wq-1)*...*RELU(Wi+2))*RELU(Wi+1) */
        for ( int j = i + 1; j < q; j ++ ) net[ j ]->TemporaryForward();
        product[ i ] = net.back()->GetTemporaryValues();
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
      Data<T> A = Jacobian( I );
      Data<T> B = Jacobian( J );
      /** KIJ = A^{T}B */
      gemm::xgemm( HMLP_OP_T, HMLP_OP_N, (T)1.0, A, B, (T)0.0, KIJ );
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

    /** [ L0, L1, ..., Lq-1 ], q layers */
    vector<Layer<FC, T>*> net;

    /** [ #W1, #W1+#W1, ..., #W1+...+#Wq-1], sorted q numbers */
    vector<size_t> index_range;

    /** [ RELU(Wq-1)*...*RELU(W1), ..., RELU(Wq-1), I ], q products */
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


    Data<T> Jacobian( const vector<size_t> &I )
    {
      size_t B = net.back()->BatchSize();
      size_t N = net.back()->NeuronSize();

      Data<T> J( N * B, I.size(), 0 );
      Data<T> JN( N, I.size(), 0 );
      Data<T> JB( B, I.size(), 0 );

      /** Iatl[ q ] contains all indices of layer q. */
      auto Iatl = Index2Layer( I );

      /** Compute Jacobian perburbation for each layer. */
      for ( size_t q = 0; q < Iatl.size(); q ++ )
      {
        /** B-by-Iatl[ q ].size() */
        Data<T> JBq = net[ q ]->ForwardPerturbation( Iatl[ q ] );
        /** Compute the rest of the feed forward network. */  
        for ( size_t j = 0; j < Iatl[ q ].size(); j ++ )
        {
          size_t lid  = get<0>( Iatl[ q ][ j ] );
          size_t lidi = get<1>( Iatl[ q ][ j ] );
          for ( size_t b = 0; b < B; b ++ ) JB( b, lid ) = JBq( b, j );
          for ( size_t i = 0; i < N; i ++ ) JN( i, lid ) = product[ q ]( i, lidi );
        }
      }

      /** Severin: compute J from JN and JB */

      return J;
    }





    /** JN is LastLayerNeuronSize-by-I.size(), JB is BatchSize-by-I.size() */
    //void Jacobian( const vector<size_t> &I, Data<T> &JN, Data<T> &JB )
    //{
    //  size_t B = net.back()->BatchSize();
    //  size_t N = net.back()->NeuronSize();

    //  JN.resize( N, I.size(), 0 );
    //  JB.resize( B, I.size(), 0 );

    //  /** */
    //  auto Iatl = Index2Layer( I );

    //  /** Compute Jacobian perburbation for each layer. */
    //  for ( size_t l = 0; l < Iatl.size(); l ++ )
    //  {
    //    Data<T> JBl = net[ l ]->ForwardPerturbation( Iatl[ l ] );
    //    /** Compute the rest of the feed forward network. */  
    //    for ( size_t j = 0; j < Iatl[ l ].size(); j ++ )
    //    {
    //      size_t lid  = get<0>( Iatl[ l ][ j ] );
    //      size_t lidi = get<1>( Iatl[ l ][ j ] );
    //      for ( size_t b = 0; b < B; b ++ ) JB( b, lid ) = JBl( b, j );
    //      for ( size_t i = 0; i < N; i ++ ) JN( i, lid ) = product[ l ]( i, lidi );
    //    }
    //    
    //  }
    //};




};





}; /** end namespace hmlp */

#endif /** define MLPGAUSSNEWTON_HPP */
