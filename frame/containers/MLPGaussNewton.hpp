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

    virtual Data<T> & GetValues() = 0;

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


    void SetValues( const Data<T> &user_x )
    {
      //assert( user_x.row() == x.row() && user_x.col() == x.col() );
      x = user_x;
    };

    Data<T> & GetValues() { return x; };

  private:

    Data<T> x;

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
      /** Initialize w as normal( 0, 0.1 ) */
      w.randn( 0.0, 0.1 );
      mpi::Bcast( w.data(), w.size(), 0, MPI_COMM_WORLD );
    };

    size_t ParameterSize() { return w.size(); };

    /** x = w  * prev.x + bias */
    void Forward() 
    {
      Data<T> &A = w;
      Data<T> &B = prev.GetValues();
      Data<T> &C = x;
      /** C = AB^{T} */
      //gemm::xgemm( HMLP_OP_N, HMLP_OP_N, (T)1.0, A, B, (T)0.0, C );
      xgemm( "N", "N", C.row(), C.col(), B.row(), 
          1.0, A.data(), A.row(), 
               B.data(), B.row(), 
          0.0, C.data(), C.row() );
      /** RELU activation function */
      nnz = 0;
      for ( auto &c : C ) 
      {
        c = std::max( c, (T)0 );
        if ( c ) nnz ++;
      }
      printf( "Layer report: %8lu/%8lu nnz\n", nnz, C.size() );
    };


    /** The tuple contains ( lid in I, idi of W, idj of W ) */
    Data<T> PerturbedGradient( bool use_reduced_format, size_t i, size_t j )
    {
      assert( i < w.row() && j < w.col() );
      Data<T> &B = prev.GetValues();
      if ( use_reduced_format )
      {
        Data<T> G( 1, this->BatchSize(), 0 );
        for ( size_t b = 0; b < this->BatchSize(); b ++ ) 
        {
          if ( x( i, b ) ) G[ b ] = B( j, b );
        }
        return G;
      }
      else
      {
        Data<T> G( this->NeuronSize(), this->BatchSize(), 0 );
        for ( size_t b = 0; b < this->BatchSize(); b ++ ) 
        {
          if ( x( i, b ) ) G( i, b ) = B( j, b );
        }
        return G;
      }
    };

    Data<T> Gradient( Data<T> &B )
    {
      Data<T> &A = w;
      Data<T> G( this->NeuronSize(), this->BatchSize(), 0 );
      //gemm::xgemm( HMLP_OP_N, HMLP_OP_N, (T)1.0, A, B, (T)0.0, G );
      xgemm( "N", "N", G.row(), G.col(), B.row(), 
          1.0, A.data(), A.row(), 
               B.data(), B.row(), 
          0.0, G.data(), G.row() );
      /** RELU gradient */
      for ( size_t i = 0; i < x.size(); i ++ ) 
      {
        if ( !x[ i ] ) G[ i ] = 0;
      }
      return G;
    };

    Data<T> Gradient( Data<T> &B, size_t q )
    {
      Data<T> &A = w;
      Data<T> G( this->NeuronSize(), this->BatchSize(), 0 );
      /** RELU gradient */
      for ( size_t j = 0; j < x.col(); j ++ )
      {
        for ( size_t i = 0; i < x.row(); i ++ )
        {
          if ( x( i, j ) ) G( i, j ) = w( i, q ) * B[ j ];
        }
      }
      return G;
    };

    Data<T> & GetValues() { return x; };

    Data<T> & GetParameters() { return w; };

    size_t NumberNonZeros() { return nnz; };

  private:

    /** N-by-B */
    Data<T> x;

    /** N-by-prev.N */
    Data<T> w;

    Data<T> bias;

    LayerBase<T> &prev;

    size_t nnz = 0;

};


template<typename T>
class MLPGaussNewton : public VirtualMatrix<T>,
                       public SPDMatrixMPISupport<T>
{
  public:

    MLPGaussNewton() 
    {
      filename = string( "/scratch/02794/ych/data/tmp/MLPGaussNewton.bin" );
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
      printf( "Layer %3lu N %8lu\n", net.size(), N ); fflush( stdout );
    };

    void Update( const Data<T> &data )
    {
      int comm_rank; mpi::Comm_rank( MPI_COMM_WORLD, &comm_rank );
      int comm_size; mpi::Comm_size( MPI_COMM_WORLD, &comm_size );

      in->SetValues( data );
      /** Feed forward */
      for ( auto layer : net ) layer->Forward();

      //fd = open( filename.data(), O_RDWR | O_CREAT, 0 );
      //assert( fd != -1 );

      //uint64_t data_size = this->col();
      //data_size *= in->BatchSize();
      //data_size *= sizeof(T);

      //if ( comm_rank == 0 )
      //{
      //  if ( lseek( fd, data_size - 1, SEEK_SET ) == -1 ) printf( "lseek failure\n" );
      //  if ( write( fd, "", 1 ) == -1 ) printf( "write failure\n" );
      //}
      //mpi::Barrier( MPI_COMM_WORLD );

      //mmappedData = (T*)mmap(0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
      //if ( mmappedData == MAP_FAILED ) printf( "mmap failure\n" );

      //#pragma omp parallel for
      //for ( size_t j = comm_rank; j < this->col(); j += comm_size )
      //{
      //  Data<T> Jj = Jacobian( vector<size_t>( 1, j ) );
      //  uint64_t offj = (uint64_t)Jj.size() * (uint64_t)j;
      //  for ( size_t i = 0; i < Jj.size(); i ++ ) mmappedData[ offj + i ] = Jj[ i ];
      //}
      //mpi::Barrier( MPI_COMM_WORLD );
    };

    void WriteJacobianToFiles( string filename )
    {
      int comm_rank; mpi::Comm_rank( MPI_COMM_WORLD, &comm_rank );
      int comm_size; mpi::Comm_size( MPI_COMM_WORLD, &comm_size );

      uint64_t nb = 2048;
      uint64_t nf = ( net.back()->NumberNonZeros() - 1 ) / nb + 1; 
      uint64_t data_size = (uint64_t)this->col() * nb * sizeof(float);

      vector<int> fd( nf );
      vector<float*> mappedData( nf );

      if ( comm_rank == 0 )
      {
        for ( size_t f = 0; f < nf; f ++ )
        {
          string filepath = filename + to_string( f * nb );
          fd[ f ] = open( filepath.data(), O_RDWR | O_CREAT, 0 );
          assert( fd[ f ] != -1 );
        }
      }
      mpi::Barrier( MPI_COMM_WORLD );
      if ( comm_rank > 0 )
      {
        for ( size_t f = 0; f < nf; f ++ )
        {
          string filepath = filename + to_string( f * nb );
          fd[ f ] = open( filepath.data(), O_RDWR, 0 );
          assert( fd[ f ] != -1 );
        }
      }
      if ( comm_rank == 0 )
      {
        for ( size_t f = 0; f < nf; f ++ )
        {
          if ( lseek( fd[ f ], data_size - 1, SEEK_SET ) == -1 ) printf( "lseek failure\n" );
          if ( write( fd[ f ], "", 1 ) == -1 ) printf( "write failure\n" );
        }
      }
      mpi::Barrier( MPI_COMM_WORLD );
      for ( size_t f = 0; f < nf; f ++ )
      {
        mappedData[ f ] = (float*)mmap( 0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd[ f ], 0 );
        if ( mappedData[ f ] == MAP_FAILED ) printf( "mmap failure\n" );
      }
      mpi::Barrier( MPI_COMM_WORLD );

      size_t zero_columns = 0;

      #pragma omp parallel for schedule(dynamic)
      for ( size_t j = comm_rank; j < this->col(); j += comm_size )
      {
        if ( j % 5000 == 0 ) printf( "%8lu columns computed (rank %3d)\n", j, comm_rank ); 
        Data<T> Jj = Jacobian( vector<size_t>( 1, j ) );
        uint64_t offj = (uint64_t)nb * (uint64_t)j;
        size_t nnz = 0;
        for ( size_t i = 0; i < Jj.size(); i ++ ) 
        {
          if ( i < Jj.size() )
          {
            mappedData[ i / nb ][ offj + ( i % nb ) ] = (float)Jj[ i ];
            if ( Jj[ i ] ) nnz ++;
          }
          else
          {
            mappedData[ i / nb ][ offj + ( i % nb ) ] = 0;
          }
        }
        if ( !nnz )
        {
          #pragma omp atomic update 
          zero_columns ++;
        }
      }
      mpi::Barrier( MPI_COMM_WORLD );
      printf( "zero column %lu\n", zero_columns );

      for ( size_t f = 0; f < nf; f ++ )
      {
        int rc = munmap( mappedData[ f ], data_size );
        assert( rc == 0 );
        close( fd[ f ] );
      }
      mpi::Barrier( MPI_COMM_WORLD );
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
      
      //size_t d = net.back()->NeuronSize() * net.back()->BatchSize();
      //Data<T> A( d, I.size() );
      //Data<T> B( d, J.size() );
      //for ( size_t j = 0; j < I.size(); j ++ )
      //{
      //  uint64_t offj = (uint64_t)d * (uint64_t)j;
      //  for ( size_t i = 0; i < d; i ++ ) A( i, j ) = mmappedData[ offj + i ];
      //}
      //for ( size_t j = 0; j < J.size(); j ++ )
      //{
      //  uint64_t offj = (uint64_t)d * (uint64_t)j;
      //  for ( size_t i = 0; i < d; i ++ ) B( i, j ) = mmappedData[ offj + i ];
      //}

      /** KIJ = A^{T}B */
      xgemm( "T", "N", KIJ.row(), KIJ.col(), B.row(), 
          1.0, A.data(), A.row(), 
               B.data(), B.row(), 
          0.0, KIJ.data(), KIJ.row() );
      return KIJ;
    };


  private:

    Layer<INPUT, T> *in = NULL;

    /** [ L0, L1, ..., Lq-1 ], q layers */
    vector<Layer<FC, T>*> net;

    /** [ #W1, #W1+#W1, ..., #W1+...+#Wq-1], sorted q numbers */
    vector<size_t> index_range;

    Data<T> Jcache;

    vector<size_t> all_rows;

    string filename;

    /** Use mmap */
    T *mmappedData = NULL;

    int fd = -1;



    /** [ RELU(Wq-1)*...*RELU(W1), ..., RELU(Wq-1), I ], q products */
    //vector<Data<T>> product;

    /** Convert global parameter index to local layer index. */
    //vector<vector<tuple<size_t, size_t, size_t>>> Index2Layer( const vector<size_t> &I )
    //{
    //  size_t n_layer = net.size();
    //  vector<vector<tuple<size_t, size_t, size_t>>> ret( n_layer );

    //  for ( size_t i = 0; i < I.size(); i ++ )
    //  {
    //    auto upper = upper_bound( index_range.begin(), index_range.end(), I[ i ] );
    //    assert( upper != index_range.end() );
    //    size_t layer = distance( index_range.begin(), upper );

    //    /** Compute local index within the layer */
    //    size_t lid = I[ i ];
    //    if ( layer ) lid -= index_range[ layer - 1 ];
    //    Data<T> &w = net[ layer ]->GetParameters();
    //    size_t lidi = lid % w.row();
    //    size_t lidj = lid / w.row();
    //    ret[ layer ].push_back( make_tuple( i, lidi, lidj ) );
    //  }

    //  return ret;
    //};

    vector<tuple<size_t, size_t, size_t>> Index2Layer( const vector<size_t> &I )
    {
      vector<tuple<size_t, size_t, size_t>> ret( I.size() );
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
        ret[ i ] = make_tuple( layer, lidi, lidj );
      }
      return ret;
    };

    Data<T> Jacobian( const vector<size_t> &I )
    {
      size_t B = net.back()->BatchSize();
      size_t N = net.back()->NeuronSize();
      size_t nnz = net.back()->NumberNonZeros();

      //Data<T> J( N * B, I.size(), 0 );
      Data<T> J( nnz, I.size(), 0 );

      /** Iatl[ q ] contains all indices of layer q. */
      auto Iatl = Index2Layer( I );
      //printf( "Index2Layer\n" ); fflush( stdout );

      //#pragma omp parallel for 
      for ( size_t b = 0; b < Iatl.size(); b ++ )
      {
        size_t l = get<0>( Iatl[ b ] );
        size_t i = get<1>( Iatl[ b ] );
        size_t j = get<2>( Iatl[ b ] );

        Data<T> Jbuff, tmp;
        
        if ( l == net.size() - 1 )
        {
          Jbuff = net[ l ]->PerturbedGradient( false, i, j );
        }
        else
        {
          Jbuff = net[ l + 0 ]->PerturbedGradient(  true, i, j );
          tmp   = net[ l + 1 ]->Gradient( Jbuff, i );
          Jbuff = tmp;
          for ( size_t layer = l + 2; layer < net.size(); layer ++ )
          {
            Data<T> tmp = net[ layer ]->Gradient( Jbuff );
            Jbuff = tmp;
          }
        }
        
        //for ( size_t layer = l + 1; layer < net.size(); layer ++ )
        //{
        //  Data<T> tmp = net[ layer ]->Gradient( Jbuff );
        //  Jbuff = tmp;
        //}
        
        assert( Jbuff.size() == N * B );
        size_t count = 0;
        for ( size_t q = 0; q < N * B; q ++ ) 
        {
          auto &lastx = net.back()->GetValues();
          if ( lastx[ q ] ) 
          {
            J( count, b ) = Jbuff[ q ];
            count ++;
          }
        }
        assert( count == nnz );
      }

      ///** Compute Jacobian perburbation for each layer. */
      //for ( size_t q = 0; q < Iatl.size(); q ++ )
      //{
      //  /** B-by-Iatl[ q ].size() */
      //  Data<T> JBq = net[ q ]->ForwardPerturbation( Iatl[ q ] );
      //  /** Compute the rest of the feed forward network. */  
      //  for ( size_t j = 0; j < Iatl[ q ].size(); j ++ )
      //  {
      //    size_t lid  = get<0>( Iatl[ q ][ j ] );
      //    size_t lidi = get<1>( Iatl[ q ][ j ] );
      //    for ( size_t b = 0; b < B; b ++ ) JB( b, lid ) = JBq( b, j );
      //    for ( size_t i = 0; i < N; i ++ ) JN( i, lid ) = product[ q ]( i, lidi );
      //  }
      //}

      ///** Severin: compute J from JN and JB */

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
