#ifndef REGRESSION_HPP
#define REGRESSION_HPP



#include <limits>

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>

#include <hmlp.h>

/** GOFMM templates */
#include <gofmm/gofmm.hpp>

#include <containers/data.hpp>
#include <containers/KernelMatrix.hpp>


using namespace std;


namespace hmlp
{

template<typename T>
class Regression
{
  public:

    Regression( size_t d, size_t n, hmlp::Data<T> *X, hmlp::Data<T> *Y )
    {
      this->d = d;
      this->n = n;
      this->X = X;
      this->Y = Y;
    };


    /**
     *  @brief 
     */ 
    Data<T> Ridge( kernel_s<T> &kernel, size_t niter )
    {
      size_t nrhs = Y->col();

      /** linear ridge regression */
      Data<T> XXt( d, d, 0.0 );
      Data<T> XY( d, nrhs, 0.0 );
      
      /** XXt + lambda * I */
      xsyrk( "Lower", "No transpose", d, n, 
          1.0, X->data(), d, 0.0, XXt.data(), d );

      for ( size_t i = 0; i < d; i ++ ) XXt( i, i ) += lambda;

      /** XY */
      xgemm( "No transpose", "No transpose", d, n, nrhs, 
          1.0, X->data(), d, 
               Y->data(), Y->row(),
          0.0, XY.data(), d );

      /** W = ( XXt + lambda * I )^{-1} * XY */
      xposv( "Lower", d, nrhs, X->data(), d, XY.data(), d );

      return XY;

    }; /** end Ridge() */



    Data<T> Lasso( kernel_s<T> &kernel, size_t niter )
    {
    }; /** end Lasso() */



    Data<T> SoftMax( kernel_s<T> &kernel, size_t nclass, size_t niter )
    {
      /** create a kernel matrix */
      KernelMatrix<T> K( n, n, d, kernel, *X );

      /** create a simple GOFMM compression */
      hmlp::gofmm::SimpleGOFMM<T, KernelMatrix<T>> H( K, 1E-3, 0.03 );

      hmlp::Data<T> W( n, nclass, 1.0 );
      hmlp::Data<T> P( n, nclass, 0.0 );

      for ( size_t it = 0; it < niter; it ++ )
      {
        hmlp::Data<T> Gradient( n, nclass, 0.0 );

        /** P = KW */
        H.Multiply( P, W );

        #pragma omp parallel for
        for ( size_t i = 0; i < n; i ++ )
        {
          T prob_all = 0.0;
          for ( size_t j = 0; j < nclass; j ++ ) prob_all  += P( i, j );
          for ( size_t j = 0; j < nclass; j ++ ) P( i, j ) /= prob_all;
          P( i, (size_t)(*Y)[ i ] ) -= 1.0;
        } 

        H.Multiply( Gradient, P );

        #pragma omp parallel for
        for ( size_t i = 0; i < n; i ++ )
        {
          for ( size_t j = 0; j < nclass; j ++ )
          {
            W( i, j ) += ( -1.0 * alpha / n ) * Gradient( i, j );
          }
        }
      }

      /** P = KW */
      H.Multiply( P, W );

      size_t n_correct = 0;
      for ( size_t i = 0; i < n; i ++ )
      {
        size_t goal = (*Y)[ i ];
        size_t pred = 0;
        T prob = 0.0;
        for ( size_t j = 0; j < nclass; j ++ )
        {
          if ( P( i, j ) > prob )
          {
            pred = j;
            prob = P( i, j );
          }
        }
        if ( pred == goal ) n_correct ++;
      } 

      printf( "Accuracy: %lf\n", (double)n_correct / n );

      {
        std::ofstream fout( "weight.dat", std::ios::out | std::ios::binary );
        fout.write( (char*)W.data(), W.size() * sizeof(T) );
        fout.close();
      }


      return W;
    };



    /**
     *  @brief gradient descent
     *
     *         w += (-1.0 / n) * K(Kw + b - Y + lambda * w)
     *         b += (-1.0 / n) *  (Kw + b - Y)
     */ 
    hmlp::Data<T> Solve( kernel_s<T> &kernel, size_t niter )
    {
      /** create a kernel matrix */
      KernelMatrix<T> K( n, n, d, kernel, *X );

      /** create a simple GOFMM compression */
      hmlp::gofmm::SimpleGOFMM<T, KernelMatrix<T>> H( K, 1E-3, 0.03 );

      hmlp::Data<T> W( n, (size_t)1.0, 0.0 );
      hmlp::Data<T> B( n, (size_t)1.0, 0.0 );

      for ( size_t it = 0; it < niter; it ++ )
      {
        hmlp::Data<T> Gradient( n, (size_t)1.0, 0.0 );

        /** ( K + lambda ) * W - Y + B */
        //K.Multiply( Gradient, W );
        H.Multiply( Gradient, W );

        /** Kw + B - Y */
        for ( size_t i = 0; i < n; i ++ )
          Gradient[ i ] += B[ i ] - (*Y)[ i ];
            
        /** update B = (-alpha / n) * ( Kw + B - Y) */
        //for ( size_t i = 0; i < n; i ++ )
        //  B[ i ] += ( -1.0 * alpha / n ) * Gradient[ i ];

        for ( size_t i = 0; i < n; i ++ )
          Gradient[ i ] += lambda * W[ i ];

        for ( size_t i = 0; i < n; i ++ )
          Gradient[ i ]  = ( -1.0 * alpha / n ) * Gradient[ i ];

        /** update W -= 1.0 * K ( ( K + lambda ) * W - Y ) */
        //K.Multiply( W, Gradient );

        hmlp::Data<T> tmp( n, (size_t)1.0, 0.0 );
        H.Multiply( tmp, Gradient );
        for ( size_t i = 0; i < n; i ++ )
          W[ i ] += tmp[ i ];


        if ( it % 100 == 0 )
        {
          /** Z = Kw + B */
          hmlp::Data<T> Z = B;
          //K.Multiply( Z, W );
          H.Multiply( Z, W );

          size_t n_correct = 0;
          for ( size_t i = 0; i < n; i ++ )
          {
            double pred = (int)( Z[ i ] + 0.5 );
            double goal = (*Y)[ i ];
            if ( pred == goal ) n_correct ++;

          }

          printf( "it %4lu Accuracy: %lf\n", it, (double)n_correct / n );
        }
      };

      /** Z = Kw + B */
      hmlp::Data<T> Z = B;
      //K.Multiply( Z, W );
      H.Multiply( Z, W );

      size_t n_correct = 0;
      for ( size_t i = 0; i < n; i ++ )
      {
        double pred = (int)( Z[ i ] + 0.5 );
        double goal = (*Y)[ i ];

        //printf( "pred %lf goal %lf\n", pred, goal );

        if ( pred == goal ) n_correct ++;
      }

      printf( "Accuracy: %lf\n", (double)n_correct / n );


      {
        std::ofstream fout( "weight.dat", std::ios::out | std::ios::binary );
        fout.write( (char*)W.data(), W.size() * sizeof(T) );
        fout.close();
      }
      {
        std::ofstream fout( "bias.dat", std::ios::out | std::ios::binary );
        fout.write( (char*)B.data(), B.size() * sizeof(T) );
        fout.close();
      }



      return W;
    };

  private:

    size_t d = 0;

    size_t n = 0;

    T lambda = 0.01;

    T alpha = 1.0;

    hmlp::Data<T> *X = NULL;

    hmlp::Data<T> *Y = NULL;

};

};

#endif /** define REGRESSION_HPP */
