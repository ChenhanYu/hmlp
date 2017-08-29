#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <limits>

#include <hmlp.h>
#include <containers/data.hpp>
#include <containers/KernelMatrix.hpp>

namespace hmlp
{

template<typename T>
class Cluster
{
  public:

    Cluster( size_t d, size_t n, size_t ncluster, 
        hmlp::Data<T> *X, std::vector<int> *assignments ) 
    {
      this->d = d;
      this->n = n;
      this->ncluster = ncluster;
      this->X = X;
      this->assignments = assignments;

      assert( assignments->size() == n );
      assert( X->size() == d * n );


      /** compute X2 */
      X2.resize( n, 1, 0.0 );
      for ( size_t i = 0; i < n; i ++ )
      {
        for ( size_t p = 0; p < d; p ++ )
        {
          auto xpi = (*X)( p, i );
          X2[ i ] += xpi * xpi;
        }
      };
    };

    void ProvideLabels( size_t nclass, std::vector<int> *labels )
    {
      this->nclass = nclass;
      this->labels = labels;
    };

    /** compute the centroids of X( :, amap ) based on the assignment */
    std::vector<size_t> Centroids( std::vector<int> &amap, 
        std::vector<T> &centroids, std::vector<T> &centroid_sqnorms )
    {
      std::vector<size_t> cluster_sizes( ncluster, 0 );
    
      /** resize centroids and centroid_sqnorms */
      centroids.resize( d * ncluster );
      centroid_sqnorms.resize( ncluster );

      /** zero out */
      for ( size_t i = 0; i < centroids.size(); i ++ ) 
        centroids[ i ] = 0.0;
      for ( size_t i = 0; i < centroid_sqnorms.size(); i ++ ) 
        centroid_sqnorms[ i ] = 0.0;

      /** compute centroids accoding to the current assignment */
      for ( size_t i = 0; i < amap.size(); i ++ )
      {
        auto pi = (*assignments)[ amap[ i ] ];
        assert( pi >= 0 && pi < ncluster );
        for ( size_t p = 0; p < d; p ++ )
        {
          centroids[ p + pi * d ] += (*X)( p, (size_t)amap[ i ] );
          cluster_sizes[ pi ] ++;
        }
      };

      /** normalize with the cluster size */
      for ( size_t j = 0; j < ncluster; j ++ )
      {
        if ( cluster_sizes[ j ] )
        {
          for ( size_t p = 0; p < d; p ++ )
            centroids[ p  + j * d ] /= cluster_sizes[ j ];
        }
        else
        {
          /** randomly assign a point to be the centroid */
          for ( size_t p = 0; p < d; p ++ )
          {
            centroids[ p + j * d ] = (*X)( p, std::rand() % n );
          }
        }
      }

      /** square 2-norm */
      for ( size_t j = 0; j < ncluster; j ++ )
        for ( size_t p = 0; p < d; p ++ )
          centroid_sqnorms[ j ] += centroids[ p + j * d ] * centroids[ p + j * d ];

      return cluster_sizes;

    }; /** end Centroids() */


    /** perform a Kmeans iteration */
    virtual void Kmeans( std::vector<int> &amap, size_t niter, T tol )
    {
      /** allocate centroids and centroid_sqnorms */
      hmlp::Data<T> centroids( d, ncluster, 0.0 );
      hmlp::Data<T> centroid_sqnorms( 1, ncluster, 0.0 );
      std::vector<int> centroid_maps( ncluster, 0 );
      for ( size_t i = 0; i < ncluster; i ++ )
        centroid_maps[ i ] = i;

      /** initialize */
      hmlp::Data<T> distance2centroids( n, 1, std::numeric_limits<T>::max() );


      //printf( "X: " );
      //X->Print();
      
      //for ( size_t i = 0; i < n; i ++ )
      //  printf( "%5.4E ", (*X)[ i ] );
      //printf( "\n" );

      /** main loop */
      for ( size_t iter = 0; iter < niter; iter ++ )
      {
        double beg_t, end_t, centroid_t, gsknn_t;

        //for ( size_t i = 0; i < n; i ++ )
        //  printf( "%d ", (*assignments)[ i ] );
        //printf( "\n" );

        /** update centroids */
        beg_t = omp_get_wtime();
        auto cluster_sizes = Centroids( amap, centroids, centroid_sqnorms );
        centroid_t = omp_get_wtime() - beg_t;

        //printf( "Centroids: " );
        //centroids.Print();


        /** perform gsknn with one neighbor */
        gsknn( amap.size(), ncluster, d, 1,
            X->data(), X2.data(), amap.data(),
            centroids.data(), centroid_sqnorms.data(), centroid_maps.data(),
            distance2centroids.data(), 
            assignments->data() );
        //printf( "d2c: " );
        //for ( size_t i = 0; i < n; i ++ )
        //  printf( "%5.4E ", distance2centroids[ i ] );
        //printf( "\n" );


        T quality = 0.00;
        for ( size_t i = 0; i < n; i ++ ) 
          quality += distance2centroids[ i ] * distance2centroids[ i ];
        quality = std::sqrt( quality );

        printf( "Kmeans iteration #%2lu quality %E\n", iter, quality );
      }
      
    }; /** end Kmeans() */


    virtual void Kmeans( size_t niter, T tol )
    {
      std::vector<int> amap( n, 0 );
      for ( size_t i = 0; i < n; i ++ ) amap[ i ] = i;
      Kmeans( amap, niter, tol );

    }; /** end Kmeans() */


    /** use simple power method with orthnormalization */
    virtual void Spectral( kernel_s<T> &kernel )
    {
      /** create a kernel matrix */
      KernelMatrix<T> K( n, n, d, kernel, *X );

      /** all one vector (indicators) */
      hmlp::Data<T> ones( n, 1, 1.0 );

      /** get the weights i.e. Degree = K * ones */
      std::vector<T> Degree( n, 0 );
      K.Multiply( 1, Degree, ones );

      /** number of eigenpairs to compute */
      size_t neig = ncluster + 2;

      /** allocate space for eigenpairs */
      hmlp::Data<T> Sigma( neig, 1, 0.0 );
      hmlp::Data<T> V( neig, n );

      /** compute ( ncluster + 2 ) eigenpairs */
      for ( size_t p = 0; p < neig; p ++ )
      {
        /** the first eigenpairs are known */
        if ( p == 0 )
        {
          Sigma[ p ] = 1.0;
          for ( size_t i = 0; i < n; i ++ ) 
            V( p, i ) = 1.0 / std::sqrt( (T)n ); 
          continue;
        }

        /** random initialization */
        hmlp::Data<T> v( n, 1 ); v.randn();

        /** power iteration */
        for ( size_t iter = 0; iter < 100; iter ++ )
        {
          for ( size_t q = 0; q < p; q ++ )
          { 
            T projection = 0.0;
            /** compute v'D^{-1}v projection */
            for ( size_t i = 0; i < n; i ++ )
            {
              projection += v[ i ] * V( q, i );
            }

            /** remove the contribution from Vq */
            for ( size_t i = 0; i < n; i ++ )
            {
              v[ i ] -= Sigma[ q ] * projection * V( q, i );
            }
          }

          /** D^{-1}KDv */
          hmlp::Data<T> DKv( n, 1, 0 );
          K.Multiply( 1, DKv, v );
          for ( size_t i = 0; i < n; i ++ ) DKv[ i ] /= Degree[ i ];

          /** compute eigenvalue = v'D^{-1}Kv / v'v */
          T sigma_p = 0.0;
          for ( size_t i = 0; i < n; i ++ ) 
            sigma_p += v[ i ] * DKv[ i ];
          Sigma[ p ] = sigma_p;

          if ( iter % 50 == 49 )
          printf( "iteration %lu #%lu eignvalue %E\n", iter, p, sigma_p );

          /** compute 2-norm */
          T DKv_norm = 0.0;
          for ( size_t i = 0; i < n; i ++ ) DKv_norm += DKv[ i ] * DKv[ i ];
          DKv_norm = std::sqrt( DKv_norm );

          /** update v with normalization */
          for ( size_t i = 0; i < n; i ++ ) v[ i ] = DKv[ i ] / DKv_norm;
        };
        printf( "\n" );

        for ( size_t i = 0; i < n; i ++ ) V( p, i ) = v[ i ]; 
      }

      /** point-wise normalization */
      for ( size_t i = 0; i < n; i ++ ) 
      {
        T vi_norm = 0.0;
        for ( size_t p = 0; p < neig; p ++ ) 
          vi_norm += V( p, i ) * V( p, i );
        for ( size_t p = 0; p < neig; p ++ ) 
          V( p, i ) /= vi_norm;
      }
      
    }; /** end Spectral() */


    std::vector<size_t> ClusterPermutation( std::vector<int> &amap,
        std::vector<std::vector<int>> &bmap )
    {
      std::vector<size_t> cluster_sizes( ncluster, 0 );

      for ( size_t j = 0; j < ncluster; j ++ )
      {
        if ( !bmap[ j ].size() ) bmap[ j ].reserve( n );
        bmap[ j ].clear();
      }

      for ( size_t i = 0; i < amap.size(); i ++ )
      {
        auto pi = (*assignments)[ amap[ i ] ];
        assert( pi >= 0 && pi < ncluster );
        bmap[ pi ].push_back( i );
        cluster_sizes[ pi ] ++;
      }

      return cluster_sizes;
    };


    virtual void KernelKmeans( kernel_s<T> &kernel, 
        std::vector<int> &amap, size_t niter, T tol )
    {
      /** create a kernel matrix */
      KernelMatrix<T> K( n, n, d, kernel, *X );

      /** get D = [ K( 0, 0 ), ..., K( n-1, n-1 ) ) */
      std::vector<T> Diag( n, 0 );
      for ( size_t i = 0; i < n; i ++ )
      {
        Diag[ i ] = K( i, i );
        assert( Diag[ i ] > 0.0 );
      }

      /** all one vector (indicators) */
      hmlp::Data<T> ones( n, 1, 1.0 );

      /** get the weights i.e. Degree = K * ones */
      std::vector<T> Degree( n, 0 );
      K.Multiply( 1, Degree, ones );

      /** cluster permutation  */
      std::vector<std::vector<int>> bmap( ncluster );

      /** similarity  */
      std::vector<hmlp::Data<T>> Similarity( ncluster );
      for ( size_t j = 0; j < ncluster; j ++ ) 
        Similarity[ j ].resize( amap.size(), 1, 0.0 );

      /** umap for similarity */
      std::vector<int> umap( amap.size(), 0 );
      for ( size_t i = 0; i < amap.size(); i ++ ) umap[ i ] = i;

      /** main loop (sequential) */
      for ( size_t iter = 0; iter < niter; iter ++ )
      {
        /** initialize */
        hmlp::Data<T> distance2centroids( n, 1, std::numeric_limits<T>::max() );

        /** centroids */
        std::vector<T> centroids( n, 0.0 );

        /** bmap permutes K into the current cluster order */
        auto cluster_sizes = ClusterPermutation( amap, bmap );

        /** compute the similarity matrix */
        for ( size_t j = 0; j < ncluster; j ++ )
        {
          T Kcc = 0.0;
          T Dcc = 0.0;

          /** clean up the similarity matrix */
          #pragma omp parallel for
          for ( size_t i = 0; i < amap.size(); i ++ )
            Similarity[ j ][ i ] = 0.0;

          //printf( "umap.size() %lu, amap.size() %lu, bmap[ j ].size() %lu\n",
          //    umap.size(), amap.size(), bmap[ j ].size() ); fflush( stdout );

          /**  sum( Kij ) = K( amap, bmap ) *  */
          K.Multiply( 
              1, Similarity[ j ], umap,
                                  amap, 
                                  bmap[ j ],
                            ones, bmap[ j ] );


          if ( amap.size() == n )
          {
            /** Kcc = sum( Similarity( bmap, j ) )*/
            for ( size_t i = 0; i < bmap[ j ].size(); i ++ ) 
              Kcc += Similarity[ j ][ bmap[ j ][ i ] ];
          }
          else
          {
            /** Kcc = sum( K( bmap, bmap ) */
            K.Multiply( 1, centroids, bmap[ j ], bmap[ j ], ones );
            for ( size_t i = 0; i < bmap[ j ].size(); i ++ ) 
              Kcc += centroids[ bmap[ j ][ i ] ];
          }

          
          /** Kcc = sum( Degree( bmap[ j ] ) )*/
          for ( size_t i = 0; i < bmap[ j ].size(); i ++ ) 
            Dcc += Degree[ bmap[ j ][ i ] ];



          /** Kii - ( 2 / n ) * sum( Kic ) + ( 1 / n^2 ) * sum( Kcc )  */
          #pragma omp parallel for
          for ( size_t i = 0; i < amap.size(); i ++ )
          {
            T Kii = Diag[ amap[ i ] ];
            T Kic = Similarity[ j ][ amap[ i ] ];
            Similarity[ j ][ amap[ i ] ] = 
              Kii / ( Degree[ amap[ i ] ] * Degree[ amap[ i ] ] ) - 
              ( 2.0 / Degree[ amap[ i ] ] ) * ( Kic / Dcc ) + 
              Kcc / ( Dcc * Dcc );

            if ( Similarity[ j ][ amap[ i ] ] <= distance2centroids[ amap[ i ] ] )
            {
              distance2centroids[ amap[ i ] ] = Similarity[ j ][ amap[ i ] ];
              (*assignments)[ amap[ i ] ] = j;
            }
          }

          //printf( "assignments\n" ); fflush( stdout );
        }
        //printf( "here\n" ); fflush( stdout );

        T quality = 0.0;
        for ( size_t i = 0; i < amap.size(); i ++ )
          quality += distance2centroids[ amap[ i ] ] * distance2centroids[ amap[ i ] ];

        printf( "KernelKmeans iteration #%2lu quality %E\n", iter, quality );
      }

    };



    virtual void KernelKmeans( kernel_s<T> &kernel, size_t niter, T tol )
    {
      std::vector<int> amap( n, 0 );
      for ( size_t i = 0; i < n; i ++ ) amap[ i ] = i;
      KernelKmeans( kernel, amap, niter, tol );

    }; /** end KernelKmeans() */



    virtual void KernelKmeansRefinement()
    {
    };

    void ComputeConfusion()
    {
      /** labels must be provided and nclass must be non-zero */
      assert( labels && nclass );

      /** ncluster-by-nclass */
      Confusion.resize( 0, 0 );
      Confusion.resize( ncluster + 1, nclass + 1, 0.0 );

      /** compute cluster_sizes, class_sizes, Confusion */
      for ( size_t i = 0; i < n; i ++ )
      {
        auto icluster = (*assignments)[ i ];
        auto iclass   = (*labels)[ i ];
        Confusion( icluster, iclass ) += 1.0;
      }

      for ( size_t q = 0; q < nclass; q ++ )
      {
        for ( size_t p = 0; p < ncluster; p ++ )
        {
          auto Cpq = Confusion( p, q );
          Confusion(        p, nclass ) += Cpq;
          Confusion( ncluster,      q ) += Cpq;
          Confusion( ncluster, nclass ) += Cpq;
        }
      }

      Confusion.Print();

      /** if nclass == ncluster then compute the accuracy */
      if ( nclass == ncluster )
      {
        T num_of_correct_assignments = 0.0;
        std::set<int> cluster_pivots;
        std::set<int> class_pivots;

        /** insert all classes to the candidate list */
        for ( int q = 0; q < nclass; q ++ )
          class_pivots.insert( q );

        /** insert all clusteres to the candidate list */
        for ( int p = 0; p < ncluster; p ++ )
          cluster_pivots.insert( p );

        while ( cluster_pivots.size() && class_pivots.size() )
        {
          int pivot_p = -1;
          int pivot_q = -1;
          size_t max_entry = 0;

          /** loop over all combination */
          for ( auto pit = class_pivots.begin(); pit != class_pivots.end(); pit ++ )
          {
            for ( auto qit = class_pivots.begin(); qit != class_pivots.end(); qit ++ )
            {
              if ( Confusion( *pit, *qit ) >= max_entry )
              {
                max_entry = Confusion( *pit, *qit );
                pivot_p = *pit;
                pivot_q = *qit;
              }
            }
          }

          num_of_correct_assignments += max_entry;
          cluster_pivots.erase( pivot_p );
          class_pivots.erase( pivot_q );
        }

        printf( "Accuracy: %4.2E\n", num_of_correct_assignments / n );
      }

    }; /** end ComputeConfusion() */



    /** normalized mutual information (true label required) */
    T NMI()
    {
      /** labels must be provided and nclass must be non-zero */
      assert( labels && nclass );

      T nmi = 0.0;
      T nmi_antecedent = 0.0;
      T nmi_consequent = 0.0;

      /** Compute confusion matrix */
      ComputeConfusion();

      /** antecedent part */
      for ( size_t q = 0; q < nclass; q ++ )
      {
        for ( size_t p = 0; p < ncluster; p ++ )
        {
          auto Cpq = Confusion(        p,      q );
          auto Cp  = Confusion(        p, nclass );
          auto Cq  = Confusion( ncluster,      q );
          if ( Cpq > 0.0 )
          {
            nmi_antecedent += (-2.0) * ( Cpq / n ) * std::log( n * Cpq / ( Cp * Cq ) );
          }
        }
      }

      /** consequent part (true class) */
      for ( size_t q = 0; q < nclass; q ++ )
      {
        auto Cq = Confusion( ncluster, q );
        nmi_consequent += ( Cq / n ) * std::log( Cq / n );
      }

      /** consequent part (cluster) */
      for ( size_t p = 0; p < ncluster; p ++ )
      {
        auto Cp = Confusion( p, nclass );
        nmi_consequent += ( Cp / n ) * std::log( Cp / n );
      }
    
      nmi = ( nmi_antecedent / nmi_consequent );

      printf( "NMI: %E / %E = %4.2E\n", nmi_antecedent, nmi_consequent, nmi );

      return nmi;

    }; /** end NMI() */


    /** normalized edge cut (no labels required) */
    virtual void NormalizedCut()
    {

    };

  private:

    size_t n = 0;

    size_t d = 0;

    size_t ncluster = 0;

    size_t nclass = 0;

    /** d-by-n */
    hmlp::Data<T> *X = NULL;

    hmlp::Data<T> X2;

    std::vector<int> *assignments = NULL;

    std::vector<int> *labels = NULL;

    /** (ncluter+1)-by-(nclass+1) */
    hmlp::Data<T> Confusion;


}; /** end class Cluster */


}; /** end namespace hmlp */

#endif /** define CLUSTER_HPP */
