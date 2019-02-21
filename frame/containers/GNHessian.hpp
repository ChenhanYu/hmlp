#ifndef GNHESSIAN_HPP
#define GNHESSIAN_HPP

/** BLAS/LAPACK support */
#include <base/blas_lapack.hpp>
/** GEMM task support */
#include <primitives/gemm.hpp>
/** GNHessian uses VirtualMatrix<T> as base */
#include <containers/VirtualMatrix.hpp>

#include <assert.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>

// pauss process for debugging
void enable_signal_handler();

// utility function
double timer() {
  double time;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  time = (double)tv.tv_sec + (double)tv.tv_usec/1.e6;
  return time;
}

//// BLAS call
//extern "C" {
//  void saxpy_(int* N, float *a, float *x, int* incrx, float *y, int *incry);
//}


template<typename T>
class Jacobian 
{
  public:
    int rows() {return nRows;}

    int cols() {return nCols;}

    void read_file(std::string filename) 
    {
      using namespace std;
      cout<< "Start reading Jacoiban from: " << filename << endl;
      ifstream file( filename.data(), ios::in|ios::binary|ios::ate );

      assert(file.is_open());

      auto size = file.tellg();
      cout << size << endl;

      int shape[2];
      file.seekg( 0, ios::beg );
      file.read( (char*)shape, 2 * sizeof( int ) );
      this->nRows = shape[ 0 ];
      this->nCols = shape[ 1 ];
      std::cout<<"Jacobian has size: "<<shape[0]<<" x "<<shape[1]<<std::endl;
      assert( nRows < nCols && "Jacobian size suspicious");

      this->Jac.resize( nRows, nCols );
      printf( "Jac %lu %lu\n", Jac.row(), Jac.col() ); fflush( stdout );
      file.read( (char*)Jac.data(), nRows * nCols * sizeof(T) );
      file.close();
    }

    Data<T> get_column(size_t idx) {
      Data<T> col(nRows, 1);
      memcpy(col.data(), Jac.columndata(idx), nRows*sizeof(T));
      return col;
    }

  private:
    int nRows;
    int nCols;
    Data<T> Jac;
};

template<typename T>
class CompressedJacobian {
  public:
    int rows() {
        return layers[L] * n;
    }

    int cols() {
        return N;
    }

    void read_file(std::string filename) {
      std::cout<<"Start reading precomputed Jacobian from: "<<filename<<std::endl;
      std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
      assert(file.is_open());

      file.read((char*)&L, sizeof(int));
      std::cout<<"# layers: "<<L<<std::endl;

      file.read((char*)&n, sizeof(int));
      std::cout<<"batch size: "<<n<<std::endl;

      assert(L > 0);
      layers.resize(L+1);
      file.read((char*)layers.data(), (L+1)*sizeof(int));
      std::cout<<"layer sizes:";
      for (int i=0; i<L+1; i++)
        std::cout<<" "<<layers[i];
      std::cout<<std::endl;
      
      // compute total number of parameters
      N = 0;
      for (int i=0; i<L; i++) {
        size_t ni = layers[i];
	size_t no = layers[i+1];
	N += no*ni+no;
      }
      std::cout<<"Total number of parameters: "<<N<<std::endl;

      assert(n > 0);
      assert(layers[0] > 0);
      x.resize(layers[0], n);
      file.read((char*)x.data(), layers[0]*n*sizeof(T));

      X.resize(L);
      for (int i=0; i<L; i++) {
        X[i].resize(layers[i+1], n);
        file.read((char*)X[i].data(), layers[i+1]*n*sizeof(T));
      }

      C.resize(L, n);
      for (int j=0; j<n; j++) { // Matlab 2D cell (in column major)
        for (int i=0; i<L; i++) {
          C(i,j).resize(layers[L], layers[i+1]);
          file.read((char*)C(i,j).data(), layers[i+1]*layers[L]*sizeof(T));
        }
      }

      // read the extra number as a sanity check
      T sum;
      file.read((char*)&sum, sizeof(T));
      std::cout<<"Sanity check: "<<sum<<std::endl;
      file.close();
    }

    Data<T> get_column(size_t idx) {
      assert(idx >= 0 && idx < N); 
      // output
      int dout = layers[L];
      Data<T> col(dout, n); // one column of the Jacobian

      // find layer and offset
      size_t l, offset;
      find_layer_and_offset(idx, l, offset);
      size_t ni = layers[l-1];
      size_t no = layers[l];

      // y = v*x + b where
      //  v: no x ni
      //  x: ni x n
      //  b: no x n
      size_t y_row; // index of the row

      // compute y
      if (offset < no*ni) { // b = 0
        // y = v*x + b = v
        // find the position of 1.0 in v
        size_t v_row = offset % no;
        size_t v_col = offset / no;

        // index of the nonzero row 
        y_row = v_row;

        // y is the v_col-th row in the input
        if (l > 1) {
          for (int i=0; i<n; i++) {
            //y(i, 0) = X[l-2](v_col, i);
            T* col_ptr = col.columndata(i);
            T* C_col_ptr = C(l-1, i).columndata(y_row);
            int one = 1;
            //saxpy_(&dout, &X[l-2](v_col, i), C_col_ptr, &one, col_ptr, &one);
            xaxpy( dout, &X[l-2](v_col, i), C_col_ptr, 1, col_ptr, 1 );
          }
        } else {
          for (int i=0; i<n; i++) {
            //y(i, 0) = x(v_col, i);
            T* col_ptr = col.columndata(i);
            T* C_col_ptr = C(l-1, i).columndata(y_row);
            int one = 1;
            //saxpy_(&dout, &x(v_col, i), C_col_ptr, &one, col_ptr, &one);
            xaxpy( dout, &x(v_col, i), C_col_ptr, 1, col_ptr, 1 );
          }
        }

      } else { // v = 0
        // y = v*x + b = b
        // index of the nonzero row
        y_row = offset - no*ni;

        for (int i=0; i<n; i++) {
          T* col_ptr = col.columndata(i);
          T* C_col_ptr = C(l-1, i).columndata(y_row);
          memcpy(col_ptr, C_col_ptr, dout*sizeof(T));
        }
      }

      return col;
    }

  private:

    void find_layer_and_offset(size_t idx, size_t &layer, size_t &offset) {
      layer = 1;
      size_t start = 0;
      // get input/output sizes
      size_t ni = layers[0];
      size_t no = layers[1];
      // find the right layer
      while (idx >= start + no*ni + no) {
        start += no*ni + no; // begining of next layer
        layer += 1; // go to next layer
        ni = layers[layer-1];
        no = layers[layer];
      }
      offset = idx - start;
    }

  private:
    int L; // # layers

    int N; // # parameters

    int n; // batch size

    vector<int> layers; // layer sizes

    Data<T> x; // input data

    vector<Data<T>> X; // activations

    Data<Data<T>> C; // precomputation, 2D cell of matrices
};


namespace hmlp
{

template<typename T>
class GNHessian : public VirtualMatrix<T>
{
  public:

    GNHessian() {
#ifdef DEBUG
      enable_signal_handler();
#endif
      this->shift = 10.0; // the default regularization
      // init timers
      this->time_total = 0.;
      this->time_jacobian = 0.;
      this->time_gemm = 0.;
      this->time_regularize = 0.;
    }

   ~GNHessian() {
      std::cout<<"\n-------- Profile Gauss-Newton computation ----------\n";
      std::cout<<"\t total time: "<<time_total<<std::endl;
      std::cout<<"\t jacobian computation: "<<time_jacobian<<std::endl;
      std::cout<<"\t GEMM: "<<time_gemm<<std::endl;
      std::cout<<"\t regularization: "<<time_regularize<<std::endl;
      std::cout<<"\n-------- End of Profile Gauss-Newton ----------\n";
    }

    /** ESSENTIAL: this is an abstract function  */
    virtual T operator()( size_t i, size_t j )
    {
      Data<T> KIJ = (*this)( vector<size_t>( 1, i ), vector<size_t>( 1, j ) );
      return KIJ[ 0 ];
    };

    /** ESSENTIAL: return a submatrix */
    virtual Data<T> operator()(const vector<size_t> &I, const vector<size_t> &J)
    {
      double t1 = timer();
      Data<T> KIJ( I.size(), J.size());
      Data<T> A = JacobianCols( I );

      Data<T> B(this->nRows, J.size());
      // The following assumes I and J are ordered
      auto p = I.begin();
      for (size_t k=0; k<J.size(); k++) {
        // find k in I
    	p = std::find(p, I.end(), k);
    	if (p != I.end()) {
	      memcpy( B.columndata(k), A.columndata(*p), nRows*sizeof(T) );
    	} else {
	      Data<T> col = JacobianOneColumn(J[k]);
    	  memcpy( B.columndata(k), col.data(), nRows*sizeof(T) );
    	}
      }

#pragma omp atomic
      this->time_jacobian += timer() - t1;      

      /** KIJ = A^{T}B */
      double t2 = timer();
      xgemm( "T", "N", KIJ.row(), KIJ.col(), B.row(), 
          1.0, A.data(), A.row(), 
               B.data(), B.row(), 
          0.0, KIJ.data(), KIJ.row() );
#pragma omp atomic
      this->time_gemm += timer() - t2;

      // add regularization
      double t3 = timer();
      std::vector<size_t> DI, DJ;
      find_diagonal_indices(I, J, DI, DJ);
      assert(DI.size() == DJ.size());
      for (size_t k=0; k<DI.size(); k++)
    	KIJ( DI[k], DJ[k] ) += this->shift;

#pragma omp atomic
      this->time_regularize += timer() - t3;
#pragma omp atomic
      this->time_total += timer() - t1;
      return KIJ;
    };
    
    void read_jacobian(std::string filename, bool type=false) {
        this->jacobian_compressed = type;
        if (jacobian_compressed == true) {
            CJac.read_file(filename);
            this->nRows = CJac.rows();
            this->nCols = CJac.cols();
        } else {
            Jac.read_file(filename);
            this->nRows = Jac.rows();
            this->nCols = Jac.cols();
        }
        VirtualMatrix<T>::resize(nRows, nRows);
    }

    void show_jacobian() {
      int N = std::min(nCols, 5);
      vector<size_t> I(N);
      for (size_t i=0; i<N; i++)
        I[i] = i;
      Data<T> J = JacobianCols(I);
      for (size_t i=0; i<N; i++) {
    	std::cout<<"column: "<<i<<std::endl;
        for (size_t j=0; j<this->nRows; j++)
	      std::cout<<J(j,i)<<" ";
    	std::cout<<std::endl;
      }
    }

    void show_hessian() {
      int N = std::min(nCols, 5);
      vector<size_t> I(N);
      for (size_t i=0; i<N; i++)
        I[i] = i;
      Data<T> H = (*this)(I, I);
      for (size_t i=0; i<N; i++) {
	std::cout<<"column: "<<i<<std::endl;
        for (size_t j=0; j<N; j++)
	  std::cout<<H(j,i)<<" ";
	std::cout<<std::endl;
      }
    }
    
  private:

    Data<T> JacobianCols(const vector<size_t> &I) {
      Data<T> cols(this->nRows, I.size());
      for (size_t i=0; i<I.size(); i++) {
        Data<T> col = JacobianOneColumn(I[i]); // col is a dout x n matrix
    	T* colptr = cols.columndata(i);
	    memcpy(colptr, col.data(), nRows*sizeof(T));
      }
      return cols;
    }

    Data<T> JacobianOneColumn(size_t idx) {
        if (jacobian_compressed == true)
            return CJac.get_column( idx );
        else
            return Jac.get_column( idx );
    }
        
    void find_diagonal_indices
    (const std::vector<size_t> &I, const std::vector<size_t> &J,
     std::vector<size_t> &DI, std::vector<size_t> &DJ) {
      // assume I and j are ordered 
      size_t i = 0, j = 0; // pointers at I and J
      while (i < I.size() && j < J.size()) {
        if (I[i] == J[j]) {
          DI.push_back(i);
          DJ.push_back(j);
          // go to next entries in I and J
          i++;
          j++;
        } else if (I[i] > J[j]) 
          j++;
        else
          i++;
      }
    }

  private:

    bool jacobian_compressed;

    CompressedJacobian<T> CJac;
    Jacobian<T> Jac;

    int nRows; // dout * batch_size
    int nCols; // # parameter

    T shift; // regularization

    double time_total;
    double time_jacobian;
    double time_gemm;
    double time_regularize;
};

}; /** end namespace hmlp */


// signal handler
static void segfault_freeze(int signal) {
  assert(signal == SIGSEGV ||
	 signal == SIGABRT ||
	 signal == SIGFPE );
  int process_id = getpid();
  char hostname[128];
  gethostname(hostname, 127);
  fprintf(stderr,"Solver process received signal %d: %s\n",
	  signal, strsignal(signal));
  fprintf(stderr,"Process %d on node %s is frozen!\n",
	  process_id, hostname);
  fflush(stderr);
  while(true)
    sleep(1);
}

void enable_signal_handler() {
  signal(SIGSEGV, segfault_freeze); // segfault
  signal(SIGABRT, segfault_freeze); // abort
  signal(SIGFPE,  segfault_freeze); // floating point exception
  std::cout<<"*********************************"<<std::endl;
  std::cout<<"**** Singal handler enabled! ****"<<std::endl;
  std::cout<<"*********************************"<<std::endl;
}

#endif /** define GNHESSIAN_HPP */
