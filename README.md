## HMLP (High Performance Machine Learning Primitives)

[![Branch](https://img.shields.io/badge/develop-branch-orange.svg)]()
[![Build Status](https://travis-ci.org/ChenhanYu/hmlp.svg?branch=develop)](https://travis-ci.org/ChenhanYu/hmlp)
[![codecov](https://codecov.io/gh/ChenhanYu/hmlp/branch/develop/graph/badge.svg)](https://codecov.io/gh/ChenhanYu/hmlp)
[![Inline docs](http://inch-ci.org/github/chenhanyu/hmlp.svg?branch=develop)](http://inch-ci.org/github/chenhanyu/hmlp)
[![Branch](https://img.shields.io/badge/master-branch-orange.svg)]()
[![Build Status](https://travis-ci.org/ChenhanYu/hmlp.svg?branch=master)](https://travis-ci.org/ChenhanYu/hmlp)
[![codecov](https://codecov.io/gh/ChenhanYu/hmlp/branch/master/graph/badge.svg)](https://codecov.io/gh/ChenhanYu/hmlp)

### WARNING

[HMLP](https://chenhanyu.github.io/hmlp/) and GOFMM are research projects 
and not in production. 
For SC'17 and SC'18 artifacts, see /artifact and our our GOFMM papers 
[SC'17](https://arxiv.org/pdf/1707.00164.pdf) and [SC'18](https://dl.acm.org/citation.cfm?id=3291676) for details.
For fast kernels, see our GSKNN paper [SC'15](http://padas.ices.utexas.edu/static/papers/sc15nn.pdf) for details.

### README

Thank you for deciding to give [HMLP](https://chenhanyu.github.io/hmlp/) a try!
This code can be used directly or pluged into many learning tasks such as
kernel summation, nearest neighbor search, k-means clustering, convolution
networks. It is also possible to create a specific general N-body operator by
using our template frameworks.

See [INSTALL](https://github.com/ChenhanYu/hmlp#install) on how to install it.
Checkout [LICENSE](https://github.com/ChenhanYu/hmlp#license) if you want to 
use or redistribute any part of HMLP. Notice that different parts of HMLP may
have different licenses. We usually annotate the specific license on the top
of the file.

HMLP (High Performance Machine Learning Primitives) is a portable framework 
that provides high performance, memory efficient matrix-matrix multiplication
like operations and their extension based on the BLIS framework. 
Currently, HMLP has implementations on Intel x86_64, Knights Landing, ARM and 
NVIDIA GPU. We may further support other architectures in the future.


Depending on your need, you may only require the basic features if you just
want to use some existing primitives. Acoording to the feature you require we
suggest that you read different part of documents. Please checkout the wiki
pages at:

[WIKI PAGE](https:://github.com/ChenhanYu/hmlp/wiki)

to see what feature better suit your need.

Architecture dependent
implementations (a.k.a microkernels or kernels in short) are identified and
saperated from the c++ loop base framework. Thus, porting any HMLP primitive
to an new architecture **only require** rewriting the kernel part. 
You are welcome to contribute more kernels beyond this list. Checkout the
guildline on implementing microkernel for HMLP at our wiki pages.

### CITATION
```
@inproceedings{yu2015performance,
  title={Performance optimization for the k-nearest neighbors kernel on x86 architectures},
  author={Yu, Chenhan D and Huang, Jianyu and Austin, Woody and Xiao, Bo and Biros, George},
  booktitle={Proceedings of the International Conference for High Performance Computing,
    Networking, Storage and Analysis},
  pages={7},
  year={2015},
  organization={ACM}
}
@inproceedings{yu2017geometry,
  title={Geometry-oblivious FMM for compressing dense SPD matrices},
  author={Yu, Chenhan D and Levitt, James and Reiz, Severin and Biros, George},
  booktitle={Proceedings of the International Conference for High Performance Computing,
    Networking, Storage and Analysis},
  pages={53},
  year={2017},
  organization={ACM}
}
@inproceedings{yu2018distributed,
  title={Distributed-memory hierarchical compression of dense SPD matrices},
  author={Yu, Chenhan D and Reiz, Severin and Biros, George},
  booktitle={Proceedings of the International Conference for High Performance Computing,
    Networking, Storage, and Analysis},
  pages={15},
  year={2018},
  organization={IEEE Press}
}
```

### DOCUMENTATION
GOFMM, MPI-GOFMM, HMLP templates, and HMLP runtime APIs are documented
by doxygen:
  * https://chenhanyu.github.io/hmlp/docs/html


## INSTALL

HMLP is tested on LINUX and OSX. Compilation **REQUIRES:**
  * Intel or GNU compilers with c++11, AVX and OpenMP support (for x86_64);
  * Arm GNU compilers (see [Cross Compilation]() for details on compilation on Android) with OpenMP support (for arm);
  * Intel-16 or later compilers (for Intel MIC, KNL);
  * nvcc (for NVIDIA GPU with capability > 2.0).


### Configuration

Edit set_env.sh for compilation options.

You MUST manually setup each environment variable in the **"REQUIRED"
CONGIFURATION** if any of those variables were not defined properly
on you system.


```
export CC             = icc    to use Intel C compilers
export CXX            = icpc   to use Intel C++ compilers
export CC             = gcc    to use GNU C compilers
export CXX            = g++    to use GNU C++ compilers
export HMLP_USE_BLAS  = false  if you don't have a BLAS library.
export MKLROOT        = xxx    to_the_path_of_intel_mkl
export OPENBLASROOT   = xxx    to_the_path_of_OpenBLAS
set HMLP_USE_CUDA     = true   to compile code with cuda templates. 
```

The default BLAS library for Intel compiler is MKL. For GNU compilers, cmake
will try to find a proper BLAS/LAPACK library on you system. If cmake fails
to locate BLAS/LAPACK, then the compilation will fail as well. You need to
manually setup the path in the cmake file.

### Cmake installation

```
source set_env.sh
mkdir build
cd build
cmake ..
make
make install
```

### Cross Compilation

If your Arm is run with OS that has native compiler and cmake support, then the
installation instruction above should work just fine.
However, while your target runs an Android OS, which currently does not have a native
C/C++ compiler, you will need to cross compile HMLP on your Linux or OSX first.
Although there are many ways to do cross compilation, we suggest that users
follow these instructions:

1. Install Android Studio with LLDB, cmake and NDK support.
2. Create stand-alone-toolchain from NDK.
3. Install adb (Android Debug Bridge)
4. Compile HMLP with cmake. It will look for your arm gcc/g++, ar and ranlib
   support.
5. Use the following instructions to push executables and scripts in hmlp/build/bin.


```
adb devices
adb push /hmlp/build/bin/* /data/local/tmp
adb shell
cd /data/local/tmp
./run_hmlp.sh
./run_gkmx.sh
```

### EXAMPLE

The default compilation will also compile all the examples in /example.
To run some basic examples from the testing drivers:

```
cd /build
./run_hmlp.sh
./run_gkmx.sh
```

To us [HMLP](https://chenhanyu.github.io/hmlp/) library you need to include the
header files <hmlp.h> and link 
${HMLP_DIR}/build/libhmlp.a statically or 
${HMLP_DIR}/build/libhmlp.so (or .dylib) dynamically.

C/C++ example:

```c++
...
#include <hmlp.h>
...
```

Static linking example:
```
icc ... -I$(HMLP_DIR)/build/include $(HMLP_DIR)/build/libhmlp.a
```

Dynamic linking example on Linux:
```
icc ... -I$(HMLP_DIR)/build/include -L$(HMLP_DIR)/build/libhmlp.so
```

Dynamic linking example on Mac:
```
icc ... -I$(HMLP_DIR)/build/include -L$(HMLP_DIR)/build/libhmlp.dylib
```

### TESTING

Following the steps in [INSTALL](https://github.com/ChenhanYu/hmlp#install)
using cmake, Google Test will be downloaded and
compiled. All testing routines located in /test will be compiled.
All executables locate in /build. To perform the whole test suits,
follow these instructions.
```
cd build
make test
```
Optionally, you can perform and coverage analysis and update the report by
```
make coverage
```

The latest coverage report can be found at
  * https://chenhanyu.github.io/hmlp/docs/html/coverage.html
  * https://chenhanyu.github.io/hmlp/docs/html/coverage_details.html


## ACKNOWLEDGEMENTS
```
The HMLP library was primarily authored by

  Chenhan D. Yu             (The University of Texas at Austin)

but many others have contributed input and feedback. Contributors
are listed acoording to the part they contribute to:


  GOFMM (Geometry-Oblivious FMM) in SC'17
  
  James Levitt              (The University of Texas at Austin)
  Severin Reiz              (Technische Universität München)
  George Biros              (The University of Texas at Austin)


  General Stride K-Nearest Neighbors in SC'15

  Jianyu Huang              (The University of Texas at Austin)
  Woody Austin              (The University of Texas at Austin)
  George Biros              (The University of Texas at Austin)


  General Stride Kernel Summation in IPDPS'15

  Bill March                (The University of Texas at Austin)
  George Biros              (The University of Texas at Austin)


  2-level Strassen in SC'16

  Jianyu Huang              (The University of Texas at Austin)
  Leslie Rice               (The University of Texas at Austin)


  HMLP on Arm

  Jianyu Huang              (The University of Texas at Austin)
  Matthew Badin             (Qualcomm Corp. Santa Clara)


  BLIS framework support

  Tyler Smith               (The University of Texas at Austin)
  Robert van de Geijn       (The University of Texas at Austin)
  Field Van Zee             (The University of Texas at Austin)
  

The gratitude especially goes to the following individual who walks
me through the whole BLIS framework.

  Field Van Zee             (The University of Texas at Austin)
```


Thank you again for being intersted in HMLP!

Best regards,

Chenhan D. Yu --- chenhan@cs.utexas.edu

