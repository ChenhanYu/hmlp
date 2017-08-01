
## HMLP (High Performance Machine Learning Primitives)

### README

Thank you for deciding to give HMLP a try!
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




### INSTALL

HMLP is tested on LINUX and OSX. Compilation **REQUIRES:**

Intel or GNU compilers with c++11, AVX and OpenMP support (for x86_64);

Arm GNU compilers (see [Cross Compilation]() for details on compilation on Android) with OpenMP support (for arm);

Intel-16 or later compilers (for Intel MIC, KNL);

nvcc (for NVIDIA GPU with capability > 2.0).


**Configuration:**

Edit set_env.sh for compilation options.

You MUST manually setup each environment variable in the "REQUIRED"
CONGIFURATION if any of those variables were not defined properly
on you system.


```
export CC             = icc    to use Intel C compilers
export CXX            = icpc   to use Intel C++ compilers
export CC             = gcc    to use GNU C compilers
export CXX            = g++    to use GNU C++ compilers
export HMLP_USE_BLAS  = false  if you don't have a BLAS library.
export MKLROOT        = xxx    to_the_path_of_intel_mkl
export OPENBLASROOT   = xxx    to_the_path_of_OpenBLAS
Set HMLP_USE_CUDA  = true  to compile code with cuda templates. 
```

The default BLAS library for Intel compiler is MKL. For GNU compilers, cmake
will try to find a proper BLAS/LAPACK library on you system. If cmake fails
to locate BLAS/LAPACK, then the compilation will fail as well. You need to
manually setup the path in the cmake file.


**Installation:**

Use cmake:

```{r, engine='bash', count_lines}
>source set_env.sh
>mkdir build
>cd build
>cmake ..
>make
>make install
```

**Cross Compilation:**

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


```{r, engine='bash', count_lines}
>adb devices
>adb push /hmlp/build/bin/* /data/local/tmp
>adb shell
>cd /data/local/tmp
>./run_hmlp.sh
>./run_gkmx.sh
```

**Testing and compilation example:**

The default compilation will also compile all the test drivers.
To run some basic examples from the testing drivers:

```{r, engine='bash', count_lines}
>cd /build/bin
>./run_hmlp.sh
>./run_gkmx.sh
```

To us HMLP library you need to include the
header files <hmlp.h> 
and link HMLP statically wich is in ${HMLP_DIR}/build/lib/libhmlp.a.

C/C++ example:

```c++
...
#include <hmlp.h>
...
```

Compilation example:
```{r, engine='bash', count_lines}
icc ... -I$(HMLP_DIR)/build/include $(HMLP_DIR)/build/lib/libhmlp.a
```

<figure class="video_container">
<iframe jsname="L5Fo6c" class="YMEQtf KfXz0b" frameborder="0"
aria-label="Chart, GSKS&lt;ADD,MUL,DOUBLE&gt;"
src="https://docs.google.com/spreadsheets/d/e/2CAIWO3elj2q1iA2x7PfXyiwRjScxokvUFH4Etki1iAyJR1PUgMpzjaiFUFtnZBAraCGhd8H0ARHGX2fYlcw/gviz/chartiframe?authuser=0&amp;autosize=true&amp;oid=21359979"
allowfullscreen></iframe>
</figure>




## LICENSE
```bash
The HMLP library is licensed under the following license, typically
known as the GPL-3.0 license.

Copyright (C) 2014-2017, The University of Texas at Austin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
```


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

