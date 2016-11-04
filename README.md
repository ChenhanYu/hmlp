
## HMLP (High Performance Machine Learning Primitives)

### README

**HMLP currently is not an open source project. Do not distribute!!!**

Thank you for deciding to give HMLP a try!
This code can be used directly or pluged into many learning tasks such as
kernel summation, nearest neighbor search, k-means clustering, convolution
networks. It is also possible to create a specific general N-body operator by
using our template frameworks.

See [INSTALL](https://github.com/ChenhanYu/hmlp#install) on how to install it.
Checkout [LICENSE](https://github.com/ChenhanYu/hmlp#license) if you want to 
use or redistribute any part of HMLP. Notice that different parts of HMLP may
have different license. We usually annotate the specific license on the top
of the file.

HMLP (High Performance Machine Learning Primitives) is a portable framework 
that provides a high performance memory efficient kernel summation based
on the BLIS framework.

HMLP has several features. For further details of this project, please
check the wiki pages at:

https:://github.com/ChenhanYu/hmlp/wiki



### INSTALL

HMLP is tested on LINUX and OSX. Compilation REQUIRES:

Intel or GNU compilers with, c++11, AVX and OpenMP support (x86_64);

Intel-16 or later compilers (Intel MIC, KNL);

nvcc (NVIDIA GPU).


**Configuration:**

Edit set_env.sh for compilation options.

```
Set HMLP_USE_INTEL = true  to use Intel compilers.
Set HMLP_USE_INTEL = false to use GNU compilers.
Set HMLP_USE_CUDA  = true  to compile code with cuda templates. 
Set HMLP_USE_BLAS  = false if you don't have a BLAS library.
Set HMLP_USE_VNL   = true  to activate Intel VML.
```

The default BLAS library for Intel compiler is MKL.

The default BLAS for GNU is Netlib (-lblas). 

The default BLAS for nvcc is CUBLAS.


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


## LICENSE
```bash
The HMLP library is licensed under the following license, typically
known as the GPL-3.0 license.

Copyright (C) 2014-2016, The University of Texas at Austin

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

but many others have contributed input and feedback, including:

  Woody Austin              (The University of Texas at Austin)
  Matthew Badin             (Qualcomm Corp. Santa Clara)
  George Biros              (The University of Texas at Austin)
  Cris Cecka                (NVIDIA Corp. Santa Clara)
  Michael Garland           (NVIDIA Corp.)
  Jianyu Huang              (The University of Texas at Austin)
  Bill March                (The University of Texas at Austin)
  Tyler Smith               (The University of Texas at Austin)
  Robert van de Geijn       (The University of Texas at Austin)
  Field Van Zee             (The University of Texas at Austin)

The gratitude especially goes to the following individual who walks
me through the whole BLIS framework.

  Field Van Zee             (The University of Texas at Austin)
```


Thank you again for being intersted in HMLP!

Best regards,

Chenhan D. Yu

chenhan@cs.utexas.edu

