## Ensure pip ≥8.1 is installed.
easy_install -U pip
## Install Cython.
pip install Cython
## Install mpi4py
#pip install mpi4py
## Cython the setup file.
LDSHARED="cc -bundle -undefined dynamic_lookup -arch x86_64 -Wl,-F." python setup.py install
LDSHARED="cc -bundle -undefined dynamic_lookup -arch x86_64 -Wl,-F." python setup.py build_ext -i
