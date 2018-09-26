#!/bin/bash

## "REQUIRED" CONFIGURATION
## ======================================

## Make sure CC and CXX are set properly in your system.
export CC=icc
export CXX=icpc
#export CC=${CC}
#export CXX=${CXX}

##
## Ingore this flag if you are "not using" MacOS.
##
## If you are using MacOS, exporting CC and CXX are useless.
## CC and CXX are two executable files in /usr/bin, and you
## cannot overwrite them without super users.
##
## Notice that HMLP "does not" support clang compilers. 
## You "must" set following flag to true to use Intel or
## flase to use GNU compilers.
##
export HMLP_USE_INTEL=true

## Whether use BLAS or not?
export HMLP_USE_BLAS=true

## Make sure MKLROOT is defined in your system. (icc/icpc)
export MKLROOT=/opt/intel/mkl
export MKLROOT=${MKLROOT}

## Make sure OPENBLASROOT is defined. (gcc/g++)
export OPENBLASROOT=${OPENBLASROOT}

## Setup the maximum number of threads.
export OMP_NUM_THREADS=2



## ARTIFACT FOR REPRODUCIABILITY
## ======================================

## If you also want to compile those artifact files, then specify the path.
export HMLP_ARTIFACT_PATH=sc18gofmm



## ADVANCE OPTIONS
## ======================================

## Manually set the target architecture.
export HMLP_GPU_ARCH_MAJOR=gpu
export HMLP_GPU_ARCH_MINOR=kepler

## (1) x86_64/sandybridge, 
## (2) x86_64/haswell, 
## (3) arm/armv8a
## (4) mic/knl
export HMLP_ARCH_MAJOR=x86_64
#export HMLP_ARCH_MINOR=sandybridge
export HMLP_ARCH_MINOR=haswell
#export HMLP_ARCH_MAJOR=mic
#export HMLP_ARCH_MINOR=knl

## Manually set the QSML path if you are using arm/armv8a architecture.
export QSMLROOT=/Users/chenhan/Documents/Projects/qsml/aarch64-linux-android

## Distributed environment poptions (if true, compile with MPI)
export HMLP_USE_MPI=false

## GPU compiler options (if true, compile the gpu library as well).
## Manually setup CUDA TOOLKIT path (otherwise cmake will try to find it).
export HMLP_USE_CUDA=false
export HMLP_CUDA_DIR=$TACC_CUDA_DIR

## MAGMA (GPU LAPACK support)
export HMLP_USE_MAGMA=false
export HMLP_MAGMA_DIR=/users/chenhan/Projects/magma-2.2.0

## Output google site data
export HMLP_ANALYSIS_DATA=false

## Build a independent sandbox
export HMLP_BUILD_SANDBOX=true

## Decide whether to compile the runtime system or not
export HMLP_HAVE_RUNTIME=true
if [ ! -f $PWD/frame/base/hmlp_runtime.cpp ]; then
  echo "Disable HMLP runtime system"
  export HMLP_HAVE_RUNTIME=false
fi


## Advance OpenMP options
export OMP_NESTED=false
export OMP_PROC_BIND=spread

## HMLP communicator
export KS_JC_NT=1
export KS_PC_NT=1
export KS_IC_NT=$OMP_NUM_THREADS
export KS_JR_NT=1



## DO NOT CHANGE ANYTHING BELOW THIS LINE
## ======================================

## Check if CC and CXX are set
echo "===================================================================="
echo "Notice: HMLP and CMAKE use variables CC and CXX to decide compilers."
echo "        If the following messages pop out:"
echo ""
echo "            Variable CC  is unset (REQUIRED) or"
echo "            Variable CXX is unset (REQUIRED),"
echo ""
echo "        then you must first export these two variables."
echo "===================================================================="
if [ -z ${CC+x} ]; 
then echo "Variable CC  is unset (REQUIRED)"; 
else echo "Variable CC  is set to '$CC'";
fi
if [ -z ${CXX+x} ]; 
then echo "Variable CXX is unset (REQUIRED)"; 
else echo "Variable CXX is set to '$CXX'";
fi
echo "===================================================================="
echo ""

## Check if MKLROOT is set
echo "===================================================================="
echo "Notice: HMLP and CMAKE use variables MKLROOT to find Intel MKL."
echo "        If you are using intel compile and seeing the following:"
echo ""
echo "            Variable MKLROOT is unset (REQUIRED by intel compilers)"
echo ""
echo "        then you must first export MKLROOT=/path_to_mkl..."
echo "===================================================================="
if [ -z ${MKLROOT+x} ]; 
then echo "Variable MKLROOT is unset (REQUIRED by intel compilers)"; 
else echo "Variable MKLROOT is set to '$MKLROOT'";
fi
echo "===================================================================="
echo ""


## Check if OPENBLASROOT is set
echo "===================================================================="
echo "Notice: HMLP and CMAKE use variables OPENBLASROOT to find OpenBLAS."
echo "        If you are using intel compile and seeing the following:"
echo ""
echo "            Variable OPENBLASROOT is unset (REQUIRED by GNU compilers)"
echo ""
echo "        then you must first export OPENBLASROOT=/path_to_OpenBLAS..."
echo "===================================================================="
if [ -z ${OPENBLASROOT+x} ]; 
then echo "Variable OPENBLASROOT is unset (REQUIRED by GNU compilers)"; 
else echo "Variable OPENBLASROOT is set to '$OPENBLASROOT'";
fi
echo "===================================================================="
echo ""




## setup project source directory
export HMLP_DIR=$PWD
echo "HMLP_DIR = $HMLP_DIR"

## Add our default building path  
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HMLP_DIR}/build/lib/
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${MKLROOT}/lib/
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${OPENBLASROOT}/:${OPENBLASROOT}/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HMLP_DIR}/build/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MKLROOT}/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENBLASROOT}/:${OPENBLASROOT}/lib/

## architecture
export HMLP_GPU_ARCH=$HMLP_GPU_ARCH_MAJOR/$HMLP_GPU_ARCH_MINOR
export HMLP_ARCH=$HMLP_ARCH_MAJOR/$HMLP_ARCH_MINOR
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "HMLP_ARCH = $HMLP_ARCH"

## Compiler options (if false, then use GNU compilers)
echo "HMLP_USE_INTEL = $HMLP_USE_INTEL"

## use blas?
echo "HMLP_USE_BLAS = $HMLP_USE_BLAS"

## Manually set the QSML path if you are using arm/armv8a architecture.
echo "QSMLROOT = $QSMLROOT"

## Distributed environment poptions (if true, compile with MPI)
echo "HMLP_USE_MPI = $HMLP_USE_MPI"

## GPU compiler options (if true, compile the gpu library as well)
echo "HMLP_USE_CUDA = $HMLP_USE_CUDA"

## Manually setup CUDA TOOLKIT path (otherwise cmake will try to find it)
echo "HMLP_CUDA_DIR = $HMLP_CUDA_DIR"

## MAGMA (GPU LAPACK support)
echo "HMLP_USE_MAGMA = $HMLP_USE_MAGMA"
echo "HMLP_MAGMA_DIR = $HMLP_MAGMA_DIR"

## Output google site data
echo "HMLP_ANALYSIS_DATA = $HMLP_ANALYSIS_DATA"

## OpenMP options
echo "OMP_PROC_BIND = $OMP_PROC_BIND"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "OMP_PLACES = $OMP_PLACES"

## HMLP communicator
echo "KS_JC_NT = $KS_JC_NT"
echo "KS_IC_NT = $KS_IC_NT"
echo "KS_JR_NT = $KS_JR_NT"

