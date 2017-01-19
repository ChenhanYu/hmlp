#!/bin/bash
export HMLP_DIR=$PWD
echo "HMLP_DIR = $HMLP_DIR"

## Manually set the target architecture.
export HMLP_GPU_ARCH_MAJOR=gpu
export HMLP_GPU_ARCH_MINOR=kepler

# export HMLP_ARCH_MAJOR=arm
# export HMLP_ARCH_MINOR=armv8a

export HMLP_ARCH_MAJOR=x86_64
export HMLP_ARCH_MINOR=sandybridge

# export HMLP_ARCH_MAJOR=x86_64
# export HMLP_ARCH_MINOR=haswell

# export HMLP_ARCH_MAJOR=mic
# export HMLP_ARCH_MINOR=knl

export HMLP_GPU_ARCH=$HMLP_GPU_ARCH_MAJOR/$HMLP_GPU_ARCH_MINOR
export HMLP_ARCH=$HMLP_ARCH_MAJOR/$HMLP_ARCH_MINOR
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "HMLP_ARCH = $HMLP_ARCH"

## Compiler options (if false, then use GNU compilers)
export HMLP_USE_INTEL=true
echo "HMLP_USE_INTEL = $HMLP_USE_INTEL"

## GPU compiler options (if true, compile the gpu library as well)
export HMLP_USE_CUDA=false
echo "HMLP_USE_CUDA = $HMLP_USE_CUDA"

## Manually setup CUDA TOOLKIT path (otherwise cmake will try to find it)
export HMLP_CUDA_DIR=$TACC_CUDA_DIR
echo "HMLP_CUDA_DIR = $HMLP_CUDA_DIR"


## Whether use BLAS or not?
export HMLP_USE_BLAS=true
echo "HMLP_USE_BLAS = $HMLP_USE_BLAS"

## Whether use VML or not? (only if you have MKL)
export HMLP_USE_VML=true
echo "HMLP_USE_VML = $HMLP_USE_VML"

## Compile with KNL -xMIC-AVX512
export HMLP_MIC_AVX512=false

## Manually set the mkl path
# export HMLP_MKL_DIR=$TACC_MKL_DIR
export HMLP_MKL_DIR=/opt/intel/mkl
echo "HMLP_MKL_DIR = $HMLP_MKL_DIR"


## Manually set the mkl path
export HMLP_QSML_DIR=/Users/chenhan/Documents/Projects/qsml/aarch64-linux-android
echo "HMLP_QSML_DIR = $HMLP_QSML_DIR"


## Parallel options
# export KMP_AFFINITY=compact
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=4
export KS_JC_NT=1
export KS_IC_NT=20
export KS_JR_NT=1
