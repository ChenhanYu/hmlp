#!/bin/bash
export HMLP_DIR=$PWD
echo "HMLP_DIR = $HMLP_DIR"

## Manually set the target architecture.
#export HMLP_ARCH_MAJOR=gpu
#export HMLP_ARCH_MINOR=kepler

# export HMLP_ARCH_MAJOR=x86_64
# export HMLP_ARCH_MINOR=sandybridge
# export HMLP_ARCH_MINOR=haswell

export HMLP_ARCH_MAJOR=mic
export HMLP_ARCH_MINOR=knl

export HMLP_ARCH=$HMLP_ARCH_MAJOR/$HMLP_ARCH_MINOR
echo "HMLP_ARCH = $HMLP_ARCH"

## Compiler options (if false, then use GNU compilers)
export HMLP_USE_INTEL=true
echo "HMLP_USE_INTEL = $HMLP_USE_INTEL"

## Whether use BLAS or not?
export HMLP_USE_BLAS=true
echo "HMLP_USE_BLAS = $HMLP_USE_BLAS"

## Whether use VML or not? (only if you have MKL)
export HMLP_USE_VML=true
echo "HMLP_USE_VML = $HMLP_USE_VML"

## Compile with KNL -xMIC-AVX512
export HMLP_MIC_AVX512=true

## Manually set the mkl path
export HMLP_MKL_DIR=$TACC_MKL_DIR
# export HMLP_MKL_DIR=/opt/intel/mkl
echo "HMLP_MKL_DIR = $HMLP_MKL_DIR"

## Parallel options
#export KMP_AFFINITY=compact
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=68
export KS_IC_NT=68
export KS_JR_NT=1
