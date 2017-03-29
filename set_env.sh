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

## MAGMA (GPU LAPACK support)
export HMLP_USE_MAGMA=false
echo "HMLP_USE_MAGMA = $HMLP_USE_MAGMA"

export HMLP_MAGMA_DIR=/users/chenhan/Projects/magma-2.2.0
echo "HMLP_MAGMA_DIR = $HMLP_MAGMA_DIR"

## Whether use BLAS or not?
export HMLP_USE_BLAS=true
echo "HMLP_USE_BLAS = $HMLP_USE_BLAS"

## Whether use VML or not? (only if you have MKL)
export HMLP_USE_VML=true
echo "HMLP_USE_VML = $HMLP_USE_VML"

## Compile with KNL -xMIC-AVX512
export HMLP_MIC_AVX512=false

## Manually set the mkl path
#export HMLP_MKL_DIR=$MKLROOT
export HMLP_MKL_DIR=$TACC_MKL_DIR
# export HMLP_MKL_DIR=/opt/intel/mkl
# export HMLP_MKL_DIR=/opt/apps/sysnet/intel/16/mkl
echo "HMLP_MKL_DIR = $HMLP_MKL_DIR"

## Manually set the qsml path
export HMLP_QSML_DIR=/Users/chenhan/Documents/Projects/qsml/aarch64-linux-android
echo "HMLP_QSML_DIR = $HMLP_QSML_DIR"

## Output google site data
export HMLP_ANALYSIS_DATA=false

echo "HMLP_ANALYSIS_DATA = $HMLP_ANALYSIS_DATA"



## Parallel options
export OMP_NESTED=false
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=24
export OMP_PLACES=
#export OMP_PLACES="{0},{2},{4},{6},{8},{10}"
#export OMP_PLACES="{0},{1},{2},{3},{4},{5}"
#export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}"
export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23}"
#export OMP_PLACES="{0},{4},{8},{12},{16},{20},{24},{28},{32},{36},{1},{5},{9},{13},{17},{21},{25},{29},{33},{37}"
echo "OMP_PROC_BIND = $OMP_PROC_BIND"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "OMP_PLACES = $OMP_PLACES"

## HMLP communicator
export KS_JC_NT=1
export KS_IC_NT=20
export KS_JR_NT=1
echo "KS_JC_NT = $KS_JC_NT"
echo "KS_IC_NT = $KS_IC_NT"
echo "KS_JR_NT = $KS_JR_NT"
