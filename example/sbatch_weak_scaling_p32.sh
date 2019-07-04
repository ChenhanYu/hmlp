#!/bin/bash
#SBATCH -A PADAS
#SBATCH -J GOFMM_WEAK_SCALING
#SBATCH -p skx-normal
#SBATCH -t 00:30:00
#SBATCH -n 32
#SBATCH -N 32
#SBATCH -o weak_scaling_p32.out

export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export OMP_NUM_THREADS=48
ulimit -Hc unlimited
ulimit -Sc unlimited

## problem size
n=4194304
## data files stored in dense d-by-N format
points="/work/02794/ych/data/XKEN${n}.points.bin"
## data dimension
d=6
## Gaussian kernel bandwidth
h=0.3
## maximum leaf node size
m=512
## maximum off-diagonal ranks
s=256
## number of neighbors
k=64
## number of right hand sides
nrhs=64
## user tolerance
stol=1E-5
## user computation budget
budget=0.00
## distance type (geometry, kernel, angle)
distance="angle"
## spdmatrix type (testsuit, dense, ooc, mlp, kernel, jacobian, userdefine)
matrixtype="kernel"
## kernelmatrix type (gaussian, laplace)
kerneltype="gaussian"

# ======= Do not change anything below this line ========
mpiexec="ibrun tacc_affinity"
executable="./test_mpigofmm"

echo "@PRIM"
echo 'gofmm'
# =======================================================

echo "@SETUP"
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "@SETUP"
echo "HMLP_ARCH = $HMLP_ARCH"
echo "@SETUP"
echo "n = $n"
echo "@SETUP"
echo "m = $m"
echo "@SETUP"
echo "s = $s"
echo "@SETUP"
echo "k = $k"
echo "@SETUP"
echo "nrhs = $nrhs"
echo "@SETUP"
echo "stol = $stol"
echo "@SETUP"
echo "budget = $budget"
echo "@SETUP"
echo "distance = $distance"
echo "@SETUP"
echo "matrixtype = $matrixtype"
# =======================================================

echo "@DATE"
date
# =======================================================
if [[ "$matrixtype" == "kernel" ]] ; then
  echo $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $kerneltype $points $d $h
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $kerneltype $points $d $h; status=$?
  echo "@STATUS"
  echo $status
fi
# =======================================================
