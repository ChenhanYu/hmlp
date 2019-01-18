#!/bin/bash
#SBATCH -A PADAS
#SBATCH -J GOFMM
#SBATCH -o gofmm_output.out
#SBATCH -p skx-dev
#SBATCH -t 00:20:00
#SBATCH -n 4
#SBATCH -N 4

export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export OMP_NUM_THREADS=48
ulimit -Hc unlimited
ulimit -Sc unlimited

## all SPD matrix files stored in dense column major format
declare -a filearray=(
"datasets/K02N4096.bin"
"datasets/K03N4096.bin"
"datasets/K04N4096.bin"
"datasets/K05N4096.bin"
"datasets/K06N4096.bin"
"datasets/K07N4096.bin"
)

## data files stored in dense d-by-N format
points="datasets/X2DN4096.points.bin"
## data dimension
d=2
## Gaussian kernel bandwidth
h=1.0


## problem size
n=4096
## maximum leaf node size
m=64
## maximum off-diagonal ranks
s=64
## number of neighbors
k=32
## number of right hand sides
nrhs=512
## user tolerance
stol=1E-5
## user computation budget
budget=0.01
## distance type (geometry, kernel, angle)
distance="angle"
## spdmatrix type (testsuit, dense, ooc, kernel, userdefine)
matrixtype="testsuit"
## kernelmatrix type (gaussian, laplace)
kerneltype="gaussian"

# ======= Do not change anything below this line ========
mpiexec="ibrun tacc_affinity"
executable="./test_mpigofmm.x"
#executable="gdb -ex run --args ./test_mpigofmm.x"
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

if [[ "$matrixtype" == "testsuit" ]] ; then
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype status=$?
  echo "@STATUS"
  echo $status
fi

if [[ "$matrixtype" == "dense" ]] ; then
	for filename in "${filearray[@]}"
	do
		echo $filename
		$mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $filename; status=$?
		echo "@STATUS"
		echo $status
	done
fi

if [[ "$matrixtype" == "ooc" ]] ; then
	for filename in "${filearray[@]}"
	do
		echo $filename
		$mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $filename; status=$?
		echo "@STATUS"
		echo $status
	done
fi

if [[ "$matrixtype" == "kernel" ]] ; then
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $kerneltype $points $d $h; status=$?
  echo "@STATUS"
  echo $status
fi
# =======================================================
