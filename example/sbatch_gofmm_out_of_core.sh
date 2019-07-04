#!/bin/bash
#SBATCH -A PADAS
#SBATCH -J GOFMM
#SBATCH -o gofmm_output.out
#SBATCH -p skx-dev
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH -N 1

export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export OMP_NUM_THREADS=48
ulimit -Hc unlimited
ulimit -Sc unlimited

## all SPD matrix files stored in dense column major format
declare -a filearray=(
"K02N1024.bin"
)

## problem size
n=1024
## maximum leaf node size
m=64
## maximum off-diagonal ranks
s=64
## number of neighbors
k=64
## number of right hand sides
nrhs=32
## user tolerance
stol=1E-3
## user computation budget
budget=0.01
## distance type (geometry, kernel, angle)
distance="angle"
## spdmatrix type (testsuit, dense, ooc, mlp, kernel, jacobian, userdefine)
matrixtype="ooc"

# ======= Do not change anything below this line ========
mpiexec="ibrun tacc_affinity"
#executable="./test_mpigofmm"
executable="./test_gofmm"

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

for filename in "${filearray[@]}"
do
  echo $filename
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $filename; status=$?
  echo "@STATUS"
  echo $status
done
# =======================================================
