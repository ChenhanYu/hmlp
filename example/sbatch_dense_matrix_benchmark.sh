#!/bin/bash
#SBATCH -A PADAS
#SBATCH -J GOFMM_DENSE_BENCHMARK
#SBATCH -o dense_matrix_benchmark_0.03.out
#SBATCH -p skx-dev
#SBATCH -t 01:00:00
#SBATCH -n 4
#SBATCH -N 4

export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export OMP_NUM_THREADS=48
ulimit -Hc unlimited
ulimit -Sc unlimited

## all SPD matrix files stored in dense column major format
declare -a filearray=(
"/work/02794/ych/data/K02N65536.bin"
"/work/02794/ych/data/K12N65536.bin"
"/work/02794/ych/data/K13N65536.bin"
"/work/02794/ych/data/K14N65536.bin"
"/work/02794/ych/data/K15N65536.bin"
"/work/02794/ych/data/data_to_use_graphs/G03N65536.bin"
)

## problem size
n=65536
## maximum leaf node size
m=512
## maximum off-diagonal ranks
s=1024
## number of neighbors
k=128
## number of right hand sides
nrhs=128
## user tolerance
stol=1E-5
## user computation budget
budget=0.03
## distance type (geometry, kernel, angle)
distance="angle"
## spdmatrix type (testsuit, dense, ooc, mlp, kernel, jacobian, userdefine)
matrixtype="dense"

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
if [[ "$matrixtype" == "dense" ]] ; then
  for filename in "${filearray[@]}"
  do
    echo $filename
    $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $filename; status=$?
    echo "@STATUS"
    echo $status
  done
fi
# =======================================================
