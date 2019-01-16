#!/bin/bash
#SBATCH -A PADAS
#SBATCH -J GOFMM
#SBATCH -o unitTest.out
#SBATCH -p skx-dev
#SBATCH -t 00:10:00
#SBATCH -n 1
#SBATCH -N 1

export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export OMP_NUM_THREADS=48
ulimit -Hc unlimited
ulimit -Sc unlimited

# ======= Do not change anything below this line ========
#mpiexec="prun"
mpiexec="ibrun tacc_affinity"
#executable="gdb -ex run --args ./unitTest"
executable="./unitTest"

echo "@DATE"
date
# =======================================================
$mpiexec $executable status=$?
echo "@STATUS"
echo $status
