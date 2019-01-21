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
"/scratch/04647/sreiz/H01N102400.bin"
"/scratch/04647/sreiz/K04N102400.bin"
"/scratch/04647/sreiz/K07N102400.bin"
"/scratch/04647/sreiz/K11N102400.bin"
"/scratch/04647/sreiz/K12N102400.bin"
)
#declare -a filearray=(
#"/scratch/02794/ych/K01N262144.bin"
#)

## data files stored in dense d-by-N format
#points="/work/02794/ych/data/X2DN1048576.points.bin"
#points="/work/02794/ych/data/X3DN2097152.points.bin"
points="/work/02794/ych/data/covtype.100k.trn.X.bin"
#points="/workspace/chenhan/data/covtype.100k.trn.X.bin"
#points="/work/02794/ych/data/covtype.100k.trn.X.bin"
#points="/work/02794/ych/data/covtype.n500000.d54.trn.X.bin"
## data dimension
d=54
## Gaussian kernel bandwidth
h=1.0


## problem size
#n=65536
#n=147456
#n=262144
n=100000
#n=102400
#n=1048576
#n=2097152
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
budget=0.0
## distance type (geometry, kernel, angle)
#distance="kernel"
distance="angle"
#distance="geometry"
## spdmatrix type (testsuit, dense, ooc, mlp, kernel, userdefine)
#matrixtype="dense"
#matrixtype="ooc"
#matrixtype="mlp"
matrixtype="kernel"
#matrixtype="testsuit"
#matrixtype="pvfmm"
## kernelmatrix type (gaussian, laplace)
kerneltype="gaussian"
#kerneltype="laplace"
## hidden layer configuration (512-512-512)
hiddenlayer="512-512-512"


# ======= Do not change anything below this line ========
mpiexec="ibrun tacc_affinity"
#mpiexec="prun"
#executable="./test_mpigofmm"
#executable="gdb -ex run --args ./test_mpigofmm.x"
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

if [[ "$matrixtype" == "mlp" ]] ; then
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $hiddenlayer $points $d $h; status=$?
  echo "@STATUS"
  echo $status
fi

if [[ "$matrixtype" == "kernel" ]] ; then
  echo $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $kerneltype $points $d $h
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $kerneltype $points $d $h; status=$?
  echo "@STATUS"
  echo $status
fi

if [[ "$matrixtype" == "pvfmm" ]] ; then
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype; status=$?
  echo "@STATUS"
  echo $status
fi
# =======================================================
