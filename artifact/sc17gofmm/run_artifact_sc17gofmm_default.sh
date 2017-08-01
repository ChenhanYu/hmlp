## all SPD matrix files stored in dense column major format
declare -a filearray=(
"/workspace/biros/sc17/data_to_use_65K/K02N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K03N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K04N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K05N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K06N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K07N65536.bin"
)

## data files stored in dense d-by-N format
points="/workspace/biros/sc17/data_to_use_65K/XKEN65536.points.bin"
## data dimension
d=6
## Gaussian kernel bandwidth
h=1.0

## problem size
n=65536
## maximum leaf node size
m=512
## maximum off-diagonal ranks
s=512
## number of neighbors
k=32
## number of right hand sides
nrhs=512
## user tolerance
stol=1E-3
## user computation budget
budget=0.03
## distance type (geometry, kernel, angle)
distance="angle"
## spdmatrix type (testsuit, dense, kernel, userdefine)
matrixtype="dense"

# ======= Do not change anything below this line ========
mpiexec=""
executable=./artifact_sc17gofmm.x
echo "@PRIM"
echo 'artifact_sc17gofmm'
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
		$mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $filename; status=$?
		echo "@STATUS"
		echo $status
	done
fi

if [[ "$matrixtype" == "kernel" ]] ; then
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $points $d $h; status=$?
  echo "@STATUS"
  echo $status
fi
# =======================================================
