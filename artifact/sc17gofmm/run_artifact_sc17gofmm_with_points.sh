## all SPD matrix files stored in dense column major format
declare -a filearray=(
"/workspace/biros/sc17/data_to_use_65K/K02N65536.bin"
)


## data points stored in dense d-by-N format
points="/workspace/biros/sc17/data_to_use_65K/X2DN65536.points.bin"
## data dimension
d=2


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
distance="geometry"
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

if [[ "$matrixtype" == "dense" ]] ; then
	for filename in "${filearray[@]}"
	do
		echo $filename
		$mpiexec $executable $n $m $k $s $nrhs $stol $budget $distance $matrixtype $filename $points $d; status=$?
		echo "@STATUS"
		echo $status
	done
fi
# =======================================================
