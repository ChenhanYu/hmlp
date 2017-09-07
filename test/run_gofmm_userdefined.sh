## all SPD matrix files stored in dense column major format
declare -a filearray=(
"/workspace/biros/sc17/data_to_use_65K/K02N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K03N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K04N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K05N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K06N65536.bin"
"/workspace/biros/sc17/data_to_use_65K/K07N65536.bin"
)

## all data files stored in dense d-by-N format
declare -a dataarray=(
"covetype100k"
)
## problem size
n=5000
## maximum leaf node size
m=64
## maximum off-diagonal ranks
s=64
## number of neighbors
k=0
## number of right hand sides
nrhs=512
## user tolerance
stol=1E-5
## user computation budget
budget=0.00
## distance type (geometry, kernel, angle)
distance="angle"
## spdmatrix type (testsuit, dense, kernel, userdefine)
matrixtype="testsuit"

# ======= Do not change anything below this line ========
if [ "${HMLP_USE_MPI}" = true ]; 
then mpiexec="mpirun -n 4"; 
else mpiexec=""; 
fi
if [ "${HMLP_USE_MPI}" = true ]; 
then executable="./test_mpigofmm.x"
else executable="./test_gofmm.x";
fi
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
# =======================================================
