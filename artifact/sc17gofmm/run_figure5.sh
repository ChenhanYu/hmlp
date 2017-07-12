declare -a filearray=(
"K02N65536.bin"
"K03N65536.bin"
"K04N65536.bin"
"K05N65536.bin"
"K06N65536.bin"
"K07N65536.bin"
"K08N65536.bin"
"K09N65536.bin"
"K10N65536.bin"
"K11N65536.bin"
"K12N65536.bin"
"K13N65536.bin"
"K14N65536.bin"
"K15N65536.bin"
"K16N65536.bin"
"K17N65536.bin"
"K18N65536.bin"
"G01N65536.bin"
"G02N65536.bin"
"G03N65536.bin"
"G04N65536.bin"
"G05N65536.bin"
)
n=65536
m=512
s=512
k=32
stol=1E-2
budget=0.03
nrhs=512
if [ -z ${HMLP_USE_MPI+x} ]; 
then mpiexec="mpirun -n 2"; 
else mpiexec=""; 
fi
# ======= Do not change anything below this line ========
executable=./test_gofmm.x
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
echo "stol = $stol"
echo "@SETUP"
echo "budget = $budget"
echo "@SETUP"
echo "nrhs = $nrhs"
# =======================================================

echo "@DATE"
date
# =======================================================

for filename in "${filearray[@]}"
do
  echo $filename
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $filename; status=$?
  echo "@STATUS"
  echo $status
done
# =======================================================
