declare -a filearray=(
"K02N65536.bin"
"K15N65536.bin"
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
# =======================================================
for filename in "${filearray[@]}"
do
  echo $filename
  $mpiexec $executable $n $m $k $s $nrhs $stol $budget $filename; status=$?
  echo "@STATUS"
  echo $status
done
# =======================================================
