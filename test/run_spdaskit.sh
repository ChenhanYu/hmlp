n=5000
m=128
s=128
stol=1E-4
budget=0.03
nrhs=128
kmin=3
kmax=4
kinc=16
# ======= Do not change anything below this line ========

echo "@PRIM"
echo 'spdaskit'
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
echo "stol = $stol"
echo "@SETUP"
echo "budget = $budget"
echo "@SETUP"
echo "nrhs = $nrhs"
echo "@SETUP"
echo "kmin = $kmin"
echo "@SETUP"
echo "kmax = $kmax"
echo "@SETUP"
echo "kinc = $kinc"
# =======================================================

echo "@DATE"
date
# =======================================================


for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_spdaskit.x $n $m $k $s $nrhs $stol $budget; status=$?
  echo "@STATUS"
  echo $status
done
# =======================================================
