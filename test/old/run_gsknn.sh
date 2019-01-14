m=4096
n=4096
r=128
kmin=4
kmax=500
kinc=31
# =======================================================

echo "@PRIM"
echo 'gsknn'
# =======================================================

echo "@SETUP"
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "@SETUP"
echo "HMLP_ARCH = $HMLP_ARCH"
echo "@SETUP"
echo "m = $m"
echo "@SETUP"
echo "n = $n"
echo "@SETIP"
echo "r = $r"
# =======================================================

echo "@DATE"
date
# =======================================================


for (( k=kmin; k<kmax; k+=kinc ))
do
  echo "@DATA"
  ./test_gsknn.x $m $n $k $r; status=$?
  echo "@STATUS"
  echo $status
done
# =======================================================
