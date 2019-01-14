m=4096
n=576
kmin=64
kmax=500
kinc=31
kernel="Gaussian"
# =======================================================

echo "@PRIM"
echo 'gsks'
# =======================================================

echo "@SETUP"
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "@SETUP"
echo "HMLP_ARCH = $HMLP_ARCH"
echo "@SETUP"
echo "m = $m"
echo "@SETUP"
echo "n = $n"
echo "@SETUP"
echo "kernel = $kernel"
# =======================================================

echo "@DATE"
date
# =======================================================

for (( k=kmin, i=0; k<kmax; k+=kinc, i+=1 ))
do
  echo "@DATA"
  ./test_gsks.x $kernel $m $n $k; status=$?
  echo "@STATUS"
  echo $status
done
# =======================================================
