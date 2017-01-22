n=5000
s=64
nrhs=128
kmin=8
kmax=64
kinc=16
# =======================================================

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
echo "s = $s"
echo "@SETIP"
echo "nrhs = $nrhs"
# =======================================================

echo "@DATE"
date
# =======================================================


for (( k=kmin; k<kmax; k+=kinc ))
do
  echo "@DATA"
  ./test_spdaskit.x $n $k $s $nrhs; status=$?
  echo "@STATUS"
  echo $status
done
# =======================================================
