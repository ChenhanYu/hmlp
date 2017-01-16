m=2048
n=2048
b=10
kmin=4
kmax=2048
kinc=31
# =======================================================



echo "@PRIM"
echo 'gkmm_gpu'
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
echo "b = $b"
# =======================================================

echo "@DATE"
date
# =======================================================

for (( k=kmin; k<kmax; k+=kinc ))
do
  echo "@DATA"
  ./test_gkmm_gpu.x $m $n $k $b; status=$?
  echo "@STATUS"
  echo $status
  #./test_strassen_gpu.x $m $n $k $b
done
# =======================================================
