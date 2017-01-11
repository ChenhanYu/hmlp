m=4096
n=4096
kmin=64
kmax=500
kinc=32
# =======================================================

echo "@PRIM"
echo 'strassen'
# =======================================================

echo "@SETUP"
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "@SETUP"
echo "HMLP_ARCH = $HMLP_ARCH"
echo "@SETUP"
echo "m = $m"
echo "@SETUP"
echo "n = $n"
# =======================================================

echo "@DATE"
date
# =======================================================

for (( k=kmin; k<kmax; k+=kinc ))
do
  echo "@DATA"
  ./test_strassen.x $m $n $k; status=$?
  echo "@STATUS"
  echo $status
done
# =======================================================




#k=300
#nmin=3
#nmax=2047
#ninc=31
#
#echo 'gemm = ['
#for (( n=nmin; n<nmax; n+=ninc ))
#do
#  ./test_strassen.x $n $n $k
#done
#echo '];'
