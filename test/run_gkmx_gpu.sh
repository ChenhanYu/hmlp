m=2048
n=2048
b=10
kmin=4
kmax=2048
kinc=31

echo 'gemm = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_gkmm_gpu.x $m $n $k $b
  ./test_strassen_gpu.x $m $n $k $b
done
echo '];'

k=1024
b=10
nmin=1024
nmax=6000
ninc=256

echo 'gemm = ['
for (( n=nmin; n<nmax; n+=ninc ))
do
  ./test_gkmm_gpu.x $n $n $k $b
  ./test_strassen_gpu.x $m $n $k $b
done
echo '];'
