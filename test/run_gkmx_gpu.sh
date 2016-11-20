m=513
n=513
b=10
kmin=4
kmax=2048
kinc=31

echo 'gemm = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_gkmm_gpu.x $m $n $k $b
done
echo '];'

k=300
b=10
nmin=3
nmax=2047
ninc=31

echo 'gemm = ['
for (( n=nmin; n<nmax; n+=ninc ))
do
  ./test_gkmm_gpu.x $n $n $k $b
done
echo '];'
