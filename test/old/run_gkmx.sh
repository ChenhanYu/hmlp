m=2047
n=2047
kmin=4
kmax=2048
kinc=31

echo 'gemm = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_gkmx.x $m $n $k
done
echo '];'

k=300
nmin=3
nmax=2047
ninc=31

echo 'gemm = ['
for (( n=nmin; n<nmax; n+=ninc ))
do
  ./test_gkmx.x $n $n $k
done
echo '];'
