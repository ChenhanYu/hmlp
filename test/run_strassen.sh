m=4096
n=4096
kmin=64
kmax=2048
kinc=64

#echo 'gemm = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_strassen.x $m $n $k
done
#echo '];'

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
