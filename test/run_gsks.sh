m=3600
n=4097
kmin=4
kmax=2048
kinc=31

echo 'Gaussian = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_gsks.x $m $n $k
done
echo '];'