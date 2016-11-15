m=4096
n=4096
r=128
kmin=4
kmax=600
kinc=31

echo 'KNN = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_gsknn.x $m $n $k $r
done
echo '];'
