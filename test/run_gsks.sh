m=3600
n=4097
kmin=4
kmax=500
kinc=31

#echo 'Gaussian = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_gsks.x Gaussian $m $n $k
done
#echo '];'

#echo 'Variable bandwidth = ['
#for (( k=kmin; k<kmax; k+=kinc ))
#do
#  ./test_gsks.x Var_bandwidth $m $n $k
#done
#echo '];'
