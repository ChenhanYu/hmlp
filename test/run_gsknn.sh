#!/bin/bash

m=4097
n=4097
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
