w0=64
h0=64
d0=3
s=1
p=1
batchSize=32
wmin=3
wmax=11
winc=2
d1=96
model='AlexNet'
# =======================================================

echo "@PRIM"
echo 'conv2d'
# =======================================================

echo "@SETUP"
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "@SETUP"
echo "HMLP_ARCH = $HMLP_ARCH"
echo "@SETUP"
echo "model = $model"
# =======================================================

echo "@DATE"
date
# =======================================================



# echo 'cond2d = ['
# for (( w=wmin; w<wmax; w+=winc ))
# do
#   ./test_conv2d.x $w0 $h0 $d0 $s $p $batchSize $w $w $d1
# done
# echo '];'



# AlexNet CONV1
echo "@DATA"
./test_conv2d.x 225 225   3 4 3 $batchSize 11 11  96; status=$?
echo "@STATUS"
echo $status

# AlexNet CONV2
echo "@DATA"
./test_conv2d.x  55  55  96 2 1 $batchSize  5  5 256; status=$?
echo "@STATUS"
echo $status

# AlexNet CONV3
echo "@DATA"
./test_conv2d.x  27  27 256 2 0 $batchSize  3  3 384; status=$?
echo "@STATUS"
echo $status

# AlexNet CONV4
echo "@DATA"
./test_conv2d.x  13  13 384 1 1 $batchSize  3  3 384; status=$?
echo "@STATUS"
echo $status

# AlexNet CONV5
echo "@DATA"
./test_conv2d.x  13  13 384 1 1 $batchSize  3  3 256; status=$?
echo "@STATUS"
echo $status
