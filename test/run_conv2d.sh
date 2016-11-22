w0=256
h0=256
d0=3

wmin=3
wmax=11
winc=2
d1=96

echo 'cond2d = ['
for (( w=wmin; w<wmax; w+=winc ))
do
  ./test_conv2d.x $w0 $h0 $d0 $w $w $d1
done
echo '];'
