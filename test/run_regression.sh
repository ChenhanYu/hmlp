
d=81
n=8182
h=0.35
niter=300
points="/Users/chenhan/Documents/Projects/opencv_practice/build/dataset/all_features.dat"
labels="/Users/chenhan/Documents/Projects/opencv_practice/build/dataset/all_labels.dat"

# =======================================================

echo "@PRIM"
echo 'regression'
# =======================================================

echo "@SETUP"
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "@SETUP"
echo "HMLP_ARCH = $HMLP_ARCH"
echo "@SETUP"
echo "d = $d"
echo "@SETUP"
echo "n = $n"
# =======================================================

echo "@DATE"
date
# =======================================================


./test_regression.x $d $n $h $niter $points $labels; status=$?
echo "@STATUS"
echo $status
# =======================================================
