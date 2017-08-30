

d=5
n=100000
ncluster=4
points="/workspace/chenhan/datasets/ball4.n100000.d5.trn.X.bin"
labels="/workspace/chenhan/datasets/ball4.n100000.d5.trn.Y.bin"

#d=2
#n=100008
#ncluster=9
#points="/workspace/chenhan/datasets/cluster9.n100008.d2.trn.X.bin"
#labels="/workspace/chenhan/datasets/cluster9.n100008.d2.trn.Y.bin"

#d=2
#n=5004
#ncluster=9
#points="/workspace/chenhan/datasets/cluster9.n5004.d2.trn.X.bin"
#labels="/workspace/chenhan/datasets/cluster9.n5004.d2.trn.Y.bin"


niter=20
tol=1E-3
# =======================================================

echo "@PRIM"
echo 'cluster'
# =======================================================

echo "@SETUP"
echo "HMLP_GPU_ARCH = $HMLP_GPU_ARCH"
echo "@SETUP"
echo "HMLP_ARCH = $HMLP_ARCH"
echo "@SETUP"
echo "d = $d"
echo "@SETUP"
echo "n = $n"
echo "@SETIP"
echo "ncluster = $ncluster"
# =======================================================

echo "@DATE"
date
# =======================================================


./test_cluster.x $d $n $ncluster $niter $tol $points $labels $ncluster; status=$?
echo "@STATUS"
echo $status
# =======================================================
