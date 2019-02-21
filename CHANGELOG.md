# Changelog


## (unreleased)

### Other

* Now secure_accuracy works for both shared and distributed memory GOFMM. [Chenhan Yu]

  However, the solver will not work. Currently the HSS solver assume that

  the low-rank factors exist on all tree nodes.

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan Yu]

* Update #2. [Chenhan Yu]

* Update. [Chenhan Yu]

* More error handle and modify set_env.sh. [Chenhan D. Yu]

* Check in nearest neighbor and fast matvec and solver unit tests. [Chenhan D. Yu]

* Prepare to merge. [Chenhan Yu]

* Now early termination in ANN has been disable. [Chenhan Yu]

* Bug fixed for the leaf node size. Will deprecate members in tree:Steup since they are duplicated with Configuration. [Chenhan Yu]

  Instead of using a fixed range to decide the middle
  of the median split, we now gradually increase the
  left and right quantiles.

  The left and right quantiles increase 10% every time
  if they ended up being the same as the median.

* Fix a configuration issue. [Chenhan Yu]

* Improving Skeletonize and id. [Chenhan Yu]

* Move to Stampede2 for MPI testing. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Update Evaluate with error handle. [Chenhan D. Yu]

* Enhance blas/lapack finding. [Chenhan D. Yu]

* Fix util.hpp. [Chenhan D. Yu]

* Now by default, we secure the accuracy (using level restriction). The relative error is also fixed to deal with zero denometer. [Chenhan D. Yu]

* Finalize badges. [Chenhan D. Yu]

* Update and to merge with master. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Visual badges from two branches. [Chenhan D. Yu]

* Change coverage comment. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Change the behavior of make coverage. [Chenhan D. Yu]

* Install gcovr. [Chenhan D. Yu]

* Try coverage. [Chenhan D. Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Try mpi. [Chenhan Yu]

* Update. [Chenhan Yu]

* Fix unitTest. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Updarte. [Chenhan D. Yu]

* Update cmake. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Update. [Chenhan D. Yu]

* Try again. [Chenhan D. Yu]

* Try switch to -coverage. [Chenhan D. Yu]

* Update the default netlib blas and lapack paths. [Chenhan Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan Yu]

* Update README with the badge. [Chenhan D. Yu]

* Try to see if blas/lapack work? [Chenhan Yu]

* Try CI on travis-CI. [Chenhan Yu]

* Deprecate. [Chenhan Yu]

  -export HMLP_NORMAL_WORKER=11
  -export HMLP_SERVER_WORKER=10
  -export HMLP_NESTED_WORKER=10
  -export KS_IC_NT=20
  -export GSKNN_IC_NT=20

* Update README and document html. [Chenhan Yu]

* Update documentation html. [Chenhan Yu]

* Update README. [Chenhan Yu]

* Update with doxygen. [Chenhan Yu]

* Improve error handling. [Chenhan Yu]

* Now gtest works on MPI functions as well. See test/unitTest.cpp and test/unit/runtime.hpp for an example. [Chenhan Yu]

* Update READMe. [Chenhan Yu]

* Update README. [Chenhan Yu]

* Set theme jekyll-theme-slate. [Chenhan D. Yu]

* Add pages. [Chenhan D. Yu]

* Set theme jekyll-theme-slate. [Chenhan D. Yu]

* Merge branch 'develop' [Chenhan Yu]

* Change /doc to /docs for Github pages. [Chenhan Yu]

* Gtest is now working. gcovr is working with GCC/G++ [Chenhan Yu]

* Prepare the branch for gtest. CMakefile has been improved. [Chenhan D. Yu]

* Make sampling consistent with evaluation. [James Levitt]

  Use near nodes to decide which row samples to exclude from sampling instead of
  pruning neighbors.

* Rename, restructur, and prepare error handling infurstructure. [Chenhan Yu]

* Update Readme. [Chenhan Yu]

* Fixed typo and added a comment in custom_kernel.cpp. [George Biros]

* A new example. [Chenhan Yu]

* Complete all requests from George. [Chenhan Yu]

* Find a bug in distributed knn. [Chenhan Yu]

* Prepare to update master and release. [Chenhan Yu]

* Fix gcc compilation errors. [Chenhan Yu]

* Fix some gcc compilation issues. [Chenhan Yu]

* Resolve the compilation problem related to the HAVE_RUNTIME option. [Chenhan Yu]

* More gofmm examples. [Chenhan D. Yu]

* Some python interface implementation. [Chenhan D. Yu]

* Create examples for GOFMM. [Chenhan Yu]

* For SC'18 artifact. [Chenhan Yu]

* New artifact for SC'18. [Chenhan Yu]

* To simplify interaction lists. [Chenhan Yu]

* Bug fixed. [Chenhan Yu]

* Update environment setup for MacOS. [Chenhan D. Yu]

* Prepare for release. [Chenhan Yu]

* Restructure and migrate to MPIObject. [Chenhan Yu]

* Clean up. [Chenhan Yu]

* All distances have been moved to VirtualMatrix.hpp. [Chenhan Yu]

* Prepare to move different distances into KIJ interfaces. [Chenhan Yu]

* Multiple touches. [Chenhan Yu]

* Clean up. [Chenhan D. Yu]

* Prepare to remove all lids. [Chenhan D. Yu]

* Clean up. [Chenhan Yu]

* Simplify several types and templates requirement. [Chenhan D. Yu]

* Clean up ULV factorization. [Chenhan Yu]

* Now add another version ULV factorization. [Chenhan Yu]

* Prepare for cleanup. [Chenhan Yu]

* Fix ComputeError. [Chenhan Yu]

* Reimplement nested tasks. [Chenhan Yu]

* Change LEVELOFFSET from 4 to 5. [Chenhan Yu]

  This takes care trees up to 32 levels

* Need to increase message number a bit. [Chenhan Yu]

* Some instructions for MLPGaussNewton. [Chenhan Yu]

* Add MLPGaussNewton. [Chenhan Yu]

* Major commit. [Chenhan Yu]

  Complete distributed async evaluation

* Pass Gaussian and Laplace 3D test on Stampede2. [Chenhan Yu]

* Clean up and move to Maverick. [Chenhan Yu]

* Complete new DistKernelMatrix interface. [Chenhan Yu]

* Now the result of distributed GOFMM with NN pruning is correct. [Chenhan Yu]

* Move to stampede2. [Chenhan D. Yu]

* Symmetry Near and Far interactions. [Chenhan Yu]

* Test distributed morton2rank. [Chenhan D. Yu]

* Update gnbx primitive. [Chenhan D. Yu]

* Update gofmm. [Chenhan Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan Yu]

* Add 12x16 skx kernel. [Chenhan D. Yu]

* Prepare for 12x16 skx kernel. [Chenhan Yu]

* Add gsks kernel for skylake. [Chenhan D. Yu]

* Add Allgather and Allgatherv support, now every MPI processes have the full copy of MortonIDs to facilitate neighbor pruning. [Chenhan Yu]

* Testing skx kernels. [Chenhan Yu]

* Add skylake kernels. [Chenhan D. Yu]

* Fix gpu compilation issues. [Chenhan D. Yu]

* Merge branch 'develop' of github.com:ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Need to fix some problem on GPUs. [Chenhan D. Yu]

* Clean up for sc17. [Chenhan D. Yu]

* Switch to two-factor skeletonization: GETMTX and SKEL. [Chenhan D. Yu]

* Complete new gnbx interface. [Chenhan D. Yu]

* Complete new gnbx. [Chenhan D. Yu]

* Change microkernel signiture. [Chenhan D. Yu]

* About to modify the micro-kernel interface. [Chenhan D. Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Now we swtich to a new communicator interface. [Chenhan D. Yu]

* Slightly change to mpi. [Chenhan D. Yu]

* May need to enforce users to provide important samples. [Chenhan D. Yu]

* Fix MPI compilation issues for -HMLP_USE_MPI=false. [Chenhan D. Yu]

* Clean Up. [Chenhan D. Yu]

* Implement batched KIJ evaluation. [Chenhan Yu]

* Add a utility executable to convert dataset into binary format. [Chenhan D. Yu]

* Fix a caching bug. [Chenhan D. Yu]

* Merge branch 'develop' of github.com:ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Include the improved pvfmm. [Chenhan D. Yu]

* Collect async and bsp results. [Chenhan D. Yu]

* Fix some indencenting problem. [Chenhan D. Yu]

* Improve ann and tree partitioning by introducing RedistributeWithPartner. [Chenhan Yu]

* Free KIJ for sapces. [Chenhan D. Yu]

* Use ls5 for scaling experiements. [Chenhan D. Yu]

* With PVFMM support. [Chenhan D. Yu]

* Remove some printf. Take care Dhairya's code in the next commit. [Chenhan D. Yu]

* Slightly improve the evaluation phase. [Chenhan D. Yu]

* Use omp_nested. [Chenhan D. Yu]

* Several optimization. [Chenhan D. Yu]

* Solvinf several compilation issues on Stampede2. [Chenhan Yu]

* Scaling experiements on Maverick. [Chenhan D. Yu]

* Now we can use hmlp_set_num_background_worker to control the amount of threads we devote. By default, we use omp_get_max_threads() / 4 + 1 workers for background during the compression and we don't need any background worker. [Chenhan D. Yu]

  Now we also use task-orientied GEMM if there is no enough
  parallelism during the tree traversal. Nested tasks will be
  created above level 2 (local tree).

* Now forward and backward permute phases of the distributed evaluation has been combined with redistribution from <RBLK,STAR> to <RIDS,STAR> [Chenhan D. Yu]

* Started to use View<T> for implicit submatrix representation. [Chenhan D. Yu]

* Adding documents for DistVirtualMatrix.hpp. [Chenhan D. Yu]

* Update the documentation for DistVirtualMatrix.hpp. [Chenhan D. Yu]

* Now we allow multiple workers to execute the background task. Reqeust for K( I, j ) a single column will use exchanging by values instead of exchaning by coordinates. [Chenhan D. Yu]

* Distributed ANN is completed, important sampling has been implemented. [Chenhan D. Yu]

* Finish RKDT, need to rewrite the KNNTask to reduce the frequency on calling K(I,J) [Chenhan D. Yu]

* Implement neighbor merging (generating snids) and important sampling. [Chenhan D. Yu]

* Update timer, need to deal with morton later. [Chenhan D. Yu]

* Update the ices setup for Severin. [Chenhan D. Yu]

* Fix a bug in DistKernelMatrix. [Chenhan D. Yu]

* Fix several bugs. Now nonblocking consensus is corret. [Chenhan D. Yu]

* Currently nonblocking consensus is too slow. In the next commit, I need to implement a software cache. [Chenhan D. Yu]

* Now we use mpitree::Setup and mpigofmm::Setup. [Chenhan D. Yu]

  In the next commit, we need to modify how we store and access
  morton[].

* DistKernelMatrix is completed. Now accessing K( i, j ) is implemented using nonblocking consensus. [Chenhan D. Yu]

* Test for MPI_TAG_UB. [Chenhan D. Yu]

* Move back to pele. [Chenhan D. Yu]

* Now TreePartition uses runtime system. [Chenhan D. Yu]

* The bug is fixed. The problem was the random seed on different MPI processes may be different such that K is actually different from MPI to MPI. Now, there is an exact Bcast to make sure K is the same on all MPI processes. [Chenhan D. Yu]

* Need to find the bug. [Chenhan D. Yu]

* Ls5 is too crowd, move to maverick. [Chenhan D. Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Implement server code for nonblocking consensus. [Chenhan D. Yu]

* Implement DistributedKernelMatrix. [Chenhan D. Yu]

  Notice that we may beed to slightly change the algorithm for
  the centersplit and randomsplit in the distributed setup.

  that is for a tree node that owns gids, we need to create
  the split using

  K( gids, samples of gids )

* Add a sbatch file for TACC systems. [Chenhan D. Yu]

* Creating HMLP sandbox. [Chenhan D. Yu]

* Add gnbx as an extension of gkmx that takes storage and packing types. [Chenhan D. Yu]

* Now distributed centersplit and randomsplit inherit the shared-memory version. [Chenhan D. Yu]

* Move to ls5. [Chenhan D. Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Simplify centersplit. [Chenhan D. Yu]

* Complete DistData<STAR,CIDS,T> and DistData<STAR,CBLK,T>. [Chenhan D. Yu]

* Clean up of the primitives. [Chenhan D. Yu]

* Now potentials (output) and weights (input) in Evaluate()   use DistData<RIDS, STAR, T> instead of Data<T>. [Chenhan D. Yu]

* Finish RBLK to and back RIDS redistribution. [Chenhan D. Yu]

* Add kernel softmax regression and ridge regression. [Chenhan D. Yu]

* Resolve compilation issue with no mpi. [Chenhan D. Yu]

* Clean up for Severin. [Chenhan D. Yu]

* Complete a simeple distributed version where K, u, w have global access. [Chenhan D. Yu]

* Not gofmm::Evaluate( tree, weitghs ) takes an n-by-nrhs column major array and output an n-by-nrhs column major array. [Chenhan D. Yu]

* The bug is fixed. In the next version, I will change the Evluation interface to take a matrix view. [Chenhan D. Yu]

* Complete direct KernelKmeans and Spectral clustering using Lanczos. [Chenhan D. Yu]

  There is a bug on inovking gofmm::Evaluate()
    multiple times. Need to fix it in the next commit.

* Abort the feature: doesn't work. [Chenhan D. Yu]

* Spectral clustering and kernel kmeans. [Chenhan D. Yu]

* Put function and rootfinder in. [Chenhan D. Yu]

* Complete simple distributed near-far nodes (HSS). [Chenhan D. Yu]

* (tree.hpp) add a new interface for TreePartition() called by mpitree::TreePartition() [Chenhan D. Yu]

  (gofmm.hpp::Skeletonize()) we now check duplication of the diagonal block
  using Morton ID instead of std::find().

  (hmlp_runtime_mpi.hpp) Allreduce and its unitype interface

  (mpigofmm.hpp) implemented distributed centersplit (assume gids are distributed
      but K( igid, jgid ) is shared)

  (tree_mpi.hpp) implemented distributed TreePartition (gids are distributed
      but morton[ gid ] is shared )

  (combinatorics.hpp) implemented distributed Select()

* - Simplify building all executables in cmake - Fix a += bug in GOFMM centersplit (ANGLE) - Initialize n_worker and n_tasks for hmlp scheudler. [Chenhan D. Yu]

* MPI infurstructure, move Mean, Scan, Select from Tree to primitives/combinatorics. [Chenhan D. Yu]

* Complete symmetric ULV factorization and solver. [Chenhan D. Yu]

* Stampede and stampede2 have passed. [Chenhan Yu]

* Maverick has passed. [Chenhan D. Yu]

* New a set_env.sh file for TACC machines. [Chenhan D. Yu]

* Make sure the code can be compiled even without BLAS (HMLP_USE_BLAS=false, but GOFMM will not be     available). [Chenhan D. Yu]

  Make sure both icc/mkl and gcc/OpenBLAS will work.

* Update license headers in all files. [Chenhan D. Yu]

* Create nested_queue for nested tasks. [Chenhan D. Yu]

* Modify readme. [Chenhan D. Yu]

* Create LICENSE. [Chenhan D. Yu]

* Update README. [Chenhan D. Yu]

* Merge branch 'develop' [Chenhan D. Yu]

* Clean up the repo for releasing. [Chenhan D. Yu]

* Remove debug flags. [Chenhan D. Yu]

* Geqp4 has some problems while being compiled on KNL. [Chenhan Yu]

  I have to manually add all external LAPACK prototypes.

  Now I gather all Netlib prototypes in blas_lapack_prototypes.hpp

* Remove xgeqrf (deprecated) from xgeqp4. [Chenhan D. Yu]

  Fix CFLAGS and LINKER for the python interface

* For sc17 artifact, I will create only 1 executable with 2 different scripts. [Chenhan D. Yu]

  The default.sh script can be executed in 3 different modes

  testsuit: run on a small 5000-by-5000 random matrix (no dataset is required)
  dense: user provides a dense matrix
  kernel: user provides points

  If users provide the dense matrix but still want to use geometry distance, then
  they can use with_points.sh script.

* Replace geqp3 in ID to geqp4. Implement FLAME GEMM algorithms with matrix view (var1, var2 and var3) [Chenhan D. Yu]

* Add dgeqp4. [Chenhan D. Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Prepare supermatrix test suit. [Chenhan D. Yu]

* Improve CMakefile. Still need to check the results of the solver. [Chenhan D. Yu]

* Now HMLP can compile with GNU compilers and OpenBLAS. [Chenhan D. Yu]

  The results of the linear solver are strange. Need to double check.

* Add sgeqp4. [Chenhan D. Yu]

* 2 files. [Chenhan D. Yu]

* Complete VirtualMatrix.hpp, rename KernelMatrix.hpp. [Chenhan D. Yu]

* Now metric type is no longer a template argument, instead it is a regular argument. [Chenhan D. Yu]

* Enhance GOFMM testing routines. [Chenhan D. Yu]

* Simplify set_env.sh. [Chenhan D. Yu]

* Complete Python interface for GOFMM. [Chenhan D. Yu]

* Try PyHMLP. [Chenhan D. Yu]

* Now we have a simple interface for Compress and Evaluate. [Chenhan D. Yu]

  Move on to Python interface.

* Change the repo structure a bit. [Chenhan D. Yu]

* Prepare to implement FLAME GEMM and SuperMatrix. [Chenhan D. Yu]

* Complete LU factorization (this version is for p-HSS matrices) [Chenhan D. Yu]

* Now traversing FactorTask and SolveTask is completed. [Chenhan D. Yu]

  Still need to create permutation tasks.

  Also, need to fix all gid and lid for the distributed version.

* Change all spdaskit to gofmm. [Chenhan D. Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Merge branch 'develop' of https://github.com/ChenhanYu/hmlp into develop. [Chenhan D. Yu]

* Need to complete the solve function first. [Chenhan D. Yu]

* Move compress to spdaskit. [Chenhan D. Yu]

* Now our cmake can compile MPI code. [Chenhan D. Yu]

* Prepare for distributed environment. [Chenhan D. Yu]

* Implement symmetric factorization. [Chenhan D. Yu]

* Create LICENSE.md. [Chenhan D. Yu]

* Modify license. [Chenhan D. Yu]

* Create LICENSE.md. [Chenhan D. Yu]

* Delete LICENSE.md. [Chenhan D. Yu]

* Rename LICENSE to LICENSE.md. [Chenhan D. Yu]

* Forgot a file. [Chenhan D. Yu]

* Moving toward factorization. [Chenhan D. Yu]

* Interface changing. Now budget and stol are input parameters. [Chenhan D. Yu]

* Merge branch 'master' of https://github.com/ChenhanYu/hmlp. [Chenhan D. Yu]

  Conflicts:
  	test/test_spdaskit.cpp

* Fix level-by-level bug and omp task. [Chenhan D. Yu]

* Try OOC on CPU-GPU hybrid machine. [Chenhan D. Yu]

* This should be the final version. [Chenhan D. Yu]

  Now we use m=4k to perform ANN, doubling m if acc < 0.8.
  This helps reduce ANN to 1 or 2 iteration.

  Now we at least use 2m rows in ID. (to match the uniform sample condition)

  The scaled_stol is finalized.

* Large angle as neighbors. [Chenhan D. Yu]

* What ever. [Chenhan D. Yu]

* Fix ann accuracy. [Chenhan D. Yu]

  Change the way we perform angle projection and angle knn

* Change BUDGET to allow 0.1% [Chenhan D. Yu]

* Fix compilation error. [Chenhan D. Yu]

  adding a prototype for magma_sgeqp3

* Merge pull request #21 from JamesLevitt/master. [Chenhan D. Yu]

  Add point-distance partitioning and NN search

* Replace std::nth_element with std::sort. [James Levitt]

  Using std::nth_element doesn't work on lonestar

* Add point-distance partitioning and NN search. [James Levitt]

* Balance the splitters and disable parallel select. [James Levitt]

  Performance can be improved by fixing the parallel select

* Refactor the kernel matrix data structure. [James Levitt]

  Change sources/targets from Data<T> to Data<T>&

* Resolve a bug in ``en extra task'' in the compression phase. Now the worker exits runtime if. [Chenhan D. Yu]

  1. n_task >= tasklist.size()
  2. there is no task in the ready queue.

* Now always prefetch all Kab to GPU. [Chenhan D. Yu]

  remove all std::vector::clear() which may potentiallu free the preallocated
  buffers.

  The behavior of GatherColumn is also changed. This prevent the system
  from allocating new page-lock memory for w_leaf.

* NN pruning based on frequency and user defined budget. [Chenhan D. Yu]

* Remove accuracy report for ANN in the expierment. [Chenhan D. Yu]

* A huge gpu commit. [Chenhan D. Yu]

* This version produces correct results on p100. [Chenhan D. Yu]

* Add hmlp_deice.* [Chenhan D. Yu]

* GPU implementation. [Chenhan D. Yu]

* Printf statement necessary to create latex tables. [severin617]

  Need to output a few values in a collected manner to grep them.

* Add some more kernel matrices. [Chenhan D. Yu]

* Producing results. [Chenhan D. Yu]

* Start to produce performance results. [Chenhan D. Yu]

* GPU implementation. [Chenhan D. Yu]

* Fuse Skeletonization with NearNodes and CacheNearNodes. [Chenhan D. Yu]

* Fix a compilation bug. [Chenhan D. Yu]

* Performance optimization on KNL is done. [Chenhan D. Yu]

* Fix the schedule. [Chenhan D. Yu]

* Some improvment. [Chenhan D. Yu]

* Change most of the gemm NT to NN. [Chenhan D. Yu]

* Explicitly permute w with w_leaf. [Chenhan D. Yu]

* Support coordinates for all type of matrices. [Chenhan D. Yu]

* Now LeavesToLeaves are splited into 4 subtasks to increase the parallelism and decrease the effect of misschudule. [Chenhan D. Yu]

* Recording events on dummy tasks seems to result in double free bugs. I comment them out now. I will fix the flops and mops counter in the future. [Chenhan D. Yu]

  Now ANN report accuracy. To control the number of testing, use the first
  template parameter of KNNTASK.

* Fix some compilation bugs. [Chenhan D. Yu]

* We are going to separate direct evaluation from SkeletonsToNodes. This will increase the parallelism. We may further split the direct interation per leaf node into multiple tasks. [Chenhan D. Yu]

* Fix compilation bug. [Chenhan D. Yu]

* Merge branch 'master' of github.com:ChenhanYu/hmlp. [Chenhan D. Yu]

* Merge pull request #20 from ChenhanYu/profiling_coneangle_ANN. [Chenhan D. Yu]

  Profiling coneangle ann (conflicts fixed by Chenhan)

* Merge branch 'master' into profiling_coneangle_ANN. [Chenhan D. Yu]

* Cone criterion now to be set as bool in main. [severin617]

* Add profiling for ANN. [severin617]

* Fix the omp depend bug for the FMM evaluation. [Chenhan D. Yu]

  Blocked kernel evaluation has a bug; thus, we will first roll back
  to the original double-loop implementation. We will wait util James
  fix the bug.

* Merge branch 'master' of github.com:ChenhanYu/hmlp. [Chenhan D. Yu]

* Merge pull request #19 from JamesLevitt/master. [Chenhan D. Yu]

  Block evaluation of kernel submatrices

* Implement block evaluation of kernel submatrices. [James Levitt]

* A few small fixes. [James Levitt]

* Update the credit list. [Chenhan D. Yu]

* Format the code layout. Fix several compilation bug on mac ox. [Chenhan D. Yu]

* Performance analysis for the evaluation. [Chenhan D. Yu]

  1/3 to 1/2 runtime goes to Kab = K( amap, bmap );

* Update GPU implementation. [Chenhan D. Yu]

* GPU facility. [Chenhan D. Yu]

* Merge branch 'master' of github.com:ChenhanYu/hmlp. [Chenhan D. Yu]

  Conflicts:
  	frame/data.hpp

* Merge branch 'master' of https://github.com/ChenhanYu/hmlp. [Chenhan D. Yu]

* Tested on KNL. [Chenhan D. Yu]

* Prepare for cuda support. [Chenhan D. Yu]

* Merge branch 'master' of https://github.com/ChenhanYu/hmlp. [Chenhan D. Yu]

* Tested on maverick. [Chenhan D. Yu]

* Some nuts destory ls5. So I have to move to maverick. [Chenhan D. Yu]

* Omp task depend is done. Now id use trsm instead of gels to save time. [Chenhan D. Yu]

  trsm is separated into Interpolate();

* Add some timers, make some change to NearNodes. [Chenhan D. Yu]

* Fix a small bug while reading target file. [Chenhan D. Yu]

* Merge pull request #18 from JamesLevitt/master. [Chenhan D. Yu]

  Kernel matrices and some skel changes

* Implement kernel matrices. [James Levitt]

* Add option to abort skel when error exceeds stol. [James Levitt]

  When askit uses adaptive rank skeletonization with adaptive level restriction
  and the error exceeds stol, the skeletonization will abort since there is no
  need to compute the projection matrix.

* Add check for n <= maxs in fixed-rank skel. [James Levitt]

* Add several counters. [Chenhan D. Yu]

* Fix a bug found by Severin in hmlp_utils.hpp. Improving documents. We will be counting K_{ij} evaluations in the next patch. [Chenhan D. Yu]

* Implement OpenMP 4.5 tasks with dependencies. Improve our runtime scheduler. Now the runtime will perform job stealing. [Chenhan D. Yu]

* Use important samples for sparse matrices. Now we also print the statistics for skeletons. [Chenhan D. Yu]

* Improving performnace for CSC and OOC. [Chenhan D. Yu]

* Done testing on Ronaldo. [Chenhan D. Yu]

* Start to implement GPU evaluation. [Chenhan D. Yu]

* We are going to try pinned allocator for CUDA. [Chenhan D. Yu]

* Complete the modification to hmlp::Data<T> [Chenhan D. Yu]

* I'm going to change hmlp::Data<T> to a general notation. [Chenhan D. Yu]

* Complete naive OOC. [Chenhan D. Yu]

* Turn off testsuits. [Chenhan D. Yu]

* Fix the bug that Split() may some time lead to an imcomplete tree. [Chenhan D. Yu]

  Implement CSC (Compressed Sparse Columns) storage type.
  Will be doing OOC later.

  Now in main(), you can choose to do

  1. random dense matrix,
  2. dense testsuits K1.dat to K13.dat, and
  3. sparse testsuits.

* Merge branch 'fixBuildNeighbors' [Chenhan D. Yu]

* Fix case k=1. [severin617]

* Merge branch 'master' of https://github.com/ChenhanYu/hmlp. [Chenhan D. Yu]

* Merge pull request #17 from ChenhanYu/improve_neighbor_merge. [Chenhan D. Yu]

  Improve leaf neighbor collection using sort.

* Improve leaf neighbor collection using sort. [severin617]

* Making some changes for Snapdragon 820. [Chenhan D. Yu]

* Update some blas and lapack functions. Add some flops and mops count. [Chenhan D. Yu]

* Now TreeParition can be parallelized using tasks. [Chenhan D. Yu]

* ADAPTIVE and LEVELRESTRICTION are completed. [Chenhan D. Yu]

  Notice that

  if ( LEVELRESTRICTION ) then ADAPTIVE must be true.

* Fix a bug: relabel skels (mapping jpvt in GEQP3 to their lids). [Chenhan D. Yu]

* Now ComputeAll() employs auto dependency analysis. [Chenhan D. Yu]

* Modify BuildNeighbors to take arbitrary type T instead of double only. This requires Skeletonize to take an extra template parameter. [Chenhan D. Yu]

  typename T. All depndents are modified as well.

* Merge pull request #16 from ChenhanYu/row-sampling. [Chenhan D. Yu]

  Row sampling by Severin. I will edit the coding style for a bit without changing the control and data flow.

* Sort snids (sampling neighbor ids) accoring to distance. [Severin Reiz]

* Build static sampling neighbor lists and use for row sampling. [Severin Reiz]

* All evaluation related tasks and traversals are now integrated in ComputeAll(). Also we add an additional option in tree traversal to let our system figure our the dependency by itself. See details in hmlp::Scheduler::DependencyAnalysis() [Chenhan D. Yu]

  /**
  *  @brief Different from DependencyAdd, which asks programmer to manually
  *         add the dependency between two tasks. DependencyAnalysis track
  *         the read/write dependencies if the data extends hmlp::Object.
  *
  *         There are 3 different data dependencies may happen:
  *         RAW: read after write a.k.a data-dependency;
  *         WAR: write after read a.k.a anti-dependency, and
  *         WAW: write after write (we don't care).
  *
  *         Take a task that read/write the object, then this task
  *         depends
  *         on the previous write task. Since it also overwrites the
  *         object,
  *         this task must wait for all other tasks that want to
  *         read the
  *         previous state of the object.
  *
  */

* Implement ComputeAll() [Chenhan D. Yu]

* The FMM resutl is now correct. [Chenhan D. Yu]

* Complete blocked FMM evaluation. Next I have to do the blocked non-FMM (unsymmetric) evaluation. [Chenhan D. Yu]

* Fix a infinite loop bug in Select. [Chenhan D. Yu]

* Fix ANN initialization bugs. [Chenhan D. Yu]

* Testsuits. [Chenhan D. Yu]

  Change these lines in the generator spdmetrices.m.
  ...
  %dlmwrite(filename,full(KA{i}));
  fileID = fopen( filename, 'w' );
  fwrite( fileID, full(KA{i}), 'double' );
  fclose( fileID );

  cp *.dat /hmlp/build/bin

  Notice that the current spliter may fail to create an even partition.
  This will lead to an incomplete binary tree.
  Currnetly I fix it by using random even split instead.

* All testsuits K1 to K13 are up. Copy all *.dat to the same directory and check out test_spdaskit.cpp. [Chenhan D. Yu]

* You can now specify if you want to use adaptive id in test_spdaskit.cpp. [Chenhan D. Yu]

  const bool ADAPTIVE = true;
  test_spdadkit<ADAPTIVE,double>( ... )

* Merge pull request #15 from JamesLevitt/master. [Chenhan D. Yu]

  Add adaptive-rank skeletonization.

* Add adaptive-rank skeletonization. [James Levitt]

* Resolve id duplication in ANN. Now you can specify if you want NN to be sorted according to the distance. eg. [Chenhan D. Yu]

  const bool SORTED = true;

  AllNesrestNeighbor<SORTED>( ... );

* Complete symmetric interaction list. Still need to do the real evaluation for the interation list. This requires all treenodes traversal without dependency. However, the output will have concurrent write. We will probably deal with this with temporary buffers. [Chenhan D. Yu]

* Multiple touches. improve spd-askit visualization. [Chenhan D. Yu]

* Now tested on lando. [Chenhan D. Yu]

* Now AllNearestNeighbor with random projection tree is implemented in Tree. Evaluation with a particular gid has the following options. [Chenhan D. Yu]

  <SYMBOLIC> if true then no evaluation occurs, only creates prunning list.
  <NNPRUNE> if true use NN in pruning.

  Now you can play with test_spdaskit.x with the following inputs.

  ./test_spdaskit.x n k s nrhs

  n (number of points)
  k (number of neighbors)
  s (number of skeletons)
  nrhs (number of right hand sides in the evaluation)

* Finish AllNearestNeighbor. [Chenhan D. Yu]

* Now you can play with Evaluate(). [Chenhan D. Yu]

  Notice that it only takes leafnodes.

* Add spd-askit approximate center splitter, updateweights, evaluation. [Chenhan D. Yu]

* Commit for discussion. [Chenhan D. Yu]

* Add a omp task implementation. [Chenhan D. Yu]

* Test on Maverick. [Chenhan D. Yu]

* Move to maverick and try it out. [Chenhan D. Yu]

* The prototypes of tree and node are finalized. [Chenhan D. Yu]

  hmlp::tree::Tree takes hmlp::tree::Setup (shared by all nodes).

  User can create a superclass of hmlp::tree::Setup to store other shared data.

  hmlp::tree::Node takes hmlp::tree::Setup (to create a call back pointer)

  and some user-define data type as per node private data.

* Forgot iaskit. [Chenhan D. Yu]

* Improve tree. [Chenhan D. Yu]

* Merge branch 'master' of https://github.com/ChenhanYu/hmlp. [Chenhan D. Yu]

* There is still a bug in gkrm_gpu. [Chenhan D. Yu]

* Try to use "using" as a template shorthand. [Chenhan D. Yu]

* Check in. I'm going to move to Maverick and finish GPU part. [Chenhan D. Yu]

* Add a new class data.hpp which extends std::vector. [Chenhan D. Yu]

  Also I reimplement the print matrix function.

* Try iframe in README. [Chenhan D. Yu]

* Now run_xxx.sh produces specific formats with tags for google spreadsheet. [Chenhan D. Yu]

* Adding google spreadsheet support. The goal is to auto report performance results (and sanity check) on websites using google spreadsheet as a simple database. [Chenhan D. Yu]

* Multiple touches. [Chenhan D. Yu]

* We are going to change ks_s and ks_t structure to take template<typename T>. [Chenhan D. Yu]

* Implement traversedown() start to implement skeletonization() update license. [Chenhan D. Yu]

* Now TraverseUp is task indepdent. Users specify a task (extension of hmlp task) and how to traverse (level-by-level or dynamic) at compile time. [Chenhan D. Yu]

* Now scheduler is allocated dynamically in Runtime::Init() and destroyed in Runtime::Finalize(). Since rt is global object, there is a bug in Intel OpenMP that will crash while calling omp_lock_destroy. [Chenhan D. Yu]

* Now the task scheduler is done. Task can be inheriated to use templates. [Chenhan D. Yu]

* Start to implement the dynamic traversal scheme. [Chenhan D. Yu]

* Some update on strassen_gpu. [Chenhan D. Yu]

* GPU strassen is good. [Chenhan D. Yu]

* Complete batch conv2d. [Chenhan D. Yu]

* Now arm can use QSML. Our asm kernel is about 7% faster. [Chenhan D. Yu]

* Fix conv2d bugs. [Chenhan D. Yu]

* New python interface. [Chenhan D. Yu]

* Implement (binary) tree partitioning. [Chenhan D. Yu]

* Reimplemented most of the runtime functionality. The next step is to figure out a new way to generating tasks and describing dependencies. I would like to keep the original CJ functionality such that auto-parallelism is possible on matrix-base algorithms. [Chenhan D. Yu]

* Conv2d seems to pack image correctly. [Chenhan D. Yu]

  Now started to port CJ runtime to HMLP.

* Merge branch 'master' of https://github.com/ChenhanYu/hmlp. [Chenhan D. Yu]

* Test GKRM with kmeans. Now the answer is correct. The next step is to implement the workspace. [Chenhan D. Yu]

* I want to change the signiture of fusedkernel such that V and ldv are hidden in auxiluary info. This facilitate the kernels that reuse C as temporart buffers. [Chenhan D. Yu]

* Add a hello world test file. Remove additional -rdynamic flag that lead to segmentation fault on TACC machine. [Chenhan D. Yu]

* Unify interface for opkernel and opreduce on CPU and GPU. [Chenhan D. Yu]

* Merge branch 'master' of https://github.com/ChenhanYu/hmlp. [Chenhan D. Yu]

* Complete gkrm gpu and reference, but there is a seg fault bug. I will deal with it later. [Chenhan D. Yu]

  It is also a good time to decide how to deal with the work space
  required by GKRM, GKMM.

  Cris suggests helper functions to help users allocate working buffers.
  These buffers will later be passed into GKMX.

* Deprecate hmlp_thread_communicator.hpp. [Chenhan D. Yu]

  now thread_communicator and thread_info are all in hmlp_thread.hpp

* Merge branch 'master' of https://github.com/ChenhanYu/hmlp. [Chenhan D. Yu]

* Fix multi-threaded gsknn. [leslierice]

* Add strassen asm kernel, but does not work when m and n are odd. [leslierice]

* Adjust cmake for arm cross compilation. [Chenhan D. Yu]

  Now hmlp_thread_info is merged into communicator file.

* Update info for cross compilation for Android OS. [Chenhan D. Yu]

* Add Arm kernels and camke. However, currently compilation for arm is still problematic. [Chenhan D. Yu]

  GKMX's marcro kernel will need to deal with the TV and TC type issue later.

  fix many type problem in STRASSEN.

* Now we will use BLIS's microkernel for rank-k update in double and single precision. Currently we still keep our rank-k update interface. [Chenhan D. Yu]

  See wiki pages for an example on how to transform a blis kernel to a HMLP
  kernel.

  Bug fix in GSKS while calling rank-k update micro kernel.

* Merge pull request #10 from SudoNohup/master. [Jianyu Huang]

  Add another strassen_internal function with amap, bmap parameters

* Add another strassen_internal function with amap, bmap parameters: for gsks, gsknn collection operations. [Jianyu Huang]

* Merge branch 'master' of github.com:ChenhanYu/hmlp. [Jianyu Huang]

* Merge branch 'master' of github.com:ChenhanYu/hmlp. [Jianyu Huang]

* GSKNN works for k > kc with Strassen and without. [leslierice]

* Rename the cublas wrapper functions. [Chenhan D. Yu]

* Start to implement blas base kmeans as a reference of gkrm. [Chenhan D. Yu]

* Forgot to change the setting back. [Chenhan D. Yu]

* Start to implement the memory pool in hmlp_runtime. [Chenhan D. Yu]

* Try to implement gkrm on x86_64. Currently implementing gkmm and gkrm using Goto algorithm is problematic due to the temporary rank-kc update in type TV. [Chenhan D. Yu]

  Given that the output c may have different types than the rank-kc update.
  The worst case is to allocate an m-by-n buffer. Several things are pretty
  sure:

  1. allocating this buffer may break user's program due to the size.
  2. allocating V may result in significant overhead.

  GPU gkmx does not have this problem because the temporary rank-kc update is
  stored in the registers. However, gkrm still need to allocate an m-by-n/4
  buffer to perform global reduction.

* Remove duplicate code. [leslierice]

* Now gkrm and its kmean instance can be compiled. I will come back and check the correctness later. [Chenhan D. Yu]

* Merge branch 'gsknn' from Leslie's branch. [Chenhan D. Yu]

* Start implementing strassen. [leslierice]

* Merge master. [leslierice]

* Try to implement for k > kc. [leslierice]

* Now working. [leslierice]

* Use bubble sort in reference soln instead, get segfault from dgsknn when moving initialization of D and I. [leslierice]

* Add heapselect. [leslierice]

* Uncomment things that break the build on my machine if commented. [Krzysztof Drewniak]

* Started to port GKRM from the origin implementation to HMLP. [Chenhan D. Yu]

* Add contributors. [Chenhan D. Yu]

* Merge pull request #8 from SudoNohup/master. [Chenhan D. Yu]

  Solve Issue #7. We need chief thread

* Solve Issue #7. We need chief thread. [Jianyu Huang]

* Merge branch 'gsknn' [Chenhan D. Yu]

* Change parameters for gsknn. [leslierice]

* Add new files for gsknn. [leslierice]

* Merge pull request #6 from SudoNohup/master. [Chenhan D. Yu]

  Try to Solve #5. Using one function for reading environment variable.

* Try to Solve #5. Using one function for reading environment variable. [Jianyu Huang]

* Update collaborator and readme. [Chenhan D. Yu]

* Try out some urls in README.md. [Chenhan D. Yu]

* Now GKMX can use STRASSEN. We will now write standard rank-k update microkernel with strassen microkernel together by overloading the operator with different arguments. [Chenhan D. Yu]

* Merge pull request #4 from SudoNohup/master. [Chenhan D. Yu]

  Create a strassen_internal function to be compatible with gkmx functi…

* Create a strassen_internal function to be compatible with gkmx function inside omp thread. [Jianyu Huang]

* Test_gkmm_gpu has passed the test. [Chenhan D. Yu]

* Compute error for gkmm_gpu. [Chenhan D. Yu]

* Again and again ... [Chenhan D. Yu]

* Updaete README again. [Chenhan D. Yu]

* Update README. [Chenhan D. Yu]

* Fix the cmake problem. [Chenhan D. Yu]

* Patch intel compiler build some? [Krzysztof Drewniak]

* Fix various compilation errors and build issues. [Krzysztof Drewniak]

  - Now uses cmake's blas finding for more portability.
  - Adds missing commas
  - Uses %= inline assembly syntax to prevent issues with duplicate labels in inlised functions

* Now there is new interface for gkmm. [Chenhan D. Yu]

* Now the GPU strassen performance is normal. [Chenhan D. Yu]

* Add a test file. [Chenhan D. Yu]

* Fix some prototype problems. [Chenhan D. Yu]

* Push strassen GPU code. [Chenhan D. Yu]

* Now we can call cublas. [Chenhan D. Yu]

* Now the GPU code is working, but I haven't double ckecked the correctness with cublas yet. The current way of maintaining gpu kernels is ugly. Also there should be a decision tree structure to implement the result of the autoaunning. [Chenhan D. Yu]

* Push more GPU code. Update the INSTALL instruction. [Chenhan D. Yu]

* Now we can compile gpu gkmx as well. Set HMLP_USE_CUDA=true. [Chenhan D. Yu]

* Add kernel/gpu/kepler. [Chenhan D. Yu]

* Now start to port gkmx. [Chenhan D. Yu]

* Merge branch 'master' of github.com:ChenhanYu/hmlp. [Chenhan D. Yu]

* Add the handler for fringes. [Jianyu Huang]

* Fix the bug for TT, TN, NT. [Jianyu Huang]

* Merge branch 'master' of github.com:ChenhanYu/hmlp. [Jianyu Huang]

* Add more fixes for trans. [Jianyu Huang]

* Fix NT, TT, TN bugs. [Jianyu Huang]

* Trying to figure out what's the problem on Maverick compute node. [Chenhan D. Yu]

* Fix some problems. [Chenhan D. Yu]

* Add conv_relu_pool. [Chenhan D. Yu]

* Proof of concept on sandybridge. [Chenhan D. Yu]

* Fix assembly code segmentation fault: C/C_ref needs to be allocated with alignment. [Jianyu Huang]

* Fix buggy cases not square: mpart is wrong. [Jianyu Huang]

* Add assembly kernel; However, segementation fault. [Jianyu Huang]

* Now it outputs correct result; Haven't deal with fringes. [Jianyu Huang]

* Add Strassen implementation: the result is wrong. [Jianyu Huang]

* Create a cnn framework for later implementation. [Chenhan D. Yu]

  Some fix to INSTALL

* Change gsks to use GetRange. [Chenhan D. Yu]

* Complete GetRange. [Chenhan D. Yu]

* New class range (loop guard) [Chenhan D. Yu]

* Complete the corner cases for gkmx. [Chenhan D. Yu]

* Major changes to the layout. [Chenhan D. Yu]

  Now /frame do not contain any arch dependent implementation.
  Instead arch dependent implementations are in /package

  eg.

  hmlp/package/x86_64/sandybridge/gsks.cpp
  hmlp/package/x86_64/sandybridge/gkmx.cpp

* Fix bugs on packA2 and packAh (wrong parameter PACK_NR->PACK_MR) [Chenhan D. Yu]

* Merge pull request #1 from SudoNohup/master. [Chenhan D. Yu]

  Fix hwb to hbw

* Fix hwb to hbw. [Jianyu Huang]

* Check in. [Chenhan D. Yu]

* Now the race condition in jc loop is fixed. 3 layers parallelization are tested to be fine. Clean up gsks.hxx. [Chenhan D. Yu]

* Now we compute all packing buffer pointers at the begining. [Chenhan D. Yu]

  Add a running script.

* Add haswell kernels. Some changes to README and INSTALL. [Chenhan D. Yu]

* KNL commit. [Chenhan D. Yu]

* Now move to KNL. [Chenhan D. Yu]

* Add several logistic files. [Chenhan D. Yu]


