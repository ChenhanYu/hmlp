import numpy as np
from hmlp import Runtime
from gofmm import PySPDMatrix
from gofmm import PyTree
from gofmm import PyCompress
from gofmm import PyEvaluate
from gofmm import PyComputeError

## input filename

## problem size
n = 1000

## number of right hand sides
nrhs = 1

## user tolerance
stol = 1E-3

## user computation budget 
budget = 0.01

## number of testing rows ( K( gid, : ) * w - H( gid, : ) * w  )
ntest = 10


## initialize hmlp runtime system
rt = Runtime()
rt.init()

## create a random SPD matrix
K = PySPDMatrix()
K.resize( n, n )
K.randspd()



## compress K 
tree = PyCompress( K, stol, budget )

## random initialize weights #
weights = np.random.rand( n , nrhs )

## potentials = K * weights
potentials = PyEvaluate( tree, weights )

print potentials.shape

total_error = 0.0
for i in range( ntest ):
	u_i = np.zeros( ( 1, nrhs ) )
  
	for j in range( nrhs ):
		u_i[ 0, j ] = potentials[ i, j ]
  
	error = PyComputeError( tree, i, u_i )
	total_error = total_error + error
	print "gid ", i, "error ", error
## end for
print total_error / ntest




## terminate hmlp runtime system
rt.finalize()

print "here"
