##  
##  HMLP (High-Performance Machine Learning Primitives)
##  
##  Copyright (C) 2014-2017, The University of Texas at Austin
##  
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##  
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
##  GNU General Public License for more details.
##  
##  You should have received a copy of the GNU General Public License
##  along with this program. If not, see the LICENSE file.
##  


import numpy as np
from hmlp import PyRuntime
from gofmm import PySPDMatrix
#from gofmm import PyTree
#from gofmm import PyCompress
#from gofmm import PyEvaluate
#from gofmm import PyComputeError

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


## Initialize hmlp runtime system
rt = PyRuntime()
rt.init()

## create a random SPD matrix
K = PySPDMatrix()
K.resize( n, n )
K.randspd()



### compress K 
#tree = PyCompress( K, stol, budget )
#
### random initialize weights #
#weights = np.random.rand( n , nrhs )
#
### potentials = K * weights
#potentials = PyEvaluate( tree, weights )
#
#print potentials.shape
#
#total_error = 0.0
#for i in range( ntest ):
#	u_i = np.zeros( ( 1, nrhs ) )
#  
#	for j in range( nrhs ):
#		u_i[ 0, j ] = potentials[ i, j ]
#  
#	error = PyComputeError( tree, i, u_i )
#	total_error = total_error + error
#	print "gid ", i, "error ", error
### end for
#print total_error / ntest
#
#
#
#
## terminate hmlp runtime system
rt.finalize()

print "here"
