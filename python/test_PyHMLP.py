import numpy as np
#import hmlp
import gofmm

m = 5
n = m

stol = 1E-3
budget = 0.01

# K = np.ndarray( shape = ( m, n ) )

K = np.zeros( ( m, n ) )

print K

# status = gofmm.Compress( K, stol, budget )
status = gofmm.Compress( K )


