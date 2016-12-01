import numpy
from conv2d import *

w0 = 5
h0 = 5
d0 = 2

w1 = 3
h1 = 3
d1 = 1



I = numpy.random.rand(  1, h0, w0, d0 ).astype( numpy.double )
W = numpy.random.rand( d1, h1, w1, d0 ).astype( numpy.double )

for j in range( h0 ):
    for i in range( w0 ):
        for p in range( d0 ):
            I[ 0, j, i, p ] = p + 1

for q in range( d1 ):
    for j in range( h1 ):
        for i in range( w1 ):
            for p in range( d0 ):
                W[ q, j, i, p ] = 1

print "dconv2d_ref:"
C_ref = dconv2d_ref( I, W )
print C_ref

# print "dconv2d:"
# C_tst = dconv2d( I, W )
# print C_tst
