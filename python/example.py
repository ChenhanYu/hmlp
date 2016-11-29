import numpy
from conv2d import *

w0 = 4
h0 = 4
d0 = 2

w1 = 3
h1 = 3
d1 = 1



I = numpy.random.rand(  1, h0, w0, d0 ).astype( numpy.double )
W = numpy.random.rand( d1, h1, w1, d0 ).astype( numpy.double )

print "dconv2d_ref:"
C_ref = dconv2d_ref( I, W )
print C_ref

print "dconv2d:"
C_tst = dconv2d( I, W )
print C_tst
