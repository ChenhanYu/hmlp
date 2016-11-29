import os
import ctypes
from numpy import *
import numpy

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d


def dconv2d_ref( I, W ):
    i_shape = I.shape
    w_shape = W.shape

    print "Theano conv2d"
    print i_shape[ 3 ], i_shape[ 2 ], i_shape[ 1 ], i_shape[ 0 ]
    print w_shape[ 3 ], w_shape[ 2 ], w_shape[ 1 ], w_shape[ 0 ]

    print i_shape
    print w_shape

    img = I.transpose( 0, 3, 1, 2 )
    weight = W.transpose( 0, 3, 1, 2 )

    print img.shape
    print weight.shape

    input = T.tensor4( name = 'input' )
    W = theano.shared( weight, name = 'W' )
    output = conv2d( input, W, border_mode = 1 )

    f = theano.function( [input], output )

    C = f( img )

    print C.shape

    return C
# end def dconv2d_ref


def dconv2d( I, W ):

    w0 = I.shape[ 2 ]
    h0 = I.shape[ 3 ]
    d0 = I.shape[ 1 ]
    b  = I.shape[ 0 ]

    w1 = W.shape[ 2 ]
    h1 = W.shape[ 3 ]
    d1 = W.shape[ 0 ]
    s  = 1
    p  = ( w1 - 1 ) / 2

    c_shape = ( b, d1, w0, h0 )
    C = numpy.ndarray( shape = c_shape, dtype = numpy.double )

    print os.environ.get( 'HMLP_DIR' );
    print os.environ.get( 'DYLD_LIBRARY_PATH' );
    print os.environ.get( 'LD_LIBRARY_PATH' );

    print os.getenv('DYLD_LIBRARY_PATH')

    libhmlp_path = os.environ.get( 'HMLP_DIR' ) + '/build/lib/libdyhmlp.dylib'
    libhmlp = ctypes.cdll.LoadLibrary( libhmlp_path )
    libhmlp.dconv2d
    (
        ctypes.c_int( w0 ),
        ctypes.c_int( h0 ),
        ctypes.c_int( d0 ),
        ctypes.c_int( s ),
        ctypes.c_int( p ),
        ctypes.c_void_p( I.ctypes.data ),
        ctypes.c_int( w1 ),
        ctypes.c_int( h1 ),
        ctypes.c_int( d1 ),
        ctypes.c_void_p( I.ctypes.data ),
        ctypes.c_void_p( C.ctypes.data ),
    )

    return C
# end def dconv2d
