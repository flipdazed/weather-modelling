import os,sys
import time

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import utils
import data
from models.logit import Logistic_Regression
from models.mlp import Hidden_Layer

__docformat__ = 'restructedtext en'

__doc__ = """
This code is adapted from a Theano tutorial found at deeplearning.net

LeNet5 is a convolutional neural network, good for classifying images.

This adapted implementation simplifies the model in the following ways:
 
 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling by max
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - http://deeplearning.net/tutorial/rbm.html
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """
    
    def __init__(self, rng, inputs, filter_shape, image_shape, poolsize=(2, 2)):
        """Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        
        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type inputs: theano.tensor.dtensor4
        :param inputs: symbolic image tensor, of shape image_shape
        
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num inputs feature maps,
                              filter height, filter width)
        
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num inputs feature maps,
                             image height, image width)
        
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        # need to make sure that we have the same number of kernels!!
        # in the images and the filters - otherwise schoolboy error
        assert image_shape[1] == filter_shape[1]
        
        self.inputs = inputs
        
        # there are "num inputs feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        
        self.w = theano.shared(
            np.asarray(
                rng.uniform(low=-w_bound, high=w_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        # convolve inputs feature maps with filters
        conv_out = conv.conv2d(
            inputs=inputs,
            filters=self.w,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            inputs=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # store parameters of this layer
        self.params = [self.w, self.b]
        pass

if __name__ == '__main__':
    pass