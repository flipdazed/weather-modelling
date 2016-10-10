import os, sys

import numpy as np
import theano
import theano.tensor as T
import caffeine

import data
import utils
import common
from models.cnn import *

__doc__ = """
Demonstrates lenet on MNIST dataset

:type learning_rate: float
:param learning_rate: learning rate used (factor for the stochastic
                      gradient)

:type n_epochs: int
:param n_epochs: maximal number of epochs to run the optimizer

:type dataset: string
:param dataset: path to the dataset used for training /testing (MNIST here)

:type nkerns: list of ints
:param nkerns: number of kernels on each layer
"""

logger = utils.logs.get_logger(__name__, update_stream_level=utils.logs.logging.DEBUG)

MODEL = os.path.join(data.model_dir, os.path.splitext(os.path.basename(__file__))[0]+'.pkl')

def calcImgReductionSize(l, pool_len, filter_len):
    """Calculate the new image size after filtering
    and max-pooling in a single particular dimension
    
    The function must be ran d times for a d-dim image
    
    :param l: length of the image
    :param pool_len: size of maxpool
    :param filter_len: length of filter
    """
    filtered_len = l-filter_len+1
    if filtered_len % pool_len == 0:
        return filtered_len // pool_len
    else:
        raise ValueError(
        'Bad params: length: {} creates filter length: {}'
        'This cannot be maxpooled with a pool size of {}'
        'as it results in a non-integer result'.format(
            l, filtered_len, pool_len))

if __name__ == "__main__":
    
    # convolution build parameters
    # first list item is layer 0; second is layer 1
    h_in            = 28
    w_in            = 28
    nkerns          = [20, 50]
    
    ## the convolutional filter dimensions
    filter_shape_h  = [5, 5]
    filter_shape_w  = [5, 5]
    
    ## the max-pooling dimensions
    poolsize_h      = [2, 2]
    poolsize_w      = [2, 2]
    
    # Multilayer Perceptron build parameters
    mlp_activation  = T.tanh
    n_hidden        = 500
    n_out           = 10
    
    # training parameters
    learning_rate   = 0.1
    n_epochs        = 200
    batch_size      = 500
    
    # early-stopping parameters
    patience = 10000                # look as this many examples regardless
    patience_increase = 5           # wait this much longer when a new best is found
    improvement_threshold = 0.995   # consider this relative improvement significant
    
    # Calculating build parameters for each layer from user values
    logger.info('Calculating build parameters ...')
    
    ## input image dimensions for each layer
    ### not entirely sure that the second argument is correct here?
    ### the default value was 5 in both cases
    conv_img_len_h  = [h_in, filter_shape_h[0]]
    conv_img_len_w  = [w_in, filter_shape_w[0]]
    
    ## image param tuple
    l0_img_shape = (batch_size,         1, conv_img_len_h[0], conv_img_len_w[0])
    l1_img_shape = (batch_size, nkerns[0], conv_img_len_h[1], conv_img_len_w[1])
    
    ## filters param tuple
    l0_filter_shape = (nkerns[0],         1, filter_shape_h[0], filter_shape_w[0])
    l1_filter_shape = (nkerns[1], nkerns[0], filter_shape_h[1], filter_shape_w[1])
    
    ## pool sizes
    l0_poolsize = (poolsize_h[0], poolsize_w[0])
    l1_poolsize = (poolsize_h[1], poolsize_w[1])
    
    ## output image dimensions for each layer
    l0_img_h = calcImgReductionSize(h_in,     poolsize_h[0], filter_shape_h[0])
    l0_img_w = calcImgReductionSize(w_in,     poolsize_w[0], filter_shape_w[0])
    l1_img_h = calcImgReductionSize(l0_img_h, poolsize_h[1], filter_shape_h[1])
    l1_img_w = calcImgReductionSize(l0_img_w, poolsize_w[1], filter_shape_w[1])
    
    # number of multilayer perceptron input nodes
    # corresponds to the output dimensions of the final conv layer
    n_in = nkerns[1] * l1_img_h * l1_img_w
    
    logger.info('Loading data ...')
    source = data.Load_Data()
    
    datasets = source.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # allocate symbolic variables for the data
    index = T.lscalar() # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    
    logger.info('Building the model ...')
    rng = np.random.RandomState(123)
    
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_inputs = x.reshape(l0_image_shape)
    
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng=rng,
        inputs=layer0_inputs,
        image_shape=l0_img_shape,
        filter_shape=l0_filter_shape,
        poolsize=l0_poolsize
    )
    
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng=rng,
        inputs=layer0.output,
        image_shape=l1_img_shape,
        filter_shape=l1_filter_shape,
        poolsize=l1_poolsize
    )
    
    # the Hidden_Layer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_inputs = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = Hidden_Layer(
        rng=rng,
        inputs=layer2_inputs,
        n_in=n_in,
        n_out=n_hidden,
        activation=mlp_activation
    )
    
    # classify the values of the fully-connected sigmoidal layer
    layer3 = Logistic_Regression(inputs=layer2.output, n_in=n_hidden, n_out=n_out)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    logger.debug('building test model')
    test_model = theano.function(
        inputs=[index],
        outputs=layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    logger.debug('building validate model')
    validate_model = theano.function(
        inputs=[index],
        outputs=layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    # create a list of gradients for all model parameters
    gparams = T.grad(cost, params)
    
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
    ]
    
    logger.debug('building training model')
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    logger.info('Training the model ...')
    utils.training.train(classifier, train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches,
        n_epochs, learning_rate,
        patience, patience_increase, improvement_threshold,
        MODEL, logger)
    
    logger.info('Testing the model ...')
    common.predict(MODEL, source, logger)
    pass