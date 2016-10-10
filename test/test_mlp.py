import os, sys

import numpy as np
import theano
import theano.tensor as T
import caffeine

import data
import utils
import common
from models.mlp import *

MODEL = os.path.join(data.model_dir, os.path.splitext(os.path.basename(__file__))[0]+'.pkl')

if __name__ == "__main__":
    learning_rate=0.01
    l1_reg=0.00
    L2_reg=0.0001
    n_epochs=1000
    batch_size=20
    n_hidden=500
    
    # early-stopping parameters
    patience = 10000                # look as this many examples regardless
    patience_increase = 5           # wait this much longer when a new best is found
    improvement_threshold = 0.995   # consider this relative improvement significant
    
    logger = utils.logs.get_logger(__name__, update_stream_level=utils.logs.logging.DEBUG)
    logger.info('Loading data ...')
    source = data.Load_Data()
    
    datasets = source.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    logger.info('Building the model ...')
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    
    rng = np.random.RandomState(1234)
    
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        inputs=x,
        n_in=28**2,
        n_hidden=n_hidden,
        n_out=10
    )
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (l1 and L2); cost is expressed
    # here symbolically
    cost = (classifier.negativeLogLikelihood(y) + l1_reg*classifier.l1 +
             L2_reg*classifier.L2_sqr)
    
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    logger.debug('building test model')
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    logger.debug('building validate model')
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]
    
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    
    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam) 
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
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