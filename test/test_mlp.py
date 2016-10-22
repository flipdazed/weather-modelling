import os, sys

import numpy as np
import theano
import theano.tensor as T
import caffeine

import data
import utils
import common
from models.mlp import *

# Save locations
## built model
MODEL = data.model_dir
MODEL_ID = os.path.splitext(os.path.basename(__file__))[0]

## visualising runtime parameters
DATA_DIR = data.data_dir
PLOT_DIR = data.plot_dir

# Parameter settings
## model parameters
n_hidden    = 625
n_in        = 28*28
n_out       = 10

## training parameters
n_epochs        = 1000
batch_size      = 20
learning_rate   = 0.01
l1_reg          = 0.0
L2_reg          = 0.0001

## early-stopping parameters
patience                = 10000 # look as this many examples regardless
patience_increase       = 2     # wait this much longer if new best is found
improvement_threshold   = 0.995 # consider this improvement significant

if __name__ == "__main__":
    logger = utils.logs.get_logger(__name__, 
        update_stream_level=utils.logs.logging.DEBUG)
    logger.info('Loading data ...')
    source = data.Load_Data()
    
    datasets = source.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] //batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] //batch_size
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
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
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
    
    # Visualise these items during training
    visualise_weights = {       # dict of images to create
        'inputLayer' + '_weights': {    # input - hiddenlayer image
            'x':classifier.hiddenLayer.w.get_value(
                borrow=True).T,         # the parameter
            'img_shape':(28, 28),       # prod. of tuple == # input nodes
            'tile_shape':(25, 25),      # Max number is # nodes in next layer
            'tile_spacing':(1, 1)       # separate imgs x,y
        },
        'logitLayer' + '_weights': {    # hidden - logistic layer
            'x':classifier.logitLayer.w.get_value(borrow=True).T,
            'img_shape':(25, 25),       # prod. of tuple == # hidden nodes
            'tile_shape':(2, 10),
            'tile_spacing':(1, 1)
        }
    }
    
    # visualise cost during runtime
    visualise_cost = {      # visualising the cost
        'cost':{'freq':1}      # frequency of sampling
        }
    
    # visualise arbitrary parameters at runtime
    visualise_params = {
        'hiddenLayer' + '_weights': {
            'freq':1,
            'x':classifier.hiddenLayer.w.get_value(borrow=True).ravel()
        },
        'hiddenLayer' + '_bias': {
            'freq':1,
            'x': classifier.hiddenLayer.b.get_value(borrow=True).ravel()
        },
        'logitLayer' + '_weights': {
            'freq':1,
            'x': classifier.logitLayer.w.get_value(borrow=True).ravel()
        },
        'logitLayer' + '_bias': {
            'freq':1,
            'x':classifier.logitLayer.b.get_value(borrow=True).ravel()
        }
    }
    
    param_man = utils.visualise.Visualise_Runtime(
        plot_dir=PLOT_DIR,
        data_dir=DATA_DIR
    )
    param_man.initalise(
        run_id = MODEL_ID,
        default_freq = min(n_train_batches, patience // 2),
        params = visualise_params,
        cost = visualise_cost,
        imgs = visualise_weights
        )
    
    utils.training.train(classifier, train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches,
        n_epochs, learning_rate,
        patience, patience_increase, improvement_threshold,
        MODEL, MODEL_ID, logger,
        visualise=param_man
    )
    logger.info('Testing the model ...')
    common.predict(os.path.join(MODEL, MODEL_ID + ".pkl"), source, logger)
    pass