import numpy as np
import theano
import theano.tensor as T
import timeit
import dill as pickle
import os, sys

import load
import utils
import common
from mlp import *

MODEL = os.path.join(load.model_dir, os.path.splitext(os.path.basename(__file__))[0]+'.pkl')

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
    
    # load the saved model
    classifier = pickle.load(open(MODEL))
    
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)
    
    # We can test it on some examples from test test
    datasets = source.mnist()
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    
    predicted_values = predict_model(test_set_x[:10])
    logger.info("Predicted vs. Actual values for the first 10 examples in test set:")
    logger.info(predicted_values)
    logger.info(test_set_y[:10].eval())
    pass
#
if __name__ == "__main__":
    logger = utils.logs.get_logger(__name__, update_stream_level=utils.logs.logging.DEBUG)
    logger.info('Loading data ...')
    data_loc = load.data_loc
    source = load.Load_Data(location=data_loc)
    
    learning_rate=0.13
    n_epochs=1000
    batch_size=600
    
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
    
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    
    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = Logistic_Regression(input=x, n_in=28**2, n_out=10)
    
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negativeLogLikelihood(y)
    
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    logger.debug('building test model')
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    logger.debug('building validate model')
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.w)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    
    
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.w, classifier.w - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
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
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995 # a relative improvement of this much is
                                  # considered significant
    common.train(classifier, train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches,
        n_epochs, learning_rate,
        patience, patience_increase, improvement_threshold, 
        MODEL, logger)
    common.predict(MODEL, source, logger)
    pass