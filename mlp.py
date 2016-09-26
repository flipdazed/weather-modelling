"""
This code is adapted from a Theano tutorial found at deeplearning.net

References:
    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2
    - http://deeplearning.net/tutorial/mlp.html
"""

__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T

from logit import Logistic_Regression

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, w=None, b=None,activation=T.tanh):
        """Initialisation of the Hidden Layer
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        
        NOTE : The nonlinearity used here is tanh
        
        Hidden unit activation is given by: tanh(dot(input,W) + b)
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        
        :type n_in: int
        :param n_in: dimensionality of input
        
        :type n_out: int
        :param n_out: number of hidden units
        
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.w = w
        self.b = b
        self.activation = activation
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        
        # `w` is initialized with `w_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if self.w is None:
            w_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
                    high=numpy.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_out)
                ),
                dtype=theano.config.floatX
            )
            if self.activation == theano.tensor.nnet.sigmoid:
                w_values *= 4
            
            self.w = theano.shared(value=w_values, name='w', borrow=True)
        
        # initialise the bias as 0
        if self.b is None:
            b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        
        lin_output = T.dot(input, self.w) + self.b
        
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        # parameters of the model
        self.params = [self.w, self.b]
        pass

class MLP(object):
    """Multi-Layer Perceptron Class
    
    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        
        :type n_hidden: int
        :param n_hidden: number of hidden units
        
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        
        """
        
        # keep track of model input
        self.input = input
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.rng = rng
        
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = Logistic_Regression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        
        # l1 norm ; one regularization option is to enforce l1 norm to
        # be small
        self.l1 = (
            abs(self.hiddenLayer.w).sum()
            + abs(self.logRegressionLayer.w).sum()
        )
        
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.w ** 2).sum()
            + (self.logRegressionLayer.w ** 2).sum()
        )
        
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negativeLogLikelihood = (
            self.logRegressionLayer.negativeLogLikelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        
        # get prediction from logistic sgd
        self.y_pred = self.logRegressionLayer.y_pred
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        pass