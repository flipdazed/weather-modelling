import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import utils
from models.logit import Logistic_Regression
from models.mlp import Hidden_Layer
from models.rbm import RBM

__docformat__ = 'restructedtext en'

__doc__ = """
This code is adapted from a Theano tutorial found at deeplearning.net

References:
    - http://deeplearning.net/tutorial/dbn.html
    
A deep belief network is obtained by stacking several RBMs on top of each
other. The hidden layer of the RBM at layer `i` becomes the inputs of the
RBM at layer `i+1`. The first layer RBM gets as inputs the inputs of the
network, and the hidden layer of the last RBM represents the output. When
used for classification, the DBN is treated as a MLP, by adding a logistic
regression layer on top.
"""

class DBN(object):
    """Deep Belief Network"""
    
    def __init__(self, np_rng, theano_rng=None, n_ins=784,hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.
        
        :type np_rng: np.random.RandomState
        :param np_rng: np random number generator used to draw initial
                    weights
        
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`
        
        :type n_ins: int
        :param n_ins: dimension of the inputs to the DBN
        
        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value
        
        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """
        
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        
        assert self.n_layers > 0
        
        if not theano_rng: 
            theano_rng = MRG_RandomStreams(np_rng.randint(2 ** 30))
        
        # allocate symbolic variables for the data
        
        # the data is presented as rasterized images
        self.x = self.inputs = T.matrix('x')
        
        # the labels are presented as 1D vector of [int] labels
        self.y = T.ivector('y')
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.
        
        for i in range(self.n_layers):
            # construct the sigmoidal layer
            
            # the size of the inputs is either the number of hidden
            # units of the layer below or the inputs size if we are on
            # the first layer
            if i == 0: inputs_size = n_ins
            else: inputs_size = hidden_layers_sizes[i - 1]
            
            # the inputs to this layer is either the activation of the
            # hidden layer below or the inputs of the DBN if you are on
            # the first layer
            if i == 0: layer_inputs = self.x
            else: layer_inputs = self.sigmoid_layers[-1].output
            
            sigmoid_layer = Hidden_Layer(
                rng=np_rng,
                inputs=layer_inputs,
                n_in=inputs_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.hard_sigmoid
            )
            
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            
            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)
            
            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(
                np_rng=np_rng,
                theano_rng=theano_rng,
                inputs=layer_inputs,
                n_visible=inputs_size,
                n_hidden=hidden_layers_sizes[i],
                w=sigmoid_layer.w,
                hbias=sigmoid_layer.b
            )
            self.rbm_layers.append(rbm_layer)
        
        # We now need to add a logistic layer on top of the MLP
        self.logitLayer = Logistic_Regression(
            inputs=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logitLayer.params)
        
        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logitLayer.negativeLogLikelihood(self.y)
        
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logitLayer.errors(self.y)
        
        # get prediction from logistic sgd
        self.y_pred = self.logitLayer.y_pred
        self.p_y_given_x = self.logitLayer.p_y_given_x
        pass
    def pretrainingFunctions(self, train_set_x, batch_size, k):
        """Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as inputs the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.
        
        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        
        """
        
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
        
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        for rbm in self.rbm_layers:
            
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.getCostUpdates(learning_rate,
                 persistent=None, k=k)
            
            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={self.x: train_set_x[batch_begin:batch_end]}
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        
        return pretrain_fns
    def buildFinetuneFunctions(self, datasets, batch_size, learning_rate):
        """Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set
        
        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        
        """
        
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x,  test_set_y)  = datasets[2]
        
        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches //= batch_size
        
        index = T.lscalar('index')  # index to a [mini]batch
        
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)
        
        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))
        
        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x:train_set_x[index*batch_size:(index+1)*batch_size],
                self.y:train_set_y[index*batch_size:(index+1)*batch_size]
            }
        )
        
        test_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x:test_set_x[index*batch_size:(index+1)*batch_size],
                self.y:test_set_y[index*batch_size:(index+1)*batch_size]
            }
        )
        
        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x:valid_set_x[index*batch_size:(index+1)*batch_size],
                self.y:valid_set_y[index*batch_size:(index+1)*batch_size]
            }
        )
        
        # both accept *args as the other routines pass an index, i
        # Create a function that scans the entire validation set
        def valid_score(*args):
            return [valid_score_i(i) for i in range(n_valid_batches)]
        
        # Create a function that scans the entire test set
        def test_score(*args):
            return [test_score_i(i) for i in range(n_test_batches)]
        
        return train_fn, valid_score, test_score

if __name__ == "__main__":
    pass