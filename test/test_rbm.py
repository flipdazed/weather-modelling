import os, timeit

import numpy as np
import dill as pickle    
try: import PIL.Image as Image
except ImportError: import Image
import theano
import theano.tensor as T
import caffeine

import data
import utils
import common
from models.rbm import *
from models import logit

__doc__ =     """
    Demonstrate how to train and afterwards sample from it using Theano.
    
    This is demonstrated on MNIST.
    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: path the the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    
    The trained RBM can be used for sampling by sharing weights with a
    n MLP + logit output layer. This structure contained as a subset of the
     dbn module and not tested here.
    """

# Save locations
## built model
MODEL = data.model_dir
MODEL_ID = os.path.splitext(os.path.basename(__file__))[0]

## visualising runtime parameters
DATA_DIR = data.data_dir
PLOT_DIR = data.plot_dir

# Model parameters
n_hidden    = 500
n_in        = 28*28
k           = 15

## training parameters
n_epochs        = 15
batch_size      = 20
learning_rate   = 0.1

## early-stopping parameters
patience                = 30000 # look as this many examples regardless
patience_increase       = 2     # wait this much longer if new best is found
improvement_threshold   = 0.995 # consider this improvement significant

# sampling parameters
n_chains = 10       # number of chains (horizontal axis)
n_samples = 10      # number of samples (vertical axis)
plot_every = 1000   # sep due to correlation of samples
sample_fname = 'samples.png'

def sampleModel(rbm_model, test_set_x,  n_chains, n_samples, np_rng, save_loc, logger):
    """Samples from the RBM"""
    
    logger.info('Sampling from RBM ...')
    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    
    # pick random test examples, with which to initialize the persistent chain
    test_idx = np_rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the sample
    # for plotting
    logger.debug('building Gibbs VHV')
    (
        [
            presig_hids, hid_mfs, hid_samples, 
            presig_vis, vis_mfs, vis_samples
        ], 
        updates
    ) = theano.scan(
            rbm_model.gibbsVHV,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=plot_every,
            name="gibbsVHV"
    )
    
    logger.debug('building Gibbs sampler function')
    # add to updates the shared variable that takes care of our persistent chain :
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function([],[vis_mfs[-1], vis_samples[-1]],
        updates=updates,
        name='sample_fn'
    )
    
    logger.info('Plotting samples ...')
    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = np.zeros((29 * n_samples + 1, 29 * n_chains - 1),
         dtype='uint8')
    
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        logger.debug('plotting sample %d' % idx)
        image_data[
            (28+1)*idx:(28+1)*idx+28, :
        ] = utils.visualise.tileRasterImages(
            x=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )
    
    # construct image
    logger.info('Plotting sample image ...')
    image = Image.fromarray(image_data)
    image.save(save_loc)
    logger.info('Saved to: {}'.format(save_loc))
    pass

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
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]//batch_size
    
    logger.info('Building the model ...')
    # allocate symbolic variables for the data
    index = T.lscalar()     # index to a [mini]batch
    x = T.matrix('x')       # the data is presented as rasterized images
    y = T.ivector('y')      # the labels are presented as 1D vector of [int]
                            # labels
    
    logger.info('Building model ...')
    np_rng = np.random.RandomState(123)
    theano_rng = T.shared_randomstreams.RandomStreams(np_rng.randint(2 ** 30))
    
    # construct the RBM class
    rbm_model = RBM(
        inputs=x,
        n_visible=n_in,
        n_hidden=n_hidden,
        np_rng=np_rng,
        theano_rng=theano_rng
    )
    
    # initialize storage for the persistent chain (state = hidden layer of chain)
    persistent_chain = theano.shared(
        np.zeros((batch_size, n_hidden),
            dtype=theano.config.floatX),
        borrow=True)
    
    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates, gparams = rbm_model.getCostUpdates(lr=learning_rate,
         persistent=persistent_chain, k=k)
    
    # it is ok for a theano function to have no output
    # the purpose of train_model is solely to update the RBM parameters
    logger.debug('building training model')
    train_model = theano.function(
        inputs=[index],
        outputs=[cost] + gparams,
        updates=updates,
        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]}
    )
    
    logger.info('Training model ...')
    
    # Visualise these items during training
    visualise_weights = {       # dict of images to create
        'inputLayer' + '_weights': {    # input - hiddenlayer image
            'x':rbm_model.w,            # the parameter
            'img_shape':(28, 28),       # prod. of tuple == # input nodes
            'tile_shape':(15, 30),      # Max number is # nodes in next layer
            'tile_spacing':(1, 1),      # separate imgs x,y
            'runtime_plots': True,
            'freq':50
        }
    }
    
    # visualise cost during runtime
    visualise_cost = {      # visualising the cost
        'cost':{'freq':50}  # frequency of sampling
        }
    
    # visualise arbitrary parameters at runtime
    visualise_params = {
        'hiddenLayer' + '_weights': {
            'freq':1,
            'x':rbm_model.w
        },
        'hiddenLayer' + '_hbias': {
            'freq':1,
            'x': rbm_model.hbias
        },
        'hiddenLayer' + '_vbias': {
            'freq':1,
            'x': rbm_model.vbias
        }
    }
    
    visualise_updates = {
        'hiddenLayer' + '_weights': {
            'update_position':0,
            'freq':1
        },
        'hiddenLayer' + '_hbias': {
            'update_position':1,
            'freq':1
        },
        'hiddenLayer' + '_vbias': {
            'update_position':2,
            'freq':1
        }
    }
    
    param_man = utils.visualise.Visualise_Runtime(
        plot_dir=PLOT_DIR,
        data_dir=DATA_DIR
    )
    param_man.initalise(
        run_id = MODEL_ID,
        default_freq = min(n_train_batches, patience//2),
        params = visualise_params,
        cost = visualise_cost,
        imgs = visualise_weights,
        updates = visualise_updates
        )
    
    utils.training.train(rbm_model, train_model, None, None,
        n_train_batches, None, None,
        n_epochs, learning_rate,
        patience, patience_increase, improvement_threshold,
        MODEL, MODEL_ID, logger,
        visualise=param_man
    )
    
    sampleModel( # sample from the trained model
        rbm_model,
        test_set_x,
        n_chains,
        n_samples,
        np_rng,
        save_loc = os.path.join(PLOT_DIR, '_'.join([MODEL_ID,sample_fname])),
        logger=logger
    )
    pass