import os, timeit
import numpy as np

import dill as pickle    
try: import PIL.Image as Image
except ImportError: import Image
import theano
import theano.tensor as T

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
    
    The trained RBM can be used for sampling by sharing weights with an MLP + logit
    output layer. This structure contained as a subset of the dbn module and not tested here.
    """

logger = utils.logs.get_logger(__name__, update_stream_level=utils.logs.logging.DEBUG)
MODEL = os.path.join(data.model_dir, os.path.splitext(os.path.basename(__file__))[0]+'.pkl')

def mkdir(d, logger):
    """Changes directory to location and creates location if not in existence
    
    :param d: directory path
    :param logger: logger class
    """
    if not os.path.isdir(d): # create directory to store data and cd to it
        logger.debug('creating directory: {}'.format(d))
        os.makedirs(d)
    pass
def trainModel(train_set_x, n_hidden, learning_rate, training_epochs, batch_size, np_rng, plot_loc=None, model_loc=MODEL, logger=logger):
    """Builds the model"""
    logger.info('Building model ...')
    
    theano_rng = T.shared_randomstreams.RandomStreams(np_rng.randint(2 ** 30))
    
    # construct the RBM class
    rbm_model = RBM(
        inputs=x,
        n_visible=28 * 28,
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
    cost, updates = rbm_model.getCostUpdates(lr=learning_rate,
         persistent=persistent_chain, k=15)
    
    # it is ok for a theano function to have no output
    # the purpose of train_model is solely to update the RBM parameters
    logger.debug('building training model')
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]}
    )
     
    if plot_loc: # append '_%d.png' % epoch to this
         base_name = os.path.join(plot_loc,'filters_at_epoch')
    
    logger.info('Training model ...')
    plotting_time = 0.
    epoch_time = 0.
    start_time = timeit.default_timer()
    
    # go through training epochs
    for epoch in range(training_epochs):
        # go through the training set
        epoch_start = timeit.default_timer()
        mean_cost = [train_model(batch_index) for batch_index in range(n_train_batches)]
        epoch_end = timeit.default_timer()
        epoch_time_i = epoch_end - epoch_start
        epoch_time += epoch_time_i
        logger.debug('epoch {}, cost is {}, time {}s'.format(
            epoch, np.mean(mean_cost), epoch_time_i))
        
        # Plot filters after each training epoch
        if plot_loc:
            logger.debug('plotting image of weights ...')
            plotting_start = timeit.default_timer()
        
            image = Image.fromarray( # Construct image from the weight matrix
                utils.visualise.tileRasterImages(
                    x=rbm_model.w.get_value(borrow=True).T,
                    img_shape=(28, 28),
                    tile_shape=(10, 10),
                    tile_spacing=(1, 1)
                )
            )
            image.save(base_name + '_%i.png' % epoch))
            plotting_end = timeit.default_timer()
            plotting_time_i = plotting_end - plotting_start
            plotting_time += plotting_time_i
            logger.debug('plotting time: {}s'.format(epoch_time_i, plotting_time_i))
    
    end_time = timeit.default_timer()
    pretraining_time = end_time - start_time # - plotting_time
    logger.info('Training took %f minutes' % (pretraining_time / 60.))
    if plot_loc: logger.debug('plotting took %f minutes' % (plotting_time / 60.))
    logger.debug('epochs took %f minutes' % (epoch_time / 60.))
    
    with open(model_loc, 'wb') as f: pickle.dump(rbm_model, f)
    logger.debug('saved model as: {}'.format(model_loc))
    
    return rbm_model
def sampleModel(rbm_model, test_set_x,  n_chains, n_samples, rng, save_loc, logger=logger):
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
    # function that does `plot_every` steps before returning the sample for plotting
    logger.debug('building Gibbs VHV')
    (
        [
            presig_hids, hid_mfs, hid_samples, presig_vis, vis_mfs, vis_samples
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
    image_data = np.zeros((29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')
    
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        logger.debug('plotting sample %d' % idx)
        image_data[29 * idx:29 * idx + 28, :] = utils.visualise.tileRasterImages(
            x=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )
    
    # construct image
    logger.info('Plotting sample image ...')
    image = Image.fromarray(image_data)
    image.save(save_loc)
    pass

if __name__ == "__main__":
    # make directory for storing data
    output_dir = os.path.join(data.plot_dir,'rbm_plots')
    mkdir(output_dir, logger)
    
    # Model parameters
    learning_rate = 0.1
    training_epochs = 15
    batch_size = 20
    n_hidden = 500
    
    # sampling parameters
    n_chains = 50       # number of chains (horizontal axis)
    n_samples = 50      # number of samples (vertical axis)
    plot_every = 1000   # sep due to correlation of samples
    sample_fname = 'samples.png'
    
    logger.info('Loading data ...')
    source = data.Load_Data()
    
    datasets = source.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    # allocate symbolic variables for the data
    index = T.lscalar()     # index to a [mini]batch
    x = T.matrix('x')       # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    
    np_rng = np.random.RandomState(123)
    
    rbm_model = trainModel( # build and train the model
        train_set_x,
        n_hidden,
        learning_rate,
        training_epochs,
        batch_size,
        np_rng,
        plot_loc = output_dir
    )
    
    sampleModel( # sample from the trained model
        rbm_model,
        test_set_x,
        n_chains,
        n_samples,
        np_rng,
        save_loc = os.path.join(output_dir, sample_fname)
    )
    
    pass