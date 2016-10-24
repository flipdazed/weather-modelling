import os, timeit
import traceback

import numpy as np
import caffeine

import data
import common
from models.dbn import *

__doc__ = """
    Demonstrates how to train and test a Deep Belief Network.
    
    This is demonstrated on MNIST.
    
    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations to run the optimiser
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a mini-batch
    """

# Save locations
## built model
MODEL = data.model_dir
MODEL_ID = os.path.splitext(os.path.basename(__file__))[0]

## visualising runtime parameters
DATA_DIR = data.data_dir
PLOT_DIR = data.plot_dir

# remove existing files
os.system('rm {}_*'.format(os.path.join(DATA_DIR, '', MODEL_ID)))

# network parameters
n_ins               = 28*28
hidden_layer_sizes  = [1000, 1000, 1000]
n_outs              = 10

# pre-training
k                   = 1     # number of Gibbs steps in CD/PCD
pretraining_epochs  = 100
pretrain_lr         = 0.01

# training (fine-tuning)
finetune_lr         = 0.1
training_epochs     = 1000
batch_size          = 10

# early-stopping parameters
patience                = 20000 # look as this many examples regardless
patience_increase       = 2     # wait this much longer if new best found
improvement_threshold   = 0.995 # consider this improvement significant
pretrain_vis_freq = 200
finetrain_vis_freq = 1

if __name__ == '__main__':
    
    logger = utils.logs.get_logger(__name__,
         update_stream_level=utils.logs.logging.INFO)
    logger.info('Loading data ...')
    source = data.Load_Data()
    
    datasets = source.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]// batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]// batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    # np random generator
    np_rng = np.random.RandomState(123)
    
    logger.info('Building the model ...')
    # construct the Deep Belief Network
    dbn = DBN(
        np_rng=np_rng,
        n_ins=n_ins,
        hidden_layers_sizes=hidden_layer_sizes,
        n_outs=n_outs
    )
    
    logger.debug('building pre-training functions')
    pretraining_fns = dbn.pretrainingFunctions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)
    
    # visualise cost during runtime
    visualise_cost = {          # visualising the cost
        'cost':{'freq':pretrain_vis_freq}       # frequency of sampling
        }
    
    # visualise arbitrary parameters at runtime
    visualise_params = {}
    param_man = utils.visualise.Visualise_Runtime(
        plot_dir=PLOT_DIR,
        data_dir=DATA_DIR
    )
    param_man.initalise(
        run_id = MODEL_ID,
        default_freq = patience,
        cost = visualise_cost
        )
    
    logger.info('Pre-training the model ...')
    start_time = timeit.default_timer()
    
    param_man.getValues(i = -1, cost = np.nan)
    param_man.writeRuntimeValues(i = -1, clean_files = True)
    
    for l in range(dbn.n_layers): # Pre-train layer-wise
        # Allows a bored person to quit the run
        # without losing everything!
        try: # control+c doesn't loose everything!
            logger.debug('Pre-training layer: {}'.format(l))
            time_pretrain_start = timeit.default_timer()
            
            new_params = {
                'hiddenLayer{:02d}'.format(l) + '_weights': {
                    'x': dbn.sigmoid_layers[l].w.get_value(
                        borrow=True).ravel(),
                    'freq':pretrain_vis_freq,
                },
                'hiddenLayer{:02d}'.format(l) + '_bias': {
                    'x': dbn.sigmoid_layers[l].b.get_value(
                        borrow=True).ravel(),
                    'freq':pretrain_vis_freq,
                }
            }
            
            param_man.initalise(
                run_id = MODEL_ID,
                default_freq = patience,
                params = new_params
                )
            
            # go through pretraining epochs 0th epoch is before start
            for epoch in range(1, pretraining_epochs+1):
                
                # go through the training set
                costs = []
                for minibatch_index in range(n_train_batches):
                    c = pretraining_fns[l](
                        index=minibatch_index,
                        lr=pretrain_lr
                    )
                    costs.append(c)
                    i = (epoch - 1) * n_train_batches + minibatch_index
                    
                    param_man.getValues(i = i, cost = np.asscalar(c))
                    param_man.writeRuntimeValues(i = i)
                
                av_cost = np.mean(costs)
                time_pretrain_end_i = timeit.default_timer()
                time_pretrain_i = time_pretrain_end_i - time_pretrain_start
                logger.debug('Pre-training layer: {}, epoch {}, cost {},'
                    ' time {}s'.format(l, epoch, av_cost, time_pretrain_i))
            
        # these sections handle errors nicely
        except KeyboardInterrupt:
            logger.warn('Manual Exit!')
            logger.warn('Moving to clean-up ...')
        except:
            logger.error('Unplanned Exit!')
            for line in traceback.format_exc().split("\n"):
                logger.error(line)
        
        del param_man.params['hiddenLayer{:02d}'.format(l) + '_weights']
        del param_man.params['hiddenLayer{:02d}'.format(l) + '_bias']
    
    end_time = timeit.default_timer()
    logger.info('The pretraining code for file '
        + os.path.split(__file__)[1]
        + ' ran for {:.2f}m'.format((end_time - start_time) / 60.))
    
    logger.info('Training (fine-tuning) the model ...')
    
    # visualise cost during runtime
    visualise_cost = {          # visualising the cost
        'cost':{'freq':finetrain_vis_freq}       # frequency of sampling
        }
    
    new_params = {
        'logitLayer' + '_weights': {
        'x': dbn.logitLayer.w.get_value(borrow=True).ravel(),
        'freq':finetrain_vis_freq,
        },
        'logitLayer' + '_bias': {
        'x':dbn.logitLayer.b.get_value(borrow=True).ravel(),
        'freq':finetrain_vis_freq,
        }
    }
    
    for l in range(dbn.n_layers):
        new_params['hiddenLayer{:02d}'.format(l) + '_weights'] = {
                'x': dbn.sigmoid_layers[l].w.get_value(
                    borrow=True).ravel(),
                'freq':finetrain_vis_freq,
            }
        new_params['hiddenLayer{:02d}'.format(l) + '_bias'] = {
                'x': dbn.sigmoid_layers[l].b.get_value(
                    borrow=True).ravel(),
                'freq':finetrain_vis_freq,
            }
    
    # Visualise these items during training
    visualise_weights = {       # dict of images to create
        'hiddenLayer00' + '_weights': {    # input - hiddenlayer image
            'x':dbn.sigmoid_layers[0].w.get_value(
                borrow=True).T,         # the parameter
            'img_shape':(28, 28),       # prod. of tuple == # input nodes
            'tile_shape':(15, 30),      # Max number is # nodes in next layer
            'runtime_plots':True
        },
        'logitLayer' + '_weights': {    # hidden - logistic layer
            'x':dbn.logitLayer.w.get_value(borrow=True).T,
            'img_shape':(40, 25),       # prod. of tuple == # hidden nodes
            'tile_shape':(20, 32),
        }
    }
    # add the weights for each hidden layer
    # This can only be done as hidden_layer_sizes are all the same
    for i in range(1, dbn.n_layers):
        visualise_weights['hiddenLayer{:02d}'.format(i) + '_weights'] = {
            'x':dbn.sigmoid_layers[i].w.get_value(borrow=True).T,
            'img_shape':(40, 25),
            'tile_shape':(25, 25),
        }
    
    param_man.initalise(
        run_id = MODEL_ID,
        default_freq = patience,
        params = new_params,
        imgs = visualise_weights,
        cost = visualise_cost
        )
    
    # get the training, validation and testing function for the model
    logger.debug('building fine-tuning functions')
    train_model, validate_model, test_model = dbn.buildFinetuneFunctions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    
    logger.debug('training')
    
    utils.training.train(dbn, train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches,
        training_epochs, finetune_lr,
        patience, patience_increase, improvement_threshold,
        MODEL, MODEL_ID, logger,
        visualise=param_man
    )
    
    logger.info('Testing the model ...')
    common.predict(os.path.join(MODEL, MODEL_ID + ".pkl"), source, logger)
    pass