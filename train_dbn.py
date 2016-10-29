from sys import platform

if platform == "linux" or platform == "linux2":
    # linux
    pass
elif platform == "darwin":
    # OS X
    import caffeine
elif platform == "win32":
    # Windows...
    pass

import os, timeit, traceback
import numpy as np

import data
import utils
from models.dbn import DBN

__docformat__ = 'restructedtext en'

__doc__ = """
This code trains a Deep Belief Network on the weather data

Data Set:
    - 582 events
    - 10092 features
    - binary result of TRUE|FALSE

The features are unknown preprocessed weather data and the result corresponds to an ice-storm event
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
n_ins               = 10092
hidden_layer_sizes  = [5000, 2500, 1000]
n_outs              = 2

# pre-training
k                   = 15     # number of Gibbs steps in CD/PCD
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
pretrain_vis_freq = False
finetrain_vis_freq = False

if __name__ == '__main__':
    
    logger = utils.logs.get_logger(__name__,
        update_stream_level=utils.logs.logging.DEBUG)
    logger.info('Loading data ...')
    source = data.Load_Data(location=data.data_loc)
    
    datasets = source.all()
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
    
    # visualise arbitrary parameters at runtime
    visualise_params = {}
    
    # visualise cost during runtime
    visualise_cost = {          # visualising the cost
        'cost':{'freq':pretrain_vis_freq}       # frequency of sampling
        }
    
    # Visualise these items during training
    visualise_weights = {
        'inputLayer' + '_weights': {
            'x':dbn.sigmoid_layers[0].w,
            'img_shape':(29*2, 29*2*3),
            'tile_shape':(25, 40),
            'tile_spacing':(1, 1),
            'freq':pretrain_vis_freq,
            'runtime_plots':True
        }
    }
    
    param_man = utils.visualise.Visualise_Runtime(
        plot_dir=PLOT_DIR,
        data_dir=DATA_DIR
    )
    param_man.initalise(
        run_id = MODEL_ID,
        imgs = visualise_weights,
        default_freq = n_train_batches//2,
        cost = visualise_cost
        )
    
    logger.info('Pre-training the model ...')
    start_time = timeit.default_timer()
    
    param_man.getValues(
            i = -1, # -1 because print at (i+1) % freq
            cost = np.nan,
            # fill with np.nan for each update that is expected
            updates = [np.nan]*len(param_man.updates.keys()))
    param_man.writeRuntimeValues(i = -1, clean_files = True)
    
    for l in range(dbn.n_layers): # Pre-train layer-wise
        # Allows a bored person to quit the run
        # without losing everything!
        try: # control+c doesn't loose everything!
            logger.debug('Pre-training layer: {}'.format(l))
            layer_start = timeit.default_timer()
            
            # this particular section pull parameters from the RBM
            # as the RBM is used for pretraining the Hidden Layers
            new_params = {
                'hiddenLayerRBM{:02d}'.format(l) + '_weights': {
                    'x': dbn.rbm_layers[l].w,
                    'freq':pretrain_vis_freq,
                },
                'hiddenLayerRBM{:02d}'.format(l) + '_hbias': {
                    'x': dbn.rbm_layers[l].hbias,
                    'freq':pretrain_vis_freq,
                },
                'hiddenLayerRBM{:02d}'.format(l) + '_vbias': {
                    'x': dbn.rbm_layers[l].vbias,
                    'freq':pretrain_vis_freq,
                }
            }
            
            # the update_position is fixed because we delete 
            # the current dict at end of loop
            visualise_updates = {
                'hiddenLayerRBM{:02d}'.format(l) + '_weights': {
                    'update_position':0,
                    'freq':pretrain_vis_freq
                },
                'hiddenLayerRBM{:02d}'.format(l) + '_hbias': {
                    'update_position':1,
                    'freq':pretrain_vis_freq
                },
                'hiddenLayerRBM{:02d}'.format(l) + '_vbias': {
                    'update_position':2,
                    'freq':pretrain_vis_freq
                }
            }
            
            param_man.initalise(
                run_id = MODEL_ID,
                default_freq = n_train_batches//2,
                params = new_params,
                updates = visualise_updates,
                cost = visualise_cost
                )
            
            # go through pretraining epochs 0th epoch is before start
            for epoch in range(1, pretraining_epochs+1):
                epoch_start = timeit.default_timer()
                # go through the training set
                costs = []
                for minibatch_index in range(n_train_batches):
                    result = pretraining_fns[l](
                        index=minibatch_index,
                        lr=pretrain_lr
                    )
                    
                    if type(result) == list:
                        # accomodates return of gparams in result
                        c = result.pop(0)
                        costs.append(c)
                        updates = [(g*pretrain_lr).mean() for g in result]
                    else:
                        # no gparams in result
                        c = result
                        costs.append(c)
                        updates = None
                    
                    i = (epoch - 1) * n_train_batches + minibatch_index
                    
                    param_man.getValues(
                        i = i,
                        cost = np.mean(costs),
                        updates = updates
                    )
                    param_man.writeRuntimeValues(i = i)
                
                av_cost = np.mean(costs)
                epoch_end = timeit.default_timer()
                epoch_time = epoch_end - epoch_start
                logger.debug('Pre-training layer: {}, epoch {}, cost {},'
                    ' time {}s'.format(l, epoch, av_cost, epoch_time))
            
        # these sections handle errors nicely
        except KeyboardInterrupt:
            logger.warn('Manual Exit!')
            logger.warn('Moving to clean-up ...')
        except:
            logger.error('Unplanned Exit!')
            for line in traceback.format_exc().split("\n"):
                logger.error(line)
        
        layer_end = timeit.default_timer()
        layer_time = layer_end - layer_start
        logger.info('done pre-training layer: {}. Time: {} mins'.format(
            l, layer_time/60.))
        
        if param_man.imgs: # only want images from first layer
            param_man.imgs = {}
        del param_man.cost['cost']
        del param_man.params['hiddenLayerRBM{:02d}'.format(l) + '_weights']
        del param_man.params['hiddenLayerRBM{:02d}'.format(l) + '_hbias']
        del param_man.params['hiddenLayerRBM{:02d}'.format(l) + '_vbias']
        del param_man.updates['hiddenLayerRBM{:02d}'.format(l) + '_weights']
        del param_man.updates['hiddenLayerRBM{:02d}'.format(l) + '_hbias']
        del param_man.updates['hiddenLayerRBM{:02d}'.format(l) + '_vbias']
    
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
            'x': dbn.logitLayer.w,
            'freq':finetrain_vis_freq,
        },
        'logitLayer' + '_bias': {
            'x':dbn.logitLayer.b,
            'freq':finetrain_vis_freq,
        }
    }
    visualise_updates = {
            'logitLayer' + '_weights': {
                'update_position':dbn.n_layers-2,
                'freq':finetrain_vis_freq
            },
            'logitLayer' + '_bias': {
                'update_position':dbn.n_layers-1,
                'freq':finetrain_vis_freq
            }
        }
    
    for l in range(dbn.n_layers):
        new_params['hiddenLayer{:02d}'.format(l) + '_weights'] = {
                'x': dbn.sigmoid_layers[l].w,
                'freq':finetrain_vis_freq,
            }
        new_params['hiddenLayer{:02d}'.format(l) + '_bias'] = {
                'x': dbn.sigmoid_layers[l].b,
                'freq':finetrain_vis_freq,
            }
        visualise_updates['hiddenLayer{:02d}'.format(l) + '_weights'] = {
                'update_position':l*2,
                'freq':finetrain_vis_freq
            }
        visualise_updates['hiddenLayer{:02d}'.format(l) + '_bias'] = {
                'update_position':l*2+1,
                'freq':finetrain_vis_freq
            }
    
    # Visualise these items during training
    visualise_weights = {       # dict of images to create
        'inputLayer' + '_weights': {    # input - hiddenlayer image
            'x':dbn.sigmoid_layers[0].w,# the parameter
            'img_shape':(29*2, 29*2*3), # prod. of tuple == # input nodes
            'tile_shape':(25, 40),    # Max number is # nodes in next layer
            'tile_spacing':(1, 1),      # separate imgs x,y
            'freq':finetrain_vis_freq,
            'runtime_plots':True
        },
        'logitLayer' + '_weights': {    # hidden - logistic layer
            'x':dbn.logitLayer.w,
            'img_shape':(40, 25),     # prod. of tuple == # hidden nodes
            'tile_shape':(1, 2),
            'freq':finetrain_vis_freq
        }
    }
    # add the weights for each hidden layer
    # This can only be done as hidden_layer_sizes are all the same
    for i in range(1, dbn.n_layers):
        visualise_weights['hiddenLayer{:02d}'.format(i) + '_weights'] = {
            'x':dbn.sigmoid_layers[i].w,
            'img_shape':(40, 25),
            'tile_shape':(25, 25),
            'freq':finetrain_vis_freq
        }
    
    param_man.initalise(
        run_id = MODEL_ID,
        default_freq = n_train_batches//2,
        params = new_params,
        imgs = visualise_weights,
        cost = visualise_cost,
        updates = visualise_updates
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
    pass