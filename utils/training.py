import os, sys, timeit
import traceback

import numpy as np
import dill as pkl

__docformat__ = 'restructedtext en'

__doc__ = """
Visualising Parameters at Runtime
===

A description of the dictionaries required for visualisation

Visualisation of weights
---
One image per node in the following layer. Each image will have `x` and `y`
dimensions such that: `x*y == nodes_in_prev_layer`

Thus for an input image of `28*28` pixels and a hidden layer of `500` nodes
this can be visualised as `500` separate images, each consisting of `728`
pixel the represent their respective weights.

# Required Input

The input expected is a `list` of dictionaries. Each dictionary contains
arguments to pass to the function `utils.visualise.tileRasterImages`
and are described there in detail.

There are two additional arguments to be supplied in the dictionary:
    
:type name: string
:param name: a name for identification and saving
:type frequency: int
:param frequency: Optional. Default is `validation_frequency` as defined
    in `utils.training.train`. Creates an image after every multiple
    of this number of minibatches.

Visualisation of Cost
---
Cost is stored as a data file at runtime when a dictionary is provided as a keyword
argument pair to the function `utils.training.train`. The dictionary should contain
the following entries:
    
:type name: string
:param name: a name for identification and saving
:param frequency: Optional. Default is `validation_frequency` as defined
    in `utils.training.train`. Creates an image after every multiple
    of this number of minibatches.

Visualisation of Parameters
---
Parameters are stored as data files at runtime when a list of dictionaries is provided. Each dictionary should be structured as follows:
    
:type param: np.ndarray
:param param: A one-dimensional array of parameter values corresponding
     to the current minibatch. Multidimensional arrays will require a 
     flattening operation e.g. `.ravel()`
:param name: as above in *Visualisation of Cost*
:param frequency: as above *Visualisation of Cost*

"""

def train(classifier,
    train_model, validate_model, test_model,
    n_train_batches, n_valid_batches, n_test_batches,
    n_epochs, learning_rate,
    patience, patience_increase, improvement_threshold,
    model_loc, model_id,
    logger,
    **kwargs):
    """Train a model with train, validation and test data sets
    
    :type visualise_weights: dict
    :param visualise_weights: key is `str` for identification and saving. 
        Value is a nested dictionary of kwargs to pass to
        `utils.visualise.tileRasterImages` with an additional kwarg `name`
        which provides a name for identification and saving.
    
    :type visualise_cost: dict
    :param visualise_cost: key is `str` for identification and saving. 
        Values are given by:
         - `frequency` Optional. integer number of minibatches to output
            data default is `validation_frequency`
    
    :type visualise_params: dict
    :param visualise_params: key is `str` for identification and saving. 
        Values are given by:
         - `param` the theano tensor shared variable
         - `frequency` Optional. default is `validation_frequency`.
            `int` number of minibatches to output data.
    """
    
    # go through this many minibatche before checking
    # the network on the validation set; in this case
    # we check every epoch
    validation_frequency = min(n_train_batches, patience // 2)
    
    start_time = timeit.default_timer()
    
    best_validation_loss = np.inf
    best_i = 0
    test_score = 0.
    
    epoch = 0
    done_looping = False
    
    if 'visualise' in kwargs:
        param_man = kwargs['visualise']
        param_man.getValues(i = -1, cost = np.nan)
        param_man.writeRuntimeValues(i= -1, clean_files = True)
    else:
        param_man = None
    
    # Allows a bored person to quit the run
    # without losing everything!
    try: # control+c doesn't loose everything!
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            
            epoch_start_i = timeit.default_timer()
            for minibatch_index in range(n_train_batches):
                
                minibatch_avg_cost = train_model(minibatch_index)
                
                # iteration number
                i = (epoch - 1) * n_train_batches + minibatch_index
                if (i+1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(j)
                        for j in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    
                    train_end_i = timeit.default_timer()
                    train_time_i = train_end_i - epoch_start_i
                    logger.debug('Epoch {:3d}, minibatch {:3d}/{:3d}: '
                        'Valid Err {:.3f}%; Cost {:8.1e} '
                        'Valid Time {:.2f} mins'.format(
                            epoch, minibatch_index + 1, n_train_batches,
                            this_validation_loss * 100.,
                            np.asscalar(minibatch_avg_cost), train_time_i/60.
                        )
                    )
                    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss
                             < best_validation_loss*improvement_threshold):
                            patience = max(patience, i * patience_increase)
                        
                        best_validation_loss = this_validation_loss
                        
                        # test it on the test set
                        test_losses = [
                            test_model(j)
                            for j in range(n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        test_end_i = timeit.default_timer()
                        test_time_i = test_end_i - epoch_start_i
                        logger.info('Epoch {:3d}, Batch     {:3d}/{:3d}: '
                            'Best Test Err {:.3f}%, '
                            'Test Time {:.2f} mins'.format(
                                epoch, minibatch_index + 1, n_train_batches,
                                test_score * 100., test_time_i/60.
                            )
                        )
                
                if param_man:
                    param_man.getValues(
                        i = i,
                        cost = np.asscalar(minibatch_avg_cost)
                    )
                    param_man.writeRuntimeValues(i=i)
                
                if patience <= i:
                    logger.info('Lost patience after {:3d} examples'.format(
                        i))
                    done_looping = True
                    break
    
    # these sections handle errors nicely
    except KeyboardInterrupt:
        logger.warn('Manual Exit!')
        logger.warn('Moving to clean-up ...')
    except:
        logger.error('Unplanned Exit!')
        for line in traceback.format_exc().split("\n"):
            logger.error(line)
        logger.warn('Moving to clean-up ...')
    
    logger.info('Running clean-up ...')
    
    # save all the recorded values
    param_man.saveValues()
    
    # save the best model
    full_path = os.path.join(model_loc, model_id + '.pkl')
    with open(full_path, 'wb') as f:
        pkl.dump(classifier, f)
    logger.info('Model saved: {}'.format(full_path))
    
    end_time = timeit.default_timer()
    logger.info('Best valid err: {:.3f} %, Test err: {:.3f} %'.format(
        best_validation_loss * 100., test_score * 100.))
    logger.info('The code run for {:0d} epochs, '
        'with {:.4f} epochs/sec'.format(
            epoch, epoch / (end_time - start_time)
        )
    )
    logger.info(('The code for file ' +
        os.path.split(__file__)[1] +
        ' ran for {:.1f}s'.format((end_time - start_time))))
    
    return param_man