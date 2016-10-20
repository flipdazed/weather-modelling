import os, sys, timeit
import traceback

import numpy as np
import dill as pickle

import utils.visualise

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
    plot_dir, data_dir,
    **kwargs):
    """Train a model with train, validation and test data sets
    
    :type visualise_weights: list of dicts
    :param visualise_weights: dictionary of kwargs to pass to
        `utils.visualise.tileRasterImages` with an additional kwarg `name`
        which provides a name for identification and saving.
    
    :type visualise_cost: dict
    :param visualise_cost: dictionary containing
         - `name` to for identification and saving
         - `frequency` Optional. integer number of minibatches to output
            data default is `validation_frequency`
    
    :type visualise_params: list of dicts
    :param visualise_params: list dictionaries containing
         - `param` the theano tensor shared variable
         - `name` to for identification and saving
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
    
    # determine if visualisation is
    # carried out at each minibatch
    if any(['visualise' in key for key in kwargs.keys()]):
        # start the parameter manager
        param_man = utils.visualise.Visualise_Runtime(
            plot_dir=plot_dir, data_dir=data_dir)
    
    if 'visualise_weights' in kwargs:
        visualise_weights = kwargs['visualise_weights']
        
        for params in visualise_weights:
            # determine how often images are sampled
            # default to the validation frequency
            if 'frequency' not in params:
                params['frequency'] = validation_frequency
            
            # create the img array
            img_arr = utils.visualise.tileRasterImages(**params)
            
            # store img array
            n = params['name']
            param_man.imgs[n] = []
            param_man.imgs[n].append(img_arr)
            
            # save the data as an image
            logger.debug('Tracking Images: {}'.format(n))
    else:
        visualise_weights = False
    
    # record the cost as it occurs
    if 'visualise_cost' in kwargs:
        visualise_cost = kwargs['visualise_cost']
        
        # determine how often images are sampled
        # default to the validation frequency
        if 'frequency' not in visualise_cost:
            visualise_cost['frequency'] = validation_frequency
        
        # store cost as `np.nan` at zeroth time point
        param_man.cost.append(np.nan)
        logger.debug('Tracking cost')
    
    # record the parameters as they are changed
    if 'visualise_params' in kwargs:
        visualise_params = kwargs['visualise_params']
        
        param_files = []
        for param in visualise_params:
            
            # determine how often images are sampled
            # default to the validation frequency
            if 'frequency' not in params:
                param['frequency'] = validation_frequency
            
            # set up param entry in param manager
            n = param['name']
            param_man.params[n] = []
            param_man.params[n].append(np.nan)
            logger.debug('Tracking param: {}'.format(n))
    
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
                
                # create a plot for each validation_frequency
                if visualise_weights:
                    for params in visualise_weights:
                        n = params['name']
                        if (i+1) % params['frequency'] == 0:
                            img_arr = utils.visualise.tileRasterImages(
                                **params)
                            param_man.imgs[n].append(img_arr)
                
                # save the cost function to file
                if visualise_cost:
                    if (i+1) % visualise_cost['frequency'] == 0:
                        v = np.asscalar(minibatch_avg_cost)
                        param_man.cost.append(v)
                
                # save the parameters to a file
                if visualise_params:
                    for param in visualise_params:
                        if (i+1) % param['frequency'] == 0:
                            n = param['name']
                            v = np.linalg.norm(param['param'])
                            param_man.params[n].append(v)
                
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
    
    # close the visualise cost file
    if visualise_cost: param_man.saveCost(run_id = model_id)
    
    # record the parameters as they are changed
    if visualise_params: param_man.saveParams(run_id = model_id)
    
    # save the best model
    full_path = os.path.join(model_loc, model_id + '.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(classifier, f)
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