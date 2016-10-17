import os, sys, timeit

import numpy as np
import dill as pickle

import utils
try: import PIL.Image as Image
except ImportError: import Image

def train(classifier, train_model, validate_model, test_model,
    n_train_batches, n_valid_batches, n_test_batches,
    n_epochs, learning_rate,
    patience, patience_increase, improvement_threshold,
    model_loc, logger, **kwargs):
    """
    
    :type visualise_weights: list of dicts
    :param visualise_weights: dictionary of kwargs to pass to `utils.visualise.tileRasterImages`
    
    :type visualise_cost: dict
    :param visualise_cost: dictionary containing 
        `save_loc` : the save location without file extension
        `frequency` (optionally) : integer number of minibatches to output data
                                    default is `validation_frequency`
    
    :type visualise_params: list of dicts
    :param visualise_params: list dictionaries containing 
        `param` : the theano tensor shared variable
        `save_loc` : the save location without file extension
        `frequency` (optionally) : integer number of minibatches to output data
                                    default is `validation_frequency`
    """
    # go through this many minibatche before checking 
    # the network on the validation set; in this case 
    # we check every epoch
    validation_frequency = min(n_train_batches, patience // 2)
    
    best_validation_loss = np.inf
    best_i = 0
    test_score = 0.
    start_time = timeit.default_timer()
    
    epoch = 0
    done_looping = False
    
    # determine if visualisation is 
    # carried out at each minibatch
    if 'visualise_weights' in kwargs:
        visualise_weights = kwargs['visualise_weights']
        
        for params in visualise_weights:
            # determine how often images are sampled
            # default to the validation frequency
            if 'frequency' not in params: 
                params['frequency'] = validation_frequency
            
            img_arr = utils.visualise.tileRasterImages(**params)
            image = Image.fromarray(img_arr)
            loc = params['save_loc'] + '_at_mb_{:04d}.png'.format(epoch)
            logger.debug('Weight visualisation saved to: {}'.format(loc))
            image.save(loc)
    else: 
        visualise_weights = False
    
    # record the cost as it occurs
    if 'visualise_cost' in kwargs:
        visualise_cost = kwargs['visualise_cost']
        
        # determine how often images are sampled
        # default to the validation frequency
        if 'frequency' not in visualise_cost: 
            visualise_cost['frequency'] = validation_frequency
        
        loc = visualise_cost['save_loc'] + '.dat'
        visualise_cost_file = open(loc, "w")
        logger.debug('Cost will be saved to: {}'.format(loc))
    
    # record the parameters as they are changed
    if 'visualise_params' in kwargs:
        visualise_params = kwargs['visualise_params']
        
        param_files = []
        for params in visualise_params:
            
            # determine how often images are sampled
            # default to the validation frequency
            if 'frequency' not in params: 
                params['frequency'] = validation_frequency
            
            loc = params['save_loc'] + '.dat'
            param_files.append(open(loc, "w"))
            logger.debug('Parameter saved to: {}'.format(loc))
    
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
                    this_validation_loss * 100., np.asscalar(minibatch_avg_cost),
                    train_time_i/60.))
                
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss
                         < best_validation_loss*improvement_threshold):
                        patience = max(patience, i * patience_increase)
                    
                    best_validation_loss = this_validation_loss
                    
                    # test it on the test set
                    test_losses = [test_model(j) for j in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    test_end_i = timeit.default_timer()
                    test_time_i = test_end_i - epoch_start_i
                    logger.info('Epoch {:3d}, Batch     {:3d}/{:3d}: ' 
                        'Best Test Err {:.3f}%, '
                        'Test Time {:.2f} mins'.format(
                        epoch, minibatch_index + 1, n_train_batches, 
                        test_score * 100., test_time_i/60.))
            
            # create a plot for each validation_frequency
            if visualise_weights:
                for params in visualise_weights:
                    
                    if (i+1) % params['frequency'] == 0:
                        img_arr = utils.visualise.tileRasterImages(**params)
                        image = Image.fromarray(img_arr)
                        image.save(params['save_loc'] + '_at_mb_{:04d}.png'.format(epoch))
            
            # save the cost function to file
            if visualise_cost:
                if (i+1) % visualise_cost['frequency'] == 0:
                    visualise_cost_file.write('{}\n'.format(minibatch_avg_cost))
            
            # save the parameters to a file
            if visualise_params:
                for param_file, param in zip(param_files, visualise_params):
                    if (i+1) % param['frequency'] == 0:
                        param_file.write('{}\n'.format(np.linalg.norm(param['param'])))
            
            if patience <= i:
                logger.info('Lost patience after {:3d} examples'.format(i))
                done_looping = True
                break
    
    # close the visualise cost file
    if visualise_cost: visualise_cost_file.close()
    
    # record the parameters as they are changed
    if visualise_params:
        for param_file in param_files:
            param_file.close()
    
    # save the best model
    with open(model_loc, 'wb') as f: 
        pickle.dump(classifier, f)
    logger.info('Model saved: {}'.format(model_loc))
    
    end_time = timeit.default_timer()
    logger.info('Best valid err: {:.3f} %, Test err: {:.3f} %'.format(
    best_validation_loss * 100., test_score * 100.))
    logger.info('The code run for {:0d} epochs, with {:.4f} epochs/sec'.format(
        epoch, epoch / (end_time - start_time)))
    logger.info(('The code for file ' + 
        os.path.split(__file__)[1] +
        ' ran for {:.1f}s'.format((end_time - start_time))))
    pass