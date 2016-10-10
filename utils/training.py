import os, sys, timeit
import numpy as np

import dill as pickle

def train(classifier, train_model, validate_model, test_model,
    n_train_batches, n_valid_batches, n_test_batches,
    n_epochs, learning_rate,
    patience, patience_increase, improvement_threshold,
    model_loc, logger):
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
    
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        
        epoch_start_i = timeit.default_timer()
        for minibatch_index in range(n_train_batches):
            
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            i = (epoch - 1) * n_train_batches + minibatch_index
            
            if (i+1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) 
                    for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                
                train_end_i = timeit.default_timer()
                train_time_i = train_end_i - epoch_start_i
                logger.debug('Epoch {:3d}, minibatch {:3d}/{:3d}: '
                'Valid Err {:.3f}%; Valid Time {:.2f} mins'.format(
                    epoch, minibatch_index + 1, n_train_batches, 
                    this_validation_loss * 100., train_time_i/60.))
                
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss
                         < best_validation_loss*improvement_threshold):
                        patience = max(patience, i * patience_increase)
                    
                    best_validation_loss = this_validation_loss
                    
                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    test_end_i = timeit.default_timer()
                    test_time_i = test_end_i - epoch_start_i
                    logger.info('Epoch {:3d}, Batch     {:3d}/{:3d}: ' 
                        'Best Test Err {:.3f}%, '
                        'Test Time {:.2f} mins'.format(
                        epoch, minibatch_index + 1, n_train_batches, 
                        test_score * 100., test_time_i/60.))
                    
            if patience <= i:
                logger.info('Lost patience after {:3d} examples'.format(i))
                done_looping = True
                break
    
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