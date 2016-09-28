import os, sys, timeit
import numpy as np

import theano
import dill as pickle

def predict(model_loc, source, logger):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
    
    # load the saved model
    classifier = pickle.load(open(model_loc))
    
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.inputs],
        outputs=classifier.y_pred)
    
    # We can test it on some examples from test test
    datasets = source.mnist()
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    
    predicted_values = predict_model(test_set_x[:10])
    logger.info("Predicted vs. Actual values for the first 10 examples in test set:")
    logger.info(predicted_values)
    logger.info(test_set_y[:10].eval())
    pass

def train(classifier, train_model, validate_model, test_model,
    n_train_batches, n_valid_batches, n_test_batches,
    n_epochs, learning_rate,
    patience, patience_increase, improvement_threshold,
    model_loc, logger):
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many minibatche before checking 
                                  # the network on the validation set; in this case 
                                  # we check every epoch
    
    best_validation_loss = np.inf
    best_i = 0
    test_score = 0.
    start_time = timeit.default_timer()
    
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            i = (epoch - 1) * n_train_batches + minibatch_index
            
            if (i+1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                
                logger.debug('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch,
                 minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
                
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss*improvement_threshold:
                        patience = max(patience, i * patience_increase)
                    
                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    
                    logger.debug(('epoch %i, batch %i/%i, '
                        'best test err: %f %%') % (epoch, minibatch_index + 1,
                        n_train_batches, test_score * 100.))
                    
            if patience <= i:
                done_looping = True
                break
    
    # save the best model
    with open(model_loc, 'wb') as f:
        pickle.dump(classifier, f)
    
    end_time = timeit.default_timer()
    logger.info('Done')
    logger.info(('Best valid err: %f %%, Test err: %f %%') % (best_validation_loss * 100.,
        test_score * 100.))
    logger.info('The code run for %d epochs, with %f epochs/sec' % (
        epoch, epoch / (end_time - start_time)))
    logger.info(('The code for file ' + 
        os.path.split(__file__)[1] +
        ' ran for %.1fs' % ((end_time - start_time))))
    logger.info('Predicting')
    pass