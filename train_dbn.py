import os, timeit
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

MODEL = os.path.join(data.model_dir, os.path.splitext(os.path.basename(__file__))[0]+'.pkl')

# network parameters
n_ins               = 10092
hidden_layer_sizes  = [15000, 15000, 15000]
n_outs              = 2

# pre-training
k                   = 1     # number of Gibbs steps in CD/PCD
pretraining_epochs  = 100
pretrain_lr         = 0.01

# training (fine-tuning)
finetune_lr         = 0.1
training_epochs     = 1000
batch_size          = 10

# early-stopping parameters
patience                = 40    # look as this many examples regardless
patience_increase       = 5     # wait this much longer when a new best is found
improvement_threshold   = 0.995 # consider this relative improvement significant

if __name__ == '__main__':
    
    logger = utils.logs.get_logger(__name__, update_stream_level=utils.logs.logging.DEBUG)
    logger.info('Loading data ...')
    source = data.Load_Data(location=data.data_loc)
    
    datasets = source.all()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    # np random generator
    np_rng = np.random.RandomState(123)
    
    logger.info('Building the model ...')
    # construct the Deep Belief Network
    dbn = DBN(np_rng=np_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layer_sizes,
              n_outs=n_outs)
    
    logger.debug('building pre-training functions')
    pretraining_fns = dbn.pretrainingFunctions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)
    
    logger.info('Pre-training the model ...')
    start_time = timeit.default_timer()
    
    for i in range(dbn.n_layers): # Pre-train layer-wise
        time_pretrain_start = timeit.default_timer()
        for epoch in range(pretraining_epochs): # go through pretraining epochs
            time_pretrain_start_i = timeit.default_timer()
            c = [pretraining_fns[i](index=batch_index, # go through the training set
                                    lr=pretrain_lr) 
                for batch_index in range(n_train_batches)]
            time_pretrain_end_i = timeit.default_timer()
            time_pretrain_i = time_pretrain_end_i - time_pretrain_start_i
            logger.debug('Pre-training layer {:0d}, epoch {:3d}, '
                'cost {:.3e}, time {:.2f} secs'.format(
                i, epoch, np.mean(c), time_pretrain_i))
    
    end_time = timeit.default_timer()
    logger.info('The pretraining code for file ' + os.path.split(__file__)[1] +
        ' ran for {:.2f} mins'.format((end_time - start_time) / 60.))
    
    logger.info('Training (fine-tuning) the model ...')
    
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
        MODEL, logger)
    pass