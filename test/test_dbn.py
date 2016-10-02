from __future__ import division

import os, timeit
import numpy as np

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
    :param training_epochs: maximal number of iations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

MODEL = os.path.join(data.model_dir, os.path.splitext(os.path.basename(__file__))[0]+'.pkl')

if __name__ == '__main__':
    pretraining_epochs=100
    pretrain_lr=0.01
    finetune_lr=0.1
    k=1
    training_epochs=1000
    batch_size=10
    
    logger = utils.logs.get_logger(__name__, update_stream_level=utils.logs.logging.DEBUG)
    logger.info('Loading data ...')
    source = data.Load_Data()
    
    datasets = source.mnist()
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
    dbn = DBN(np_rng=np_rng, n_ins=28 * 28,
              hidden_layers_sizes=[1000, 1000, 1000],
              n_outs=10)
    
    logger.debug('building pre-training functions')
    pretraining_fns = dbn.pretrainingFunctions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)
    
    logger.info('Pre-training the model ...')
    start_time = timeit.default_timer()
    
    for i in range(dbn.n_layers): # Pre-train layer-wise
        time_pretrain_start = timeit.default_timer()
        for epoch in range(pretraining_epochs): # go through pretraining epochs
            c = [pretraining_fns[i](index=batch_index, # go through the training set
                                    lr=pretrain_lr) for batch_index in range(n_train_batches)]
            time_pretrain_end_i = timeit.default_timer()
            time_pretrain_i = time_pretrain_end_i - time_pretrain_start
            logger.debug('Pre-training layer {}, epoch {}, cost {}, time {}s'.format(i, epoch, np.mean(c), time_pretrain_i))
    
    end_time = timeit.default_timer()
    logger.info('The pretraining code for file ' + os.path.split(__file__)[1] +
        ' ran for {:.2f}m'.format((end_time - start_time) / 60.))
    
    logger.info('Training (fine-tuning) the model ...')
    
    # get the training, validation and testing function for the model
    logger.debug('building fine-tuning functions')
    train_model, validate_model, test_model = dbn.buildFinetuneFunctions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    
    # early-stopping parameters
    patience_increase = 2.          # wait this much longer when a new best is found
    improvement_threshold = 0.995   # consider this relative improvement significant
    patience = 4 * n_train_batches  # look as this many examples regardless
    
    logger.debug('training')
    common.train(dbn, train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches,
        training_epochs, finetune_lr,
        patience, patience_increase, improvement_threshold, 
        MODEL, logger)
    
    logger.info('Testing the model ...')
    common.predict(MODEL, source, logger)
    pass