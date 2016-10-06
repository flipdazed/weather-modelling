#!/usr/bin/env python
import argparse

import numpy as np
import theano
import dill as pickle
import pandas as pd

import data
import utils

__docformat__ = 'restructedtext en'

__doc__ = """
Runs a specified model across the weather dataset.

Run from command line as,
    
    ./predict_dbn.py -m dump/models/result_dbn.pkl -s 0,-1,10

this will run the stored DBN model across the sliced test set
of data corresponding to `test_data[slice(0,-1,10)]`
"""

def predict(model_loc, data_set_x, data_set_y, logger, test_slice=slice(0,10)):
    """
    load a trained model and use it to predict labels
    
    :param model_loc: location of .pkl model
    :type data_set_x: theano.tensor.TensorType
    :param data_set_x: Shared var. that contains test data points
    :type data_set_y: theano.tensor.TensorType
    :param data_set_y: Shared var. that contains test data labels
    :param logger: logger instance
    :type test_slice: slice
    :param test_slice: section of the test data to test on
    """
    
    # load the saved model
    logger.debug('loading model')
    classifier = pickle.load(open(model_loc))
    
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.inputs],
        outputs=classifier.y_pred)
    
    # We can test it on some examples from test test
    x = data_set_x.get_value()
    actual_values = data_set_y[test_slice].eval()
    
    logger.debug('running model')
    predicted_values = predict_model(x[test_slice])
    
    # create a pandas dataframe of the results
    df = pd.DataFrame(
        {
            'Predicted':predicted_values, 
            'Actual':actual_values
        },
        index = range(*test_slice.indices(x.shape[0]))
    )
    df['Correct?'] = np.where(df['Predicted']==df['Actual'], True, False)
    correct = df['Correct?'].sum()
    
    # output full results
    output_lines = df.to_string().split('\n')
    for line in output_lines: logger.debug(line)
    
    logger.info(
        "Prediction Accuracy: "
        "{:4d}/{:4d} ({:.2f} %)".format(
            correct,
            predicted_values.size,
            correct/float(predicted_values.size)*100.
        )
    )
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Runs a stored model across the test weather data'
    )
    parser.add_argument("-m", "--model-loc",
        help = "location of the model e.g. dump/models/result_mlp.pkl",
        required=True
    )
    parser.add_argument("-s", "--slice",
        help = "comma sep. arguments to slice() to specify section of test data set"
        " e.g. 0, -1 gives slice(0,-1)",
        required=False,
        default='0, -1, 4'
    )
    args = vars(parser.parse_args())
    model_loc = args['model_loc']
    data_slice = slice(*(int(i.strip()) for i in args['slice'].split(',')))
    
    logger = utils.logs.get_logger(__name__,
        update_stream_level=utils.logs.logging.DEBUG)
    logger.info('Loading data ...')
    source = data.Load_Data(location=data.data_loc)
    
    datasets = source.all()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    predict(model_loc, test_set_x, test_set_y, logger, data_slice)
    pass