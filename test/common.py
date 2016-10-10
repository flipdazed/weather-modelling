import theano
import dill as pickle
__docformat__ = 'restructedtext en'

__doc__ = """
Functions that would otherwise be repeated across tests
"""

def predict(model_loc, source, logger, test_slice=slice(0,10)):
    """
    An example of how to load a trained model and use it
    to predict labels.
    
    :param model_loc: location of .pkl model
    :param source: data source, instance of Load_Data()
    :param logger: logger instance
    :param slice_obj: a slice() object to index the test data
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
    
    predicted_values = predict_model(test_set_x[test_slice])
    logger.info("Predicted vs. Actual values for the first "
                "{:0d} examples in test set:".format(predicted_values.size))
    logger.info(predicted_values)
    logger.info(test_set_y[test_slice].eval())
    pass