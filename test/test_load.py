from load import *

#
if __name__ == '__main__':
    logger = utils.logs.get_logger(__name__, update_stream_level=utils.logs.logging.DEBUG)
    
    logger.info('Testing Load_Data ...')
    
    source = Load_Data(location=data_loc)
    
    logger.info('Test data pull ...')
    test_data = source.test()
    
    # logger.info('Test All data pull ...')
    # allData = source.all()
    
    logger.info('Test MNIST data pull ...')
    mnist = source.mnist()
    pass