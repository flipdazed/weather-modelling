import utils
from configparser import ConfigParser
from glob import glob

import numpy as np
import theano
import theano.tensor as T

# This is just required by the MNIST section
import cPickle
import gzip
import os
import urllib
# end collection

# get configuration
config = ConfigParser()
config.read(u'config.ini')
mnist_url       = config.get('Load Data', 'mnist_url')
mnist_loc       = config.get('Load Data', 'mnist_loc')
icestorm_loc    = config.get('Load Data', 'icestorm_loc')
log             = config.get('Global', 'log')
utils.logs.getHandlers(filename=log, mode='w+')

class Load_Data(object):
    """Loading Data for Classification
    
    Stores the location of the datasets and generates list of data files
    
    The methods get_all() and get_test() are used to call the datasets
    """
    def __init__(self, location, searchParam='day*'):
        
        # Set up logger to track progress
        utils.logs.get_logger(self=self)
        
        # set location
        self.logger.debug("Data location: %s" % location)
        self.location = location
        
        # get data files as list
        self.dataFiles = glob(location+searchParam)
        
        # check files exits
        if self.dataFiles:
            self.logger.debug("Obtained %d files like '%s'" % (len(self.dataFiles), searchParam))
        else:
            self.logger.error("No data files found")
            raise ValueError("No data files found in location:\n\t%s\n\tlike: '%s'" % \
                 (location, searchParam))
        pass
    def all(self):
        """Loads test data"""
        self.logger.info('Loading all datasets')
        datasets = self._get(test=False)
        self.logger.info('Done')
        return datasets
    def test(self):
        """Loads test data"""
        self.logger.info('Loading test dataset')
        datasets = self._get(test=True)
        self.logger.info('Done')
        return datasets
    def mnist(self, dataset='mnist.pkl.gz', static_origin=mnist_url):
        ''' Loads the MNIST dataset
        
        :type dataset: string
        :param dataset: the path to the dataset (here MNIST)
        '''
        self.logger.info('Loading the MNIST test dataset')
        
        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(dataset)
        # if data_dir == "" and not os.path.isfile(dataset):
        #
        #     # Check if dataset is in the data directory
        #     new_path = os.path.join(
        #         os.path.split(__file__)[0],
        #         "..",
        #         "data",
        #         dataset)
        #     if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
        #         dataset = new_path
        
        if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
            self.logger.debug('Downloading data from %s' % static_origin)
            urllib.urlretrieve(static_origin, dataset)
    
        self.logger.debug('Reading data')

        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        self.logger.debug('Partitioning data to list of [Train, Valid, Test]')
        test_set_x, test_set_y = self._shared_dataset(test_set)
        valid_set_x, valid_set_y = self._shared_dataset(valid_set)
        train_set_x, train_set_y = self._shared_dataset(train_set)
        
        datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        
        self.logger.info('Done')
        return datasets
    def _get(self, test):
        """
        Loads data, randomly partitions (train, valid, test) & splits X,Y variables
        
        Returns a list of datasets which contains:
            [0]: Train (x,y)
            [1]: Valid (x,y)
            [2]: Test (x,y)
        
        """
        # read data from file
        data = self._parse(test=test)
        
        # partition data
        datasets = self._random_partition(data)
        
        # split into (x, y) tuple
        datasets = [self._split_xy(dataset) for dataset in datasets]
        
        return datasets
    def _parse(self, comments='@', test=True):
        """Returns data parsed from datafile(s)
        
        Specify only one data file with test=True
        """
        
        self.logger.debug('Parsing %s data' % ('test' if test else 'all'))
        
        # test switch - if testing it only grabs the first match
        files = [self.dataFiles[0]] if test else self.dataFiles
        
        # load data files and concatenate inplace
        allData = []
        for file in files:
            self.logger.debug("Reading from '%s'" % file)
            allData.append(
                    np.genfromtxt(file, delimiter=',', comments=comments)
                    )
        # concatenate data along axis = 1
        data = np.concatenate(allData, axis=0)
        return data
    def _split_xy(self, data):
        """Splits x and y into a tuple of two datasets"""
        
        self.logger.debug('Splitting data into tuple of (x,y)')
        
        # split for x, y
        y = (data[:, -1] + 1 / 2.).reshape(data.shape[0], 1)
        x = data[:, :-1]
        
        # create shared dataset
        x,y = self._shared_dataset((x,y), borrow=True)
        
        return (x, y)
    def _random_partition(self, data, n=3):
        """
        Randomly partitions data into list of n sets of equal outcome ratios
        
        Assumes that data is split into two sets of Event1 and Event2
        defined by the [-1]th variable taking values of (1, -1) 
        
        Returns a numpy array of shape (n, data[0] // 3, data[1]),
        which can be split as:
            partition1, partition2, partition3 = returnedArray
        """
        
        self.logger.debug('Randomly partitioning data to list of [Train, Valid, Test]')
        
        # assumes that True events are 1
        # uses a count method on data[]
        storm = data[np.where((data[:,-1]==1))]
        noStorm = data[np.where(~(data[:,-1]==1))]
        
        # This loops through each of the storm and noStorm
        # data and splits into threee equal parts
        # these three equal parts are bundled in a list
        # for a np.concatenate() along the second axis
        # after the loop
        datasets = []
        for split in (storm, noStorm):
            
            # Shuffle each data split along axis 0
            np.random.shuffle(split)
            
            # get size of data split
            size = split.shape[0]
            
            # integer divison of dataset by n
            intSeg = size // n
            
            gen = zip(range(0, size, intSeg),range(intSeg, size, intSeg)[:-1]+[size])
            
            # splits data into three equal segments
            partitions = []
            for i,f in gen:
                partition_i = split[i:f, :]
                partitions.append(partition_i)
            
            datasets.append(partitions)
        
        # Re-combine the data along axis=0
        # resulting in datasets containg n partitions
        # with ~equal ratios of storm:noStorm
        partitionsCombined = [np.concatenate([d1,d2], axis=0) for d1,d2 in zip(*datasets)]
        
        # note: doesn't really matter which order:
        # test, valid, train are pulled from datasets
        # split by: train, valid, test = datasets
        return partitionsCombined
    def _shared_dataset(self, data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        data_y = data_y.reshape(data_y.shape[0])
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        self.logger.debug('X shape: %s' % str(data_x.shape))
        self.logger.debug('Y shape: %s' % str(data_y.shape))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')
#
if __name__ == '__main__':
    
    logger = utils.logs.get_logger(__name__)
    
    logger.info('Testing Load_Data ...')
    
    source = Load_Data(location=icestorm_loc)
    
    logger.info('Test data pull ...')
    testData = source.test()
    
    # logger.info('Test All data pull ...')
    # allData = source.all()
    
    logger.info('Test MNIST data pull ...')
    mnist = source.mnist(dataset=mnist_loc)