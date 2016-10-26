import os, sys

import numpy as np
try: import PIL.Image as Image
except ImportError: import Image
import cPickle as pkl

import logs

__docformat__ = 'restructedtext en'

__doc__ = """ 
This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example `tileRasterImages` helps in generating a easy to grasp
image from a set of samples or weights.
"""

class Visualise_Runtime(object):
    """Handles visualisation data I/O
    
    `self.params` contains the runtime parameter set
    `self.imgs` contains runtime weight-image visualisation
    `self.cost` contains the runtime cost
    
    """
    def __init__(self, plot_dir, data_dir, data_ext='.pkl', img_ext='.png'):
        """Initialises `self.params` as an empty `dict`
        
        Each dictionary key contains the name of the parameter with the 
        respective value being the data set as a `list`.
        
        :type data_ext: str
        :param data_ext: the file extension used for the data
        
        :type data_ext: str
        :param data_ext: the file extension used for the images
        """
        
        # Set up logger to track progress
        logs.get_logger(self=self)
        self.logger.info('Started runtime parameter manager ...')
        
        # set save directories
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        
        # set up the data structure to hold 
        ## the runtime data
        self.params = {}
        
        ## The image data
        self.imgs = {}
        
        ## The cost data
        self.cost = {}
        
        ## the updates data
        self.updates = {}
        
        self.attrs = [self.params, self.cost, self.imgs, self.updates]
        self.attr_names = ['params', 'cost', 'imgs', 'updates']
        self.attr_dirs = [self.data_dir]*2 + [self.plot_dir, self.data_dir]
        
        # set the extension type
        self.data_ext = data_ext
        self.imgs_ext  = img_ext
        
        # run time file attributes
        self.runtime_files = {}
        self.runtime_files_error = False
        pass
    def initalise(self, run_id, default_freq, params=False, cost=False, imgs=False, updates=False):
        """
        
        :type run_id: str
        :param run_id: a string associated with the data set for 
            identification that will form part of the save names. This
            will generally be the root file name give by 
            `os.path.basename(__file__)`
        
        :type default_freq: int
        :param param: The default freq for validations
        """
        self.logger.info('Tagging data under Run Id: {}'.format(run_id))
        
        # iterate through cost, imgs, params settings
        for input_attr, attr, attr_name, save_dir in zip(
            [params, cost, imgs, updates],
            self.attrs,
            self.attr_names,
            self.attr_dirs
        ):
            if input_attr: # only run if the attr is passed into the func
                for n, attr_settings in input_attr.iteritems():
                    # send settings to a settings
                    attr[n] = {}
                    attr[n]['settings'] = attr_settings
                    # determine how often images are sampled
                    # default to the validation freq
                    if 'freq' not in attr_settings:
                        attr[n]['settings']['freq'] = default_freq
                    
                    attr[n]['data'] = []
                    attr[n]['settings']['save_path'] = os.path.join(
                        save_dir,
                        '_'.join([run_id, attr_name, n])
                    )
                    
                    self.logger.debug(
                        'Tracking {}, id: {}'.format(attr_name,n)
                    )
            else:
                self.logger.warn('No parameters for {}'.format(attr_name))
        pass
    def getValues(self, i, **runtime_only_values):
        """store values at runtime
        
        :type i: int 
        :param i: iteration counter
        
        :type runtime_only_values: dict
        :param runtime_only_values: add runtime values as kwarg=arg pairs
            such that the kwarg identifies the local attribute e.g.
            self.cost would be identified by cost=arg.
        """
        # save the weights
        for n, params in self.imgs.iteritems():
            if (i+1) % params['settings']['freq'] == 0:
                img_arr = tileRasterImages(**params['settings'])
                self.imgs[n]['data'].append(img_arr)
        
        # save the parameters
        for n, params in self.params.iteritems():
            if (i+1) % params['settings']['freq'] == 0:
                v = np.mean(params['settings']['x'])
                self.params[n]['data'].append(v)
        
        # save the cost
        if self.cost and ('cost' in runtime_only_values):
            n = 'cost'
            v = runtime_only_values[n]
            if (i+1) % self.cost[n]['settings']['freq'] == 0:
                self.cost[n]['data'].append(v)
        
        # save the updates
        if self.updates and ('updates' in runtime_only_values):
            n = 'updates'
            v = runtime_only_values[n]
            
            # error handling if the training func is not properly set up
            if v is None:
                self.logger.error('Updates specified but training '
                    'function needs a list output:')
                self.logger.error('\toutputs = [cost] + gradients')
                self.logger.error('... where gradients is a list of'
                    ' T.tensors')
                raise ValueError('Correct Training function!')
            
            # iterate through the update param names
            # and extract the relevant order from settings
            for n, settings in self.updates.iteritems():
                
                if (i+1) % settings['settings']['freq'] == 0:
                    
                    # pull the update order index
                    # which is the expected position in the list
                    # of updates being received as an argument
                    idx = settings['settings']['update_position']
                    
                    # append the relevant update using the idx
                    # mean must be taken as above with params
                    self.updates[n]['data'].append(np.mean(v[idx]))
        pass
    def writeRuntimeValues(self, i, clean_files = False):
        """Creates file objects to write to for cost and params
        
        :type i: int 
        :param i: iteration counter
        """
        write_type = ['a', 'w'][clean_files]
        
        for param_name, settings in self.cost.iteritems():
            if (i+1) % settings['settings']['freq'] == 0:
                save_loc = settings['settings']['save_path']+'.dat'
                data = settings['data'][-1]
                with open(save_loc, write_type) as f:
                    f.write('{0:12.6e}\n'.format(data))
        
        for param_name, settings in self.params.iteritems():
            if (i+1) % settings['settings']['freq'] == 0:
                save_loc = settings['settings']['save_path']+'.dat'
                data = settings['data'][-1]
                with open(save_loc, write_type) as f:
                    f.write('{0:12.6e}\n'.format(data))
        
        for param_name, settings in self.updates.iteritems():
            if (i+1) % settings['settings']['freq'] == 0:
                save_loc = settings['settings']['save_path']+'.dat'
                data = settings['data'][-1]
                with open(save_loc, write_type) as f:
                    f.write('{0:12.6e}\n'.format(data))
        
        for param_name, settings in self.imgs.iteritems():
            if 'runtime_plots' not in settings['settings']: continue
            if not settings['settings']['runtime_plots']: continue
            
            if (i+1) % settings['settings']['freq'] == 0:
                save_loc = settings['settings']['save_path']+'.dat'
                data = settings['data'][-1]
                np.savetxt(save_loc, data, fmt='%03d')
        pass
    def saveValues(self):
        """save all the recorded values"""
        if self.cost: self.saveCost()
        if self.params: self.saveParams()
        if self.updates: self.saveUpdates()
        if self.imgs: self.saveImgs()
        pass
    def saveImgs(self, param_name=None, erase = False):
        """Saves the parameters to file iteratively. The dictionary of
         parameters is specified by `self.params[param_name]`
        
        :type param_name: str
        :param param_name: Optional. Default is `None`. When `None` all
            parameters are saved. When `param_name` is specified as a key
            entry to `self.params`, only that parameter will be affected.
        
        :type erase: bool
        :param erase: Optional. Default `False`. When `True` the data 
            holder is emptied / erased after the data is written to file.
        """
        
        if param_name is None: # iterate all parameters in dict
            self.logger.info('Saving all Weight Image Sets ...')
            for param_name, settings in self.imgs.iteritems():
                save_path = settings['settings']['save_path']
                data = settings['data']
                if data:
                    self._imageSave(data, save_path)
                    if erase: self.imgs[param_name]['data'] = []
                else:
                    self.logger.warn('No data captured for: '
                        '{}'.format(param_name))
        
        else:                   # only save the specified paramter
            self.logger.info('Saving Weight Image Set: {} ...'.format(
                param_name))
            settings = self.imgs[param_name]
            save_path = os.path.join(self.plot_dir,
                '_'.join([run_id, param_name]))
            if data:
                self._imageSave(data, save_path)
                if erase: self.imgs[param_name]['data'] = []
            else:
                self.logger.warn('No data captured for: '
                    '{}'.format(param_name))
        pass
    def saveCost(self, param_name='cost', erase = False):
        """Saves the parameters to file iteratively. The dictionary of
        parameters is specified by `self.params[param_name]`
        
        :type param_name: str
        :param param_name: Optional. Default is `None`. When `None` all
            parameters are saved. When `param_name` is specified as a key
            entry to `self.params`, only that parameter will be affected.
        
        :type erase: bool
        :param erase: Optional. Default `False`. When `True` the data 
            holder is emptied / erased after the data is written to file.
        """
        
        self.logger.info('Saving Cost ...')
        settings = self.cost
        save_path = settings[param_name]['settings']['save_path']
        self._dumpMethod(settings[param_name]['data'], save_path)
        if erase: settings[param_name]['data'] = []
        pass
    def saveParams(self, param_name = None, erase = False):
        """Saves the parameters to file iteratively. The dictionary of
        parameters is specified by `self.params[param_name]`
            
            :type param_name: str
            :param param_name: Optional. Default is `None`. When `None` all
                parameters are saved. When `param_name` is specified as a key
                entry to `self.params`, only that parameter will be affected.
            
            :type erase: bool
            :param erase: Optional. Default `False`. When `True` the data 
                holder is emptied / erased after the data is written to file.
        """
        source = self.params
        
        if param_name is None: # iterate all parameters in dict
            self.logger.info('Saving all Parameters ...')
            for param_name, settings in source.iteritems():
                save_path = settings['settings']['save_path']
                self._dumpMethod(settings['data'], save_path)
                if erase: source[param_name] = []
                    
        else:                   # only save the specified paramter
            self.logger.info('Saving Parameter: {} ...'.format(param_name))
            settings = source[param_name]
            save_path = settings['settings']['save_path']
            self._dumpMethod(settings['data'], save_path)
            if erase: source[param_name]['data'] = []
        pass
    def saveUpdates(self, param_name = None, erase = False):
        """Saves the parameters to file iteratively. The dictionary of
        parameters is specified by `self.params[param_name]`
            
            :type param_name: str
            :param param_name: Optional. Default is `None`. When `None` all
                parameters are saved. When `param_name` is specified as a key
                entry to `self.params`, only that parameter will be affected.
            
            :type erase: bool
            :param erase: Optional. Default `False`. When `True` the data 
                holder is emptied / erased after the data is written to file.
        """
        source = self.updates
        
        if param_name is None: # iterate all parameters in dict
            self.logger.info('Saving all Updates ...')
            for param_name, settings in source.iteritems():
                save_path = settings['settings']['save_path']
                self._dumpMethod(settings['data'], save_path)
                if erase: self.params[param_name] = []
                    
        else:                   # only save the specified paramter
            self.logger.info('Saving Update: {} ...'.format(param_name))
            settings = source[param_name]
            save_path = settings['settings']['save_path']
            self._dumpMethod(settings['data'], save_path)
            if erase: source[param_name]['data'] = []
        pass
    def _dumpMethod(self, data, full_path_without_extension):
        """The method that data is dumped defined in one place.
        
        :param data: a dataset / array
        :type full_path_without_extension: str
        :param full_path_without_extension: the full file path without
            the extension e.g. `/Users/me/path/to/data/file_name`
        """
        full_path = full_path_without_extension + self.data_ext
        with open(full_path, 'wb') as f:
            pkl.dump(data, f)
        self.logger.debug('saved: {}'.format(full_path))
        pass
    def _loadMethod(self, full_path_without_extension):
        """Loads and returns stored data
        
        :type full_path_without_extension: str
        :param full_path_without_extension: the full file path without
            the extension e.g. `/Users/me/path/to/data/file_name`
        """
        full_path = full_path_without_extension + self.data_ext
        with open(full_path, 'rb') as f:
            data = pkl.load(f)
        self.logger.debug('loaded: {}'.format(full_path))
        return data
    def _imageSave(self, img_arrs, full_path_without_extension):
        """Converts an array to Image and saves using `PIL` via `Pillow`_
        
        :param img_arras: list
        :param img_arrs: image data
        :type full_path_without_extension: str
        :param full_path_without_extension: the full file path without
            the extension e.g. `/Users/me/path/to/data/file_name`
        
        .. _PIL: https://github.com/python-pillow/Pillow
        """
        
        for i, img_arr in enumerate(img_arrs):
            save_path = full_path_without_extension \
                + '_{:04d}{}'.format(i, self.imgs_ext)
            Image.fromarray(img_arr).save(save_path)
        
        self.logger.debug('saved {} images: {}_*{}'.format(i,
            full_path_without_extension, self.imgs_ext))
        pass
#
def scaleToUnitInterval(ndar, eps=1e-8):
    """Scales all values in ndar to be between 0 and 1"""
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tileRasterImages(x, img_shape, tile_shape, tile_spacing=(0, 0), scale_rows_to_unit_interval=True, output_pixel_vals=True, *args, **kwargs):
    """Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    
    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    
    :type x: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param x: a 2-D array in which every row is a flattened image.
    
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats
    
    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not
    
    
    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as x.
    
    """
    
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2
    
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]
    
    if isinstance(x, tuple):
        assert len(x) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=x.dtype)
        
        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals: channel_defaults = [0, 0, 0, 255]
        else: channel_defaults = [0., 0., 0., 1.]
        
        for i in xrange(4):
            if x[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals: dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape, dtype=dt) \
                    + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tileRasterImages(
                    x[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
    else:
        # if we are dealing with only one channel
        h, w = img_shape
        hs, ws = tile_spacing
        
        # generate a matrix to store the output
        dt = x.dtype
        if output_pixel_vals: dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)
        
        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < x.shape[0]:
                    this_x = x[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scaleToUnitInterval`
                        # function
                        this_img = scaleToUnitInterval(this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals: c = 255
                    out_array[
                        tile_row * (h + hs): tile_row * (h + hs) + h,
                        tile_col * (w + ws): tile_col * (w + ws) + w
                    ] = this_img * c
    return out_array
