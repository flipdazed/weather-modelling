import os

import numpy as np
try: import PIL.Image as Image
except ImportError: import Image
import hickle as hkl

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
    def __init__(self, plot_dir, data_dir, file_ext='.hkl', img_ext='.png'):
        """Initialises `self.params` as an empty `dict`
        
        Each dictionary key contains the name of the parameter with the 
        respective value being the data set as a `list`.
        
        :type file_ext: str
        :param file_ext: the file extension used for the data
        
        :type file_ext: str
        :param file_ext: the file extension used for the images
        """
        
        # set save directories
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        
        # set up the data structure to hold 
        ## the runtime data
        self.params = {}
        
        ## The image data
        self.imgs = {}
        
        ## The cost data
        self.cost = []
        
        # set the extension type
        self.file_ext = file_ext
        self.imgs_ext  = img_ext 
        pass
    
    def saveWeightImgs(self, run_id, param_name='filters', erase = False):
        """Saves the parameters to file iteratively. The dictionary of
             parameters is specified by `self.params[param_name]`
            
            :type run_id: str
            :param run_id: a string associated with the data set for 
                identification that will form part of the save names. This
                will generally be the root file name give by 
                `os.path.basename(__file__)`
            
            :type param_name: str
            :param param_name: Optional. Default is `None`. When `None` all
                parameters are saved. When `param_name` is specified as a key
                entry to `self.params`, only that parameter will be affected.
            
            :type erase: bool
            :param erase: Optional. Default `False`. When `True` the data 
                holder is emptied / erased after the data is written to file.
        """
        
        if param_name is None: # iterate all parameters in dict
            for param_name, img_arrs in self.imgs.iteritems():
                save_path = os.path.join(self.data_dir,
                    '_'.join([run_id,param_name]))
                self._imageSave(img_arrs, save_path)
                if erase: self.imgs[param_name] = []
            
        else:                   # only save the specified paramter
            img_arrs = self.imgs[param_name]
            save_path = os.path.join(self.data_dir,
                '_'.join([run_id,param_name]))
            self._imageSave(img_arrs, save_path)
            if erase: self.imgs[param_name] = []
        pass
    
    def saveCost(self, run_id, param_name='cost', erase = False):
        """Saves the parameters to file iteratively. The dictionary of
             parameters is specified by `self.params[param_name]`
            
            :type run_id: str
            :param run_id: a string associated with the data set for 
                identification that will form part of the save names. This
                will generally be the root file name give by 
                `os.path.basename(__file__)`
            
            :type param_name: str
            :param param_name: Optional. Default is `None`. When `None` all
                parameters are saved. When `param_name` is specified as a key
                entry to `self.params`, only that parameter will be affected.
            
            :type erase: bool
            :param erase: Optional. Default `False`. When `True` the data 
                holder is emptied / erased after the data is written to file.
        """
        save_path = os.path.join(self.data_dir, '_'.join([run_id,param_name]))
        self._dumpMethod(self.cost, save_path)
        if erase: self.cost = []
        pass
    
    def saveParams(self, run_id, param_name = None, erase = False):
        """Saves the parameters to file iteratively. The dictionary of
             parameters is specified by `self.params[param_name]`
            
            :type run_id: str
            :param run_id: a string associated with the data set for 
                identification that will form part of the save names. This
                will generally be the root file name give by 
                `os.path.basename(__file__)`
            
            :type param_name: str
            :param param_name: Optional. Default is `None`. When `None` all
                parameters are saved. When `param_name` is specified as a key
                entry to `self.params`, only that parameter will be affected.
            
            :type erase: bool
            :param erase: Optional. Default `False`. When `True` the data 
                holder is emptied / erased after the data is written to file.
        """
        
        if param_name is None: # iterate all parameters in dict
            for param_name, data in self.params.iteritems():
                save_path = os.path.join(self.data_dir, 
                    '_'.join([run_id,param_name]))
                self._dumpMethod(data, save_path)
                if erase: self.params[param_name] = []
                    
        else:                   # only save the specified paramter
            save_path = os.path.join(self.data_dir,
                '_'.join([run_id,param_name]))
            self._dumpMethod(data, save_path)
            if erase: self.params[param_name] = []
        pass
    def _dumpMethod(self, data, full_path_without_extension):
        """The method that data is dumped defined in one place. Data is 
        stored is HDF5 with the library `hickle`_
        
        :param data: a dataset / array
        :type full_path_without_extension: str
        :param full_path_without_extension: the full file path without
            the extension e.g. `/Users/me/path/to/data/file_name`
        
        .. _hickle: https://github.com/telegraphic/hickle
        """
        
        hkl.dump(data, full_path_without_extension + self.file_ext)
        pass
    def _loadMethod(self, full_path_without_extension):
        """Returns data stored is HDF5 with the library `hickle`_
        
        :type full_path_without_extension: str
        :param full_path_without_extension: the full file path without
            the extension e.g. `/Users/me/path/to/data/file_name`
        
        .. _hickle: https://github.com/telegraphic/hickle
        """
        
        data = hkl.load(
            data,
            full_path_without_extension + self.file_ext
            )
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
            save_path = full_path_without_extension + '_{:04d}'.format(i)
            Image.fromarray(img_arrs).save(save_path)
        pass
#
def scaleToUnitInterval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
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
