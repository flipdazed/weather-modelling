import numpy as np

__doc__ = """ This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tileRasterImages`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

def scaleToUnitInterval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
def tileRasterImages(x, img_shape, tile_shape, tile_spacing=(0, 0), scale_rows_to_unit_interval=True, output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
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
    
    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]
    
    if isinstance(x, tuple):
        assert len(x) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=x.dtype)
        
        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals: channel_defaults = [0, 0, 0, 255]
        else: channel_defaults = [0., 0., 0., 1.]
        
        for i in xrange(4):
            if x[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape, dtype=dt) + channel_defaults[i]
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
