import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random
import cv2

from typing import Tuple, List, Union, Callable, Optional

from Afterthought.Miscellaneous import kernel_fabricator, motion_kernel, blur_edge


def jitter(delta: int = 6, seed = None) -> Callable:
    '''
    Inputs
    ----------
    delta - Value which will be cropped from the image

    seed - Random crop function seed
    '''
        
    def jitter_helper(image: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        shape = tf.shape(image)

        return tf.image.random_crop(image, (shape[-4], shape[-3] - delta, shape[-2] - delta, shape[-1]), seed = seed)
    
    return jitter_helper


def scale(scales: Union[float, List[float]], seed = None) -> Callable:
    '''
    Inputs
    ----------
    scales - List or single value for the scaling factors (e.g: [1, 2, 3] or 1.4)

    seed - Uniform function seed
    '''
        
    def scale_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        t = tf.convert_to_tensor(images, dtype = tf.float32)
        
        if isinstance(scales, list):
            scale = tf.constant(scales)[tf.random.uniform((), 0, len(scales), "int32", seed = seed)]

        else:
            scale = tf.constant(list(scales))[tf.random.uniform((), 0, 1, "int32", seed = seed)]
        
        shape = tf.shape(t)
        scale_shape = tf.cast(scale * tf.cast(shape[-3:-1], "float32"), "int32")

        return tf.compat.v1.image.resize_bilinear(t, scale_shape)

    return scale_helper


def flip(horizontal: bool = True, vertical: bool = False, seed = None) -> Callable:
    '''
    Inputs
    ----------
    horizontal - Flag for horizontal flip (default: False)

    vertical - Flag for vertical flip (default: True)

    seed - Random flip function seed
    '''
        
    def flip_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        if horizontal:
            images = tf.image.random_flip_left_right(images, seed = seed)

        if vertical:
            images = tf.image.random_flip_up_down(images, seed = seed)

        return images
    return flip_helper


def padding(size: int = 6, pad_value: float = 0.0) -> Callable:
    '''
    Inputs
    ----------
    size - Padding size

    pad_value - Scalar padding value
    '''
        
    pad_array = [(0, 0), (size, size), (size, size), (0, 0)]
    pad_value = tf.cast(pad_value, tf.float32)

    def padding_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        return tf.pad(images, pad_array, mode = "CONSTANT", constant_values = pad_value)

    return padding_helper


def apply_kernel(size: int, type: str) -> Callable:
    '''
    Inputs
    ----------
    size - Kernel size

    type - Technique used (can be 'BOX', 'SHARPNESS', 'HARD-SHARPNESS', 'MOTION-BLUR')
    '''
        
    if type == 'BOX':
        kernel = tf.ones((size, size, 1, 1)) / (size ** 2)

    elif type == 'SHARPNESS':
        kernel = kernel_fabricator(size, 5, -1)
        kernel[np.ix_((0,-1), (0,-1))] = 0.0
        kernel /= tf.reduce_sum(tf.cast(kernel, tf.float32))

    elif type == 'HARD-SHARPNESS':
        kernel = kernel_fabricator(size, 9, -1)
        kernel /= tf.reduce_sum(tf.cast(kernel, tf.float32))

    elif type == 'GAUSSIAN-BLUR':
        x = cv2.getGaussianKernel(size, -1)
        kernel = tf.cast(x.dot(x.T), tf.float32)

    elif type == 'MOTION-BLUR':
        kernel = np.flip(np.identity(size), axis = 0)
        kernel /= tf.reduce_sum(tf.cast(kernel, tf.float32))

    else:
        kernel = None

    if kernel != None:
        kernel = tf.reshape(kernel, (size, size, 1, 1))
        kernel = tf.tile(kernel, [1, 1, 3, 1])

    def apply_kernel_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        return tf.nn.depthwise_conv2d(images, kernel, strides = [1, 1, 1, 1], padding = 'SAME') if kernel != None else images
    
    return apply_kernel_helper



def apply_double_kernel(size: int, type: str) -> Callable:
    '''
    Inputs
    ----------
    size - Kernel size

    type - Technique used (can be 'EMBOSS')
    '''
        
    if type == 'EMBOSS':
        kernel1 = np.ones((size, size))

        kernel1[0] = -1
        kernel1[:, 0] = -1

        kernel1[0, 0] = -2
        kernel1[-1, -1] = 2
        kernel1[0, -1] = 0
        kernel1[-1, 0] = 0

        kernel2 = np.ones((size, size))

        kernel2[-1] = -1
        kernel2[:, 0] = -1

        kernel2[0, -1] = 2
        kernel2[-1, 0] = -2
        kernel2[0, 0] = 0
        kernel2[-1, -1] = 0

        kernel1 = kernel1 / np.sum(kernel1)
        kernel2 = kernel2 / np.sum(kernel2)

        
    kernel1 = tf.cast(kernel1, tf.float32)
    kernel1 = tf.reshape(kernel1, (size, size, 1, 1))
    kernel1 = tf.tile(kernel1, [1, 1, 3, 1])

    kernel2 = tf.cast(kernel2, tf.float32)
    kernel2 = tf.reshape(kernel2, (size, size, 1, 1))
    kernel2 = tf.tile(kernel2, [1, 1, 3, 1])

    def apply_double_kernel_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        images = tf.nn.depthwise_conv2d(images, kernel1, strides = [1, 1, 1, 1], padding = 'SAME')
        return tf.nn.depthwise_conv2d(images, kernel2, strides = [1, 1, 1, 1], padding = 'SAME')
    
    return apply_double_kernel_helper


def mean(size: int = 1) -> Callable:
    '''
    Inputs
    ----------
    size - Kernel size
    '''
        
    def mean_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        return tfa.image.mean_filter2d(images, (size, size))

    return mean_helper


def median(size: int = 1) -> Callable:
    '''
    Inputs
    ----------
    size - Kernel size
    '''

    def median_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        return tfa.image.median_filter2d(images, (size, size))

    return median_helper


def deconvolution(image_size: int, size: int = 1) -> Callable:
    '''
    Inputs
    ----------
    image_size - Image resizing resolution (e.g: 512 -> (512, 512, 3))

    size - Kernel size for motion and edge blur
    '''
        
    ANGLE = np.deg2rad(135)
    D = 22

    def deconvolution_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        images = tf.image.resize(images, [image_size, image_size])
        images = blur_edge(images, size)

        images = tf.transpose(images, (0, 3, 1, 2))

        h, w = images.shape[-2:]

        fft = tf.signal.rfft2d(images)
        spectrum1 = tf.complex(tf.math.real(fft), tf.math.imag(fft))

        images = tf.transpose(images, (0, 2, 3, 1))

        psf = motion_kernel(ANGLE, D, size)
        psf /= psf.sum()
        kh, kw = psf.shape

        psf = np.tile(np.reshape(psf, (kh, kw, 1)), 3)

        psf_pad = np.zeros((h, w, 3))
        psf_pad[:kh, :kw] = psf

        psf_pad = tf.transpose(psf_pad, (2, 0, 1))
        fft = tf.signal.rfft2d(psf_pad)
        spectrum2 = tf.cast(tf.complex(tf.math.real(fft), tf.math.imag(fft)), tf.complex64)
        
        s = tf.signal.irfft2d(spectrum1 * spectrum2)
        s /= tf.reduce_max(s)
        
        return tf.image.resize(tf.transpose(s, (0, 2, 3, 1)), [h, w])

    return deconvolution_helper


def maco_standard(size: int, crops: int, steps: int, box_size: Union[list, float] = None, std: Union[list, float] = None, seed: int = 0):
    '''
    Inputs
    ----------
    size - Image resolution

    crops - Number of crops to make

    steps - Activation maximization steps, to know the size of the linspaces

    std - Noise standard deviation

    seed - Randomness seed
    '''
    box_size_values = None
    noise_values = None

    if isinstance(box_size, list):
        box_size_values = box_size
        
    else:
        box_size_values = list(np.linspace(0.5, 0.05, steps))
    
    if isinstance(std, list):
        noise_values = std
        
    else:
        noise_values = list(np.logspace(0, -4, steps))
    
    def maco_standard_helper(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:
        new_box_size = box_size if (box_size and isinstance(box_size, float)) else tf.gather(box_size_values, step)
        noise_std = std if (std and isinstance(std, float) ) else tf.gather(noise_values, step)

        center_x = 0.5 + tf.random.normal((crops, ), stddev = 0.15, seed = seed)
        center_y = 0.5 + tf.random.normal((crops, ), stddev = 0.15, seed = seed)
        
        delta_x = tf.random.normal((crops, ), stddev = 0.05, mean = new_box_size, seed = seed)
        delta_x = tf.clip_by_value(delta_x, 0.05, 1.0)
        delta_y = delta_x

        box_indices = tf.zeros(shape = (crops, ), dtype = tf.int32)
        boxes = tf.stack([center_x - delta_x * 0.5,
                          center_y - delta_y * 0.5,
                          center_x + delta_x * 0.5,
                          center_y + delta_y * 0.5], -1)

        images = tf.image.crop_and_resize(images, boxes, box_indices, (size, size))

        images += tf.random.normal(images.shape, stddev = noise_std, mean = 0.0, seed = seed)
        images += tf.random.uniform(images.shape, minval = -noise_std / 2.0, maxval = noise_std / 2.0, seed = seed)

        return images
    
    return maco_standard_helper


def composition(input_shape: Tuple, transformations: List[Callable]) -> Callable:
    '''
    Inputs
    ----------
    input_shape - Model input shape for image preparation

    transformations - List of transformation functions to compose, sequentially
    '''

    def compose(images: tf.Tensor, step: Optional[tf.Tensor]) -> tf.Tensor:     
        for func in transformations:
            images = func(images, step)

        return tf.image.resize(images, tf.cast([input_shape[0], input_shape[1]], tf.int32))

    return compose


def standard(input_shape: Tuple, unit: int):
    '''
    Inputs
    ----------
    input_shape - Model input shape for image preparation

    unit - Default factor to use on the transformations
    '''

    unit = int(unit / 16) 
    return composition(input_shape,
        [
            padding(unit * 4),
            jitter(unit * 2, seed = 0),
            jitter(unit * 2, seed = 0),
            jitter(unit * 4, seed = 0),
            jitter(unit * 4, seed = 0),
            jitter(unit * 4, seed = 0),
            scale((0.92, 0.96), seed = 0),
            apply_kernel(3, 'BOX'),
            jitter(unit, seed = 0),
            jitter(unit, seed = 0),
            flip(seed = 0)
        ])