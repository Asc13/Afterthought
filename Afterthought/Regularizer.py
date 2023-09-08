import tensorflow as tf

from typing import Callable

from Afterthought.Miscellaneous import *
from Afterthought.Transformation import *


def L1(factor: float = 1.0) -> Callable:
    '''
    Inputs
    ----------
    factor - Regularizer power (default: 1)
    ''' 
            
    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.reduce_mean(tf.abs(image))
    
    return reg


def L2(factor: float = 1.0) -> Callable:
    '''
    Inputs
    ----------
    factor - Regularizer power (default: 1)
    ''' 
        
    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.sqrt(tf.reduce_mean(image ** 2))
    
    return reg


def Linf(factor: float = 1.0) -> Callable:
    '''
    Inputs
    ----------
    factor - Regularizer power (default: 1)
    ''' 

    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.reduce_max(tf.abs(image))
    
    return reg


def total_variation(factor: float = 1.0) -> Callable:
    '''
    Inputs
    ----------
    factor - Regularizer power (default: 1)
    ''' 

    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.image.total_variation(image)[0] / (image.shape[-1] * image.shape[-2] * image.shape[-3])
    
    return reg


def kernel(size: int, type: str, factor: float = 1.0) -> Callable:
    '''
    Inputs
    ----------
    size - Kernel size

    type - Technique used (can be 'BOX', 'SHARPNESS', 'HARD-SHARPNESS', 'GAUSSIAN-BLUR', 'MOTION-BLUR', 'EMBOSS')

    factor - Regularizer power (default: 1)
    '''
    transformation = apply_kernel(size, type)

    if type == 'EMBOSS':
        transformation = apply_double_kernel(size, type)

    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.reduce_mean(tf.abs(transformation(image, 0)))

    return reg


def gauss_diff(higher: int, lower: int, factor: float = 1.0) -> Callable:
    '''
    Inputs
    ----------
    higher - Higher gaussian kernel size

    lower - Lower gaussian kernel size

    factor - Regularizer power (default: 1)
    '''
        
    x = cv2.getGaussianKernel(higher, -1)
    kernel1 = tf.cast(x.dot(x.T), tf.float32)
    kernel1 = tf.reshape(kernel1, (higher, higher, 1, 1))
    kernel1 = tf.tile(kernel1, [1, 1, 3, 1])

    x = cv2.getGaussianKernel(lower, -1)
    kernel2 = tf.cast(x.dot(x.T), tf.float32)
    kernel2 = tf.reshape(kernel2, (lower, lower, 1, 1))
    kernel2 = tf.tile(kernel2, [1, 1, 3, 1])

    def reg(image: tf.Tensor) -> tf.Tensor:
        i1 = tf.nn.depthwise_conv2d(image, kernel2, strides = [1, 1, 1, 1], padding = 'SAME')
        i2 = tf.nn.depthwise_conv2d(image, kernel1, strides = [1, 1, 1, 1], padding = 'SAME')
        
        return factor * tf.reduce_mean(tf.abs(i2 - i1))

    return reg