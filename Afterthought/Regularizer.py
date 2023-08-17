import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2

from typing import Callable

from src.Transformation import apply_kernel
from src.Miscellaneous import kernel_fabricator, addWeigthed

        
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
        return factor * tf.image.total_variation(image)[0]
    
    return reg