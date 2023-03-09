import tensorflow as tf
import numpy as np
import cv2

from typing import Callable

from Miscellaneous import blur_conv


def L1(factor: float = 1.0) -> Callable:
    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.reduce_mean(tf.abs(image))
    
    return reg


def L2(factor: float = 1.0) -> Callable:
    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.sqrt(tf.reduce_mean(image ** 2))
    
    return reg


def Linf(factor: float = 1.0) -> Callable:
    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.reduce_max(tf.abs(image))
    
    return reg


def total_variation(factor: float = 1.0) -> Callable:
    def reg(image: tf.Tensor) -> tf.Tensor:
        return factor * tf.image.total_variation(image)
    
    return reg


def boundary_complexity(width: int = 20, C: float = 0.5, 
                        factor: float = 1.0) -> Callable:
    
    def reg(image: tf.Tensor) -> tf.Tensor:
        mask = np.ones(image.shape[1:])
        mask[:, width:-width, width:-width] = 0.0

        blur = blur_conv(image, width = 5)
        diffs = (blur - image) ** 2
        diffs += 0.8 * (image - C) ** 2
        
        return factor * tf.reduce_sum(diffs * mask)
    return reg


def gauss_diff(higher: int, lower: int, factor: float = 1.0) -> Callable:
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
        
        return factor * tf.reduce_sum(i2 - i1)

    return reg