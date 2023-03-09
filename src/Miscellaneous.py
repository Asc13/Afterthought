import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2
import math


@tf.function
def similarity(tensor_a: tf.Tensor, tensor_b: tf.Tensor) -> tf.Tensor:
    tensor_a = tf.nn.l2_normalize(tensor_a)
    tensor_b = tf.cast(tf.nn.l2_normalize(tensor_b), tf.float32)
    
    return tf.reduce_sum(tensor_a * tensor_b)


@tf.function
def dot(tensor_a: tf.Tensor, tensor_b: tf.Tensor, cossim_pow: float = 2.0) -> tf.Tensor:
    sim = tf.maximum(similarity(tensor_a, tensor_b), 1e-1) ** cossim_pow
    
    dot = tf.reduce_sum(tensor_a * tf.cast(tensor_b, tf.float32))

    return dot * sim


@tf.function
def blur_conv(x, width = 3):
  depth = x.shape[-1]
  k = np.zeros([width, width, depth, depth])

  for ch in range(depth):
    k_ch = k[:, :, ch, ch]
    k_ch[:,:] = 0.5
    k_ch[1:-1, 1:-1] = 1.0

  conv_k = lambda t: tf.nn.conv2d(t, k, [1, 1, 1, 1], "SAME")
  return conv_k(x) / conv_k(tf.ones_like(x))


def composite_activation(x):
  x = tf.atan(x)

  return tf.concat([x / 0.67, (x * x) / 0.6], -1)


def composite_activation_unbiased(x):
  x = tf.atan(x)

  return tf.concat([x / 0.67, (x * x - 0.45) / 0.396], -1)


def relu_normalized(x):
  x = tf.nn.relu(x)

  return (x - 0.40) / 0.58



def kernel_fabricator(size: int, center: int, border):
    a1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))

    a1[:, [0, -1]] = 0
    a1[[0, -1], :] = 0
    a1 = np.where(a1 == 1, center, border)

    c = int(size / 2)

    a3 = np.zeros((size, size))

    if c > 2:
        for i in range(c - 1):
            o = c - i - 1
            a2 = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

            fixV = np.concatenate([np.arange(0, i + 2),  np.arange(size - 1, size - 3 - i, -1)], axis = 0)
            fixH = np.concatenate([np.arange(0, o),  np.arange(size - 1, size - o - 1, -1)], axis = 0)

            a2[:, fixV] = 0
            a2[fixH, :] = 0

            a2 = np.where(a2 == 1, center, border)

            if i == 0:
                a3 = np.maximum(a2, a1)

            else:
                a3 = np.maximum(a3, a2)
    else:
        return a1

    return a3


def blur_edge(images: tf.Tensor, size: int = 31) -> tf.Tensor:
    _, h, w, _ = images.shape

    pad = tf.pad(images, [(0, 0), (size, size), (size, size), (0, 0)])
    gauss = tfa.image.gaussian_filter2d(pad, (2 * size + 1, 2 * size + 1))
    gauss = tf.image.resize(gauss, [h, w])
    
    y, x = np.indices((h, w))
    dist = np.dstack([x, w - x - 1, y, h - y - 1]).min(-1)
    m = np.minimum(np.float32(dist) / size, 1.0)
    
    m = tf.reshape(m, (1, h, w, 1))
    m = tf.tile(m, [1, 1, 1, 3])
    
    return images * m + gauss * (1 - m)


def motion_kernel(angle, d, sz = 1):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    
    A[:, 2] = (sz2, sz2) - np.dot(A[:,:2], ((d - 1) * 0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags = cv2.INTER_CUBIC)
    
    return kern


def gauss(std, x, y):
    d = x ** 2 + y ** 2
    g = math.exp(-d / (2 * std ** 2))

    return g


def high_pass(std, img_shape):
    return np.asarray([[1 - gauss(std, x - img_shape[0] / 2, y - img_shape[1] / 2)  for y in range(img_shape[1])] for x in range(img_shape[0])])        


def low_pass(std, img_shape):
    return np.asarray([[gauss(std, x - img_shape[0]/2, y - img_shape[1]/2)  for y in range(img_shape[1])] for x in range(img_shape[0])])
