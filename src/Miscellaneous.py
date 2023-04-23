import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from math import *

from typing import Tuple, Union


def load_image(shape: Tuple, path: str) -> tf.Tensor:
    _, x, y, z = shape

    image = np.float32(cv2.imread(path)) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    min_side = min(h, w)
    max_side_center = max(h, w) // 2.0
    
    min_cut = int(max_side_center - min_side // 2)
    max_cut = int(max_side_center + min_side // 2)
    
    image = image[:, min_cut:max_cut] if w > h else image[min_cut:max_cut]
    
    image = cv2.resize(image, (x, y))
    
    return tf.reshape(image, (1, x, y, z))
    

def load_files(path: str, size: int = 10, has_titles: bool = False):
    subfolders = [f.path for f in os.scandir(path) if f.is_file()]

    x = len(subfolders)
    sqr = int(sqrt(x)) + 1
    fig = plt.figure(figsize = (size, size))

    for n, sf in enumerate(subfolders):
        image = np.float32(cv2.imread(sf)) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig.add_subplot(sqr, sqr, n + 1)
        plt.imshow(image)

        if has_titles:
            plt.title(sf.split('/')[-1].split('.')[0])
            
        plt.axis('off')


@tf.function(reduce_retracing = True)
def compute_gradients_fmaps(model: tf.keras.Model,
                            images: tf.Tensor,
                            targets: tf.Tensor) -> Tuple:

    with tf.GradientTape(watch_accessed_variables = False) as tape:
        tape.watch(images)
        feature_maps, predictions = model(images)
        score = tf.reduce_sum(tf.multiply(predictions, targets), axis = -1)

    return feature_maps, tape.gradient(score, feature_maps)


@tf.function(reduce_retracing = True)
def compute_gradients(model: tf.keras.Model,
                      images: tf.Tensor,
                      targets: tf.Tensor) -> Tuple:

    with tf.GradientTape(watch_accessed_variables = False) as tape:
        tape.watch(images)
        score = tf.reduce_sum(tf.multiply(model(images), targets), axis = 1)

    return tape.gradient(score, images)


def repeat_labels(labels: tf.Tensor, repetitions: int) -> tf.Tensor:
    labels = tf.expand_dims(labels, axis = 1)
    labels = tf.repeat(labels, repeats = repetitions, axis = 1)
    
    return tf.reshape(labels, (-1, *labels.shape[2:]))


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
    g = exp(-d / (2 * std ** 2))

    return g


def high_pass(std, img_shape):
    return np.asarray([[1 - gauss(std, x - img_shape[0] / 2, y - img_shape[1] / 2)  for y in range(img_shape[1])] for x in range(img_shape[0])])        


def low_pass(std, img_shape):
    return np.asarray([[gauss(std, x - img_shape[0]/2, y - img_shape[1]/2)  for y in range(img_shape[1])] for x in range(img_shape[0])])



def normalize_image(image: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
    image = np.array(image, np.float32)

    image -= image.min()
    image /= image.max()

    return image


def clip_heatmap(heatmap: Union[tf.Tensor, np.ndarray],
                 percentile: float) -> np.ndarray:
    
    clip_min = np.percentile(heatmap, percentile)
    clip_max = np.percentile(heatmap, 100.0 - percentile)

    return np.clip(heatmap, clip_min, clip_max)


def plot_attribution(explanation: Union[tf.Tensor, np.ndarray],
                     image: Union[tf.Tensor, np.ndarray],
                     cmap: str = "jet",
                     alpha: float = 0.5,
                     clip_percentile: float = 0.1,
                     absolute_value: bool = False,
                     **plot_kwargs):
    
    plt.imshow(normalize_image(image))

    if explanation.shape[-1] == 3:
        explanation = np.mean(explanation, -1)

    if absolute_value:
        explanation = np.abs(explanation)

    if clip_percentile:
        explanation = clip_heatmap(explanation, clip_percentile)

    plt.imshow(normalize_image(explanation), cmap = cmap, alpha = alpha, **plot_kwargs)
    plt.axis('off')


def plot_all(images: tf.Tensor):
    fig = plt.figure(figsize = (20, 20))
    sqr = int(sqrt(len(images))) + 1

    for n, image in enumerate(images):
        a = fig.add_subplot(sqr, sqr, n + 1)
        plt.imshow(image[0])
        plt.title('batch ' + str(n))
        plt.axis('off')

    plt.show()