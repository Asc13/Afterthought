import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from math import *
from typing import Tuple, Union
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join


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


def create_adversarial(image1, image2, ratio = 0.1, position = (0.0, 0.0)):
    h, w, c = image1.shape
    
    example = np.array(image1)

    if ratio > 0:
        height, width = int(ratio * h), int(ratio * w)

        position_x = int(position[0] * (h - height))
        position_y = int(position[1] * (w - width))

        adversarial = Image.fromarray(np.uint8(image2 * 255.0))
        adversarial = adversarial.resize((width, height), Image.LANCZOS)
        adversarial = np.array(adversarial) / 255.0

        example[position_y:position_y + width, position_x:position_x + height, 0:c] = adversarial
    
    return example


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


@tf.function
def similarity(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = tf.nn.l2_normalize(x)
    y = tf.cast(tf.nn.l2_normalize(y), tf.float32)
    
    return tf.reduce_sum(x * y)


@tf.function
def dot(x: tf.Tensor, y: tf.Tensor, cossim_pow: float = 2.0) -> tf.Tensor:
    sim = tf.maximum(similarity(x, y), 1e-1) ** cossim_pow
    dot = tf.reduce_sum(x * tf.cast(y, tf.float32))

    return dot * sim


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    
    return result / (num_locations)


def composite_activation(x):
  x = tf.atan(x)

  return (x * x) / 0.6


def composite_activation_unbiased(x):
  x = tf.atan(x)

  return (x * x - 0.45) / 0.396


def relu_normalized(x):
  x = tf.nn.relu(x)

  return (x - 0.40) / 0.58


def circle(center, r, size):
    a = np.zeros(size, dtype = int)

    for angle in range(0, 360, 5):
        x = r * sin(radians(angle)) + center
        y = r * cos(radians(angle)) + center
        
        a[int(round(x)) + int(sqrt(size)) * int(round(y))] = 1

    return a


def kernel_fabricator(size: int, center: int, border: int):
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
                a3 = np.maximum(a2, a1) if center >= 0 else np.minimum(a2, a1)

            else:
                a3 = np.maximum(a3, a2) if center >= 0 else np.minimum(a3, a2)
    else:
        return a1

    return a3


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
                     absolute_value: bool = False):
    
    plt.imshow(normalize_image(image))

    if explanation.shape[-1] == 3:
        explanation = np.mean(explanation, -1)

    if absolute_value:
        explanation = np.abs(explanation)

    if clip_percentile:
        explanation = clip_heatmap(explanation, clip_percentile)

    plt.imshow(normalize_image(explanation), cmap = cmap, alpha = alpha)
    plt.axis('off')


def plot_all_attributions(explanations: Union[tf.Tensor, np.ndarray],
                          image: Union[tf.Tensor, np.ndarray],
                          cmap: str = "jet",
                          alpha: float = 0.5,
                          clip_percentile: float = 0.1,
                          absolute_value: bool = False):
    
    fig = plt.figure(figsize = (20, 20))
    sqr = sqrt(len(explanations))
    perfect_square = sqr.is_integer() and sqr > 1
    sqr = int(sqr)
    sqr += 0 if perfect_square else 1

    for n, explanation in enumerate(explanations):
        a = fig.add_subplot(sqr, sqr, n + 1)
        plot_attribution(explanation[0], image, cmap, alpha, clip_percentile, absolute_value)
    
    plt.show()



def plot_all(images: tf.Tensor, verbose: bool = True):
    fig = plt.figure(figsize = (20, 20))
    sqr = sqrt(len(images))
    perfect_square = sqr.is_integer() and sqr > 1
    sqr = int(sqr)
    sqr += 0 if perfect_square else 1

    for n, image in enumerate(images):
        a = fig.add_subplot(sqr, sqr, n + 1)
        plt.imshow(image[0])

        if verbose:
            plt.title('slot ' + str(n))

        plt.axis('off')

    plt.show()


def plot_all_maco(images: tf.Tensor, alphas: tf.Tensor, percentile_image: float = 1.0, 
                  percentile_alpha: float = 80, verbose: bool = True):
    
    fig = plt.figure(figsize = (20, 20))
    sqr = sqrt(len(images))
    perfect_square = sqr.is_integer() and sqr > 1
    sqr = int(sqr)
    sqr += 0 if perfect_square else 1

    for n, image in enumerate(images):
        a = fig.add_subplot(sqr, sqr, n + 1)

        image = np.array(image[0]).copy()
        image = clip_heatmap(image, percentile_image)

        alpha = np.mean(np.array(alphas[n][0]).copy(), -1, keepdims = True)
        alpha = np.clip(alpha, 0, np.percentile(alpha, percentile_alpha))
        alpha = alpha / alpha.max()

        image = image * alpha
        image = normalize_image(image)

        if verbose:
            plt.title('slot ' + str(n))

        plt.imshow(np.concatenate([image, alpha], -1))
        plt.axis('off')
    
    plt.show()


def compress(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for f in files:
        image = Image.open(path + '\\' + f)
        image = image.convert('RGB')

        image.save(path + '\\' + f,
                   'JPEG',
                   optimize = True,
                   quality = 100)