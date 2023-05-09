import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Tuple, List, Callable

from src.Miscellaneous import composite_activation, relu_normalized


imagenet_color_correlation = tf.cast(
    [[0.56282854, 0.58447580, 0.58447580],
    [0.19482528, 0.00000000,-0.19482528],
    [0.04329450,-0.10823626, 0.06494176]], tf.float32
)


def recorrelate_colors(image: tf.Tensor) -> tf.Tensor:
    image_flat = tf.reshape(image, [-1, 3])
    image_flat = tf.matmul(image_flat, imagenet_color_correlation)

    return tf.reshape(image_flat, tf.shape(image))


def fft_2d_freq(width: int, height: int) -> np.ndarray:
    freq_y = np.fft.fftfreq(height)[:, np.newaxis]

    cut_off = int(width % 2 == 1)
    freq_x = np.fft.fftfreq(width)[:width // 2 + 1 + cut_off]

    return np.sqrt(freq_x ** 2 + freq_y ** 2)


def get_fft_scale(width: int, height: int, decay_power: float = 1.0) -> tf.Tensor:
    frequencies = fft_2d_freq(width, height)
    fft_scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    fft_scale = fft_scale * np.sqrt(width * height)

    return tf.convert_to_tensor(fft_scale, dtype = tf.complex64)


def fft_to_rgb(image: tf.Tensor, shape: Tuple, 
               fft_scale: tf.Tensor) -> tf.Tensor:
    
    batch, _, width, height, channels = shape

    spectrum = tf.complex(image[0], image[1]) * fft_scale

    image = tf.signal.irfft2d(spectrum)
    image = tf.transpose(image, (0, 2, 3, 1))

    return image / 4.0


def fft_image(shape: Tuple, std: float = 0.01) -> tf.Tensor:
    batches, _, width, height, channels = shape
    
    frequencies = fft_2d_freq(width, height)

    return tf.random.normal((batches, 2, 1, channels) + frequencies.shape, stddev = std)


def cppn(size: Tuple, batches: int,
         num_output_channels: int, 
         num_hidden_channels: int, 
         num_layers: int, 
         activation_function: Callable, 
         seed: int = 0):

    r = 3.0 ** 0.5
    coord_range_x = tf.linspace(-r, r, size[0])
    coord_range_y = tf.linspace(-r, r, size[1])

    y, x = tf.meshgrid(coord_range_x, coord_range_y, indexing = "ij")

    inputs = [x, y, tf.math.sqrt(x ** 2 + y ** 2)]
    inputs = tf.transpose(tf.stack(inputs), [1, 2, 0])
    inputs = tf.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[-1]))

    if seed is not None:
        tf.random.set_seed(seed)
    
    net = tf.convert_to_tensor(inputs)

    for i in range(0, num_layers):
        if i == num_layers - 1:
            W = tf.random.normal((net.shape[1], num_output_channels))
        
        else:
            W = tf.random.normal((net.shape[1], num_hidden_channels))
    
        net = tf.matmul(net, W)
        activation_function = tf.tanh 
        net = activation_function(net)

    net = (1 + net) / 2.0
    net = tf.cast(tf.reshape(net, (1, size[0], size[1], net.shape[-1])), dtype = tf.float32)

    return tf.tile(net, [batches, 1, 1, 1])


def product(l):
  prod = 1

  for x in l:
    prod *= x

  return prod


def collapse_shape(shape, a, b):
  shape = list(shape)
  if a < 0:
    n_pad = -a
    pad = n_pad * [1]
    return collapse_shape(pad + shape, a + n_pad, b + n_pad)
  
  if b > len(shape):
    n_pad = b - len(shape)
    pad = n_pad * [1]
    return collapse_shape(shape + pad, a, b)
 
  return [product(shape[:a])] + shape[a:b] + [product(shape[b:])]


def resize_bilinear_nd(t, target_shape):
  shape = t.get_shape().as_list()
  target_shape = list(target_shape)
  assert len(shape) == len(target_shape)

  d = 0
  while d < len(shape):
    if shape[d] == target_shape[d]:
      d += 1
      continue

    new_shape = shape[:]
    new_shape[d : d+2] = target_shape[d : d+2]

    shape_ = collapse_shape(shape, d, d+2)
    new_shape_ = collapse_shape(new_shape, d, d+2)

    t_ = tf.reshape(t, shape_)
    t_ = tf.compat.v1.image.resize_bilinear(t_, new_shape_[1:3])

    t = tf.reshape(t_, new_shape)
    shape = new_shape
    d += 2

  return t



def lowres_tensor(shape, underlying_shape, offset=None, sd=None):
    sd = sd or 0.01
    init_val = sd * np.random.randn(*underlying_shape).astype("float32")
    underlying_t = tf.Variable(init_val)
    t = resize_bilinear_nd(underlying_t, shape)

    if offset is not None:
        if not isinstance(offset, list):
            offset = len(shape) * [offset]

        for n in range(len(offset)):
            if offset[n] is True:
                offset[n] = shape[n] / underlying_shape[n] / 2
            if offset[n] is False:
                offset[n] = 0
            offset[n] = int(offset[n])

        padding = [(pad, 0) for pad in offset]
        t = tf.pad(t, padding, "SYMMETRIC")
        begin = len(shape) * [0]
        t = tf.slice(t, begin, shape)
        
    return t


def to_valid_rgb(image: tf.Tensor, normalizer: Union[str, Callable], 
                 values_range: Tuple[float, float]) -> tf.Tensor:
    image = recorrelate_colors(image)

    if normalizer == 'sigmoid':
        image = tf.nn.sigmoid(image)

    elif normalizer == 'clip':
        image = tf.clip_by_value(image, values_range[0], values_range[1])
        
    else:
        image = normalizer(image)

    image = image - tf.reduce_min(image, keepdims = True)
    image = image / tf.reduce_max(image, keepdims = True)
    image *= values_range[1] - values_range[0]
    image += values_range[0]

    return image


class Parameterization:

    def __init__(self, images: List[tf.Tensor],
                       function: List[Callable]):
        self.images = images
        self.function = function


    def __add__(self, new):
        return Parameterization(
            images = self.images + new.images,
            function = self.function + new.function
        )
    
    @staticmethod
    def image(image: tf.Tensor, size: int, batches: int = 1,
              normalizer: str = 'sigmoid', 
              values_range: Tuple[float, float] = (0, 1)):

        values_range = (min(values_range), max(values_range))
        
        images = tf.tile(tf.image.resize(image, [size, size]), [batches, 1, 1, 1])
        images = tf.reshape(images, (images.shape[0], 1,) + images.shape[1:])

        function = lambda images: to_valid_rgb(images, normalizer, values_range)

        return Parameterization(list(images), [function] * batches)
    

    @staticmethod
    def image_normal(size: int, batches: int = 1, std : float = 0.01,
                     normalizer: str = 'sigmoid', 
                     values_range: Tuple[float, float] = (0, 1)):

        values_range = (min(values_range), max(values_range))

        images = tf.random.normal((batches, 1, size, size, 3), std, dtype = tf.float32)

        function = lambda images: to_valid_rgb(images, normalizer, values_range)

        return Parameterization(list(images), [function] * batches)


    @staticmethod
    def image_fft(size: int, batches: int = 1, 
                  std: float = 0.01, fft_decay: float = 0.85,
                  normalizer: str = 'sigmoid', 
                  values_range: Tuple[float, float] = (0, 1)):

        values_range = (min(values_range), max(values_range))

        shape = (batches, 1, size, size, 3)

        images = fft_image(shape, std)  
        fft_scale = get_fft_scale(shape[2], shape[3], decay_power = fft_decay)
        
        function = lambda images: to_valid_rgb(fft_to_rgb(images, shape, fft_scale), normalizer, values_range)

        return Parameterization(list(images), [function] * batches)


    @staticmethod
    def image_cppn(size: int, batches: int = 1, 
                   num_output_channels: int = 3, 
                   num_hidden_channels: int = 24, 
                   num_layers: int = 8, 
                   activation_func: Callable = composite_activation, 
                   seed: int = 0):
    
        images = cppn((size, size), batches, num_output_channels, 
                      num_hidden_channels, num_layers, 
                      activation_func, seed)
        
        shape = (batches, 1, size, size, 3)
        images = tf.reshape(images, shape)
              
        function = lambda images: images

        return Parameterization(list(images), [function] * batches)
    

    @staticmethod
    def image_laplacian_pyramid(size: int, batches: int = 1,
                                levels: int = 4, sd: int = 0.01,
                                normalizer: str = 'sigmoid', 
                                values_range: Tuple[float, float] = (0, 1)):
        
        values_range = (min(values_range), max(values_range))

        shape = (batches, 1, size, size, 3)

        batch_dims = shape[:-3]
        w, h, ch = shape[-3:]
        pyramid = 0

        for n in range(levels):
            k = 2 ** n
            pyramid += lowres_tensor(shape, batch_dims + (w // k, h // k, ch), sd = sd)
    
        pyramid *= 255.0

        function = lambda pyramid: to_valid_rgb(pyramid, normalizer, values_range)

        return Parameterization(list(pyramid), [function] * batches)
    

    @staticmethod
    def image_cppn_fft(size: int, batches: int = 1,
                       fft_decay: float = 0.85,
                       num_output_channels: int = 3, 
                       num_hidden_channels: int = 24, 
                       num_layers: int = 8, 
                       activation_func: Callable = composite_activation, 
                       seed: int = 0,
                       normalizer: str = 'sigmoid', 
                       values_range: Tuple[float, float] = (0, 1)):
    
        values_range = (min(values_range), max(values_range))

        frequencies = fft_2d_freq(size, size)

        shape = (batches, 2, 1, 3) + frequencies.shape

        images = cppn(frequencies.shape, batches * 2, num_output_channels, 
                      num_hidden_channels, num_layers, 
                      activation_func, seed)
        
        images = tf.reshape(images, shape)

        fft_scale = get_fft_scale(size, size, decay_power = fft_decay)

        shape = (batches, 1, size, size, 3)

        function = lambda images: to_valid_rgb(fft_to_rgb(images, shape, fft_scale), normalizer, values_range)

        return Parameterization(list(images), [function] * batches)


    @staticmethod
    def image_laplacian_pyramid_fft(size: int, batches: int = 1,
                                    levels: int = 4, sd: int = 0.01,
                                    fft_decay: float = 0.85,
                                    normalizer: str = 'sigmoid', 
                                    values_range: Tuple[float, float] = (0, 1)):
        
        values_range = (min(values_range), max(values_range))
        
        frequencies = fft_2d_freq(size, size)
        shape = (batches, 2, 1, 3) + frequencies.shape

        pyramid = 0

        for n in range(levels):
            k = 2 ** n
            pyramid += lowres_tensor(shape, (batches, 2, 1, ) + (3, size // k, size // k), sd = sd)
        
        images = tf.reshape(pyramid * 255.0, shape)

        fft_scale = get_fft_scale(size, size, decay_power = fft_decay)

        shape = (batches, 1, size, size, 3)

        function = lambda images: to_valid_rgb(fft_to_rgb(images, shape, fft_scale), normalizer, values_range)

        return Parameterization(list(images), [function] * batches)


    @staticmethod
    def shared_parameterization(size: int, dividers: List[int], 
                                batches: int = 1,
                                fft_decay: float = 0.85,
                                normalizer: str = 'sigmoid', 
                                values_range: Tuple[float, float] = (0, 1)):
        
        values_range = (min(values_range), max(values_range))
        
        frequencies = fft_2d_freq(size, size)
        shape = (batches, 2, 1, 3) + frequencies.shape

        shared = []

        for b in range(batches):
            shared.append(lowres_tensor((batches, size, size, 3), (1, size // dividers[b], size // dividers[b], 3)))
        
        shared = sum(shared)
        
        fft_scale = get_fft_scale(size, size, decay_power = fft_decay)

        shape = (batches, 1, size, size, 3)
        images = fft_image(shape)
        
        function = lambda images: to_valid_rgb(fft_to_rgb(images, shape, fft_scale) + shared, normalizer, values_range)

        return Parameterization(list(images), [function] * batches)
    

    @staticmethod
    def image_style_transfer(image: tf.Tensor, style: tf.Tensor,
                             std: float = 0.01, fft_decay: float = 0.85,
                             normalizer: str = 'sigmoid', 
                             values_range: Tuple[float, float] = (0, 1)):

        values_range = (min(values_range), max(values_range))

        shape = (1,) + image.shape

        images = fft_image(shape, std)  
        fft_scale = get_fft_scale(shape[2], shape[3], decay_power = fft_decay)

        function = lambda images: to_valid_rgb(fft_to_rgb(images, shape, fft_scale), normalizer, values_range)
        identity = lambda style: style

        return Parameterization(list([images[0], image, style]), [function, identity, identity])