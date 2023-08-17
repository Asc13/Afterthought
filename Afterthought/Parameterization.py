import tensorflow as tf
import numpy as np

from typing import Union, Tuple, List, Callable, Any

from src.Miscellaneous import *
from src.Transformation import apply_kernel


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


def fft_to_rgb(image: tf.Tensor, fft_scale: tf.Tensor) -> tf.Tensor:
    spectrum = tf.complex(image[0], image[1]) * fft_scale

    image = tf.signal.irfft2d(spectrum)
    image = tf.transpose(image, (0, 2, 3, 1))

    return image / 4.0


def fft_image(shape: Tuple, std: float = 0.01, seed: int = 0) -> tf.Tensor:
    slots, _, width, height, channels = shape
    
    frequencies = fft_2d_freq(width, height)

    return tf.random.normal((slots, 2, 1, channels) + frequencies.shape, stddev = std, seed = seed)


def cppn(net: tf.Tensor, size: int,
         num_output_channels: int, 
         num_hidden_channels: int, 
         num_layers: int, 
         activation_function: Callable, 
         std: float,
         seed: int) -> tf.Tensor:

    net = tf.reshape(net, (size * size, num_output_channels))

    for i in range(0, num_layers):

        if i == num_layers - 1:
            W = tf.random.normal((net.shape[1], num_output_channels), seed = seed, stddev = std)
        
        else:
            W = tf.random.normal((net.shape[1], num_hidden_channels), seed = seed, stddev = std)
    
        net = activation_function(tf.matmul(net, W))

    return tf.cast(tf.reshape(net, (size, size, num_output_channels)), dtype = tf.float32)


def product(l: tf.Tensor) -> int:
    prod = 1

    for x in l:
        prod *= x

    return prod


def collapse_shape(shape, a: int, b: int):
    shape = list(shape)

    if a < 0:
        n_pad = -a
        pad = n_pad * [1]
        return collapse_shape(pad + shape, a + n_pad, b + n_pad)
  
    if b > len(shape):
        n_pad = b - len(shape)
        pad = n_pad * [1]
        return collapse_shape(shape + pad, a, b)
    
    return [product(shape[:a])] + shape[a: b] + [product(shape[b:])]


def resize_bilinear(t, target_shape):
    shape = t.get_shape().as_list()
    target_shape = list(target_shape)
    assert len(shape) == len(target_shape)

    d = 0
    while d < len(shape):
        if shape[d] == target_shape[d]:
            d += 1
            continue

        new_shape = shape[:]
        new_shape[d: d + 2] = target_shape[d: d + 2]

        shape_ = collapse_shape(shape, d, d + 2)
        new_shape_ = collapse_shape(new_shape, d, d + 2)

        t_ = tf.reshape(t, shape_)
        t_ = tf.compat.v1.image.resize_bilinear(t_, new_shape_[1: 3])

        t = tf.reshape(t_, new_shape)
        shape = new_shape
        d += 2

    return t


def low_resolution(shape, underlying_shape, std, seed,
                   offset = None) -> tf.Tensor:

    init_val = tf.random.normal(underlying_shape, std, dtype = tf.float32, seed = seed)
    underlying_t = tf.Variable(init_val)
    t = resize_bilinear(underlying_t, shape)

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


def convert_to_rgb(image: tf.Tensor, normalizer: Union[str, Callable], 
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


def maco_fft(spectrum, phase):
    phase = phase - tf.reduce_mean(phase)
    phase = phase / (tf.math.reduce_std(phase) + 1e-5)

    spectrum = tf.complex(tf.cos(phase) * spectrum, tf.sin(phase) * spectrum)
    image = tf.signal.irfft2d(spectrum)
    image = tf.transpose(image, (0, 2, 3, 1))

    image = image - tf.reduce_mean(image)
    image = image / (tf.math.reduce_std(image) + 1e-5)

    return image * 2


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
    def image_normal(size: int, slots: int = 1, std : float = 0.01,
                     seed: int = 0, normalizer: str = 'sigmoid', 
                     values_range: Tuple[float, float] = (0, 1),):
        '''
        Inputs
        ----------
        size - Image resolution ([size, size, 3])

        slots - Number of slots/repetitions of the image tensor

        std - Noise normal function standard deviation

        seed - Noise normal function generation seed

        normalizer - Color normalization function (e.g: sigmoid)

        values_range - Image values range (e.g: between 0 and 1)
        ''' 

        values_range = (min(values_range), max(values_range))

        images = tf.random.normal((slots, 1, size, size, 3), std, dtype = tf.float32, seed = seed)

        function = lambda images: convert_to_rgb(images, normalizer, values_range)

        return Parameterization(list(images), [function] * slots)


    @staticmethod
    def image_fft(size: int, slots: int = 1, 
                  std: float = 0.01, seed: int = 0,
                  fft_decay: float = 0.85,
                  normalizer: str = 'sigmoid', 
                  values_range: Tuple[float, float] = (0, 1)):
        '''
        Inputs
        ----------
        size - Image resolution ([size, size, 3])

        slots - Number of slots/repetitions of the image tensor

        std - Noise normal function standard deviation

        seed - Noise normal function generation seed

        fft_decay - Fast fourier transform decay power

        normalizer - Color normalization function (e.g: sigmoid)

        values_range - Image values range (e.g: between 0 and 1)
        ''' 

        values_range = (min(values_range), max(values_range))

        shape = (slots, 1, size, size, 3)

        images = fft_image(shape, std, seed)  
        fft_scale = get_fft_scale(shape[2], shape[3], decay_power = fft_decay)
        
        function = lambda images: convert_to_rgb(fft_to_rgb(images, fft_scale), normalizer, values_range)

        return Parameterization(list(images), [function] * slots)


    @staticmethod
    def image_cppn(size: int, slots: int = 1, 
                   num_output_channels: int = 3, 
                   num_hidden_channels: int = 24, 
                   num_layers: int = 8, 
                   activation_func: Callable = composite_activation,
                   std: float = 1,
                   seed: int = 0):
        '''
        Inputs
        ----------
        size - Image resolution ([size, size, 3])

        slots - Number of slots/repetitions of the image tensor

        num_output_channels - CPPN network output channels number

        num_hidden_channels - CPPN network hidden channels number

        num_layers - CPPN network layers (iterations) number

        activation_func - CPPN network final activation function (e.g: tanh)
        
        seed - CPPN network noise normal function generation seed
        '''

        r = 3.0 ** 0.5
        coord_range_x = tf.linspace(-r, r, size)
        coord_range_y = tf.linspace(-r, r, size)

        y, x = tf.meshgrid(coord_range_x, coord_range_y, indexing = "ij")

        images = [x, y, tf.math.sqrt(x ** 2 + y ** 2)]
        images = tf.transpose(tf.stack(images), [1, 2, 0])

        images = tf.cast(tf.reshape(images, (1, 1, size, size, num_output_channels)), dtype = tf.float32)
        images = tf.tile(images, [slots, 1, 1, 1, 1])
        
        function = lambda images: tf.map_fn(lambda image: cppn(image, size, num_output_channels, 
                                                               num_hidden_channels, num_layers, 
                                                               activation_func, std, seed), images)

        return Parameterization(list(images), [function] * slots)
    

    @staticmethod
    def image_laplacian_pyramid(size: int, slots: int = 1,
                                levels: int = 4, std: int = 0.01, seed = 0,
                                normalizer: str = 'sigmoid', 
                                values_range: Tuple[float, float] = (0, 1)):
        '''
        Inputs
        ----------
        size - Image resolution ([size, size, 3])

        slots - Number of slots/repetitions of the image tensor

        levels - Pyramiding levels (iterations)

        std - Noise normal function standard deviation

        seed - Noise normal function generation seed

        normalizer - Color normalization function (e.g: sigmoid)

        values_range - Image values range (e.g: between 0 and 1)
        ''' 
            
        values_range = (min(values_range), max(values_range))

        shape = (slots, 1, size, size, 3)

        slot_dims = shape[:-3]
        w, h, ch = shape[-3:]
        pyramid = 0

        for n in range(levels):
            k = 2 ** n
            pyramid += low_resolution(shape, slot_dims + (w // k, h // k, ch), std = std, seed = seed)

        function = lambda pyramid: convert_to_rgb(pyramid, normalizer, values_range)

        return Parameterization(list(pyramid), [function] * slots)
    

    @staticmethod
    def image_style_transfer(image: tf.Tensor, style: tf.Tensor,
                             std: float = 0.01, fft_decay: float = 0.85,
                             normalizer: str = 'sigmoid', 
                             values_range: Tuple[float, float] = (0, 1)):
        '''
        Inputs
        ----------
        image - Image to use as parameterization tensor

        style - Style image to use as parameterization tensorvalues_range

        std - Noise normal function standard deviation

        fft_decay - Fast fourier transform decay power

        normalizer - Color normalization function (e.g: sigmoid)

        values_range - Image values range (e.g: between 0 and 1)
        ''' 

        values_range = (min(values_range), max(values_range))

        shape = (1,) + image.shape

        images = fft_image(shape, std)
        fft_scale = get_fft_scale(shape[2], shape[3], decay_power = fft_decay)

        function = lambda images: convert_to_rgb(fft_to_rgb(images, fft_scale), normalizer, values_range)
        identity = lambda style: style

        return Parameterization(list([images[0], image, style]), [function, identity, identity])
    

    @staticmethod
    def image_maco(size: int, spectrum_path: str,
                   slots: int = 1, 
                   std: float = 1,
                   normalizer: str = 'sigmoid', 
                   values_range: Tuple[float, float] = (0, 1)):
        '''
        Inputs
        ----------
        size - Image resolution ([size, size, 3])

        slots - Number of slots/repetitions of the image tensor

        std - Noise normal function standard deviation

        normalizer - Color normalization function (e.g: sigmoid)

        values_range - Image values range (e.g: between 0 and 1)
        ''' 

        values_range = (min(values_range), max(values_range))

        shape = (slots, size, size // 2 + 1, 3)

        phases = tf.random.normal((slots, 1, shape[3], shape[1], shape[2]), stddev = std)

        spectrum = np.load(spectrum_path)
        spectrum = tf.image.resize(np.moveaxis(spectrum, 0, -1), [shape[1], shape[2]]).numpy()
        spectrum = np.moveaxis(spectrum, -1, 0)

        spectrum, phases = tf.cast(spectrum, tf.float32), tf.cast(phases, tf.float32)
        
        function = lambda phases: convert_to_rgb(maco_fft(spectrum, phases), normalizer, values_range)

        return Parameterization(list(phases), [function] * slots)