import tensorflow as tf

from typing import Tuple, List, Union, Callable


def blur_T(sigma_range: Tuple[float, float] = (1.0, 2.0), kernel_size: int = 10) -> Callable:
    uniform = tf.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    uniform_xx, uniform_yy = tf.meshgrid(uniform, uniform)

    kernel_size = tf.cast(kernel_size, tf.float32)
    sigma_min = tf.cast(max(sigma_range[0], 0.1), tf.float32)
    sigma_max = tf.cast(max(sigma_range[1], 0.1), tf.float32)

    def blur_conv_T(images: tf.Tensor) -> tf.Tensor:
        sigma = tf.random.uniform([], minval = sigma_min, maxval = sigma_max, dtype = tf.float32)

        kernel = tf.exp(-0.5 * (uniform_xx ** 2 + uniform_yy ** 2) / sigma ** 2)
        kernel /= tf.reduce_sum(kernel)

        kernel = tf.reshape(kernel, (kernel_size, kernel_size, 1, 1))
        kernel = tf.tile(kernel, [1, 1, 3, 1])

        return tf.nn.depthwise_conv2d(images, kernel, strides = [1, 1, 1, 1], padding = 'SAME')

    return blur_conv_T


def jitter(delta: int = 6, seed = None) -> Callable:
    def jitter_helper(image: tf.Tensor) -> tf.Tensor:
        t_image = tf.convert_to_tensor(image, dtype = tf.float32)
        t_shp = tf.shape(t_image)

        crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - delta, t_shp[-1:]], 0)
        crop = tf.image.random_crop(t_image, crop_shape, seed = seed)
        shp = t_image.get_shape().as_list()

        mid_shp_changed = [
            shp[-3] - delta if shp[-3] is not None else None,
            shp[-2] - delta if shp[-3] is not None else None,
        ]
        crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])

        return crop
    
    return jitter_helper


def scale(scales: Union[float, List[float]], seed = None) -> Callable:
    def scale_helper(images: tf.Tensor) -> tf.Tensor:
        t = tf.convert_to_tensor(images, dtype = tf.float32)
        
        if isinstance(scales, list):
            scale = tf.constant(scales)[tf.random.uniform((), 0, len(scales), "int32", seed = seed)]

        else:
            scale = tf.constant(list(scales))[tf.random.uniform((), 0, 1, "int32", seed = seed)]
        
        shp = tf.shape(t)
        scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")

        return tf.compat.v1.image.resize_bilinear(t, scale_shape)

    return scale_helper


def flip(horizontal: bool = True, vertical: bool = False, seed = None) -> Callable:
    def flip_helper(images: tf.Tensor) -> tf.Tensor:
        if horizontal:
            images = tf.image.random_flip_left_right(images, seed = seed)

        if vertical:
            images = tf.image.random_flip_up_down(images, seed = seed)

        return images
    return flip_helper


def padding(size: int = 6, pad_value: float = 0.0) -> Callable:
    pad_array = [(0, 0), (size, size), (size, size), (0, 0)]
    pad_value = tf.cast(pad_value, tf.float32)

    def padding_helper(images: tf.Tensor) -> tf.Tensor:
        return tf.pad(images, pad_array, mode = "CONSTANT", constant_values = pad_value)

    return padding_helper


def apply_kernel(size: int, type: str) -> Callable:
    if type == 'BOX':
        kernel = tf.ones((size, size, 1, 1)) / (size ** 2)

    elif type == 'SIMPLE-LAPLACE':
        kernel = tf.cast(kernel_fabricator(size, 8, -1), tf.float32)
    
    elif type == 'HARD-LAPLACE':
        kernel = kernel_fabricator(size, -4, 1)
        kernel[np.ix_((0,-1), (0,-1))] = 0.0
        kernel = tf.cast(kernel, tf.float32)

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

    if kernel:
        kernel = tf.reshape(kernel, (size, size, 1, 1))
        kernel = tf.tile(kernel, [1, 1, 3, 1])

    def apply_kernel_helper(images: tf.Tensor) -> tf.Tensor:
        return tf.nn.depthwise_conv2d(images, kernel, strides = [1, 1, 1, 1], padding = 'SAME') if kernel else images
    
    return apply_kernel_helper


def apply_double_kernel(size: int, type: str) -> Callable:
    if type == 'PREWIT':
        kernel1 = np.zeros((size, size))
        kernel1[0] = -1
        kernel1[-1] = 1
        kernel1 = tf.cast(kernel1, tf.float32)

        kernel2 = np.zeros((size, size))
        kernel2[:, 0] = -1
        kernel2[:, -1] = 1
        kernel2 = tf.cast(kernel2, tf.float32)

    elif type == 'SCHARR':
        kernel1 = np.zeros((size, size))
        kernel1[0] = -3
        kernel1[-1] = 3
        kernel1[0, np.arange(1, size - 1)] = -10
        kernel1[-1, np.arange(1, size - 1)] = 10
        kernel1 = tf.cast(kernel1, tf.float32)

        kernel2 = np.zeros((size, size))
        kernel2[:, 0] = -3
        kernel2[:, -1] = 3
        kernel2[np.arange(1, size - 1), 0] = -10
        kernel2[np.arange(1, size - 1), -1] = 10
        kernel2 = tf.cast(kernel2, tf.float32)
    
    elif type == 'LAPLACIAN-ALTERNATIVE':
        pass

    kernel1 = tf.reshape(kernel1, (size, size, 1, 1))
    kernel1 = tf.tile(kernel1, [1, 1, 3, 1])

    kernel2 = tf.reshape(kernel2, (size, size, 1, 1))
    kernel2 = tf.tile(kernel2, [1, 1, 3, 1])

    def apply_double_kernel_helper(images: tf.Tensor) -> tf.Tensor:
        images = tf.nn.depthwise_conv2d(images, kernel1, strides = [1, 1, 1, 1], padding = 'SAME')
        return tf.nn.depthwise_conv2d(images, kernel2, strides = [1, 1, 1, 1], padding = 'SAME')
    
    return apply_double_kernel_helper


def mean(size: int = 1) -> Callable:
    def mean_helper(images: tf.Tensor) -> tf.Tensor:
        return tfa.image.mean_filter2d(images, (size, size))

    return mean_helper


def median(size: int = 1) -> Callable:
    def median_helper(images: tf.Tensor) -> tf.Tensor:
        return tfa.image.median_filter2d(images, (size, size))

    return median_helper


def deconvolution(size: int = 1) -> Callable:
    ANGLE = np.deg2rad(135)
    D = 22

    def deconvolution_helper(images: tf.Tensor) -> tf.Tensor:
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


def composition(input_shape: Tuple, transformations: List[Callable]) -> Callable:
    def compose(images: tf.Tensor) -> tf.Tensor:     
        for func in transformations:
            images = func(images)

        return tf.image.resize(images, tf.cast([input_shape[0], input_shape[1]], tf.int32))

    return compose


def standard(input_shape: Tuple, unit: int):
    unit = int(unit / 16) 
    return composition(input_shape,
        [
            padding(unit * 4),
            jitter(unit * 2),
            jitter(unit * 2),
            jitter(unit * 4),
            jitter(unit * 4),
            jitter(unit * 4),
            flip(seed = 0),
            scale((0.92, 0.96), seed = 0),
            blur_T(sigma_range = (1.0, 1.1)),
            jitter(unit),
            jitter(unit),
            flip(seed = 0)
        ])