import tensorflow as tf
import numpy as np
import cv2

from abc import ABC, abstractmethod
from tensorflow.keras import Model
from tensorflow.keras.losses import cosine_similarity
from skimage.segmentation import quickshift
from sklearn import linear_model
from typing import Union, Any

from Miscellaneous import *


class Attribution(ABC):

    def __init__(self, model: Model,
                       batches: int = 1):
        
        self.model = model
        self.batches = batches


    @abstractmethod
    def explainer(self, images: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                        labels: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        
        raise NotImplementedError()
    

    def __call__(self, images: tf.Tensor,
                       labels: tf.Tensor) -> tf.Tensor:
            
        return self.explainer(images, labels)
    


class GradientCAM(Attribution):

    def __init__(self, model: Model, 
                       layer: Union[int, str] = None, 
                       batches: int = 1):
        
        super().__init__(model, batches)

        if type(layer) is str:
            self.layer = model.get_layer(name = layer)
           
        elif type(layer) is int:
            self.layer = model.get_layer(index = layer)
           
        else:
            self.layer = next(layer for layer in model.layers[::-1] if hasattr(layer, 'filters'))

        self.model = tf.keras.Model(model.input, [self.layer.output, model.output])


    def explainer(self, images: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                        labels: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        
        explainers = None

        for x, y in tf.data.Dataset.from_tensor_slices((images, labels)).batch(self.batches):

            feature_maps, gradients = compute_gradients_fmaps(self.model, x, y)
            weights = GradientCAM.compute_weights(gradients, feature_maps)
            applied_weights = tf.nn.relu(tf.reduce_sum(feature_maps * weights, axis = -1))

            explainers = applied_weights \
                         if explainers is None \
                         else tf.concat([explainers, applied_weights], axis = 0)

        return np.array([cv2.resize(e, (*images.shape[1:3],), interpolation = cv2.INTER_CUBIC)
                        for e in explainers.numpy()])
    

    @staticmethod
    @tf.function(reduce_retracing = True)
    def compute_weights(gradients: tf.Tensor,
                        feature_maps: tf.Tensor) -> tf.Tensor:
        
        return tf.reduce_mean(gradients, axis = (1, 2), keepdims = True)
    


class GradientCAMPP(GradientCAM):

    @staticmethod
    @tf.function(reduce_retracing = True)
    def compute_weights(gradients: tf.Tensor,
                        feature_maps: tf.Tensor) -> tf.Tensor:
     
        denominator = 2.0 * tf.pow(gradients, 2) + \
                      tf.pow(gradients, 3) * \
                      tf.reduce_mean(feature_maps, axis = (1, 2), keepdims = True)
        
        denominator += tf.cast(denominator == 0, tf.float32) * tf.constant(1e-4)

        alphas = tf.pow(gradients, 2) / denominator * tf.nn.relu(gradients)
        weights = tf.reduce_mean(alphas, axis = (1, 2))
       
        return tf.reshape(weights, (weights.shape[0], 1, 1, weights.shape[-1]))
    


class GradientInput(Attribution):

    def explainer(self, images: Union[tf.data.Dataset, tf.Tensor, np.ndarray], 
                        labels: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        
        explainers = None

        for x, y in tf.data.Dataset.from_tensor_slices((images, labels)).batch(self.batches):

            gradients = compute_gradients(self.model, x, y)
        
            explainers = gradients \
                         if explainers is None \
                         else tf.concat([explainers, gradients], axis = 0)
            
        return tf.multiply(explainers, images)
    

class IntegradtedGradient(Attribution):

    def __init__(self, model: tf.keras.Model,
                       batches: int = 1,
                       steps: int = 10,
                       base: float = 0.0):
        
        super().__init__(model, batches)
        self.steps = steps
        self.base = base


    def explainer(self, images: Union[tf.data.Dataset, tf.Tensor, np.ndarray], 
                        labels: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        
        explainers = None
        baseline = tf.ones((*images.shape[1:],)) * self.base

        for x, y in tf.data.Dataset.from_tensor_slices((images, labels)).batch(max(self.batches // self.steps, 1)):
            
            alpha = tf.reshape(tf.linspace(0.0, 1.0, self.steps), (1, -1, *(1,) * len(x.shape[1:])))

            ix = tf.expand_dims(x, axis = 1)
            ix = tf.repeat(ix, repeats = self.steps, axis = 1)
            ix = baseline + alpha * (ix - baseline)
            ix = tf.reshape(ix, (-1, *ix.shape[2:]))

            ry = repeat_labels(y, self.steps)

            gradients = compute_gradients(self.model, ix, ry)
            gradients = tf.reshape(gradients, (-1, self.steps, *gradients.shape[1:]))

            gradients = gradients[:, :-1] + gradients[:, 1:]
            gradients = tf.reduce_mean(gradients, axis = 1) * 0.5

            gradients = (x - baseline) * gradients

            explainers = gradients \
                         if explainers is None \
                         else tf.concat([explainers, gradients], axis = 0)
            
        return explainers
            


class LIME(Attribution):

    def __init__(self, model: Model,
                       batches: int = 1,
                       interpretable_model: Any = linear_model.Ridge(alpha = 2),
                       kernel: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = None,
                       pertubation: Callable[[Union[int, tf.Tensor],int], tf.Tensor] = None,
                       map: Callable[[tf.Tensor], tf.Tensor] = None,
                       reference_value: np.ndarray = None,
                       samples: int = 150,
                       distance_mode: str = "euclidean",
                       kernel_width: float = 45.0,
                       probability: float = 0.5):

        super().__init__(model, batches)

        self.map = map
        self.interpretable_model = interpretable_model
        self.kernel = LIME.get_kernel(distance_mode, kernel_width) if kernel is None else kernel
        self.pertubation = LIME.get_pertubation(probability) if pertubation is None else pertubation
        self.reference_value = reference_value
        self.samples = samples


    def explainer(self, images: Union[tf.data.Dataset, tf.Tensor, np.ndarray], 
                        labels: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        
        if self.map is None:
            self.map = lambda m: tf.cast(quickshift(m.numpy().astype('double'), ratio = 0.5, kernel_size = 2), tf.int32)

        self.reference_value = tf.zeros(images.shape[-1]) if self.reference_value is None \
                                                          else tf.cast(self.reference_value, tf.float32)
            
        explainers = []

        for x, y in tf.data.Dataset.from_tensor_slices((images, labels)):
            
            mapping = self.map(x)
            features = tf.reduce_max(mapping) + tf.ones(1, dtype = tf.int32)
            interpretation_samples = self.pertubation(features, self.samples)

            perturbed_labels = []
            similarities = []

            for i in tf.data.Dataset.from_tensor_slices(interpretation_samples).batch(self.batches):

                masks =  tf.gather(i, indices = mapping, axis = 1)

                perturbed_samples = tf.repeat(tf.expand_dims(x, axis = 0), repeats = masks.shape[0], axis = 0)
                masks_samples = tf.repeat(tf.expand_dims(masks, axis = -1), repeats = x.shape[-1], axis=-1)

                perturbed_samples = perturbed_samples * tf.cast(masks_samples, tf.float32)
                perturbed_samples += (tf.ones((masks_samples.shape)) - tf.cast(masks_samples, tf.float32)) * \
                                      tf.reshape(self.reference_value, (1, 1, 1, x.shape[-1]))

                augmented_labels = tf.repeat(tf.expand_dims(y, axis = 0), len(perturbed_samples), axis = 0)

                perturbed_labels.append(tf.reduce_sum(self.model(perturbed_samples) * augmented_labels, axis = -1))
                similarities.append(self.kernel(x, i, perturbed_samples))

            perturbed_labels = tf.concat(perturbed_labels, axis = 0)
            similarities = tf.concat(similarities, axis = 0)

            self.interpretable_model.fit(interpretation_samples.numpy(),
                                         perturbed_labels.numpy(),
                                         sample_weight = similarities.numpy())
            
            explainers.append(tf.gather(tf.cast(self.interpretable_model.coef_, dtype = tf.float32), 
                                                indices = mapping, axis = 0))

        return tf.stack(explainers, axis = 0)


    @staticmethod
    def get_kernel(distance_mode: str, kernel_width: float) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:

        kernel_width = tf.cast(kernel_width, dtype = tf.float32)

        def kernel(image: tf.Tensor, 
                   samples: tf.Tensor, 
                   perturbed_samples: tf.Tensor) -> tf.Tensor:
            
            augmented_input = tf.expand_dims(image, axis = 0)
            augmented_input = tf.repeat(augmented_input, repeats = len(samples), axis = 0)

            flatten_inputs = tf.reshape(augmented_input, [len(samples), -1])
            flatten_samples = tf.reshape(perturbed_samples, [len(samples), -1])

            if distance_mode == "euclidean":
                distances = tf.norm(flatten_inputs - flatten_samples, ord = 'euclidean', axis = 1)               

            elif distance_mode == "cosine":
                distances = 1.0 - cosine_similarity(flatten_inputs, flatten_samples, axis = 1)

            else:
                raise ValueError()
            
            return tf.exp(-1.0 * (distances ** 2) / (kernel_width ** 2))
        
        return kernel
    

    @staticmethod
    def get_pertubation(probability: float) -> Callable[[Union[int, tf.Tensor],int], tf.Tensor]:

        probability = tf.cast(probability, dtype = tf.float32)
        
        @tf.function
        def pertubation(nb_features: Union[int, tf.Tensor], nb_samples: int) -> tf.Tensor:
            
            probabilities = tf.ones(nb_features, tf.float32) * tf.cast(probability, tf.float32)
            uniform_sampling = tf.random.uniform(shape = [nb_samples, tf.squeeze(nb_features)], dtype = tf.float32, minval = 0, maxval = 1, seed = 0)
            
            return tf.cast(tf.greater(probabilities, uniform_sampling), dtype = tf.int32)

        return pertubation



class KernelSHAP(LIME):
    
    def __init__(self, model: Model, 
                       batches: int = 1,
                       map: Callable[[tf.Tensor], tf.Tensor] = None, 
                       samples: int = 150, 
                       reference_value: np.ndarray = None):
        
        super().__init__(model, 
                         batches, 
                         interpretable_model = linear_model.LinearRegression(), 
                         kernel = KernelSHAP.get_kernel, 
                         pertubation = KernelSHAP.get_pertubation, 
                         map = map, 
                         reference_value = reference_value, 
                         samples = samples)

    
    @staticmethod
    @tf.function
    def get_kernel(image: tf.Tensor, 
                   samples: tf.Tensor, 
                   perturbed_samples: tf.Tensor) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        
        return tf.ones(len(samples), dtype = tf.float32)


    @staticmethod
    @tf.function
    def get_pertubation(nb_features: Union[int, tf.Tensor],
                        nb_samples: int) -> Callable[[Union[int, tf.Tensor], int], tf.Tensor]:

        nb_features = tf.cast(tf.squeeze(nb_features), dtype = tf.int32)

        list_features_indexes = tf.range(1, nb_features)
        denominator = tf.multiply(list_features_indexes, (nb_features - list_features_indexes)) 
        probabilities = tf.cast(tf.concat([[0.0], tf.divide(nb_features - 1, denominator)], 0), dtype = tf.float32)

        nb_selected_features = tf.random.categorical(tf.math.log([probabilities]), nb_samples, dtype = tf.int32)
        nb_selected_features = tf.reshape(nb_selected_features, [nb_samples])
        nb_selected_features = tf.one_hot(nb_selected_features, nb_features, dtype = tf.int32)

        random_values = tf.random.normal([nb_samples, nb_features], seed = 0)
        sorted_values = tf.argsort(random_values, axis = 1, direction = 'DESCENDING')

        threshold = sorted_values * nb_selected_features
        threshold = random_values * tf.one_hot(tf.reduce_sum(threshold, axis = 1), nb_features)
        threshold = tf.repeat(tf.expand_dims(tf.reduce_sum(threshold, axis = 1), axis = 1), 
                              repeats = nb_features, axis = 1)
    
        return tf.cast(tf.greater(random_values, threshold), dtype = tf.int32)
    


class RISE(Attribution):

    def __init__(self, model: Model, 
                       batches: int = 1,
                       samples: int = 10,
                       grid_size: Union[int, Tuple[int]] = 7,
                       probability: float = 0.5):
        
        super().__init__(model, batches)

        self.samples = samples

        shape = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
        self.masks = tf.random.uniform((samples, *shape, 1), minval = 0, maxval = 1, seed = 0) < probability


    def explainer(self, images: Union[tf.data.Dataset, tf.Tensor, np.ndarray], 
                        labels: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        
        explainers = None
        
        for x, y in zip(images, labels):

            nominator = denominator = tf.zeros((*x.shape[:-1], 1))

            for m in tf.data.Dataset.from_tensor_slices(self.masks).batch(self.batches or self.samples):
                
                upsampled_size = tf.cast((int(x.shape[0] * (1.0 + 1.0 / m.shape[1])),
                                          int(x.shape[1] * (1.0 + 1.0 / m.shape[2]))), tf.int32)

                upsampled_masks = tf.image.resize(tf.cast(m, tf.float32), upsampled_size)
                masks = tf.image.random_crop(upsampled_masks, (m.shape[0], *x.shape[:-1], 1), seed = 0)

                mx = tf.expand_dims(x, 0) * masks
                y = repeat_labels(y[tf.newaxis, :], len(m))

                predictions = tf.reduce_sum(self.model(mx) * y, axis = -1)

                nominator += tf.reduce_sum(tf.reshape(predictions, (-1, 1, 1, 1)) * masks, 0)
                denominator += tf.reduce_sum(masks, 0)

            rise_map = nominator / (denominator + tf.constant(1e-4))
            rise_map = rise_map[tf.newaxis, :, :, 0]

            explainers = rise_map \
                         if explainers is None \
                         else tf.concat([explainers, rise_map], axis = 0)
            
        return explainers