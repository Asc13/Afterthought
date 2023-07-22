import tensorflow as tf
import numpy as np
import cv2

from abc import ABC, abstractmethod
from tensorflow.keras import Model
from tensorflow.keras.losses import cosine_similarity
from skimage.segmentation import quickshift
from sklearn import linear_model
from typing import Union, Any, Callable

from src.Miscellaneous import *
from src.Wrapper import *


class Attribution(ABC):

    def __init__(self, model: Wrapper,
                       batches: int = 1):
        
        self.model = Model(model.get_input(), model.get_output()) 
        self.batches = batches


    @abstractmethod
    def explainer(self, images: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                        labels: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        
        raise NotImplementedError()
    

    def __call__(self, images: tf.Tensor,
                       labels: tf.Tensor) -> tf.Tensor:
            
        return self.explainer(images, labels)
    


class GradientCAM(Attribution):

    def __init__(self, model: Wrapper, 
                       layer: Union[int, str] = None, 
                       batches: int = 1):
        '''
        Inputs
        ----------
        model: Model wrapper

        layer: Layer name or index to use on attribution

        batches: Number of batches
        ''' 

        super().__init__(model, batches)

        if type(layer) is str or type(layer) is int:
            self.layer = model.get_layer(layer)
           
        else:
            self.layer = next(layer for layer in model.get_layers()[::-1] if hasattr(layer, 'filters'))

        self.model = Model(model.get_input(), [self.layer.output, model.get_output()])


    def explainer(self, images: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                        labels: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        '''
        Inputs
        ----------
        images - Images tensor to use on attribution

        labels - Label one hot encoding tensor
        ''' 
        
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
    def compute_weights(gradients: tf.Tensor, feature_maps: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(gradients, axis = (1, 2), keepdims = True)