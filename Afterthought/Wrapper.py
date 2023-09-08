import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras import Model, Sequential, applications
from typing import Any, Union, Tuple


class Wrapper:

    def __init__(self, model: Union[str, Model, Sequential],
                       weights: Any = 'imagenet',
                       classes: int = 1000):
        
        if isinstance(model, str):
            
            func = getattr(applications, f'{model}')
            self.model = func(include_top = True, weights = weights, classes = classes)
            self.name = model
            self.classes = classes

        else:
            self.model = model
            self.name = 'None'
            self.classes = classes

    
    def layers(self):
        return [layer.name for layer in self.model.layers]
    
    
    def get_layers(self):
        return self.model.layers
    
    
    def get_layer(self, layer: Union[str, int]) -> Layer:
        if isinstance(layer, str):
            return self.model.get_layer(name = layer)
        
        elif layer == -1:
            for layer in self.model.layers:
                if layer.output.shape[-1] == self.classes:
                    return layer
                
        return self.model.get_layer(index = layer)
    
    

    def predict(self, image: tf.Tensor) -> tf.Tensor:
        return self.model(image)
    

    def decode(self, probabilities: np.ndarray, top: int = 5) -> np.ndarray:
        return applications.imagenet_utils.decode_predictions(probabilities, top = top)[0]


    def get_input(self):
        return self.model.input
    
    
    def get_output(self):
        return self.model.output
    

    def get_input_shape(self) -> Tuple:
        return self.model.input.shape
    

    def get_layer_output_shape(self, layer) -> Tuple:
        return self.get_layer(layer).output.shape

