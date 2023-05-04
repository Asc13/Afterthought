import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from typing import List, Tuple, Callable, Union

from src.Miscellaneous import dot
from src.Wrapper import *


class Objective:

    def __init__(self, model: Wrapper,
                       layers: List[Layer],
                       function: List[Callable],
                       indexes: List[int]):
        self.model = model
        self.layers = layers
        self.function = function
        self.indexes = indexes


    def __add__(self, new):
        if isinstance(new, (int, float)):
            layers = self.layers
            function = list(map(lambda x: x + new, self.function))
            indexes = self.indexes
        
        else:
            layers = self.layers + new.layers
            function = self.function + new.function
            indexes = self.indexes + [list(map(lambda x: x + self.indexes[-1][-1] + 1, new.indexes[-1]))]

        return Objective(
            self.model,
            layers = layers,
            function = function,
            indexes = indexes
        )


    def __sub__(self, new):
        return self + (-1 * new)


    def __neg__(self):
        return -1 * self


    def __mul__(self, new):
        if isinstance(new, (int, float)):
            layers = self.layers
            function = list(map(lambda m: lambda x, y, z: m(x, y, z) * new, self.function))
            indexes = self.indexes

        else:
            layers = self.layers + new.layers
            function = list(map(lambda m: (lambda x, y, z: m(x, y, z) * new.function[0](x, y, z)), self.function)) * 2
            indexes =  self.indexes + [list(map(lambda x: x + self.indexes[-1][-1] + 1, new.indexes[-1]))]
    
        return Objective(
            self.model,
            layers = layers,
            function = function,
            indexes = indexes
        )


    def __rmul__(self, new):
        return self.__mul__(new)


    def __radd__(self, new):
        return self.__add__(new)


    def sum(new):
        layers = []
        function = []
        indexes = []

        for n in new:
            layers += n.layers
            function += n.function
            indexes += n.indexes

        return Objective(
            new[0].model,
            layers = layers,
            function = function,
            indexes = indexes
        )
    

    def compile(self, batches) -> Tuple[Model, Callable, Tuple]:
        feature_extractor = Model(inputs = self.model.get_input(), outputs = [*self.layers])

        def objective_function(model_outputs, batch):
            loss = 0.0
            
            for index in range(len(self.indexes)):
                loss += self.function[index](model_outputs, batch, self.indexes[index])

            return loss
        
        input_shape = (batches, *feature_extractor.input.shape[1:])
        
        return feature_extractor, objective_function, input_shape
    

    @staticmethod
    def layer(model: Wrapper,
              layer: Union[str, int],
              deepDream: bool = False,
              batches: Union[int, List[int]] = -1):

        layer = model.get_layer(layer)
        power = 2.0 if deepDream else 1.0

        def optimization_function(model_outputs, batch, indexes):
            if(isinstance(batches, list) and (batch in batches or -1 in batches)) or \
              ((isinstance(batches, int)) and (batch == batches or -1 == batches)):
                return tf.reduce_mean(model_outputs[batch][indexes[0]] ** power)

            return tf.constant(0.0)

        return Objective(model, [layer.output], [optimization_function], [[0]])


    @staticmethod
    def channel(model: Wrapper,
                layer: Union[str, int],
                channels: Union[int, List[int]],
                batches: Union[int, List[int]] = -1):

        layer = model.get_layer(layer)
        shape = layer.output.shape

        if len(shape) > 2:
            if type(channels) is int:
                if channels < 0 or channels > shape[3]:
                    channels = shape[3] // 2 

            else:
                for i in range(len(channels)):
                    if channels[i] < 0 or channels[i] > shape[3]:
                        channels[i] = shape[3] // 2

        def optimization_function(model_outputs, batch, indexes):
            loss = 0.0
        
            if(isinstance(batches, list) and (batch in batches or -1 in batches)) or \
              ((isinstance(batches, int)) and (batch == batches or -1 == batches)):

                model_outputs = model_outputs[batch][indexes[0]]

                if type(channels) is int:
                    return tf.reduce_mean(model_outputs[..., channels])
                
                else:
                    for c in channels:
                        loss += tf.reduce_mean(model_outputs[..., c])

                    return loss
                
            return tf.constant(loss)

        return Objective(model, [layer.output], [optimization_function], [[0]])

 
    @staticmethod
    def spatial(model: Wrapper,
                layer: Union[str, int],
                spatials: Union[int, List[int]],
                batches: Union[int, List[int]] = -1):

        layer = model.get_layer(layer)
        shape = layer.output.shape

        coords = []

        if len(shape) > 2:
            if type(spatials) is int:
                coords.append((int(spatials / shape[1]), 
                                spatials % shape[1]))

            else:
                for s in spatials:
                    coords.append((int(s / shape[1]), 
                                    s % shape[1]))


        def optimization_function(model_outputs, batch, indexes):
            loss = 0.0
        
            if(isinstance(batches, list) and (batch in batches or -1 in batches)) or \
              ((isinstance(batches, int)) and (batch == batches or -1 == batches)):

                model_outputs = model_outputs[batch][indexes[0]]

                for c in coords:
                    loss += tf.reduce_mean(model_outputs[..., c[0], c[1], :])

                return loss
                
            return tf.constant(loss)

        return Objective(model, [layer.output], [optimization_function], [[0]])
    
    
    @staticmethod
    def neuron(model: Wrapper,
               layer: Union[str, int],
               channels: Union[int, List[int]],
               x: int = None, 
               y: int = None, 
               batches: Union[int, List[int]] = -1):

        layer = model.get_layer(layer)
        shape = layer.output.shape

        if len(shape) > 2:
            _x = shape[1] // 2 if x is None else x
            _y = shape[2] // 2 if x is None else y

            if type(channels) is int:
                if channels < 0 or channels > shape[3]:
                    channels = shape[3] // 2

            else:
                for i in range(len(channels)) :
                    if channels[i] < 0 or channels[i] > shape[3]:
                        channels[i] = shape[3] // 2

        def optimization_function(model_outputs, batch, indexes):
            loss = 0.0
            
            if(isinstance(batches, list) and (batch in batches or -1 in batches)) or \
              ((isinstance(batches, int)) and (batch == batches or -1 == batches)):

                model_outputs = model_outputs[batch][indexes[0]]

                if type(channels) is int:
                    return tf.reduce_mean(model_outputs[..., _x, _y, channels])
                
                else:
                    for c in channels:
                        loss += tf.reduce_mean(model_outputs[..., _x, _y, c])

                    return loss
                
            return tf.constant(loss)
            
        return Objective(model, [layer.output], [optimization_function], [[0]])
    

    @staticmethod
    def direction(model: Wrapper,
                  layer: Union[str, int],
                  vectors: Union[tf.Tensor, List[tf.Tensor]],
                  batches: Union[int, List[int]] = -1,
                  power: float = 0.0):
        
        layer = model.get_layer(layer)
        vectors = vectors.astype("float32")

        def optimization_function(model_outputs, batch, indexes):
            loss = 0.0
            
            if(isinstance(batches, list) and (batch in batches or -1 in batches)) or \
              ((isinstance(batches, int)) and (batch == batches or -1 == batches)):

                model_outputs = model_outputs[batch][indexes[0]]

                if type(vectors) is not list:
                    return dot(model_outputs, vectors, power)
                
                else:
                    for v in vectors:
                        loss += dot(model_outputs, v, power)

                    return loss
        
            return tf.constant(loss)
        
        return Objective(model, [layer.output], [optimization_function], [[0]])
    

    @staticmethod
    def channel_interpolate(model: Wrapper,
                            layer1: Union[str, int],
                            channel1: int,
                            layer2: Union[str, int],
                            channel2: int):

        layer1 = model.get_layer(layer1)
        layer2 = model.get_layer(layer2)
        
        def optimization_function(model_outputs, batch, indexes):
            S = 0

            batches = len(model_outputs)
            weights = (np.arange(batches) / float(batches - 1))

            model1_outputs = model_outputs[batch][indexes[0]]
            model2_outputs = model_outputs[batch][indexes[1]]

            S += (1 - weights[batch]) * tf.reduce_mean(model1_outputs[..., channel1])
            S += weights[batch] * tf.reduce_mean(model2_outputs[..., channel2])

            return S
        
        return Objective(model, [layer1.output, layer2.output], [optimization_function], [[0, 1]])
    

    @staticmethod
    def diversity(model: Wrapper,
                  layer: Union[str, int]):
        
        layer = model.get_layer(layer)
        
        def optimization_function(model_outputs, batch, indexes):
            batches = len(model_outputs)
            outputs = []

            for m in model_outputs:
                outputs.append(m[indexes[0]])

            flattened = tf.reshape(outputs, [batches, -1, model_outputs[batch][indexes[0]].shape[-1]])

            grams = tf.matmul(flattened, flattened, transpose_a = True)
            grams = tf.nn.l2_normalize(grams, axis = [1, 2], epsilon = 1e-10)

            return sum([sum([tf.reduce_sum(grams[i] * grams[j])
                        for j in range(batches) if j != i])
                        for i in range(batches)]) / batches

        return Objective(model, [layer.output], [optimization_function], [[0]])
    

    @staticmethod
    def alignment(model: Wrapper,
                  layer: Union[str, int],
                  decay: int = 2):

        layer = model.get_layer(layer)

        def optimization_function(model_outputs, batch, indexes):
            batches = len(model_outputs)
            loss = 0.0

            for d in np.arange(1, batches, 1):
                for i in range(batches - d):
                    arr1, arr2 = model_outputs[i][indexes[0]], model_outputs[i + d][indexes[0]]
                    loss += tf.reduce_mean((arr1 - arr2) ** 2) / decay ** float(d)
            
            return loss

        return Objective(model, [layer.output], [optimization_function], [[0]])
    

    @staticmethod
    def activation_difference(model: Wrapper,
                              layers: List[Union[str, int]],
                              transform: Callable = None, 
                              index: int = 1):
        outs = []

        for l in layers:
            outs.append(model.get_layer(l).output)

        def optimization_function(model_outputs, batch, indexes):
            loss = 0.0

            if batch == 0:
                activations = [model_outputs[index][i] for i in indexes]
                transfer_activations = [model_outputs[0][i] for i in indexes]

                if transform is not None:
                    activations = [transform(activation) for activation in activations]
                    transfer_activations = [transform(activation) for activation in transfer_activations]

                def mean_L1(a, b):
                    return tf.reduce_mean(tf.abs(a - b))

                return tf.add_n([mean_L1(a, b) for a, b in zip(activations, transfer_activations)])

            return tf.constant(loss)

        return Objective(model, outs, [optimization_function], [list(range(0, len(outs)))])