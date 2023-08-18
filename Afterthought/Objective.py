import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from typing import List, Tuple, Callable, Union

from Afterthought.Miscellaneous import dot
from Afterthought.Wrapper import *


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
            
            indexes = self.indexes
            last = self.indexes[-1][-1]

            for i in new.indexes:
                indexes += [list(map(lambda x: x + last + 1, i))]

        return Objective(
            self.model,
            layers = layers,
            function = function,
            indexes = indexes
        )


    def __mul__(self, new):
        if isinstance(new, (int, float)):
            layers = self.layers
            function = list(map(lambda m: lambda x, y, z: m(x, y, z) * new, self.function))
            indexes = self.indexes

        else:
            layers = self.layers + new.layers
            function = list(map(lambda m: (lambda x, y, z: m(x, y, z) * new.function[0](x, y, z)), self.function)) * 2
            
            indexes = self.indexes
            last = self.indexes[-1][-1]

            for i in new.indexes:
                indexes += [list(map(lambda x: x + last + 1, i))]

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

    
    def __sub__(self, new):
        return self + (-1 * new)


    def __neg__(self):
        return -1 * self
    

    def sum(new):
        i = new[0]

        for n in range(1, len(new)):
            i += new[n]

        return i
    

    def compile(self, slots) -> Tuple[Model, Callable, Tuple]:
        feature_extractor = Model(inputs = self.model.get_input(), outputs = [*self.layers])

        def objective_function(model_outputs, slot):
            loss = 0.0
            
            for index in range(len(self.indexes)):
                loss += self.function[index](model_outputs, slot, self.indexes[index])

            return loss
        
        input_shape = (slots, *feature_extractor.input.shape[1:])
        
        return feature_extractor, objective_function, input_shape
    

    @staticmethod
    def layer(model: Wrapper,
              layer: Union[str, int],
              squared: bool = False,
              slots: Union[int, List[int]] = -1):

        '''
        Inputs
        ----------
        model - Model wrapper

        layer - Layer to optimize (name or index)

        squared - Flag to raise the activation to 2

        slots - List of slots that use this objetive (-1 to use all)
        ''' 
                
        layer = model.get_layer(layer)
        power = 2.0 if squared else 1.0

        def optimization_function(model_outputs, slot, indexes):
            if(isinstance(slots, list) and (slot in slots or -1 in slots)) or \
              ((isinstance(slots, int)) and (slot == slots or -1 == slots)):
                return tf.reduce_mean(model_outputs[slot][indexes[0]] ** power)

            return tf.constant(0.0)

        return Objective(model, [layer.output], [optimization_function], [[0]])


    @staticmethod
    def channel(model: Wrapper,
                layer: Union[str, int],
                channels: Union[int, List[int]],
                slots: Union[int, List[int]] = -1):

        '''
        Inputs
        ----------
        model - Model wrapper

        layer - Layer to optimize (name or index)

        channel - Channel list to optimize (activations shape = (x, y, channels))

        slots - List of slots that use this objetive (-1 to use all)
        ''' 

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

        def optimization_function(model_outputs, slot, indexes):
            loss = 0.0
        
            if(isinstance(slots, list) and (slot in slots or -1 in slots)) or \
              ((isinstance(slots, int)) and (slot == slots or -1 == slots)):

                model_outputs = model_outputs[slot][indexes[0]]
                
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
                slots: Union[int, List[int]] = -1):
        '''
        Inputs
        ----------
        model - Model wrapper

        layer - Layer to optimize (name or index)

        spatials - Spatial list to optimize

        slots - List of slots that use this objetive (-1 to use all)
        ''' 

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


        def optimization_function(model_outputs, slot, indexes):
            loss = 0.0
        
            if(isinstance(slots, list) and (slot in slots or -1 in slots)) or \
              ((isinstance(slots, int)) and (slot == slots or -1 == slots)):

                model_outputs = model_outputs[slot][indexes[0]]

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
               slots: Union[int, List[int]] = -1):
        '''
        Inputs
        ----------
        model - Model wrapper

        layer - Layer to optimize (name or index)

        channels - Channel list to optimize which the neurons come from

        x and y - Neuron coordinates inside the channel (activations shape = (x, y, channels))

        slots - List of slots that use this objetive (-1 to use all)
        ''' 

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

        def optimization_function(model_outputs, slot, indexes):
            loss = 0.0
            
            if(isinstance(slots, list) and (slot in slots or -1 in slots)) or \
              ((isinstance(slots, int)) and (slot == slots or -1 == slots)):

                model_outputs = model_outputs[slot][indexes[0]]

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
                  slots: Union[int, List[int]] = -1,
                  power: float = 0.0):
        '''
        Inputs
        ----------
        model - Model wrapper

        layer - Layer to optimize (name or index)

        vectors - Vectors that define the direction to each channel tensor

        slots - List of slots that use this objetive (-1 to use all)

        power - Dot product power
        ''' 
                
        layer = model.get_layer(layer)
        vectors = vectors.astype("float32")

        def optimization_function(model_outputs, slot, indexes):
            loss = 0.0
            
            if(isinstance(slots, list) and (slot in slots or -1 in slots)) or \
              ((isinstance(slots, int)) and (slot == slots or -1 == slots)):

                model_outputs = model_outputs[slot][indexes[0]]

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
        '''
        Inputs
        ----------
        model - Model wrapper

        layer1 - Interpolation first extreme layer to optimize (name or index)

        channel1 - Interpolation first extreme channel to optimize

        layer2 - Interpolation second extreme layer to optimize (name or index)

        channel2 - Interpolation second extreme channel to optimize
        ''' 

        layer1 = model.get_layer(layer1)
        layer2 = model.get_layer(layer2)
        
        def optimization_function(model_outputs, slot, indexes):
            S = 0

            slots = len(model_outputs)
            weights = (np.arange(slots) / float(slots - 1))

            model1_outputs = model_outputs[slot][indexes[0]]
            model2_outputs = model_outputs[slot][indexes[1]]

            S += (1 - weights[slot]) * tf.reduce_mean(model1_outputs[..., channel1])
            S += weights[slot] * tf.reduce_mean(model2_outputs[..., channel2])

            return S
        
        return Objective(model, [layer1.output, layer2.output], [optimization_function], [[0, 1]])
    

    @staticmethod
    def diversity(model: Wrapper,
                  layer: Union[str, int]):
        '''
        Inputs
        ----------
        model - Model wrapper

        layer - Layer to optimize (name or index)
        '''

        layer = model.get_layer(layer)
        
        def optimization_function(model_outputs, slot, indexes):
            slots = len(model_outputs)
            outputs = []

            for m in model_outputs:
                outputs.append(m[indexes[0]])

            flattened = tf.reshape(outputs, [slots, -1, model_outputs[slot][indexes[0]].shape[-1]])

            grams = tf.matmul(flattened, flattened, transpose_a = True)
            grams = tf.nn.l2_normalize(grams, axis = [1, 2], epsilon = 1e-10)

            return sum([sum([tf.reduce_sum(grams[i] * grams[j])
                        for j in range(slots) if j != i])
                        for i in range(slots)]) / slots

        return Objective(model, [layer.output], [optimization_function], [[0]])


    @staticmethod
    def activation_difference(model: Wrapper,
                              layers: List[Union[str, int]],
                              transform: Callable = None, 
                              index: int = 1):
        '''
        Inputs
        ----------
        model - Model wrapper

        layers - Layers to optimize (name or index)

        transform - transform function for style tranfer

        index - slot index to optimize for the activations
        ''' 

        outs = []

        for l in layers:
            outs.append(model.get_layer(l).output)

        def optimization_function(model_outputs, slot, indexes):
            loss = 0.0

            if slot == 0:
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