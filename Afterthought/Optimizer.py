import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from typing import Callable
from tensorflow.keras.optimizers import Adam, Nadam, Optimizer
from typing import Union, Tuple, List, Callable

from Afterthought.Objective import Objective
from Afterthought.Parameterization import *
from Afterthought.Regularizer import *
from Afterthought.Transformation import *
from Afterthought.Miscellaneous import *
from Afterthought.Wrapper import *


def run(objective: Objective,
        parameterization: Parameterization,
        optimizer: Optimizer = None,
        steps: int = 256,
        learning_rate: float = 0.05,
        transformations: Union[List[Callable], str] = 'standard',
        regularizers: List[Callable] = [],
        image_shape: int = 512,
        normalize_gradients: bool = False,
        save_steps: List[int] = None,
        only_first_slot: bool = False,
        verbose = True) -> List[tf.Tensor]:
    '''
    Inputs
    ----------
    objective - Objective to optimize (e.g: channel, layer, direction, etc ...)

    parameterization - Image parameterization to use on optimization (e.g: fft, normal, cppn, etc ...)

    optimizer - Gradient ascent optimizer (eg: Adam)

    steps - Algorithm iterations (also known as epochs)

    learning_rate - Gradient ascent optimizer learning rate (default: 0.05)

    transformations - List of transformations to use on the optimization process (default: standard, which uses a default set on the toolkit)

    regularizers - List of regularizers to use on the optimization process (default: no regularizers)

    image_shape - Images resolution (height and width, default: 512)

    normalize_gradients - Normalize gradients each step (default: False)

    save_steps - Number of steps list that the algorithm uses to save the output (eg: [50, 100, 150], makes the algorithm save the output save at 50, 100 and 150)
    
    only_first_slot - Flag that ensures that only the first slot is optimizable (default: False, only used for style transfer)

    verbose - Algorithm verbosity (disable for no console output)
    '''
        
    model, objective_function, input_shape = objective.compile(len(parameterization.images))

    if optimizer is None:
        optimizer = Adam(learning_rate)

    else:
        optimizer = optimizer(learning_rate)
        
    shape = input_shape

    if image_shape:
        shape = (shape[0], image_shape, image_shape, shape[-1])

    if transformations == 'standard':
        transformations = standard((model.input.shape[1], model.input.shape[2]), shape[1])

    else:
        transformations = composition((model.input.shape[1], model.input.shape[2]), transformations)

    image_param = parameterization.function
    input = parameterization.images

    ascent = gradient_ascent(objective_function, image_param, shape, 
                             transformations, regularizers, 
                             normalize_gradients, only_first_slot)

    images = []
    inputs = []

    for slot in range(shape[0]):
        inputs.append(tf.Variable(input[slot]))

    for step in range(steps):
        if verbose:
            print('Step ' + str(step))

        gradients = ascent(model, inputs, tf.constant(step)) 

        for slot in range(shape[0]):
            if gradients[slot] is not None:
                if only_first_slot and slot == 0:
                    optimizer.apply_gradients([(-gradients[slot], inputs[slot])])

                elif not only_first_slot:
                    optimizer.apply_gradients([(-gradients[slot], inputs[slot])])

            last_iteration = step == steps - 1
            should_save = save_steps and (step + 1) in save_steps
            
            if should_save or last_iteration:
                images.append(image_param[slot](inputs[slot]))

    return images


def gradient_ascent(objective_function : Callable,
                    image_param: List[Callable],
                    input_shape: Tuple,
                    transformations: Callable,
                    regularizers: List[Callable],
                    normalize_gradients: bool,
                    only_first_slot: bool) -> Callable:
    
    @tf.function
    def step(model, inputs, step_index):
        with tf.GradientTape() as tape:
            tape.watch(inputs)

            model_outputs = []
            imgs = []
            
            for index in range(len(inputs)):
                i = image_param[index](inputs[index])

                if transformations:
                    if only_first_slot and index == 0:
                        i = transformations(i, step_index)

                    elif not only_first_slot:
                        i = transformations(i, step_index)

                    else:
                        model_shape = model.input.shape
                        i = tf.image.resize(i, tf.cast([model_shape[1], model_shape[2]], tf.int32))

                model_outputs.append(model(i))
                imgs.append(tf.image.resize(i, (input_shape[1], input_shape[2])))

            loss = []
            
            for i in range(len(model_outputs)):
                l = objective_function(model_outputs, i)

                for r in regularizers:
                    l -= r(imgs[i])
                
                loss.append(l)
            
            gradients = tape.gradient(loss, inputs)

            if normalize_gradients:
                gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients
    
    return step


def run_activation_atlas(path: str,
                         model: Wrapper,
                         layer: Union[int, str],
                         steps: int = 256,
                         learning_rate: int = 0.05,
                         batches: int = 10,
                         samples: int = -1,
                         image_shape: int = 512,
                         normalize_gradients: bool = False,
                         verbose = True) -> tf.Tensor:
    '''
    Inputs
    ----------
    path - Filesystem path for the image that will activate the model 

    model - Model wrapper

    layer - Layer to optimize (name or index)

    steps - Algorithm iterations (also known as epochs)

    learning_rate - Gradient ascent optimizer learning rate (default: 0.05)

    batches - Number of batches that will be used on the optimization (default: -1 to use all - it takes a long time)

    samples - Number of samples per batch (for performance purposes)

    image_shape - Images resolution (height and width, default: 512)

    normalize_gradients - Normalize gradients each step (default: False)

    verbose - Algorithm verbosity (disable for no console output)
    '''
    
    l = model.get_layer(layer)
    image = load_image(model.get_input_shape(), path)
    feature_extractor = Model(inputs = model.get_input(), outputs = l.output)
    
    if verbose:
        plt.imshow(image[0])
        plt.show()

        prediction_probabilities = model.predict(image)
        predictions = model.decode(prediction_probabilities.numpy())

        print([(class_name, prob) for (_, class_name, prob) in predictions])

    acts = feature_extractor(image)[0]
    acts_flat = tf.reshape(acts, ([-1] + [acts.shape[2]]))
    act_split = np.array_split(acts_flat, batches)

    unit = int(image_shape / 128)

    transformations = [
        padding(unit * 4),
        jitter(unit * 2, seed = 0),
        scale((0.92, 0.96), seed = 0),
        jitter(unit, seed = 0)
    ]

    images_flat = []

    samples = samples if len(act_split) > samples else len(act_split)

    for s in act_split:
        if samples != -1:
            act_samples = np.array(random.sample(list(s), samples if len(s) > samples else len(s)))

        else:
            act_samples = np.array(list(s))

        parameterization = Parameterization.image_fft(image_shape, slots = len(act_samples))

        objectives = Objective.sum([
            Objective.direction(model, layer, vectors = np.asarray(v), slots = i)
            for i, v in enumerate(act_samples)
        ])

        images = run(objectives, parameterization, steps = steps, 
                     learning_rate = learning_rate,
                     transformations = transformations,
                     image_shape = image_shape,
                     normalize_gradients = normalize_gradients,
                     verbose = verbose)
        
        for image in images:
            images_flat.append(image)

    images_flat = np.array(images_flat)

    return images_flat


def run_activation_layer(path: str,
                         model: Wrapper,
                         layers: List[Union[int, str]],
                         steps: int = 256,
                         learning_rate: int = 0.05,
                         image_shape: int = 512,
                         normalize_gradients: bool = False,
                         verbose = True) -> tf.Tensor:
    '''
    Inputs
    ----------
    path - Filesystem path for the image that will activate the model 

    model - Model wrapper

    layers - List of the layers to optimize (name or index)

    steps - Algorithm iterations (also known as epochs)

    learning_rate - Gradient ascent optimizer learning rate (default: 0.05)

    image_shape - Images resolution (height and width, default: 512)

    normalize_gradients - Normalize gradients each step (default: False)

    verbose - Algorithm verbosity (disable for no console output)
    '''

    image = load_image(model.get_input_shape(), path)

    if verbose:
        plt.imshow(image[0])
        plt.show()

        prediction_probabilities = model.predict(image)
        predictions = model.decode(prediction_probabilities.numpy())
    
        print([(class_name, prob) for (_, class_name, prob) in predictions])
    
    acts = []
    objectives = []

    for n, e in enumerate(layers):
        l = model.get_layer(e)    
        feature_extractor = Model(inputs = model.get_input(), outputs = l.output)  
        acts = feature_extractor(image)[0]

        objectives.append(Objective.direction(model, e, np.array(acts), slots = n))
    
    objectives = Objective.sum(objectives)

    parameterization = Parameterization.image_fft(image_shape, slots = len(layers))

    unit = int(image_shape / 128)

    transformations = [
        padding(unit * 4),
        jitter(unit * 2, seed = 0),
        scale((0.92, 0.96), seed = 0),
        jitter(unit, seed = 0)
    ]

    images = run(objectives, parameterization, steps = steps, 
                 learning_rate = learning_rate,
                 transformations = transformations,
                 image_shape = image_shape,
                 normalize_gradients = normalize_gradients,
                 verbose = verbose)

    return images


def run_style_transfer(image_path: str,
                       style_path: str,
                       model: Wrapper,
                       layers: List[Union[int, str]],
                       style_layers: List[Union[int, str]],
                       power: float = 100,
                       style_power: float = 1,
                       steps: int = 256,
                       learning_rate: int = 0.1,
                       image_shape: int = 512,
                       normalize_gradients: bool = False,
                       save_steps: List[int] = None,
                       verbose = True) -> tf.Tensor:
    '''
    Inputs
    ----------
    image_path - Filesystem path for the image that will activate the model 

    style_path - Filesystem path for the image that will activate the model for style purposes

    model - Model wrapper

    layers - List of the layers to optimize (name or index)

    style_layers - List of the layers to optimize for style purposes (name or index)

    power - Content layers power

    style_power - Style layers power

    steps - Algorithm iterations (also known as epochs)

    learning_rate - Gradient ascent optimizer learning rate (default: 0.1)

    image_shape - Images resolution (height and width, default: 512)

    normalize_gradients - Normalize gradients each step (default: False)

    save_steps - Number of steps list that the algorithm uses to save the output (eg: [50, 100, 150], makes the algorithm save the output save at 50, 100 and 150)

    verbose - Algorithm verbosity (disable for no console output)
    '''

    image = load_image(model.get_input_shape(), image_path)
    style = load_image(model.get_input_shape(), style_path)

    objective = power * Objective.activation_difference(model, layers, index = 1) +\
                style_power * Objective.activation_difference(model, style_layers, transform = gram_matrix, index = 2)
    
    parameterization = Parameterization.image_style_transfer(image, style)

    unit = int(image_shape / 128)

    transformations = [
        padding(unit * 4),
        jitter(unit * 2, seed = 0),
        scale((0.92, 0.96), seed = 0),
        jitter(unit, seed = 0)
    ]

    images = run(objective, parameterization,
                 steps = steps,
                 transformations = transformations,
                 learning_rate = learning_rate,
                 image_shape = image_shape,
                 normalize_gradients = normalize_gradients,
                 save_steps = save_steps,
                 verbose = verbose,
                 only_first_slot = True)
       
    return images


def run_maco(objective: Objective,
             spectrum_path: str,
             optimizer: Optimizer = None,
             steps: int = 256,
             std: Union[float, List[float]] = None,
             box_size: Union[float, List[float]] = None,
             crops: int = 32,
             learning_rate: float = 1.0,
             image_shape: int = 512,
             save_steps: List[int] = None,
             verbose = True) -> List[tf.Tensor]:
        
    '''
    Inputs
    ----------
    objective - Objective to optimize (e.g: channel, layer, direction, etc ...)

    spectrum_path - Decorrelation spectrum file path

    optimizer - Gradient ascent optimizer (eg: Adam)

    steps - Algorithm iterations (also known as epochs)

    std - Noise intensity value or list (standard deviation)
    
    box - Crop boxes size percentage or list

    crops - Number of crops each step

    learning_rate - Gradient ascent optimizer learning rate (default: 0.05)

    image_shape - Images resolution (height and width, default: 512)

    save_steps - Number of steps list that the algorithm uses to save the output (eg: [50, 100, 150], makes the algorithm save the output save at 50, 100 and 150)
  
    verbose - Algorithm verbosity (disable for no console output)
    '''
    
    model, objective_function, _ = objective.compile(slots = 1)

    if optimizer is None:
        optimizer = Nadam(learning_rate)

    else:
        optimizer = optimizer(learning_rate)
    
    parameterization = Parameterization.image_maco(image_shape, spectrum_path, 1)
    transformations = composition((model.input.shape[1], model.input.shape[2]), 
                                  [maco_standard(image_shape, crops, steps, box_size, std)])

    image_param = parameterization.function[0]
    input = tf.Variable(parameterization.images[0])

    transparency = tf.zeros((1, image_shape, image_shape, 3))

    images = []
    transparencies = []

    ascent = maco_gradient_ascent(objective_function, image_param, transformations)

    for step in range(steps):
        if verbose:
            print('Step ' + str(step))

        gradients, gradients_images = ascent(model, input, tf.constant(step))

        optimizer.apply_gradients([(-gradients, input)])
        transparency += tf.abs(gradients_images)

        last_iteration = step == steps - 1
        should_save = save_steps and (step + 1) in save_steps
        
        if should_save or last_iteration:
            transparencies.append(transparency)
            images.append(image_param(input))

    return images, transparencies


def maco_gradient_ascent(objective_function : Callable,
                         image_param: Callable,
                         transformations: Callable) -> Callable:

    @tf.function
    def maco_step(model, input, step_index):
        with tf.GradientTape() as tape:
            tape.watch(input)

            image = image_param(input)

            crops = transformations(image, step_index)

            loss = objective_function([model(crops)], 0)

            gradients = tape.gradient(loss, [input, image])
            gradients_phases, gradients_images = gradients

        return gradients_phases, gradients_images
    
    return maco_step