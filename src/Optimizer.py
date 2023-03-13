import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2


from typing import Callable
from tensorflow.keras.optimizers import Adam, Optimizer
from typing import Union, Tuple, List, Callable, Optional
from tensorflow.keras import Model, applications


from Objective import Objective
from Parameterization import Parameterization
from Transformation import *
from Miscellaneous import load_image


def run(objective: Objective,
        parameterization: Parameterization,
        optimizer: Optional[Optimizer] = None,
        steps: int = 256,
       learning_rate: float = 0.05,
        transformations: Union[List[Callable], str] = 'standard',
        regularizers: List[Callable] = [],
        image_shape: Optional[Tuple] = (512, 512),
        threshold: Optional[int] = None,
        verbose = True) -> List[tf.Tensor]:

    model, objective_function, input_shape = objective.compile(len(parameterization.images))

    if optimizer is None:
        optimizer = Adam(learning_rate)

    shape = input_shape

    if image_shape:
        shape = (shape[0], *image_shape, shape[-1])

    if transformations == 'standard':
        transformations = standard((model.input.shape[1], model.input.shape[2]), shape[1])

    else:
        transformations = composition((model.input.shape[1], model.input.shape[2]), transformations)

    image_param = parameterization.function
    input = parameterization.images
    
    ascent = _gradient_ascent(objective_function, image_param, shape, 
                              transformations, regularizers)

    images = []
    inputs = []

    for batch in range(shape[0]):
        inputs.append(tf.Variable(input[batch]))

    for step in range(steps):
        if verbose:
            print('Step ' + str(step))

        gradients = ascent(model, inputs)

        for batch in range(shape[0]):
            if gradients[batch] is not None:
                optimizer.apply_gradients([(-gradients[batch], inputs[batch])])

            last_iteration = step == steps - 1
            should_save = threshold and (step + 1) % threshold == 0
            
            if should_save or last_iteration:
                imgs = image_param[batch](inputs[batch])
                images.append(imgs)

    return images


def _gradient_ascent(objective_function : Callable,
                     image_param: Callable,
                     input_shape: Tuple,
                     transformations: Callable,
                     regularizers: List[Callable]) -> Callable:
    
    @tf.function
    def step(model, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)

            model_outputs = []
            imgs = []

            for index in range(len(inputs)):
                i = image_param[index](inputs[index])

                if transformations:
                    i = transformations(i)

                model_outputs.append(model(i))

                imgs.append(tf.image.resize(i, (input_shape[1], input_shape[2])))

            loss = []
            
            for i in range(len(model_outputs)):

                l = objective_function(model_outputs, i)

                for r in regularizers:
                    l += r(imgs[i])
                    
                loss.append(l)
                
            gradients = tape.gradient(loss, inputs)

        return gradients
    
    return step


def run_activation_atlas(path: str,
                         model: Model,
                         layer: Union[int, str],
                         steps: int = 256,
                         samples: int = 5,
                         image_shape: Optional[Tuple] = (512, 512),
                         verbose = True) -> tf.Tensor:
    
        if type(layer) is str:
            l = model.get_layer(name = layer)
        
        else:
            l = model.get_layer(index = layer)

        image = load_image(model.input.shape, path)
        feature_extractor = Model(inputs = model.input, outputs = l.output)
        
        if verbose:
            plt.imshow(image[0])
            plt.show()

            prediction_probabilities = model(image)
            predictions = applications.inception_v3.decode_predictions(prediction_probabilities.numpy())[0]
        
            print([(class_name, prob) for (_, class_name, prob) in predictions])

        acts = feature_extractor(image)[0]

        acts_flat = tf.reshape(acts, ([-1] + [acts.shape[2]]))

        parameterization = Parameterization.image_fft(image_shape[0], batches = acts_flat.shape[0])

        objectives = Objective.sum([
            Objective.direction(model, layer, vectors = np.asarray(v), batches = n)
            for n, v in enumerate(acts_flat)
        ])
        
        r = random.sample(range(0, acts_flat.shape[0]), samples)

        parameterization.images = [parameterization.images[nd] for nd in r]
        objectives.layers = [objectives.layers[nd] for nd in r]

        regularizers = []

        transformations = [
            padding(16),
            jitter(8, seed = 0),
            scale((0.92, 0.96), seed = 0),
            jitter(4, seed = 0) 
        ]

        images = run(objectives, parameterization, steps = steps, 
                     regularizers = regularizers,
                     transformations = transformations,
                     image_shape = image_shape,
                     verbose = verbose)
        
        return images


def run_activation_layer(path: str,
                         model: Model,
                         layer: Union[int, str],
                         batches: int = 1,
                         steps: int = 256,
                         image_shape: Optional[Tuple] = (512, 512),
                         verbose = True) -> tf.Tensor:

    if type(layer) is str:
            l = model.get_layer(name = layer)
        
    else:
        l = model.get_layer(index = layer)

    image = load_image(model.input.shape, path)
    feature_extractor = Model(inputs = model.input, outputs = l.output)
        
    if verbose:
        plt.imshow(image[0])
        plt.show()

        prediction_probabilities = model(image)
        predictions = applications.inception_v3.decode_predictions(prediction_probabilities.numpy())[0]
    
        print([(class_name, prob) for (_, class_name, prob) in predictions])

    acts = feature_extractor(image)[0]

    objective = Objective.direction(model, layer, np.array(acts))
    parameterization = Parameterization.image_fft(image_shape[0], batches = batches)

    regularizers = []

    transformations = [
        padding(16),
        jitter(8, seed = 0),
        scale((0.92, 0.96), seed = 0),
        jitter(4, seed = 0) 
    ]

    images = run(objective, parameterization, steps = steps, 
                 regularizers = regularizers,
                 transformations = transformations,
                 image_shape = image_shape,
                 verbose = verbose)

    return images


def model_predictions(path, model):
    _, x, y, z = model.input.shape

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (x, y))
    image = tf.reshape(image, (1, x, y, z))

    plt.imshow(image[0])
    plt.show()