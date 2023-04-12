import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os


from typing import Callable
from tensorflow.keras.optimizers import Adam, Optimizer
from typing import Union, Tuple, List, Callable, Optional
from tensorflow.keras import Model, applications, preprocessing


from Objective import Objective
from Parameterization import Parameterization
from Transformation import *
from Miscellaneous import load_image
from Wrapper import *


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
                         model: Wrapper,
                         layer: Union[int, str],
                         steps: int = 256,
                         samples: int = 5,
                         image_shape: Optional[Tuple] = (512, 512),
                         verbose = True) -> tf.Tensor:
    
        l = model.get_layer(layer)
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

        if samples != -1:
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
                         model: Wrapper,
                         layers: List[Union[int, str]],
                         steps: int = 256,
                         image_shape: Optional[Tuple] = (512, 512),
                         verbose = True) -> tf.Tensor:

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
        l = model.get_layer()    
        feature_extractor = Model(inputs = model.get_input(), outputs = l.output)  
        acts = feature_extractor(image)[0]

        objectives.append(Objective.direction(model, e, np.array(acts), batches = n))
    
    objective = objectives[0]

    for o in range(1, len(objectives)):
        objective += objectives[o]

    parameterization = Parameterization.image_fft(image_shape[0], batches = len(layers))

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


def search_class(path: str,
                 classes_path: str,
                 model: Wrapper,
                 nb_classes: int,
                 batch_size: int = 50,
                 image_shape: Optional[Tuple] = (512, 512),
                 verbose = True) -> tf.Tensor:

    image = load_image(model.get_input_shape(), path)

    original_prediction_probabilities = model.predict(image)
    original_predictions = model.decode(original_prediction_probabilities.numpy(), top = 1)
    original_stats = [(class_name, prob) for (_, class_name, prob) in original_predictions][0]
    
    if verbose:
        plt.imshow(image[0])
        plt.show()

        print(original_stats)

    classes = np.arange(nb_classes)
    np.random.shuffle(classes)
    
    indexes_splitted = np.array_split(classes, batch_size)
    flag = True
    final_image = None

    _, x, y, _ = model.get_input_shape()

    unit = int(512 / 16)
    transformations = [
        padding(unit * 4, pad_value = 0.5),
        jitter(unit * 2, seed = 0),
        jitter(unit * 2, seed = 0),
        jitter(unit * 4, seed = 0),
        jitter(unit * 4, seed = 0),
        jitter(unit * 4, seed = 0),
        padding(unit * 6),
        scale((1.1, 1.8), seed = 0),
        blur_T(sigma_range = (1.0, 1.1)),
        flip(seed = 0),
        padding(unit * 2),
        jitter(unit * 2, seed = 0),
        jitter(unit * 2, seed = 0),
        padding(unit * 2),
    ]

    if classes_path is None:
        for indexes in indexes_splitted:
            if flag:
                parameterization = Parameterization.image_fft(image_shape[0], batches = len(indexes))
                objective = Objective.neuron(model, 'conv_preds', int(indexes[0]), batches = 0)

                for i in range(1, len(indexes)):
                    objective += Objective.neuron(model, 'conv_preds', int(indexes[i]), batches = i)
                    
                images = run(objective, parameterization, 
                             transformations = transformations,
                             steps = 1000, image_shape = image_shape,
                             verbose = False)
                
                for n, image in enumerate(images):
                    new = tf.image.resize(image, [x, y])
                    prediction_probabilities = model.predict(new)
                    predictions = model.decode(prediction_probabilities.numpy(), top = 1)

                    stats = [(class_name, prob) for (_, class_name, prob) in predictions][0]
                    
                    if verbose:
                        preprocessing.image.save_img(f'tests/classes/{indexes[n]}.png', image[0])
                        print(str(indexes[n]), stats)
                            
                    if stats[0] == original_stats[0]: 
                        final_image = new
                        flag = False
                        break
            else:
                break
    
    else:
        subfiles = [f.path for f in os.scandir(classes_path) if f.is_file()]

        avg = 0.0

        for n, f in enumerate(subfiles, start = 1):
            image = load_image(model.get_input_shape(), f)

            prediction_probabilities = model.predict(image)
            predictions = model.decode(prediction_probabilities.numpy(), top = 1)
            
            stats = [(class_name, prob) for (_, class_name, prob) in predictions][0]
            
            avg += stats[1]

            if stats[0] == original_stats[0]: 
                final_image = image
                flag = False
                print(f, ' Avg: ', avg / n)
                break
        
        plt.imshow(final_image[0])
        plt.show()
        

    return final_image