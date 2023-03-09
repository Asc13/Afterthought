import tensorflow as tf

from typing import Callable
from tensorflow.keras.optimizers import Adam, Optimizer
from typing import Union, Tuple, List, Callable, Optional

import Objective
import Parameterization
from Miscellaneous import blur_conv
from Transformation import composition, standard


def run(objective: Objective,
        parameterization: Parameterization,
        optimizer: Optional[Optimizer] = None,
        steps: int = 256,
        learning_rate: float = 0.1,
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