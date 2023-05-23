import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, Conv2D, Dense, Flatten, BatchNormalization, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from src.Miscellaneous import plot_all

from typing import Tuple


class DGN_AM():
    def __init__(self, latent_dimension: int = 256, batches: int = 1, image_shape: Tuple = (512, 512, 3)):
        self.latent_dimension = latent_dimension
        self.batches = batches

        self.discriminator = DGN_AM.discriminator_function(image_shape)
        self.generator = DGN_AM.generator_function(latent_dimension, image_shape)

        self.cross_entropy = BinaryCrossentropy()

        self.generator_optimizer = Adam(learning_rate = 2e-4, beta_1 = 0.5, decay = 2e-4 / 50)
        self.discriminator_optimizer = Adam(learning_rate = 2e-4, beta_1 = 0.5, decay = 2e-4 / 50)

    
    @staticmethod
    def discriminator_function(image_shape: Tuple) -> Model:
        discriminator = tf.keras.Sequential(name = 'Discriminator')

        discriminator.add(Conv2D(64, (5, 5), strides = 2, padding = 'same', input_shape = image_shape))
        discriminator.add(LeakyReLU())

        discriminator.add(Conv2D(64, (5, 5), strides = 2, padding = 'same'))
        discriminator.add(LeakyReLU())

        discriminator.add(Flatten())
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU())
        discriminator.add(Dense(1, activation = 'sigmoid'))

        return discriminator


    @staticmethod
    def generator_function(latent_dimension: int, image_shape: Tuple) -> Model:
        generator = tf.keras.Sequential(name = 'Generator')

        generator.add(Dense(512, input_shape = (latent_dimension, )))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization())

        size_X = int(image_shape[0] / 4)
        size_Y = int(image_shape[1] / 4)

        generator.add(Dense(size_X * size_Y * 64))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization())

        generator.add(Reshape((size_X, size_Y, 64)))
        generator.add(Conv2DTranspose(32, (5, 5), strides = 2, padding = 'same'))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization())

        generator.add(Conv2DTranspose(3, (5, 5), strides = 2, padding = 'same', activation = 'sigmoid'))

        return generator
    

    def discriminator_loss(self, real_output: tf.Tensor, fake_output: tf.Tensor) -> float:
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)

        return real_loss + fake_loss
    

    def generator_loss(self, fake_output: tf.Tensor) -> float:
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


    @tf.function
    def train_step(self, images: tf.Tensor) -> Tuple:
        noise = tf.random.normal([self.batches, self.latent_dimension])
        
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_images = self.generator(noise)

            real_output = self.discriminator(images)
            fake_output = self.discriminator(generated_images)

            disc_loss = self.discriminator_loss(real_output, fake_output)
            gen_loss = self.generator_loss(fake_output)

        generator_gradient = generator_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        return gen_loss, disc_loss
    

    def train(self, dataset: tf.Tensor, epochs: int, verbose: bool = True):
        input = tf.random.normal([self.batches, self.latent_dimension])

        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_dics_loss = 0

            for batch in dataset:
                gen_loss, disc_loss = self.train_step(batch)
                epoch_gen_loss += gen_loss
                epoch_dics_loss += disc_loss

            if verbose:
                predictions = self.generator(input, training = False)

                shape = predictions.shape
                predictions = tf.reshape(predictions, (shape[0], 1,) + shape[1:])

                plot_all(predictions)
                
                print(f'Epoch {epoch + 1}: generator_loss = {epoch_gen_loss}, discriminator_loss = {epoch_dics_loss}')


    def evaluate(self, images: tf.Tensor, samples: int = 100):
        x_real = images[np.random.choice(images.shape[0], samples, replace = True), :]

        y_real = np.ones((samples, 1))

        self.discriminator.compile(loss = self.cross_entropy, optimizer = self.discriminator_optimizer, metrics = ['accuracy'])
        _, real_accuracy = self.discriminator.evaluate(x_real, y_real, verbose = 0)

        latent_output = np.random.randn(self.latent_dimension * samples)
        latent_output = latent_output.reshape(samples, self.latent_dimension)

        x_fake = self.generator.predict(latent_output)

        y_fake = np.zeros((samples, 1))

        _, fake_accuracy = self.discriminator.evaluate(x_fake, y_fake, verbose = 0)

        print('Discriminator real accuracy: ', real_accuracy)
        print('Discriminator fake accuracy: ', fake_accuracy)


    def save(self, path: str):
        self.generator.save(path + '/' + self.generator.name)
        self.discriminator.save(path + '/' + self.discriminator.name)


    def load(self, path: str):
        self.generator.load_weights(path + '/' + self.generator.name)
        self.discriminator.load_weights(path + '/' + self.discriminator.name)