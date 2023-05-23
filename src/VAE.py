import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU, Conv2D, Dense, Flatten, BatchNormalization, Reshape, Conv2DTranspose, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, binary_crossentropy

from src.Miscellaneous import plot_all

from typing import Tuple


class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_logvar = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape = (batch, dim))

        return z_mean + tf.exp(0.5 * z_logvar) * epsilon


class VAE():
    def __init__(self, num_classes: int, latent_dimension: int = 256, batches: int = 1, image_shape: Tuple = (512, 512, 3)):
        self.latent_dimension = latent_dimension
        self.batches = batches

        image_input, self.encoder, z_layer = VAE.encoder_function(latent_dimension, image_shape)
        class_input, self.decoder = VAE.decoder_function(num_classes, latent_dimension, image_shape)
        self.vae = Model([image_input, class_input], self.decoder((z_layer, class_input)), name = 'VAE')

        self.cross_entropy = binary_crossentropy
        self.optimizer = Adam(learning_rate = 1e-3)
    
    @staticmethod
    def encoder_function(latent_dimension: int, image_shape: Tuple) -> Model:
        image_input = Input(shape = image_shape, name = 'image_input')

        encoder_input = Conv2D(64, (5, 5), padding = 'same')(image_input)
        encoder = Conv2D(32, 3, strides = 2, padding = 'same', activation = 'relu')(encoder_input)
        encoder = Conv2D(64, 3, strides = 2, padding = 'same', activation = 'relu')(encoder)
        encoder = Flatten()(encoder)

        z_mean = Dense(latent_dimension)(encoder)
        z_logvar = Dense(latent_dimension)(encoder)
        z_layer = Sampling()([z_mean, z_logvar])
    
        return image_input, Model(image_input, [z_layer, z_mean, z_logvar], name = "Encoder"), z_layer


    @staticmethod
    def decoder_function(num_classes: int, latent_dimension: int, image_shape: Tuple) -> Model:
        class_input = Input(shape = (num_classes), name = 'class_input')

        z_input = Input(shape = (latent_dimension))

        decoder_input = tf.keras.layers.concatenate(inputs = [z_input, class_input])

        size_X = int(image_shape[0] / 4)    
        size_Y = int(image_shape[1] / 4)

        decoder = Dense(units = size_X * size_Y * 32, activation = tf.nn.relu)(decoder_input)
        decoder = Reshape(target_shape = (size_X, size_Y, 32))(decoder)
        decoder = Conv2DTranspose(64, 3, strides = 2, padding = "same",  activation = 'relu')(decoder)
        decoder = Conv2DTranspose(32, 3, strides = 2, padding = "same",  activation = 'relu')(decoder)

        decoder_output = Conv2DTranspose(3, 3, strides = (1, 1), padding = "same", activation = 'sigmoid')(decoder)

        return class_input, Model([z_input, class_input], decoder_output, name = 'Decoder')

    

    def kl_loss(self, z_logvar: tf.Tensor, z_mean: tf.Tensor) -> float:
        kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))

        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis = 1))
    
    
    def reconstruction_loss(self, train_x: tf.Tensor, x: tf.Tensor) -> float:  
        return tf.reduce_mean(tf.reduce_sum(self.cross_entropy(train_x, x)))


    @tf.function
    def train_step(self, train_x: tf.Tensor, train_y: tf.Tensor) -> Tuple:
        
        with tf.GradientTape() as tape:
            z, z_mean, z_logvar = self.encoder(train_x)
            x = self.decoder([z, train_y])

            kl_loss = self.kl_loss(z_logvar, z_mean)
            reconstruction_loss = self.reconstruction_loss(train_x, x)
            loss = kl_loss + reconstruction_loss

        gradients = tape.gradient(loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))
            
        return tf.reduce_mean(loss), tf.reduce_mean(kl_loss), tf.reduce_mean(reconstruction_loss)
    

    def train(self, path_to_save: str, dataset: tf.Tensor, epochs: int, verbose: bool = True):
        for epoch in range(epochs):
            epoch_loss = 0
            rec_loss = 0
            kls_loss = 0
            batch = 1

            for train_x, train_y in dataset:

                loss = self.train_step(train_x, train_y)
                epoch_loss += loss[0]
                kls_loss += loss[1]
                rec_loss += loss[2]
                batch += 1

            if verbose:
                samples = []

                z, _, _ = self.encoder(train_x)
                samples.extend(self.decoder([z, train_y]).numpy())

                samples = np.array(samples)
                shape = samples.shape
                samples = tf.reshape(samples, (shape[0], 1,) + shape[1:])

                plot_all(samples)
                
                print(f'Epoch {epoch + 1}: loss: {epoch_loss / batch}, Reconst loss: {rec_loss / batch}, KL loss: {kls_loss / batch}')

        self.save(path_to_save)


    def save(self, path: str):
        self.encoder.save(path + '/' + self.encoder.name)
        self.decoder.save(path + '/' + self.decoder.name)
        self.vae.save(path + '/' + self.vae.name)


    def load(self, path: str):
        self.encoder.load_weights(path + '/' + self.encoder.name)
        self.decoder.load_weights(path + '/' + self.decoder.name)
        self.vae.load_weights(path + '/' + self.vae.name)