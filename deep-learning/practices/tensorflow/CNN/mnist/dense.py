import tensorflow as tf
import h5py
import numpy as np

class Dense(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Dense, self).__init__()
        self.units = units


    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.initializers.he_uniform()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                dtype='float32'),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs, self.w) + self.b


