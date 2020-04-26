import tensorflow as tf
import h5py
import numpy as np

class Dense(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Dense, self).__init__()
        self.units = units


    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
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



# Instantiates the layer.
linear_layer = Dense(4)



# This will also call `build(input_shape)` and create the weights.
#y = linear_layer(tf.ones((4, 3)))



class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = Dense(30)
        self.dense2 = Dense(20)
        self.dense3 = Dense(1)


    def call(self, inputs):
        x = self.dense1(inputs)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        x = tf.nn.relu(x)
        x = self.dense3(x)
        return tf.nn.sigmoid(x)


m = MyModel()
