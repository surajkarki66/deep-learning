import numpy as np
import tensorflow as tf


class ResnetBlock:

    def identity_block(self, X, f, filters, stage, block):
        """ Identity Block of ResNet """
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        F1, F2, F3 = filters
        
        X_shortcut = X
        
        # first component of the main path
        X = tf.keras.layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1),
                                padding = 'valid', name = conv_name_base + '2a',
                                kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        
        X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        # Second Component of the main path
        X = tf.keras.layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1),
                                padding = 'same', name = conv_name_base + '2b',
                                kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        
        X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        # third component of the main path
        X = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1),
                                padding = 'valid', name = conv_name_base + '2c',
                                kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
        
        # Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.Activation('relu')(X)
        
        return X



    def convolutional_block(self, X, f, filters, stage, block, s = 2):
        """ Convolution Block of ResNet """
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        F1, F2, F3 = filters
        
        X_shortcut = X

        # First component        
        X = tf.keras.layers.Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        
        X = tf.keras.layers.Conv2D(F2, (f, f), strides = (1,1), padding='same', name = conv_name_base + '2b', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = tf.keras.layers.Activation('relu')(X)

        X = tf.keras.layers.Conv2D(F3, (1, 1), strides = (1,1), padding='valid', name = conv_name_base + '2c', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        X_shortcut = tf.keras.layers.Conv2D(F3, (1, 1), strides = (s,s), padding='valid', name = conv_name_base + '1', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        # Add shortcut value to main path
        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.Activation('relu')(X)
        
        
        return X

            

        