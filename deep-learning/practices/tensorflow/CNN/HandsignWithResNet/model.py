import tensorflow as tf
import logging

from resnet_block import ResnetBlock


class Model:
    def __init__(self):
        self.resnet_block = ResnetBlock()

    def ResNet50(self, input_shape=(64, 64, 3), classes=6):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
         -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        """
        X_input = tf.keras.layers.Input(input_shape)
        X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

        # CONV2D -> BATCHNORM -> RELU - MAXPOOL
        X = tf.keras.layers.Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

        # CONVBLOCK -> IDBLOCK * 2
        X = self.resnet_block.convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = self.resnet_block.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.resnet_block.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # CONVBLOCK -> IDBLOCK * 3
        X = self.resnet_block.convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = self.resnet_block.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.resnet_block.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.resnet_block.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # CONVBLOCK -> IDBLOCK * 5
        X = self.resnet_block.convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = self.resnet_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.resnet_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.resnet_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.resnet_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.resnet_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # CONVBLOCK -> IDBLOCK * 2
        X = self.resnet_block.convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = self.resnet_block.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.resnet_block.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # Average Pool
        X = tf.keras.layers.AveragePooling2D()(X)

        # output layer
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

        # Create model
        model  = tf.keras.models.Model(inputs = X_input, outputs = X, name='ResNet50')
        model.summary()
        
        return model

        

    







    
        



