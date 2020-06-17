import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model

def vgg19(input_shape=None, classes=None):
    try:
        input_data = Input(shape=input_shape)
        
        # First convolution block
        conv1_1 = Conv2D(64, kernel_size=(3,3), padding= 'same',
                        activation='relu', name='conv1_1')(input_data)
        conv1_2 = Conv2D(64, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv1_2')(conv1_1)
        maxpool_1 = MaxPooling2D(pool_size=(2,2), strides= (2,2),
                                name='maxpool_1')(conv1_2)
        
        # Second convolution block
        conv2_1 = Conv2D(128, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv2_1')(maxpool_1)
        conv2_2 = Conv2D(128, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv2_2')(conv2_1)
        maxpool_2 = MaxPooling2D(pool_size=(2,2), strides= (2,2),
                                name='maxpool_2')(conv2_2)
        
        # Third convolution block
        conv3_1 = Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv3_1')(maxpool_2)
        conv3_2 = Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv3_2')(conv3_1)
        conv3_3 = Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv3_3')(conv3_2)
        conv3_4 = Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv3_4')(conv3_3)
        maxpool_3 = MaxPooling2D(pool_size=(2,2), strides= (2,2),
                                 name='maxpool_3')(conv3_4)
        
        # Fourth convolution block
        conv4_1 = Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv4_1')(maxpool_3)
        conv4_2 = Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv4_2')(conv4_1)
        conv4_3 = Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv4_3')(conv4_2)
        conv4_4 = Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv4_4')(conv4_3)
        maxpool_4 = MaxPooling2D(pool_size=(2,2), strides= (2,2),
                                 name='maxpool_4')(conv4_4)
        
        # Fifth convolution block
        conv5_1 = Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv5_1')(maxpool_4)
        conv5_2 = Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv5_2')(conv5_1)
        conv5_3 = Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv5_3')(conv5_2)
        conv5_4 = Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', name='conv5_4')(conv5_3)
        maxpool_5 = MaxPooling2D(pool_size=(2,2), strides= (2,2),
                                name='maxpool_5')(conv5_4)

        flatten = Flatten()(maxpool_5)
        fc1 = Dense(4096, activation= 'relu')(flatten)
        drop1 = Dropout(0.5)(fc1)
        fc2 = Dense(4096, activation= 'relu')(drop1)
        drop2 = Dropout(0.5)(fc2)
        output_data = Dense(classes, activation= 'softmax')(drop2)
        
        model = Model(inputs=input_data, outputs=output_data)
        
        return model
    except ValueError:
        print("Please provide input_shape and classes")
		
    
        