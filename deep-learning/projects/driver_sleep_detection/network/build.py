from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Input, Activation, MaxPooling2D


def build_model(input_shape=None, classes=2):
    """ Building our model"""

    input_data = Input(shape=input_shape, name="input")

    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_data)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = Conv2D(32,(3,3),activation='relu')(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,1))(x)

    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=input_data, outputs=x)

    return model

