import tensorflow as tf
import numpy as np

from utils import preproccess
from model import Model
from imagedetection import image




if __name__ == "__main__":
    x_train, y_train, x_test, y_test, classes = preproccess()
    
    m = Model()
    model = m.ResNet50(input_shape = (64, 64, 3), classes = 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10)
    model.save('handsign.h5')

    
