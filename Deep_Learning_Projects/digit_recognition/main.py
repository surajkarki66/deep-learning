import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from model import build_model

class Model:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = build_model(input_shape= self.input_shape)

    def compile(self, learning_rate=0.001, optimizer=None, loss=None):
        """
        Configures the Model for training/predict.

        :param optimizer: optimizer for training
        @param learning_rate:
        """
        if optimizer == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return self.model

    
    def fit(self, epochs=1, train_ds = None, test_ds = None):
        history = self.model.fit(train_ds, epochs=epochs, validation_data = test_ds)
        return history

    def save(self, name=None):
        self.model.save(filepath=name)
        return "Your model saved"
        

    def predict(self, input):
        prediction = self.model.predict(input)
        
        return prediction






        


