import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from model import Model

class NN:
    def __init__(self):
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        self.train_loss = tf.keras.metrics.Mean(name= 'train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.model = Model()

    @tf.function
    def train_step(self, inputs, outputs):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss(outputs, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(outputs, predictions)


        
    @tf.function
    def test_step(self, inputs, outputs):
        predictions = self.model(inputs)
        t_loss = self.loss(outputs, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(outputs, predictions)


    
    def train(self, epochs=1, train_ds = None, test_ds = None):
        for epoch in range(epochs):
            for inputs, outputs in train_ds:
                self.train_step(inputs, outputs)

            for test_inputs, test_outputs in test_ds:
                self.test_step(test_inputs, test_outputs)



            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy:{}'
            print(template.format(epoch+1,self.train_loss.result(),self.train_accuracy.result()*100,self.test_loss.result(),self.test_accuracy.result()*100))

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()


    def save(self, name=None):
        self.model.save(filepath=name)
        return "Your model saved"
        

    def predict(self, input):
        prediction = self.model.predict(input)
        
        return prediction






        


