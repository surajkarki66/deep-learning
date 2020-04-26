import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# data 
"""
observations = 1000
x = np.random.uniform(low=-10, high=10, size=(observations, 1))
z = np.random.uniform(-10, 10, (observations, 1))

generated_inputs = np.column_stack((x, z))

noise = np.random.uniform(-1, 1, (observations, 1))

generated_targets = 2 * x - 3 * z + 5 + noise

np.savez('TF1', inputs=generated_inputs, targets=generated_targets)

"""

training_data = np.load('TF1.npz')
input_size = 2
output_size = 1

# build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size)
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(training_data['inputs'], training_data['targets'], epochs=100)

# extracting weights and biases
print(model.layers[0].get_weights())

# predictions 
print(model.predict_on_batch(training_data['inputs']))